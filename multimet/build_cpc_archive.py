# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ETL pipeline to build the unified Open-MultiMet daily CPC surface archive.

Ingests NOAA CPC Global Unified Daily Precipitation from the NOAA Physical
Sciences Laboratory (PSL) yearly NetCDF archives published at
``https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc``.

Standardizes spatial dimensions to the Caravan MultiMet specification:

- Dimensions: ``(time, latitude, longitude)``
- Latitude: 360 points from -89.75 to 89.75 (0.5 deg resolution, ascending)
- Longitude: 720 points from -179.75 to 179.75 (0.5 deg, shifted from
  ``[0, 360)``)
- Variable: ``cpc_precipitation`` (float32, mm/day, NaN over missing values)
"""

from __future__ import annotations

import argparse
import datetime
import functools
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from multimet import storage

# NOAA PSL publishes CPC Global Unified Precipitation from 1979 onwards.
DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = datetime.date.today().year

# Standard CPC 0.5 deg coordinates.
CPC_LATS = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
CPC_LONS = np.linspace(-179.75, 179.75, 720, dtype=np.float32)

# Name of the single data variable written to the archive.
CPC_VARIABLE = "cpc_precipitation"

NOAA_PSL_URL_TEMPLATE = (
    "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc"
)


def ensure_psl_cpc_netcdf(
    year: int,
    cache_dir: str,
    url_template: str = NOAA_PSL_URL_TEMPLATE,
    max_retries: int = 4,
) -> str:
  """Downloads and caches a yearly NOAA PSL CPC NetCDF file if not present.

  Args:
    year: Four-digit calendar year to fetch.
    cache_dir: Local directory where downloaded NetCDF files are cached.
    url_template: URL template accepting ``{year}``.
    max_retries: Maximum number of download attempts for transient errors.

  Returns:
    Path to the local ``precip.{year}.nc`` file.

  Raises:
    storage.UpstreamDataMissingError: If the upstream server returns HTTP 404.
    RuntimeError: If the download fails after ``max_retries`` attempts.
  """
  os.makedirs(cache_dir, exist_ok=True)
  local_path = os.path.join(cache_dir, f"precip.{year}.nc")
  if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
    return local_path

  url = url_template.format(year=year)
  temp_path = f"{local_path}.tmp.{os.getpid()}.{time.time_ns()}"
  logging.info("Downloading NOAA PSL CPC NetCDF for %d from %s...", year, url)

  last_error: Exception | None = None
  for attempt in range(max_retries):
    try:
      if (
          os.path.exists(local_path)
          and os.path.getsize(local_path) > 1024 * 1024
      ):
        return local_path
      req = urllib.request.Request(
          url,
          headers={"User-Agent": "OpenMultiMet/1.1 (Google Research)"},
      )
      with urllib.request.urlopen(req, timeout=180) as response, open(
          temp_path, "wb"
      ) as out_f:
        shutil.copyfileobj(response, out_f)
      if not (
          os.path.exists(local_path)
          and os.path.getsize(local_path) > 1024 * 1024
      ):
        os.replace(temp_path, local_path)
      logging.info(
          "Cached %s (%.1f MB)", local_path, os.path.getsize(local_path) / 1e6
      )
      return local_path
    except urllib.error.HTTPError as err:
      if os.path.exists(temp_path):
        os.remove(temp_path)
      if err.code == 404:
        raise storage.UpstreamDataMissingError(
            f"NOAA PSL CPC NetCDF for year {year} not published (HTTP 404): "
            f"{url}"
        ) from err
      last_error = err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
      last_error = err
      if os.path.exists(temp_path):
        os.remove(temp_path)

    wait_s = 3 * (2**attempt)
    logging.warning(
        "Failed download attempt %d/%d for year %d: %s. Retrying in %ds...",
        attempt + 1,
        max_retries,
        year,
        last_error,
        wait_s,
    )
    time.sleep(wait_s)

  raise RuntimeError(
      f"Failed to download NOAA PSL NetCDF for {year} after {max_retries} "
      f"attempts: {last_error}"
  ) from last_error


def process_cpc_netcdf_to_dataset(
    nc_path: str,
    target_start_date: pd.Timestamp | None = None,
    target_end_date: pd.Timestamp | None = None,
    *,
    trim_trailing_unpublished: bool = False,
) -> xr.Dataset | None:
  """Reads a yearly NOAA PSL NetCDF file and standardizes it to MultiMet schema.

  Performs spatial transformation:
  1. Inverts latitude axis (PSL: ``[89.75 .. -89.75]`` -> ``[-89.75 .. 89.75]``).
  2. Shifts longitude axis (PSL: ``[0.25 .. 359.75]`` -> ``[-179.75 .. 179.75]``).
  3. Masks missing values (``< 0`` or fill values) to ``np.nan``.

  Args:
    nc_path: Path to local ``precip.{year}.nc`` file.
    target_start_date: Optional inclusive lower bound filter on dates.
    target_end_date: Optional inclusive upper bound filter on dates.
    trim_trailing_unpublished: If ``True`` and the dataset contains valid dates
      followed by trailing all-NaN slices (such as future pre-allocated dates in
      NOAA PSL's current-year file), strips the trailing all-NaN dates.

  Returns:
    Standardized ``xarray.Dataset`` with dimensions
    ``(time, latitude, longitude)``, or ``None`` if no dates match.
  """
  with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
    precip_raw = ds["precip"].values
    time_raw = pd.to_datetime(ds["time"].values)
    stations_raw = None
    for st_cand in ("cpc_num_stations", "num_stations", "gcount"):
      if st_cand in ds.data_vars:
        stations_raw = ds[st_cand].values
        break

  # Invert latitude: north->south [89.75 .. -89.75] to [-89.75 .. 89.75].
  precip_lat_inv = precip_raw[:, ::-1, :]

  # Shift longitude: split at 180 deg (index 360) and concatenate [360:] + [:360].
  precip_shifted = np.concatenate(
      [precip_lat_inv[:, :, 360:], precip_lat_inv[:, :, :360]], axis=2
  )

  # Mask missing values (< 0) with NaN.
  precip_clean = np.where(precip_shifted < 0, np.nan, precip_shifted).astype(
      np.float32
  )

  stations_clean = None
  if stations_raw is not None:
    st_lat_inv = stations_raw[:, ::-1, :]
    st_shifted = np.concatenate(
        [st_lat_inv[:, :, 360:], st_lat_inv[:, :, :360]], axis=2
    )
    stations_clean = np.where(st_shifted < 0, np.nan, st_shifted).astype(
        np.float32
    )

  dates = pd.DatetimeIndex(time_raw.strftime("%Y-%m-%d"))

  if target_start_date is not None or target_end_date is not None:
    mask = np.ones(len(dates), dtype=bool)
    if target_start_date is not None:
      mask &= dates >= target_start_date
    if target_end_date is not None:
      mask &= dates <= target_end_date

    dates = dates[mask]
    precip_clean = precip_clean[mask]
    if stations_clean is not None:
      stations_clean = stations_clean[mask]

  if trim_trailing_unpublished and len(dates) > 0:
    finite_per_day = np.isfinite(precip_clean).any(axis=(1, 2))
    if finite_per_day.any():
      last_valid_idx = int(np.where(finite_per_day)[0][-1])
      dates = dates[: last_valid_idx + 1]
      precip_clean = precip_clean[: last_valid_idx + 1]
      if stations_clean is not None:
        stations_clean = stations_clean[: last_valid_idx + 1]
    else:
      return None

  if len(dates) == 0:
    return None

  data_vars = {
      CPC_VARIABLE: (
          ["time", "latitude", "longitude"],
          precip_clean,
          {
              "units": "mm/day",
              "long_name": "CPC Global Unified Gauge-Based Daily Precipitation",
              "standard_name": "precipitation_amount",
          },
      ),
  }
  if stations_clean is not None:
    data_vars["cpc_num_stations"] = (
        ["time", "latitude", "longitude"],
        stations_clean,
        {
            "units": "count",
            "long_name": "CPC Reporting Rain Gauge Stations per Grid Cell",
        },
    )

  return xr.Dataset(
      data_vars=data_vars,
      coords={
          "time": dates.values,
          "latitude": CPC_LATS,
          "longitude": CPC_LONS,
      },
      attrs={
          "title": (
              "Open-MultiMet NOAA CPC Global Unified Daily Precipitation"
              " Archive"
          ),
          "spatial_resolution": "0.50 degree",
          "description": (
              "Daily gauge-based global precipitation analysis from NOAA PSL"
              " standardized to Caravan MultiMet v1.1"
          ),
          "license": (
              "Usage Restrictions: None (Public Domain / NOAA PSL: "
              "https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html)"
          ),
          "institution": "NOAA PSL / CPC / Open-MultiMet",
          "citation": (
              "Chen et al. (2008) J. Geophys. Res. 113, D04110; Xie et al."
              " (2007) J. Hydrometeorol. 8, 607-626."
          ),
          "product": "CPC",
          "version": "1.1",
      },
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    is_initial_write: bool = False,
    consolidated: bool = False,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store."""
  chunk_days = min(30, len(ds_batch["time"]))
  storage.write_dataset_batch_to_zarr(
      ds_batch,
      target_zarr_url,
      project=project,
      is_initial_write=is_initial_write,
      consolidated=consolidated,
      time_chunk_size=chunk_days,
      max_retries=max_retries,
  )


def _extract_single_year_task(
    year: int,
    *,
    cache_dir: str,
    url_template: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    cleanup_cache: bool,
    year_starts: dict[int, pd.Timestamp],
) -> tuple[int, xr.Dataset | None]:
  """Worker function to download and transform a single year of CPC data."""
  t0 = time.time()
  logging.info(
      "Worker [%d] downloading CPC PSL NetCDF for year %d...",
      os.getpid(),
      year,
  )
  try:
    nc_path = ensure_psl_cpc_netcdf(
        year, cache_dir=cache_dir, url_template=url_template
    )
  except storage.UpstreamDataMissingError as err:
    logging.warning("Skipping unpublished year %d: %s", year, err)
    return year, None

  y_start = year_starts.get(year, start_date)
  ds_year = process_cpc_netcdf_to_dataset(
      nc_path,
      target_start_date=y_start,
      target_end_date=end_date,
      trim_trailing_unpublished=True,
  )

  if cleanup_cache and os.path.exists(nc_path):
    os.remove(nc_path)
    logging.info("Cleaned up cached file: %s", nc_path)

  logging.info(
      "Worker [%d] finished year %d in %.2fs",
      os.getpid(),
      year,
      time.time() - t0,
  )
  return year, ds_year


def build_cpc_archive(
    target_zarr: str,
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    start_date: str | None = None,
    end_date: str | None = None,
    project: str | None = None,
    cache_dir: str | None = None,
    source_url_template: str = NOAA_PSL_URL_TEMPLATE,
    cleanup_cache: bool = False,
    overwrite: bool = False,
    num_workers: int | None = None,
) -> None:
  """Executes the NOAA CPC daily gridded archive build.

  Args:
    target_zarr: Destination Zarr store URI or local directory path.
    start_year: First NOAA PSL yearly file to ingest.
    end_year: Last NOAA PSL yearly file to ingest (inclusive).
    start_date: Optional inclusive lower bound date filter (YYYY-MM-DD).
    end_date: Optional inclusive upper bound date filter (YYYY-MM-DD).
    project: Optional GCP project for GCS billing/authentication.
    cache_dir: Local directory for staging downloaded NetCDF files. If ``None``,
      an isolated temporary directory is created and cleaned up on exit.
    source_url_template: Upstream URL template accepting ``{year}``.
    cleanup_cache: Whether to delete cached NetCDF files after ingestion.
    overwrite: Whether to delete and rebuild an existing target store.
    num_workers: Number of parallel worker processes.
  """
  full_target_url, _, store_exists, has_consolidated, mapper = (
      storage.inspect_zarr_store(target_zarr, project=project, overwrite=overwrite)
  )

  if num_workers is None or num_workers <= 0:
    num_workers = min(32, os.cpu_count() or 4)

  t_start_filter = pd.Timestamp(start_date) if start_date else None
  t_end_filter = pd.Timestamp(end_date) if end_date else None

  if t_start_filter:
    start_year = max(start_year, t_start_filter.year)
  if t_end_filter:
    end_year = min(end_year, t_end_filter.year)

  years = list(range(start_year, end_year + 1))
  logging.info(
      "Building CPC archive for %d years: %d to %d (Workers: %d)",
      len(years),
      start_year,
      end_year,
      num_workers,
  )
  logging.info("Target: %s", full_target_url)

  is_first_write = not store_exists
  processed_years: set[int] = set()
  year_starts: dict[int, pd.Timestamp] = {}
  in_place_dates_set: set[str] = set()
  date_to_idx: dict[str, int] = {}

  if store_exists:
    with xr.open_zarr(mapper, consolidated=has_consolidated) as existing_ds:
      existing_times = pd.to_datetime(existing_ds["time"].values)
      date_to_idx = {
          t.strftime("%Y-%m-%d"): idx for idx, t in enumerate(existing_times)
      }
      # Detect any trailing all-NaN slices at the end of the store so they are
      # re-fetched and updated in-place rather than locking out future updates.
      trailing_nan_dates: list[pd.Timestamp] = []
      for idx in range(len(existing_times) - 1, -1, -1):
        sub_vals = np.asarray(existing_ds[CPC_VARIABLE].isel(time=idx).values)
        if not np.isfinite(sub_vals).any():
          trailing_nan_dates.append(pd.Timestamp(existing_times[idx]))
          in_place_dates_set.add(existing_times[idx].strftime("%Y-%m-%d"))
        else:
          break

      valid_times = (
          existing_times[: len(existing_times) - len(trailing_nan_dates)]
          if trailing_nan_dates
          else existing_times
      )

    for y in years:
      y_dates = valid_times[valid_times.year == y]
      expected_days = (
          366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
      )
      if len(y_dates) >= expected_days:
        processed_years.add(y)
      elif len(y_dates) > 0:
        next_date = pd.Timestamp(y_dates.max()) + pd.Timedelta(days=1)
        year_starts[y] = (
            max(t_start_filter, next_date)
            if t_start_filter is not None
            else next_date
        )
        logging.info(
            "Year %d partially present (%d valid dates up to %s). Resuming "
            "from %s.",
            y,
            len(y_dates),
            y_dates.max().strftime("%Y-%m-%d"),
            year_starts[y].strftime("%Y-%m-%d"),
        )

    remaining_years = [y for y in years if y not in processed_years]
    if not remaining_years:
      logging.info("Store already contains all requested years. Done!")
      return
    years = remaining_years
    is_first_write = False

  temp_cache_ctx = (
      tempfile.TemporaryDirectory(prefix="cpc_cache_")
      if cache_dir is None
      else None
  )
  effective_cache_dir = (
      temp_cache_ctx.name if temp_cache_ctx is not None else str(cache_dir)
  )

  total_days_processed = 0
  t0_total = time.time()
  worker_fn = functools.partial(
      _extract_single_year_task,
      cache_dir=effective_cache_dir,
      url_template=source_url_template,
      start_date=t_start_filter,
      end_date=t_end_filter,
      cleanup_cache=cleanup_cache,
      year_starts=year_starts,
  )

  try:
    if num_workers > 1 and len(years) > 1:
      mp_ctx = mp.get_context("spawn")
      with mp_ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap(worker_fn, years, chunksize=1)
        for y, ds_year in tqdm.tqdm(
            iterator,
            total=len(years),
            desc=f"Processing CPC ({num_workers} workers)",
        ):
          if ds_year is None:
            logging.warning(
                "No valid data returned for year %d within date filters.", y
            )
            continue
          total_days_processed += _write_cpc_year_dataset(
              ds_year,
              full_target_url=full_target_url,
              project=project,
              is_first_write=is_first_write,
              has_consolidated=has_consolidated,
              in_place_dates_set=in_place_dates_set,
              date_to_idx=date_to_idx,
          )
          is_first_write = False
    else:
      for y in tqdm.tqdm(years, desc="Processing CPC (sequential)"):
        _, ds_year = worker_fn(y)
        if ds_year is None:
          logging.warning(
              "No valid data returned for year %d within date filters.", y
          )
          continue
        total_days_processed += _write_cpc_year_dataset(
            ds_year,
            full_target_url=full_target_url,
            project=project,
            is_first_write=is_first_write,
            has_consolidated=has_consolidated,
            in_place_dates_set=in_place_dates_set,
            date_to_idx=date_to_idx,
        )
        is_first_write = False

    logging.info(
        "CPC Archive build complete! Processed %d total days across %d years "
        "in %.1fs.",
        total_days_processed,
        len(years),
        time.time() - t0_total,
    )
  finally:
    if temp_cache_ctx is not None:
      temp_cache_ctx.cleanup()
    elif cleanup_cache and cache_dir and os.path.exists(cache_dir):
      shutil.rmtree(cache_dir)
      logging.info("Cleaned up cache directory: %s", cache_dir)


def _write_cpc_year_dataset(
    ds_year: xr.Dataset,
    *,
    full_target_url: str,
    project: str | None,
    is_first_write: bool,
    has_consolidated: bool,
    in_place_dates_set: set[str],
    date_to_idx: dict[str, int],
) -> int:
  """Writes a processed CPC year dataset, splitting in-place vs append slices."""
  year_times = pd.to_datetime(ds_year["time"].values)
  in_place_mask = np.array(
      [t.strftime("%Y-%m-%d") in in_place_dates_set for t in year_times],
      dtype=bool,
  )
  append_mask = np.array(
      [t.strftime("%Y-%m-%d") not in date_to_idx for t in year_times],
      dtype=bool,
  )

  days_written = 0
  if in_place_mask.any():
    ds_in_place = ds_year.isel(time=in_place_mask)
    storage.write_dataset_batch_in_place(
        ds_in_place,
        full_target_url,
        project=project,
        date_to_idx=date_to_idx,
    )
    days_written += len(ds_in_place["time"])

  if append_mask.any():
    ds_append = ds_year.isel(time=append_mask)
    write_batch_to_zarr(
        ds_append,
        full_target_url,
        project=project,
        is_initial_write=is_first_write,
        consolidated=has_consolidated,
    )
    days_written += len(ds_append["time"])

  return days_written


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the CPC archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-cpc-archive",
      description="Build the unified CPC daily precipitation archive.",
  )
  parser.add_argument(
      "--target_zarr",
      type=str,
      required=True,
      help="Destination Zarr store (local path or explicit gs:// URI).",
  )
  parser.add_argument(
      "--start_year",
      type=int,
      default=DEFAULT_START_YEAR,
      help="First NOAA PSL yearly file to ingest (e.g. 1979).",
  )
  parser.add_argument(
      "--end_year",
      type=int,
      default=DEFAULT_END_YEAR,
      help="Last NOAA PSL yearly file to ingest (inclusive).",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=None,
      help="Optional lower bound date filter within the year range.",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=None,
      help="Optional upper bound date filter within the year range.",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="Optional GCP project used for billing/auth of gs:// requests.",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      default=None,
      help="Local directory used to cache downloaded NOAA PSL NetCDF files.",
  )
  parser.add_argument(
      "--source_url_template",
      type=str,
      default=NOAA_PSL_URL_TEMPLATE,
      help="Upstream NOAA PSL URL template containing {year}.",
  )
  parser.add_argument(
      "--cleanup_cache",
      action="store_true",
      help="Delete downloaded NetCDF files once they have been written.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Delete and rebuild the target store instead of resuming it.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=min(32, os.cpu_count() or 4),
      help="Number of parallel extraction worker processes.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-cpc-archive``."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  build_cpc_archive(
      target_zarr=args.target_zarr,
      start_year=args.start_year,
      end_year=args.end_year,
      start_date=args.start_date,
      end_date=args.end_date,
      project=args.project,
      cache_dir=args.cache_dir,
      source_url_template=args.source_url_template,
      cleanup_cache=args.cleanup_cache,
      overwrite=args.overwrite,
      num_workers=args.num_workers,
  )


if __name__ == "__main__":
  main()
