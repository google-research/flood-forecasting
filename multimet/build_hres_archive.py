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

"""ETL pipeline to build the unified Open-MultiMet daily HRES surface archive.

Ingests ECMWF IFS HRES 0.25-degree forecasts from public cloud archives:

#. WeatherBench 2 (2016-01-01 to 2023-01-10): public Zarr archive.
#. ECMWF Open Data (2023-07-13 to present): operational 0.25-degree GRIB2
   archive (``ifs/0p25/oper``).

The intermediate window (2023-01-11 to 2023-07-12) between the end of the
WeatherBench 2 archive and the start of the ECMWF Open Data archive is
initialized with NaN slices when spanned by a multi-year build so that the
daily time coordinate stays contiguous and can be backfilled in-place.
Trailing unpublished dates at the end of a requested date range are never
written as NaN slices, preventing premature time-axis initialization.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing as mp
import os
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

try:
  import eccodes  # type: ignore[import-untyped]
except ImportError:
  eccodes = None

try:
  import gcsfs  # type: ignore[import-untyped]
except ImportError:
  gcsfs = None

from multimet import storage

WB2_HRES_ZARR = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
ECMWF_OPEN_DATA_BUCKET = "ecmwf-open-data"

# First forecast initialization date available in WeatherBench 2.
DEFAULT_START_DATE = "2016-01-01"

WB2_CUTOFF_DATE = pd.Timestamp("2023-01-10")
OPEN_DATA_START_DATE = pd.Timestamp("2023-07-13")

LEAD_STEPS_WB2 = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

# Number of daily lead steps (lead day 1 .. 10) stored for every init date.
NUM_LEAD_DAYS = 10

# Canonical 0.25 degree HRES grid used by the unified archive.
HRES_LATS = np.linspace(-90.0, 90.0, 721, dtype=np.float32)
HRES_LONS = np.linspace(0.0, 359.75, 1440, dtype=np.float32)

# Surface variables written to the archive, in canonical order.
HRES_VARIABLES = (
    "temperature_2m",
    "surface_pressure",
    "total_precipitation",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
)

HRES_ATTRS = {
    "title": "Open-MultiMet ECMWF HRES Daily Surface Forecast Archive",
    "spatial_resolution": "0.25 degree",
    "description": (
        "Daily-aggregated surface forecast variables (lead days 1..10) from"
        " ECMWF IFS HRES"
    ),
    "license": "CC-BY-4.0",
    "institution": "ECMWF / Open-MultiMet",
    "Sources": (
        "Canonical upstream source is European Centre for Medium-Range Weather"
        " Forecasts (ECMWF) Operational IFS High-Resolution (HRES) 00Z"
        " Forecasts (0.25 degree, lead days 1..10). Ingested from three"
        " contiguous archives with no date gaps:\n"
        "1. 2016-01-01 to 2023-01-10: WeatherBench 2 ECMWF IFS HRES 0.25"
        " degree Public Zarr Archive"
        " (gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr;"
        " provides temperature_2m, surface_pressure, and total_precipitation;"
        " surface_net_solar_radiation and surface_net_thermal_radiation are not"
        " included in WeatherBench 2 and are NaN only during 2016-01-01 to"
        " 2023-01-10).\n"
        "2. 2023-01-11 to 2023-07-11: Google Flood Forecasting internal ECMWF"
        " IFS HRES 00Z daily surface NetCDF archive"
        " (gs://ecmwf-downloads/flood-forecasting/single-levels/daily-surface-regridded/{YYYY-MM-DD}-tp-2t-sp-ssr-str-sf.nc;"
        " provides all 5 variables: temperature_2m, surface_pressure,"
        " total_precipitation, surface_net_solar_radiation, and"
        " surface_net_thermal_radiation).\n"
        "3. 2023-07-12 to present: ECMWF Open Data Operational IFS HRES GRIB2"
        " Archive (gs://ecmwf-open-data/<YYYYMMDD>/00z/,"
        " https://www.ecmwf.int/en/forecasts/datasets/open-data,"
        " 0p4-beta/oper on 2023-07-12 and ifs/0p25/oper from 2023-07-13 onward;"
        " provides all 5 variables: temperature_2m, surface_pressure,"
        " total_precipitation, surface_net_solar_radiation, and"
        " surface_net_thermal_radiation)."
    ),
    "Code_Repository": "https://github.com/google-research/flood-forecasting",
    "Code_Package": (
        "https://github.com/google-research/flood-forecasting/tree/main/multimet"
    ),
    "Generated_By": "multimet.build_hres_archive",
}


def deaccumulate(
    accumulated: np.ndarray, clip_negative: bool = False
) -> np.ndarray:
  """Converts a run-cumulative forecast stack into per-lead-day increments.

  ECMWF reports ``tp``, ``ssr`` and ``str`` as totals accumulated since the
  forecast initialization time, so lead day ``d`` must be differenced against
  lead day ``d - 1`` to recover the value for that day alone. Lead day 1 is
  already a single-day total and is passed through unchanged.

  Args:
    accumulated: Array whose leading axis is the lead-day axis.
    clip_negative: If ``True``, negative increments are floored at zero. Use
      this for precipitation, where a negative increment can only be numerical
      noise. Radiation fluxes are genuinely signed and must not be clipped.

  Returns:
    An array of the same shape holding per-lead-day increments.
  """
  daily = np.empty_like(accumulated)
  daily[0] = accumulated[0]
  difference = accumulated[1:] - accumulated[:-1]
  daily[1:] = np.maximum(0.0, difference) if clip_negative else difference
  return daily


class WeatherBench2Source:
  """Extracts daily aggregates from the WeatherBench 2 HRES Zarr archive."""

  def __init__(self, zarr_path: str = WB2_HRES_ZARR):
    full_path, is_remote = storage.resolve_zarr_target(zarr_path)
    if is_remote and full_path.startswith("gs://"):
      if gcsfs is None:
        raise ImportError("gcsfs is required to read WeatherBench 2 from GCS.")
      fs = gcsfs.GCSFileSystem(token="anon")
      self.mapper = fs.get_mapper(full_path.removeprefix("gs://"))
      self.ds = xr.open_zarr(self.mapper, decode_timedelta=False)
    else:
      self.ds = xr.open_zarr(full_path, decode_timedelta=False)
    self.latitudes = self.ds["latitude"].values.astype(np.float32)
    self.longitudes = self.ds["longitude"].values.astype(np.float32)

  def extract_date(self, date: pd.Timestamp) -> dict[str, np.ndarray] | None:
    """Extracts 10 lead days for a single forecast date."""
    target_time = np.datetime64(date.strftime("%Y-%m-%dT00:00:00"))
    if target_time not in self.ds["time"].values:
      return None

    day_sel = self.ds.sel(time=target_time)

    # 1. Precipitation: total_precipitation_24hr is precomputed at 24h multiples.
    tp_daily = (
        day_sel["total_precipitation_24hr"]
        .sel(prediction_timedelta=LEAD_STEPS_WB2)
        .values.astype(np.float32)
    )

    # 2. Temperature and Pressure: 24h mean over each lead day.
    temp_slices = []
    pres_slices = []
    for d in range(1, 11):
      steps = [d * 24 - 18, d * 24 - 12, d * 24 - 6, d * 24]
      t_mean = (
          day_sel["2m_temperature"]
          .sel(prediction_timedelta=steps)
          .mean(dim="prediction_timedelta")
          .values
      )
      p_mean = (
          day_sel["surface_pressure"]
          .sel(prediction_timedelta=steps)
          .mean(dim="prediction_timedelta")
          .values
      )
      temp_slices.append(t_mean)
      pres_slices.append(p_mean)

    temp_daily = np.stack(temp_slices, axis=0).astype(np.float32)
    pres_daily = np.stack(pres_slices, axis=0).astype(np.float32)

    # Solar and thermal radiation are unavailable in WB2 HRES.
    nan_grid = np.full(
        (10, len(self.latitudes), len(self.longitudes)),
        np.nan,
        dtype=np.float32,
    )

    return {
        "temperature_2m": temp_daily,
        "surface_pressure": pres_daily,
        "total_precipitation": tp_daily,
        "surface_net_solar_radiation": nan_grid,
        "surface_net_thermal_radiation": nan_grid.copy(),
    }


class ECMWFOpenDataSource:
  """Extracts 0.25-degree daily surface aggregates from ECMWF Open Data GRIB2."""

  def __init__(self, bucket: str = ECMWF_OPEN_DATA_BUCKET):
    if gcsfs is None:
      raise ImportError("gcsfs is required to read ECMWF Open Data from GCS.")
    self.bucket = bucket.removeprefix("gs://").strip("/")
    self.fs = gcsfs.GCSFileSystem(token="anon")
    self.lead_steps = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

  def extract_date(
      self,
      date: pd.Timestamp,
      target_lat: np.ndarray,
      target_lon: np.ndarray,
  ) -> dict[str, np.ndarray] | None:
    """Extracts 10 lead days for ``date`` from the 0.25-degree IFS stream.

    Does not fall back to coarser grids (such as ``0p4-beta``) or interpolate.
    Raises if any GRIB message is corrupted or missing required parameters.
    """
    if eccodes is None:
      raise ImportError(
          "eccodes is required to decode ECMWF Open Data GRIB2 files. "
          "Install python-eccodes / eccodes."
      )

    date_str = date.strftime("%Y%m%d")
    prefix = f"{self.bucket}/{date_str}/00z/ifs/0p25/oper/{date_str}000000"
    if not self.fs.exists(f"{prefix}-24h-oper-fc.index"):
      return None

    expected_shape = (len(target_lat), len(target_lon))
    raw_steps: dict[str, list[np.ndarray]] = {
        var: [] for var in ("2t", "sp", "tp", "ssr", "str")
    }

    for step in self.lead_steps:
      idx_path = f"{prefix}-{step}h-oper-fc.index"
      grib_path = f"{prefix}-{step}h-oper-fc.grib2"

      offsets: dict[str, tuple[int, int]] = {}
      with self.fs.open(idx_path, "r") as f:
        for line in f:
          msg = json.loads(line)
          param = msg.get("param")
          if msg.get("levtype") == "sfc" and param in raw_steps:
            offsets[param] = (int(msg["_offset"]), int(msg["_length"]))

      missing_params = [p for p in raw_steps if p not in offsets]
      if missing_params:
        raise RuntimeError(
            f"Missing surface parameters {missing_params} in {idx_path}"
        )

      with self.fs.open(grib_path, "rb") as f:
        for param, param_steps in raw_steps.items():
          off, length = offsets[param]
          f.seek(off)
          raw_bytes = f.read(length)
          gid = eccodes.codes_new_from_message(raw_bytes)
          try:
            vals = eccodes.codes_get_values(gid)
          finally:
            eccodes.codes_release(gid)

          if vals.size != expected_shape[0] * expected_shape[1]:
            raise ValueError(
                f"Grid size mismatch for {param} at {date_str} +{step}h: "
                f"got {vals.size} values, expected {expected_shape}"
            )
          param_steps.append(vals.reshape(expected_shape).astype(np.float32))

    return {
        "temperature_2m": np.stack(raw_steps["2t"], axis=0),
        "surface_pressure": np.stack(raw_steps["sp"], axis=0),
        "total_precipitation": deaccumulate(
            np.stack(raw_steps["tp"], axis=0), clip_negative=True
        ),
        "surface_net_solar_radiation": deaccumulate(
            np.stack(raw_steps["ssr"], axis=0)
        ),
        "surface_net_thermal_radiation": deaccumulate(
            np.stack(raw_steps["str"], axis=0)
        ),
    }


_worker_wb2: WeatherBench2Source | None = None
_worker_open_data: ECMWFOpenDataSource | None = None
_target_lat: np.ndarray = HRES_LATS
_target_lon: np.ndarray = HRES_LONS


def _init_worker(
    project: str | None = None,
    wb2_zarr: str = WB2_HRES_ZARR,
    ecmwf_open_data_bucket: str = ECMWF_OPEN_DATA_BUCKET,
    need_wb2: bool = True,
    need_open_data: bool = True,
) -> None:
  """Initializes worker-process source clients only for required windows."""
  global _worker_wb2, _worker_open_data, _target_lat, _target_lon  # noqa: PLW0603
  del project  # Public upstream sources use anonymous access.
  _worker_wb2 = WeatherBench2Source(wb2_zarr) if need_wb2 else None
  _worker_open_data = (
      ECMWFOpenDataSource(ecmwf_open_data_bucket) if need_open_data else None
  )
  if _worker_wb2 is not None:
    _target_lat = _worker_wb2.latitudes
    _target_lon = _worker_wb2.longitudes
  else:
    _target_lat = HRES_LATS
    _target_lon = HRES_LONS


def _make_nan_date_payload(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> dict[str, np.ndarray]:
  """Creates an unaliased all-NaN date dictionary for a missing date."""
  shape = (NUM_LEAD_DAYS, len(latitudes), len(longitudes))
  return {
      var: np.full(shape, np.nan, dtype=np.float32) for var in HRES_VARIABLES
  }


def _is_date_payload_all_nan(payload: dict[str, np.ndarray]) -> bool:
  """Returns True if every variable in ``payload`` is entirely NaN."""
  return all(bool(np.isnan(arr).all()) for arr in payload.values())


def _extract_single_date(
    dt: pd.Timestamp,
    max_retries: int = 3,
) -> tuple[pd.Timestamp, dict[str, np.ndarray]]:
  """Extracts a single forecast date, retrying transient operational errors.

  Genuinely absent upstream dates (where the source returns ``None`` or raises
  :class:`storage.UpstreamDataMissingError`) return an all-NaN dictionary so
  the caller can decide whether the date is an interior gap (written as NaN and
  logged in ``missing_dates``) or a trailing unpublished date (not written).
  Any operational, decode, or network error that persists across ``max_retries``
  is raised rather than swallowed as a NaN slice.
  """
  last_err: Exception | None = None
  for extract_attempt in range(max_retries):
    try:
      if dt <= WB2_CUTOFF_DATE:
        if _worker_wb2 is None:
          raise RuntimeError("WeatherBench2Source was not initialized.")
        date_data = _worker_wb2.extract_date(dt)
      elif dt < OPEN_DATA_START_DATE:
        # Gap between WeatherBench 2 (2023-01-10) and ECMWF Open Data (2023-07-13).
        date_data = None
      else:
        if _worker_open_data is None:
          raise RuntimeError("ECMWFOpenDataSource was not initialized.")
        date_data = _worker_open_data.extract_date(
            dt, _target_lat, _target_lon
        )
      if date_data is None:
        logging.warning(
            "No upstream data published for date %s.",
            dt.strftime("%Y-%m-%d"),
        )
        return dt, _make_nan_date_payload(_target_lat, _target_lon)
      return dt, date_data
    except storage.UpstreamDataMissingError:
      return dt, _make_nan_date_payload(_target_lat, _target_lon)
    except ImportError:
      raise
    except Exception as err:  # noqa: BLE001
      last_err = err
      logging.warning(
          "Error extracting date %s (attempt %d/%d): %s",
          dt.strftime("%Y-%m-%d"),
          extract_attempt + 1,
          max_retries,
          err,
      )
      if extract_attempt < max_retries - 1:
        time.sleep(2 * (2**extract_attempt))

  raise RuntimeError(
      f"Failed to extract HRES data for {dt.strftime('%Y-%m-%d')} after "
      f"{max_retries} attempts: {last_err}"
  ) from last_err


def build_batch_dataset(
    batch_dates: Sequence[pd.Timestamp],
    batch_data: dict[str, Sequence[np.ndarray]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> xr.Dataset:
  """Assembles one write batch into the canonical HRES archive schema."""
  return xr.Dataset(
      data_vars={
          var: (
              ["time", "lead_time", "latitude", "longitude"],
              np.stack(values, axis=0),
          )
          for var, values in batch_data.items()
      },
      coords={
          "time": list(batch_dates),
          "lead_time": np.arange(1, NUM_LEAD_DAYS + 1, dtype=np.int32),
          "latitude": latitudes,
          "longitude": longitudes,
      },
      attrs=dict(HRES_ATTRS),
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    is_initial_write: bool = False,
    consolidated: bool = False,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store with retries."""
  storage.write_dataset_batch_to_zarr(
      ds_batch,
      target_zarr_url,
      project=project,
      is_initial_write=is_initial_write,
      consolidated=consolidated,
      time_chunk_size=1,
      max_retries=max_retries,
  )


def write_batch_in_place(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    date_to_idx: dict[str, int] | None = None,
    max_retries: int = 5,
) -> None:
  """Writes a batch of dates directly in-place into existing Zarr slices."""
  storage.write_dataset_batch_in_place(
      ds_batch,
      target_zarr_url,
      project=project,
      date_to_idx=date_to_idx,
      max_retries=max_retries,
  )


def build_hres_archive(
    start_date: str,
    end_date: str,
    target_zarr: str,
    *,
    project: str | None = None,
    wb2_zarr: str = WB2_HRES_ZARR,
    ecmwf_open_data_bucket: str = ECMWF_OPEN_DATA_BUCKET,
    batch_size: int = 10,
    overwrite: bool = False,
    in_place: bool = False,
    num_workers: int | None = None,
    failure_log: str | None = None,
) -> None:
  """Executes the HRES daily surface forecast archive build.

  Args:
    start_date: First forecast initialization date (YYYY-MM-DD).
    end_date: Last forecast initialization date (YYYY-MM-DD).
    target_zarr: Destination Zarr store path or explicit ``gs://`` URI.
    project: Optional GCP project for GCS billing/authentication.
    wb2_zarr: Input Zarr URI/path for WeatherBench 2 HRES archive.
    ecmwf_open_data_bucket: Input GCS bucket/prefix for ECMWF Open Data.
    batch_size: Number of forecast dates accumulated per Zarr write.
    overwrite: Whether to delete and rebuild the target store from scratch.
    in_place: Whether to rewrite existing dates in place.
    num_workers: Number of parallel extraction worker processes.
    failure_log: Optional path to write a JSON report of missing/failed dates.
  """
  full_target_url, _, store_exists, has_consolidated, mapper = (
      storage.inspect_zarr_store(
          target_zarr, project=project, overwrite=overwrite
      )
  )

  if num_workers is None or num_workers <= 0:
    num_workers = min(32, os.cpu_count() or 4)

  requested_dates = pd.date_range(start_date, end_date, freq="1D")
  date_to_idx: dict[str, int] = {}
  in_place_dates = pd.DatetimeIndex([])
  append_dates = requested_dates

  if in_place:
    if not store_exists:
      raise ValueError(
          f"Cannot run --in_place: target store {full_target_url} does not exist."
      )
    with xr.open_zarr(mapper, consolidated=False) as existing_ds:
      time_pd = pd.to_datetime(existing_ds["time"].values)
      date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)}
    in_place_dates = requested_dates
    append_dates = pd.DatetimeIndex([])
  elif store_exists:
    in_place_dates, append_dates, date_to_idx = storage.plan_archive_resume(
        mapper, requested_dates, has_consolidated=has_consolidated
    )
    if len(in_place_dates) == 0 and len(append_dates) == 0:
      logging.info(
          "Store already contains all valid dates up to %s. Nothing to do!",
          end_date,
      )
      return

  all_work_dates = in_place_dates.union(append_dates)
  need_wb2 = bool((all_work_dates <= WB2_CUTOFF_DATE).any())
  need_open_data = bool((all_work_dates >= OPEN_DATA_START_DATE).any())

  is_first_write = not store_exists
  newly_valid_dates: list[str] = []
  newly_missing_dates: list[str] = []
  newly_failed_dates: list[str] = []

  def _run_extraction(
      dates_to_run: pd.DatetimeIndex,
  ) -> list[tuple[pd.Timestamp, dict[str, np.ndarray]]]:
    if len(dates_to_run) == 0:
      return []
    if num_workers > 1 and len(dates_to_run) > 1:
      mp_ctx = mp.get_context("spawn")
      with mp_ctx.Pool(
          processes=min(num_workers, len(dates_to_run)),
          initializer=_init_worker,
          initargs=(
              project,
              wb2_zarr,
              ecmwf_open_data_bucket,
              need_wb2,
              need_open_data,
          ),
      ) as pool:
        iterator = pool.imap(_extract_single_date, dates_to_run, chunksize=1)
        return list(
            tqdm.tqdm(
                iterator,
                total=len(dates_to_run),
                desc=f"Processing HRES ({num_workers} workers)",
            )
        )
    _init_worker(
        project,
        wb2_zarr,
        ecmwf_open_data_bucket,
        need_wb2,
        need_open_data,
    )
    return [
        _extract_single_date(dt)
        for dt in tqdm.tqdm(dates_to_run, desc="Processing HRES (sequential)")
    ]

  try:
    # 1. Execute any in-place backfills (either explicit --in_place or healing
    # previously missing/trailing-NaN slices on resume).
    if len(in_place_dates) > 0:
      in_place_results = _run_extraction(in_place_dates)
      target_lat = _target_lat
      target_lon = _target_lon
      ip_dates: list[pd.Timestamp] = []
      ip_data: dict[str, list[np.ndarray]] = {var: [] for var in HRES_VARIABLES}

      for dt, date_data in in_place_results:
        d_str = dt.strftime("%Y-%m-%d")
        if _is_date_payload_all_nan(date_data):
          newly_missing_dates.append(d_str)
          if not in_place:
            continue
        else:
          newly_valid_dates.append(d_str)

        ip_dates.append(dt)
        for k, k_list in ip_data.items():
          k_list.append(date_data[k])
        if len(ip_dates) >= batch_size:
          ds_batch = build_batch_dataset(
              ip_dates, ip_data, target_lat, target_lon
          )
          write_batch_in_place(
              ds_batch, full_target_url, project=project, date_to_idx=date_to_idx
          )
          ip_dates = []
          ip_data = {var: [] for var in HRES_VARIABLES}

      if ip_dates:
        ds_batch = build_batch_dataset(
            ip_dates, ip_data, target_lat, target_lon
        )
        write_batch_in_place(
            ds_batch, full_target_url, project=project, date_to_idx=date_to_idx
        )

    # 2. Execute appends for dates beyond the store's current max timestamp.
    # Trailing all-NaN (unpublished) dates are held in pending_missing and ONLY
    # written if followed by at least one valid date (proving they are an
    # interior gap rather than future/unpublished tail dates).
    if len(append_dates) > 0:
      append_results = _run_extraction(append_dates)
      target_lat = _target_lat
      target_lon = _target_lon

      batch_dates: list[pd.Timestamp] = []
      batch_data: dict[str, list[np.ndarray]] = {
          var: [] for var in HRES_VARIABLES
      }
      pending_missing: list[tuple[pd.Timestamp, dict[str, np.ndarray]]] = []

      def _flush_append_batch() -> None:
        nonlocal batch_dates, batch_data, is_first_write
        if not batch_dates:
          return
        ds_batch = build_batch_dataset(
            batch_dates, batch_data, target_lat, target_lon
        )
        write_batch_to_zarr(
            ds_batch,
            full_target_url,
            project=project,
            is_initial_write=is_first_write,
            consolidated=has_consolidated,
        )
        is_first_write = False
        batch_dates = []
        batch_data = {var: [] for var in HRES_VARIABLES}

      for dt, date_data in append_results:
        if _is_date_payload_all_nan(date_data):
          pending_missing.append((dt, date_data))
        else:
          # A valid date arrived after pending_missing: flush those interior
          # gap dates as NaN slices to keep the daily time axis contiguous.
          for m_dt, m_data in pending_missing:
            newly_missing_dates.append(m_dt.strftime("%Y-%m-%d"))
            batch_dates.append(m_dt)
            for k, k_list in batch_data.items():
              k_list.append(m_data[k])
            if len(batch_dates) >= batch_size:
              _flush_append_batch()
          pending_missing.clear()

          newly_valid_dates.append(dt.strftime("%Y-%m-%d"))
          batch_dates.append(dt)
          for k, k_list in batch_data.items():
            k_list.append(date_data[k])
          if len(batch_dates) >= batch_size:
            _flush_append_batch()

      if pending_missing:
        logging.warning(
            "Skipping %d trailing unpublished/missing dates at end of range "
            "(%s to %s) so the Zarr time axis is not initialized with trailing "
            "NaNs.",
            len(pending_missing),
            pending_missing[0][0].strftime("%Y-%m-%d"),
            pending_missing[-1][0].strftime("%Y-%m-%d"),
        )

      _flush_append_batch()

    if not is_first_write:
      storage.update_store_tracking_attrs(
          mapper,
          newly_valid_dates=newly_valid_dates,
          newly_missing_dates=newly_missing_dates,
          newly_failed_dates=newly_failed_dates,
      )
  finally:
    storage.write_failure_log(
        failure_log,
        failed_dates=newly_failed_dates,
        missing_dates=newly_missing_dates,
    )

  logging.info(
      "HRES archive build complete for %s to %s!", start_date, end_date
  )


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the HRES archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-hres-archive",
      description="Build the unified HRES daily surface forecast archive.",
  )
  parser.add_argument(
      "--target_zarr",
      type=str,
      required=True,
      help="Destination Zarr store (local path or explicit gs:// URI).",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=DEFAULT_START_DATE,
      help="First forecast initialization date to build (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=datetime.date.today().isoformat(),
      help="Last forecast initialization date to build (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="Optional GCP project used for billing/auth of gs:// requests.",
  )
  parser.add_argument(
      "--wb2_zarr",
      type=str,
      default=WB2_HRES_ZARR,
      help="Upstream WeatherBench 2 HRES Zarr path or URI.",
  )
  parser.add_argument(
      "--ecmwf_open_data_bucket",
      type=str,
      default=ECMWF_OPEN_DATA_BUCKET,
      help="Upstream ECMWF Open Data GCS bucket or prefix.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=10,
      help="Number of forecast dates accumulated before each Zarr write.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Delete and rebuild the target store instead of resuming it.",
  )
  parser.add_argument(
      "--in_place",
      action="store_true",
      help="Rewrite dates that already exist in the target store in place.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=min(32, os.cpu_count() or 4),
      help="Number of parallel extraction worker processes.",
  )
  parser.add_argument(
      "--failure_log",
      type=str,
      default=None,
      help="Optional JSON file path to record missing and failed dates.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-hres-archive``."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  build_hres_archive(
      start_date=args.start_date,
      end_date=args.end_date,
      target_zarr=args.target_zarr,
      project=args.project,
      wb2_zarr=args.wb2_zarr,
      ecmwf_open_data_bucket=args.ecmwf_open_data_bucket,
      batch_size=args.batch_size,
      overwrite=args.overwrite,
      in_place=args.in_place,
      num_workers=args.num_workers,
      failure_log=args.failure_log,
  )


if __name__ == "__main__":
  main()
