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

"""ETL pipeline to build the unified Open-MultiMet daily IMERG surface archive.

Ingests NASA GPM IMERG Early V07 daily precipitation at its native 0.1 degree
resolution:

- Spatial resolution: 0.1 degree x 0.1 degree global grid
  - ``latitude``: 1,800 points from -89.95 to 89.95 (ascending)
  - ``longitude``: 3,600 points from -179.95 to 179.95 (ascending)
- Temporal resolution: Daily aggregation (00:00:00 UTC to 24:00:00 UTC)
- Variable: ``imerg_precipitation`` (mm/day, ``float32``)

Supported upstream sources:

#. ``gesdisc``: Downloads the official Level 3 Daily NetCDF-4 product
   (``3B-DAY-E.MS.MRG.3IMERG.*.V07*.nc4``) from NASA GES DISC using Earthdata
   Login credentials (via ``~/.netrc``, environment variables, or CLI flags).
#. ``local``: Ingests pre-downloaded daily NetCDF-4 files (``.nc4`` / ``.nc``)
   or 48 half-hourly HDF5 granules (``.RT-H5`` / ``.HDF5``) from a local
   directory. All 48 half-hourly observations at a grid cell must be valid for
   the daily total at that cell to be finite; any cell with missing half-hour
   observations is masked to ``NaN``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import http.client
import io
import logging
import netrc
import os
import random
import shutil
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Sequence

try:
  import h5py  # type: ignore[import-untyped]
except ImportError:
  h5py = None

import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr

from multimet import storage

DEFAULT_GESDISC_URL = (
    "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07"
)
DEFAULT_CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
IMERG_HHR_SHORT_NAME = "GPM_3IMERGHHE"
IMERG_DAILY_SHORT_NAME = "GPM_3IMERGDE"
DEFAULT_START_DATE = "2000-06-01"

LAT_COUNT = 1800
LON_COUNT = 3600
IMERG_LATS = np.linspace(-89.95, 89.95, LAT_COUNT, dtype=np.float32)
IMERG_LONS = np.linspace(-179.95, 179.95, LON_COUNT, dtype=np.float32)
IMERG_VARIABLE = "imerg_precipitation"

IMERG_ATTRS = {
    "title": "Open-MultiMet NASA GPM IMERG Early V07 Daily Surface Archive",
    "spatial_resolution": "0.1 degree x 0.1 degree",
    "temporal_resolution": (
        "Daily (00:00:00 UTC to 24:00:00 UTC, left-labeled)"
    ),
    "units": "precipitation [mm]",
    "citation": (
        "Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, Jackson"
        " Tan (2024), GPM IMERG Early Precipitation L3 Half Hourly 0.1"
        " degree x 0.1 degree V07, Greenbelt, MD, GES DISC,"
        " 10.5067/GPM/IMERG/3B-HH-E/07"
    ),
    "license": "CC-BY-4.0",
    "institution": "NASA GSFC / Open-MultiMet",
}

_netcdf_lock = threading.Lock()


def query_cmr_granules(
    short_name: str,
    date: pd.Timestamp,
    version: str = "07",
    cmr_url: str = DEFAULT_CMR_GRANULES_URL,
    timeout: int = 30,
) -> list[str]:
  """Queries NASA's Common Metadata Repository (CMR) for IMERG granule URLs."""
  dt = pd.Timestamp(date)
  start_iso = dt.strftime("%Y-%m-%dT00:00:00Z")
  end_iso = dt.strftime("%Y-%m-%dT23:59:59Z")
  params = {
      "short_name": short_name,
      "version": version,
      "temporal": f"{start_iso},{end_iso}",
      "page_size": 200,
  }
  resp = requests.get(cmr_url, params=params, timeout=timeout)
  resp.raise_for_status()
  entries = resp.json().get("feed", {}).get("entry", [])
  urls: list[str] = []
  for entry in entries:
    for link in entry.get("links", []):
      href = link.get("href", "")
      rel = link.get("rel", "")
      if (
          href.startswith("https://")
          and "data#" in rel
          and not href.endswith((".xml", ".dmrpp", ".s3"))
          and href.endswith((".RT-H5", ".HDF5", ".h5", ".nc4", ".nc"))
      ):
        urls.append(href)
        break
  return sorted(set(urls))


def get_earthdata_credentials_from_netrc(
    netrc_path: str | None = None,
) -> tuple[str | None, str | None]:
  """Reads NASA Earthdata credentials from ``.netrc`` if present.

  Raises if ``netrc_path`` was explicitly provided by the user and cannot be
  parsed or does not exist.
  """
  if netrc_path is not None:
    if not os.path.exists(netrc_path):
      raise FileNotFoundError(f"Specified netrc_path does not exist: {netrc_path}")
    parsed = netrc.netrc(netrc_path)
  else:
    default_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(default_path):
      return None, None
    parsed = netrc.netrc(default_path)

  for host in ("urs.earthdata.nasa.gov", "gpm1.gesdisc.eosdis.nasa.gov"):
    auth_info = parsed.authenticators(host)
    if auth_info:
      return auth_info[0], auth_info[2]
  return None, None


class EarthdataSession(requests.Session):
  """Custom ``requests.Session`` that preserves auth across NASA URS redirects."""

  AUTH_HOST = "urs.earthdata.nasa.gov"

  def __init__(
      self,
      username: str | None = None,
      password: str | None = None,
      token: str | None = None,
      netrc_path: str | None = None,
  ):
    super().__init__()
    token = token or os.environ.get("EARTHDATA_TOKEN")
    username = username or os.environ.get("EARTHDATA_USERNAME")
    password = password or os.environ.get("EARTHDATA_PASSWORD")

    if not (username and password) and not token:
      netrc_user, netrc_pass = get_earthdata_credentials_from_netrc(netrc_path)
      if netrc_user and netrc_pass:
        username, password = netrc_user, netrc_pass

    self.token = token
    self.username = username
    self.password = password

    if token:
      self.headers.update({"Authorization": f"Bearer {token}"})
    elif username and password:
      self.auth = (username, password)

  def rebuild_auth(
      self,
      prepared_request: requests.PreparedRequest,
      response: requests.Response,
  ) -> None:
    """Preserves Authorization header across redirects to/from NASA URS."""
    headers = prepared_request.headers
    url = prepared_request.url

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.hostname == self.AUTH_HOST:
      if self.token:
        headers["Authorization"] = f"Bearer {self.token}"
      elif self.username and self.password:
        prepared_request.prepare_auth((self.username, self.password))
      return

    if "Authorization" in headers:
      original_parsed = urllib.parse.urlparse(response.request.url)
      redirect_parsed = urllib.parse.urlparse(url)
      if (
          original_parsed.hostname != redirect_parsed.hostname
          and redirect_parsed.hostname != self.AUTH_HOST
          and original_parsed.hostname != self.AUTH_HOST
      ):
        del headers["Authorization"]

    super().rebuild_auth(prepared_request, response)


def download_daily_imerg(
    url: str,
    dest_path: str,
    session: requests.Session | None = None,
    max_retries: int = 8,
) -> str:
  """Downloads a daily IMERG NetCDF4 file from NASA GES DISC with retries."""
  if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
    return dest_path

  if session is None:
    session = EarthdataSession()

  os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
  temp_path = f"{dest_path}.tmp.{os.getpid()}.{time.time_ns()}"

  last_err: Exception | None = None
  for attempt in range(max_retries):
    try:
      if attempt == 0:
        time.sleep(random.uniform(0.05, 0.5))
      else:
        sleep_sec = (2**attempt) + random.uniform(0.5, 2.0)
        logging.warning(
            "Retrying NASA GES DISC download (%d/%d) in %.1fs for: %s",
            attempt + 1,
            max_retries,
            sleep_sec,
            url,
        )
        time.sleep(sleep_sec)

      with session.get(url, stream=True, timeout=120) as resp:
        if resp.status_code in (401, 403):
          raise PermissionError(
              f"NASA GES DISC returned HTTP {resp.status_code} Unauthorized "
              f"for URL:\n  {url}\nAccess to NASA IMERG data requires NASA "
              "Earthdata Login authentication."
          )
        if resp.status_code == 404:
          raise storage.UpstreamDataMissingError(f"404 Not Found: {url}")
        if resp.status_code in (429, 500, 502, 503, 504):
          last_err = requests.HTTPError(
              f"{resp.status_code} Server Error: {resp.reason} for url: {url}",
              response=resp,
          )
          continue

        resp.raise_for_status()
        with open(temp_path, "wb") as f:
          for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
              f.write(chunk)

        if not (
            os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024
        ):
          os.replace(temp_path, dest_path)
        return dest_path
    except (storage.UpstreamDataMissingError, FileNotFoundError, PermissionError):
      raise
    except (
        requests.RequestException,
        http.client.RemoteDisconnected,
        TimeoutError,
    ) as err:
      last_err = err
      continue
    finally:
      if os.path.exists(temp_path):
        os.remove(temp_path)

  if last_err is not None:
    raise last_err
  raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


def parse_imerg_netcdf_to_grid(nc_path: str) -> np.ndarray:
  """Reads a NASA IMERG V07 daily NetCDF4 file into a (1800, 3600) float32 grid.

  Strictly requires the IMERG V07 ``precipitation`` variable; raises ``KeyError``
  if given a legacy V06 file containing only ``precipitationCal``.
  """
  with _netcdf_lock, xr.open_dataset(nc_path) as ds:
    if "precipitation" not in ds:
      raise KeyError(
          f"IMERG V07 variable 'precipitation' not found in {nc_path} "
          f"(found variables: {list(ds.data_vars)}). Legacy V06 "
          "'precipitationCal' files are not supported."
      )
    da = ds["precipitation"]
    if "time" in da.dims:
      da = da.squeeze("time")
    if da.dims == ("lon", "lat"):
      da = da.transpose("lat", "lon")

    vals = da.values.astype(np.float32)
    return np.where(vals < 0.0, np.nan, vals)


def _read_and_parse_h5_granule(fpath: str) -> np.ndarray:
  """Reads a single half-hourly IMERG V07 HDF5 granule into a (lat, lon) array.

  Raises immediately if ``h5py`` is not installed, if the file is corrupt, or
  if the V07 ``/Grid/precipitation`` dataset is missing.
  """
  if h5py is None:
    raise ImportError("h5py is required to parse raw HDF5 granules.")

  with open(fpath, "rb") as f:
    content = f.read()

  with h5py.File(io.BytesIO(content), "r") as h5:
    if "Grid" not in h5 or "precipitation" not in h5["Grid"]:
      raise KeyError(
          f"IMERG V07 '/Grid/precipitation' dataset not found in {fpath}. "
          "Legacy V06 'precipitationCal' granules are not supported."
      )
    ds = h5["Grid"]["precipitation"]
    raw = np.squeeze(ds[()])
    return np.transpose(raw).astype(np.float32)


class GESDISCImergSource:
  """Downloads and extracts daily gridded precipitation from NASA GES DISC."""

  def __init__(
      self,
      cache_dir: str,
      base_url: str = DEFAULT_GESDISC_URL,
      username: str | None = None,
      password: str | None = None,
      token: str | None = None,
      netrc_path: str | None = None,
      cleanup_cache: bool = False,
  ):
    self.base_url = base_url.rstrip("/")
    self.username = username
    self.password = password
    self.token = token
    self.netrc_path = netrc_path
    self.cleanup_cache = cleanup_cache
    self._thread_local = threading.local()
    self.cache_dir = cache_dir
    os.makedirs(self.cache_dir, exist_ok=True)

  @property
  def session(self) -> EarthdataSession:
    if not hasattr(self._thread_local, "session"):
      self._thread_local.session = EarthdataSession(
          username=self.username,
          password=self.password,
          token=self.token,
          netrc_path=self.netrc_path,
      )
    return self._thread_local.session

  def extract_date(self, date: pd.Timestamp) -> np.ndarray | None:
    """Downloads daily NetCDF-4 granule and returns (1800, 3600) array in mm."""
    date = pd.to_datetime(date)
    date_str = date.strftime("%Y%m%d")
    year = date.year
    month = date.month

    candidate_suffixes = (
        ["V07C", "V07B", "V07", "V07A"]
        if year >= 2026
        else ["V07B", "V07C", "V07", "V07A"]
    )

    cached_path = None
    for suffix in candidate_suffixes:
      cand_fn = (
          f"3B-DAY-E.MS.MRG.3IMERG.{date_str}-S000000-E235959.{suffix}.nc4"
      )
      cand_path = os.path.join(self.cache_dir, cand_fn)
      if os.path.exists(cand_path) and os.path.getsize(cand_path) > 1000:
        cached_path = cand_path
        break

    if cached_path is None:
      for suffix in candidate_suffixes:
        cand_fn = (
            f"3B-DAY-E.MS.MRG.3IMERG.{date_str}-S000000-E235959.{suffix}.nc4"
        )
        cand_url = f"{self.base_url}/{year}/{month:02d}/{cand_fn}"
        cand_path = os.path.join(self.cache_dir, cand_fn)
        try:
          download_daily_imerg(cand_url, cand_path, session=self.session)
          cached_path = cand_path
          break
        except (storage.UpstreamDataMissingError, FileNotFoundError):
          continue

    if cached_path is None or not os.path.exists(cached_path):
      logging.warning(
          "No published IMERG V07 granule found for %s (HTTP 404).",
          date_str,
      )
      return None

    try:
      return parse_imerg_netcdf_to_grid(cached_path)
    finally:
      if self.cleanup_cache and cached_path and os.path.exists(cached_path):
        os.remove(cached_path)


class LocalImergSource:
  """Extracts daily gridded precipitation from a local directory."""

  def __init__(self, local_dir: str, granule_workers: int = 8):
    if not os.path.isdir(local_dir):
      raise FileNotFoundError(
          f"Local IMERG source directory does not exist: {local_dir}"
      )
    self.local_dir = local_dir
    self.granule_workers = max(1, granule_workers)

  def extract_date(self, date: pd.Timestamp) -> np.ndarray | None:
    """Extracts a daily grid from local NetCDF-4 or 48 half-hourly HDF5 files."""
    date = pd.to_datetime(date)
    date_str = date.strftime("%Y%m%d")
    month_str = date.strftime("%Y%m")

    # 1. Check for daily NetCDF-4 files in local_dir.
    nc_matches = [
        os.path.join(self.local_dir, fname)
        for fname in sorted(os.listdir(self.local_dir))
        if date_str in fname and fname.endswith((".nc4", ".nc"))
    ]
    if nc_matches:
      return parse_imerg_netcdf_to_grid(nc_matches[0])

    # 2. Check for 48 half-hourly HDF5 granules in local_dir or local_dir/{YYYYMM}.
    search_dirs = [self.local_dir, os.path.join(self.local_dir, month_str)]
    h5_files: list[str] = []
    for dpath in search_dirs:
      if os.path.isdir(dpath):
        for fname in os.listdir(dpath):
          if date_str in fname and fname.endswith((".RT-H5", ".HDF5", ".h5")):
            h5_files.append(os.path.join(dpath, fname))
        if h5_files:
          break

    if not h5_files:
      return None

    h5_files = sorted(h5_files)
    if len(h5_files) != 48:
      raise ValueError(
          f"Incomplete half-hourly HDF5 granules for {date.strftime('%Y-%m-%d')} "
          f"in {self.local_dir}: found {len(h5_files)} granules, expected 48."
      )

    if self.granule_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(48, self.granule_workers)
      ) as pool:
        granule_grids = list(pool.map(_read_and_parse_h5_granule, h5_files))
    else:
      granule_grids = [_read_and_parse_h5_granule(fpath) for fpath in h5_files]

    daily_sum = np.zeros((LAT_COUNT, LON_COUNT), dtype=np.float64)
    valid_counts = np.zeros((LAT_COUNT, LON_COUNT), dtype=np.int32)

    for grid_rate in granule_grids:
      valid_mask = (grid_rate >= 0.0) & (~np.isnan(grid_rate))
      daily_sum[valid_mask] += grid_rate[valid_mask] * 0.5
      valid_counts[valid_mask] += 1

    # Missing data in ALWAYS means missing data out: require all 48 half-hour
    # observations at a grid cell to be valid for its daily sum to be finite.
    complete_mask = valid_counts == 48
    result = np.full((LAT_COUNT, LON_COUNT), np.nan, dtype=np.float32)
    result[complete_mask] = daily_sum[complete_mask].astype(np.float32)
    return result


def build_batch_dataset(
    batch_dates: Sequence[pd.Timestamp],
    batch_grids: Sequence[np.ndarray],
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
) -> xr.Dataset:
  """Assembles a batch of daily IMERG grids into the canonical Zarr schema."""
  lats = IMERG_LATS if latitudes is None else latitudes
  lons = IMERG_LONS if longitudes is None else longitudes
  return xr.Dataset(
      data_vars={
          IMERG_VARIABLE: (
              ["time", "latitude", "longitude"],
              np.stack(batch_grids, axis=0).astype(np.float32),
          ),
      },
      coords={
          "time": list(batch_dates),
          "latitude": lats,
          "longitude": lons,
      },
      attrs=dict(IMERG_ATTRS),
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    is_initial_write: bool = False,
    consolidated: bool = True,
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


def build_imerg_archive(
    target_zarr: str,
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    source_type: str = "gesdisc",
    project: str | None = None,
    gesdisc_url: str = DEFAULT_GESDISC_URL,
    batch_size: int = 30,
    num_workers: int = 4,
    granule_workers: int = 8,
    cache_dir: str | None = None,
    cleanup_cache: bool = False,
    overwrite: bool = False,
    in_place: bool = False,
    local_dir: str | None = None,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    earthdata_token: str | None = None,
    netrc_path: str | None = None,
    failure_log: str | None = None,
) -> None:
  """Builds or updates the unified IMERG daily native-resolution Zarr archive."""
  if source_type not in ("gesdisc", "local"):
    raise ValueError(
        f"Invalid source_type={source_type!r}. Must be 'gesdisc' or 'local'."
    )
  if local_dir is not None:
    source_type = "local"
  if source_type == "local" and not local_dir:
    raise ValueError("--local_dir must be provided when --source=local.")

  full_target_url, _, store_exists, has_consolidated, mapper = (
      storage.inspect_zarr_store(
          target_zarr, project=project, overwrite=overwrite
      )
  )

  if end_date is None:
    end_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

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

  temp_cache_ctx = (
      tempfile.TemporaryDirectory(prefix="imerg_cache_")
      if cache_dir is None
      else None
  )
  effective_cache_dir = (
      temp_cache_ctx.name if temp_cache_ctx is not None else str(cache_dir)
  )

  if source_type == "local":
    logging.info("Using local directory archive from %s", local_dir)
    source: LocalImergSource | GESDISCImergSource = LocalImergSource(
        local_dir=str(local_dir), granule_workers=granule_workers
    )
  else:
    logging.info("Using NASA GES DISC public daily NetCDF archive")
    source = GESDISCImergSource(
        cache_dir=effective_cache_dir,
        base_url=gesdisc_url,
        username=earthdata_username,
        password=earthdata_password,
        token=earthdata_token,
        netrc_path=netrc_path,
        cleanup_cache=cleanup_cache,
    )

  is_first_write = not store_exists
  newly_valid_dates: list[str] = []
  newly_missing_dates: list[str] = []
  newly_failed_dates: list[str] = []

  def _extract_chunk(
      chunk_dates: pd.DatetimeIndex,
  ) -> list[tuple[pd.Timestamp, np.ndarray | None]]:
    if num_workers > 1 and len(chunk_dates) > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(len(chunk_dates), num_workers)
      ) as executor:
        futures = {
            executor.submit(source.extract_date, dt): dt for dt in chunk_dates
        }
        results_map: dict[pd.Timestamp, np.ndarray | None] = {}
        for future in concurrent.futures.as_completed(futures):
          dt_item = futures[future]
          results_map[dt_item] = future.result()
        return [(dt, results_map[dt]) for dt in chunk_dates]
    return [(dt, source.extract_date(dt)) for dt in chunk_dates]

  try:
    # 1. Execute in-place backfills (either explicit --in_place or healing
    # previously missing/trailing-NaN slices on resume).
    if len(in_place_dates) > 0:
      for i in tqdm.trange(
          0, len(in_place_dates), batch_size, desc="IMERG In-Place Backfill"
      ):
        chunk = in_place_dates[i : i + batch_size]
        extracted = _extract_chunk(chunk)
        ip_dates: list[pd.Timestamp] = []
        ip_grids: list[np.ndarray] = []
        for dt, grid in extracted:
          d_str = dt.strftime("%Y-%m-%d")
          if grid is None or bool(np.isnan(grid).all()):
            newly_missing_dates.append(d_str)
            if not in_place:
              continue
            grid = np.full((LAT_COUNT, LON_COUNT), np.nan, dtype=np.float32)
          else:
            newly_valid_dates.append(d_str)
          ip_dates.append(dt)
          ip_grids.append(grid)
        if ip_dates:
          ds_batch = build_batch_dataset(ip_dates, ip_grids)
          write_batch_in_place(
              ds_batch, full_target_url, project=project, date_to_idx=date_to_idx
          )

    # 2. Execute appends for new dates, holding any missing dates in
    # pending_missing so trailing unpublished dates are never written as NaN.
    if len(append_dates) > 0:
      batch_dates: list[pd.Timestamp] = []
      batch_grids: list[np.ndarray] = []
      pending_missing: list[pd.Timestamp] = []

      def _flush_append() -> None:
        nonlocal batch_dates, batch_grids, is_first_write
        if not batch_dates:
          return
        ds_batch = build_batch_dataset(batch_dates, batch_grids)
        write_batch_to_zarr(
            ds_batch,
            full_target_url,
            project=project,
            is_initial_write=is_first_write,
            consolidated=has_consolidated or is_first_write,
        )
        is_first_write = False
        batch_dates = []
        batch_grids = []

      for i in tqdm.trange(
          0, len(append_dates), batch_size, desc="Processing IMERG Batches"
      ):
        chunk = append_dates[i : i + batch_size]
        extracted = _extract_chunk(chunk)
        for dt, grid in extracted:
          if grid is None or bool(np.isnan(grid).all()):
            pending_missing.append(dt)
          else:
            for m_dt in pending_missing:
              newly_missing_dates.append(m_dt.strftime("%Y-%m-%d"))
              batch_dates.append(m_dt)
              batch_grids.append(
                  np.full((LAT_COUNT, LON_COUNT), np.nan, dtype=np.float32)
              )
              if len(batch_dates) >= batch_size:
                _flush_append()
            pending_missing.clear()

            newly_valid_dates.append(dt.strftime("%Y-%m-%d"))
            batch_dates.append(dt)
            batch_grids.append(grid)
            if len(batch_dates) >= batch_size:
              _flush_append()

      if pending_missing:
        logging.warning(
            "Skipping %d trailing unpublished/missing IMERG dates at end of "
            "range (%s to %s) so the Zarr time axis is not initialized with "
            "trailing NaNs.",
            len(pending_missing),
            pending_missing[0].strftime("%Y-%m-%d"),
            pending_missing[-1].strftime("%Y-%m-%d"),
        )

      _flush_append()

    if not is_first_write:
      storage.update_store_tracking_attrs(
          mapper,
          newly_valid_dates=newly_valid_dates,
          newly_missing_dates=newly_missing_dates,
          newly_failed_dates=newly_failed_dates,
      )

    logging.info(
        "IMERG archive build complete for %s to %s!", start_date, end_date
    )
  finally:
    storage.write_failure_log(
        failure_log,
        failed_dates=newly_failed_dates,
        missing_dates=newly_missing_dates,
    )
    if temp_cache_ctx is not None:
      temp_cache_ctx.cleanup()
    elif cleanup_cache and cache_dir and os.path.exists(cache_dir):
      shutil.rmtree(cache_dir)
      logging.info("Cleaned up cache directory: %s", cache_dir)


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the IMERG archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-imerg-archive",
      description="Build the unified IMERG daily surface precipitation archive.",
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
      help="First date to ingest (YYYY-MM-DD, default 2000-06-01).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=None,
      help="Last date to ingest (YYYY-MM-DD, default yesterday UTC).",
  )
  parser.add_argument(
      "--source",
      type=str,
      default="gesdisc",
      choices=["gesdisc", "local"],
      help="Upstream source type: 'gesdisc' (NASA GES DISC) or 'local'.",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="Optional GCP project used for billing/auth of gs:// requests.",
  )
  parser.add_argument(
      "--gesdisc_url",
      type=str,
      default=DEFAULT_GESDISC_URL,
      help="Base URL for NASA GES DISC IMERG V07 daily archive.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=30,
      help="Number of daily slices accumulated before each Zarr write.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=4,
      help="Number of concurrent date download/extraction workers.",
  )
  parser.add_argument(
      "--granule_workers",
      type=int,
      default=8,
      help="Number of concurrent HDF5 granule workers per date (local mode).",
  )
  parser.add_argument(
      "--cache_dir",
      "--local_cache",
      dest="cache_dir",
      type=str,
      default=None,
      help="Local directory used to stage downloaded NASA GES DISC files.",
  )
  parser.add_argument(
      "--cleanup_cache",
      action="store_true",
      help="Delete downloaded NetCDF files after processing and on exit.",
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
      "--local_dir",
      type=str,
      default=None,
      help="Local directory containing pre-downloaded NetCDF-4 or HDF5 files.",
  )
  parser.add_argument(
      "--earthdata_username",
      type=str,
      default=None,
      help="NASA Earthdata Login username.",
  )
  parser.add_argument(
      "--earthdata_password",
      type=str,
      default=None,
      help="NASA Earthdata Login password.",
  )
  parser.add_argument(
      "--earthdata_token",
      type=str,
      default=None,
      help="NASA Earthdata Bearer token.",
  )
  parser.add_argument(
      "--netrc_path",
      type=str,
      default=None,
      help="Custom path to a .netrc file containing Earthdata credentials.",
  )
  parser.add_argument(
      "--failure_log",
      type=str,
      default=None,
      help="Optional JSON file path to record missing and failed dates.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-imerg-archive``."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  build_imerg_archive(
      target_zarr=args.target_zarr,
      start_date=args.start_date,
      end_date=args.end_date,
      source_type=args.source,
      project=args.project,
      gesdisc_url=args.gesdisc_url,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      granule_workers=args.granule_workers,
      cache_dir=args.cache_dir,
      cleanup_cache=args.cleanup_cache,
      overwrite=args.overwrite,
      in_place=args.in_place,
      local_dir=args.local_dir,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
      failure_log=args.failure_log,
  )


if __name__ == "__main__":
  main()
