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

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime
import io
import logging
import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import urllib.parse

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr

from multimet.base import BaseExtractor
from multimet.config import DEFAULT_STORAGE_PATHS, Product
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix


def get_earthdata_credentials_from_netrc(
    netrc_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
  """Reads Earthdata credentials from ~/.netrc."""
  path = netrc_path or os.path.expanduser("~/.netrc")
  if not os.path.exists(path):
    return None, None
  import netrc
  n = netrc.netrc(path)
  for host in ("urs.earthdata.nasa.gov", "gpm1.gesdisc.eosdis.nasa.gov"):
    auth_info = n.authenticators(host)
    if auth_info:
      return auth_info[0], auth_info[2]
  return None, None


class EarthdataSession(requests.Session):
  """Custom requests.Session that preserves authentication on NASA Earthdata redirects."""

  AUTH_HOST = "urs.earthdata.nasa.gov"

  def __init__(
      self,
      username: Optional[str] = None,
      password: Optional[str] = None,
      token: Optional[str] = None,
      netrc_path: Optional[str] = None,
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

  def rebuild_auth(self, prepared_request, response):
    """Preserves Authorization header across redirects to/from NASA Earthdata auth host."""
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
    session: Optional[requests.Session] = None,
) -> str:
  """Downloads a daily IMERG NetCDF4 file from NASA GES DISC with auth handling."""
  if session is None:
    session = EarthdataSession()

  os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
  with session.get(url, stream=True, timeout=120) as resp:
    if resp.status_code in (401, 403):
      raise PermissionError(
          f"NASA GES DISC returned HTTP {resp.status_code} Unauthorized for URL:\n  {url}\n\n"
          "Access to NASA IMERG data requires NASA Earthdata Login authentication.\n"
          "To resolve this, do one of the following:\n"
          "1. Add your Earthdata credentials to ~/.netrc:\n"
          "   machine urs.earthdata.nasa.gov login <username> password <password>\n"
          "   chmod 600 ~/.netrc\n"
          "2. Set the EARTHDATA_TOKEN (or EARTHDATA_USERNAME and EARTHDATA_PASSWORD) environment variable.\n"
          "3. Or pre-download the daily NetCDF4 (.nc4) files to a local directory and pass --data_dir=<path>."
      )
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
      for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
          f.write(chunk)
  return dest_path


def _weighted_mean_valid(vals: np.ndarray, weights: np.ndarray) -> float:
  """Computes weighted mean over non-NaN grid cells, normalizing by valid weights."""
  valid = ~np.isnan(vals)
  if not np.any(valid):
    return np.nan
  w_valid = weights[valid]
  sum_w = np.sum(w_valid)
  if sum_w <= 0.0:
    return np.nan
  return float(np.sum(vals[valid] * w_valid) / sum_w)


def resolve_date_to_imerg_files(
    storage_dir: str, dt: pd.Timestamp
) -> List[str]:
  """Resolves a date to its corresponding IMERG HDF5 file(s)."""
  dt = pd.to_datetime(dt)
  month_str = dt.strftime("%Y%m")
  day_str = dt.strftime("%Y%m%d")
  m_dir = os.path.join(storage_dir, month_str)

  files = []
  if os.path.exists(m_dir):
    for fname in os.listdir(m_dir):
      if day_str in fname and fname.endswith((".RT-H5", ".HDF5")):
        files.append(os.path.join(m_dir, fname))

  return sorted(files)


class IMERGExtractor(BaseExtractor):
  """Extractor for NASA GPM IMERG Early Precipitation (0.1 deg, [-60, 60] latitude)."""

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "auto",
      username: Optional[str] = None,
      password: Optional[str] = None,
      token: Optional[str] = None,
      netrc_path: Optional[str] = None,
  ):
    super().__init__(Product.IMERG, data_dir)
    default_url = DEFAULT_STORAGE_PATHS[Product.IMERG].get(
        "gesdisc_url",
        "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07/",
    )
    self.data_dir = data_dir if data_dir is not None else default_url

    if source in ("auto", "default", "gesdisc", "public", "nasa"):
      self.source = "gesdisc"
    elif source == "dynamical":
      self.source = "dynamical"
    elif source in ("h5", "local"):
      self.source = "h5"
    else:
      self.source = source

    self.session = EarthdataSession(
        username=username, password=password, token=token, netrc_path=netrc_path
    )

    # Standard IMERG grid: 1800 lats x 3600 lons (0.1 deg)
    # Latitudes: -89.95 to 89.95 (south to north)
    self.lats = np.linspace(-89.95, 89.95, 1800, dtype=np.float64)
    # Longitudes: -179.95 to 179.95
    self.lons = np.linspace(-179.95, 179.95, 3600, dtype=np.float64)
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.1, cell_res_lon=0.1
    )

  def get_daily_file(self, dt: pd.Timestamp) -> str:
    """Finds or downloads the daily IMERG NetCDF4 file for a given date."""
    dt = pd.to_datetime(dt)
    date_str = dt.strftime("%Y%m%d")
    year = dt.year
    month = dt.month

    # 1. Check if data_dir is a local directory containing matching daily files
    if os.path.isdir(self.data_dir):
      for fname in os.listdir(self.data_dir):
        if date_str in fname and fname.endswith((".nc4", ".nc", ".HDF5", ".h5")):
          return os.path.join(self.data_dir, fname)
      ym_dir = os.path.join(self.data_dir, str(year), f"{month:02d}")
      if os.path.isdir(ym_dir):
        for fname in os.listdir(ym_dir):
          if date_str in fname and fname.endswith((".nc4", ".nc", ".HDF5", ".h5")):
            return os.path.join(ym_dir, fname)

    # 2. Check local cache directory
    cache_dir = os.environ.get("MULTIMET_IMERG_CACHE", "/tmp/multimet_imerg_cache")
    filename = f"3B-DAY-E.MS.MRG.3IMERG.{date_str}-S000000-E235959.V07B.nc4"
    cached_path = os.path.join(cache_dir, filename)
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1000:
      return cached_path

    if os.path.isdir(cache_dir):
      for fname in os.listdir(cache_dir):
        if date_str in fname and fname.endswith((".nc4", ".nc")):
          f_path = os.path.join(cache_dir, fname)
          if os.path.getsize(f_path) > 1000:
            return f_path

    # 3. If not cached locally, download from NASA GES DISC
    base_url = (
        self.data_dir
        if (
            self.data_dir.startswith("http://")
            or self.data_dir.startswith("https://")
        )
        else "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07"
    )
    url = f"{base_url.rstrip('/')}/{year}/{month:02d}/{filename}"

    return download_daily_imerg(url, cached_path, session=self.session)

  def extract_day_from_nc4(
      self,
      nc_path: str,
      basins_gdf: gpd.GeoDataFrame,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of IMERG precipitation from a daily NetCDF4 file."""
    with xr.open_dataset(nc_path) as ds:
      if "precipitation" in ds:
        da = ds["precipitation"]
      elif "precipitationCal" in ds:
        da = ds["precipitationCal"]
      elif "MWprecipitation" in ds:
        da = ds["MWprecipitation"]
      else:
        raise KeyError(
            f"Could not find precipitation variable in {nc_path}. "
            f"Variables: {list(ds.data_vars.keys())}"
        )

      if "time" in da.dims:
        da = da.squeeze("time")
      if da.dims == ("lon", "lat"):
        da = da.transpose("lat", "lon")

      min_lon, min_lat, max_lon, max_lat = basins_gdf.total_bounds
      pad = 0.2
      lat_vals = da.lat.values
      lon_vals = da.lon.values

      if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(float(max_lat + pad), float(min_lat - pad))
      else:
        lat_slice = slice(float(min_lat - pad), float(max_lat + pad))

      if lon_vals[0] > lon_vals[-1]:
        lon_slice = slice(float(max_lon + pad), float(min_lon - pad))
      else:
        lon_slice = slice(float(min_lon - pad), float(max_lon + pad))

      sub_da = da.sel(lat=lat_slice, lon=lon_slice).load()
      sub_lats = sub_da.lat.values
      sub_lons = sub_da.lon.values

      raw_vals = sub_da.values.astype(np.float32)
      raw_vals = np.where(raw_vals < 0, np.nan, raw_vals)

      if weights_matrix is not None and (
          weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
          and np.allclose(weights_matrix.lats, sub_lats)
          and np.allclose(weights_matrix.lons, sub_lons)
      ):
        matrix = weights_matrix
      else:
        matrix = ZonalWeightMatrix.from_geodataframe(
            basins_gdf, sub_lats, sub_lons, cell_res_lat=0.1, cell_res_lon=0.1
        )

      reduced = matrix.reduce_2d(raw_vals)
      return {"imerg_precipitation": reduced.astype(np.float32)}

  @staticmethod
  def parse_imerg_h5_bytes(content: bytes) -> np.ndarray:
    """Parses an IMERG HDF5 byte stream and returns a 2D (1800, 3600) array in mm/hr."""
    if h5py is None:
      raise ImportError("h5py is required to parse raw HDF5 files.")
    with h5py.File(io.BytesIO(content), "r") as h5:
      if "Grid/precipitation" in h5:
        ds = h5["Grid/precipitation"]
      elif "Grid/precipitationCal" in h5:
        ds = h5["Grid/precipitationCal"]
      else:
        raise KeyError("Could not find precipitation band in IMERG HDF5 file.")

      raw = np.squeeze(ds[()])
      transposed = np.transpose(raw).astype(np.float32)
      transposed[transposed < 0] = np.nan
      return transposed

  def extract_day_from_imerg_files(
      self,
      imerg_files: List[str],
      basin_ids: List[str],
      weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of IMERG precipitation across all basins from 48 half-hourly files.

    Strict validation: requires exactly 48 half-hourly files for a full day.
    If any file is missing or unreadable, returns all NaNs.
    """
    num_basins = len(basin_ids)
    res = np.full(num_basins, np.nan, dtype=np.float32)

    if len(imerg_files) != 48:
      return {"imerg_precipitation": res}

    daily_basin_sums = np.zeros(num_basins, dtype=np.float64)
    basin_has_nan = np.zeros(num_basins, dtype=bool)

    for fpath in imerg_files:
      with open(fpath, "rb") as f:
        content = f.read()
      grid_2d = self.parse_imerg_h5_bytes(content)

      for b_idx, b_id in enumerate(basin_ids):
        if b_id not in weights_dict:
          basin_has_nan[b_idx] = True
          continue
        lat_idx, lon_idx, w = weights_dict[b_id]
        val = _weighted_mean_valid(grid_2d[lat_idx, lon_idx], w)
        if np.isnan(val):
          basin_has_nan[b_idx] = True
        else:
          daily_basin_sums[b_idx] += val * 0.5

    for b_idx in range(num_basins):
      if not basin_has_nan[b_idx]:
        res[b_idx] = np.float32(daily_basin_sums[b_idx])

    return {"imerg_precipitation": res}

  def extract_for_basins_dynamical(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_dt: pd.Timestamp,
      end_dt: pd.Timestamp,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> xr.Dataset:
    """Extracts IMERG precipitation using dynamical.org cloud-optimized Zarr on S3."""
    import dynamical_catalog

    ds = dynamical_catalog.open("nasa-imerg-analysis-early")
    basin_ids = list(basins_gdf.index)

    min_lon, min_lat, max_lon, max_lat = basins_gdf.total_bounds
    pad = 0.2

    lat_slice = slice(float(max_lat + pad), float(min_lat - pad))
    lon_slice = slice(float(min_lon - pad), float(max_lon + pad))
    time_slice = slice(
        f"{start_dt.strftime('%Y-%m-%d')}T00:00:00",
        f"{end_dt.strftime('%Y-%m-%d')}T23:30:00",
    )

    sub_da = ds["precipitation_surface"].sel(
        time=time_slice,
        latitude=lat_slice,
        longitude=lon_slice,
    ).load()

    sub_lats = sub_da.latitude.values
    sub_lons = sub_da.longitude.values
    if weights_matrix is not None and (
        weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
        and np.allclose(weights_matrix.lats, sub_lats)
        and np.allclose(weights_matrix.lons, sub_lons)
    ):
      matrix = weights_matrix
    else:
      matrix = ZonalWeightMatrix.from_geodataframe(
          basins_gdf, sub_lats, sub_lons, cell_res_lat=0.1, cell_res_lon=0.1
      )

    daily_depth = (sub_da * 1800.0).resample(time="1D").sum(dim="time")

    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    precip_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )

    reduced_depth = matrix.reduce_3d(daily_depth.values)
    daily_times = pd.to_datetime(daily_depth.time.values)

    for t_idx, t_val in enumerate(daily_times):
      dt_target = pd.to_datetime(t_val.strftime("%Y-%m-%d"))
      if dt_target in date_idx:
        d_pos = date_idx.get_loc(dt_target)
        precip_matrix[:, d_pos] = reduced_depth[:, t_idx]

    return xr.Dataset(
        data_vars={
            "imerg_precipitation": (["basin", "date"], precip_matrix),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      matrix: Optional[ZonalWeightMatrix] = None,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of IMERG precipitation across basins."""
    dt = pd.to_datetime(dt)
    if self.source in ("gesdisc", "auto", "default", "public", "nasa"):
      nc_path = self.get_daily_file(dt)
      return self.extract_day_from_nc4(
          nc_path, basins_gdf, weights_matrix=matrix
      )
    elif self.source in ("dynamical", "cloud"):
      ds = self.extract_for_basins_dynamical(
          basins_gdf, start_dt=dt, end_dt=dt, weights_matrix=matrix
      )
      return {
          "imerg_precipitation": ds["imerg_precipitation"].values[:, 0].astype(
              np.float32
          )
      }
    else:
      imerg_files = resolve_date_to_imerg_files(self.data_dir, dt)
      basin_ids = list(basins_gdf.index)
      if weights_dict is None:
        weights_dict = {}
        for b_id in basin_ids:
          geom = basins_gdf.loc[b_id].geometry
          w = self.zonal_calc.compute_weights(b_id, geom)
          if w is not None:
            weights_dict[b_id] = w
      return self.extract_day_from_imerg_files(
          imerg_files, basin_ids, weights_dict
      )

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> xr.Dataset:
    """Extracts daily accumulated IMERG precipitation for given basins."""
    basin_ids = list(basins_gdf.index)

    if start_date is not None:
      start_dt = pd.to_datetime(start_date)
    else:
      start_dt = pd.to_datetime("2000-06-01")

    if end_date is not None:
      end_dt = pd.to_datetime(end_date)
    else:
      end_dt = pd.to_datetime("today")

    if self.source == "dynamical":
      return self.extract_for_basins_dynamical(
          basins_gdf, start_dt, end_dt, weights_matrix=weights_matrix
      )

    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    precip_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )

    if self.source in ("gesdisc", "auto", "default", "public", "nasa"):
      for d_idx, dt in enumerate(
          tqdm.tqdm(
              date_idx,
              desc=(
                  f"IMERG GES-DISC [{start_dt.strftime('%Y-%m-%d')} to"
                  f" {end_dt.strftime('%Y-%m-%d')}]"
              ),
              unit="day",
              leave=True,
          )
      ):
        day_res = self.extract_day(dt, basins_gdf, matrix=weights_matrix)
        precip_matrix[:, d_idx] = day_res["imerg_precipitation"]
    else:
      weights_dict = {}
      for b_id in basin_ids:
        geom = basins_gdf.loc[b_id].geometry
        w = self.zonal_calc.compute_weights(b_id, geom)
        if w is not None:
          weights_dict[b_id] = w

      for d_idx, dt in enumerate(
          tqdm.tqdm(
              date_idx,
              desc=(
                  f"IMERG [{start_dt.strftime('%Y-%m-%d')} to"
                  f" {end_dt.strftime('%Y-%m-%d')}]"
              ),
              unit="day",
              leave=True,
          )
      ):
        sub_files = resolve_date_to_imerg_files(self.data_dir, dt)
        if sub_files:
          day_res = self.extract_day_from_imerg_files(
              sub_files, basin_ids, weights_dict
          )
          precip_matrix[:, d_idx] = day_res["imerg_precipitation"]

    ds = xr.Dataset(
        data_vars={
            "imerg_precipitation": (["basin", "date"], precip_matrix),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    return ds
