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

import datetime
import glob
import gzip
import io
import os
import re
import shutil
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from multimet.base import BaseExtractor
from multimet.config import DEFAULT_STORAGE_PATHS, Product
from multimet.spatial import slice_coordinates_by_bounds
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix

import netCDF4

import logging
import time
import urllib.request

logger = logging.getLogger(__name__)


def ensure_psl_cpc_netcdf(year: int, cache_dir: str = "/tmp/cpc_cache") -> str:
  """Downloads and caches yearly NOAA PSL CPC NetCDF file if not already present."""
  os.makedirs(cache_dir, exist_ok=True)
  local_path = os.path.join(cache_dir, f"precip.{year}.nc")
  if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
    return local_path

  url = f"https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc"
  temp_path = f"{local_path}.tmp.{os.getpid()}.{time.time_ns()}"
  try:
    if not (os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024):
      logger.info("Downloading NOAA PSL CPC NetCDF for %d from %s...", year, url)
      with urllib.request.urlopen(url, timeout=120) as response, open(temp_path, "wb") as out_f:
        shutil.copyfileobj(response, out_f)
      if not (os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024):
        os.replace(temp_path, local_path)
        logger.info("Cached %s (%.1f MB)", local_path, os.path.getsize(local_path) / 1e6)
  finally:
    if os.path.exists(temp_path):
      try:
        os.remove(temp_path)
      except OSError:
        pass
  return local_path


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


def resolve_date_to_cpc_file(
    storage_dir: str, dt: pd.Timestamp
) -> Optional[str]:
  """Resolves a date to its corresponding CPC binary file path."""
  dt = pd.to_datetime(dt)
  yr_dir = os.path.join(storage_dir, str(dt.year))
  dt_str = dt.strftime("%Y%m%d")
  # File format: PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx.YYYYMMDD.gz or .RT
  candidate_gz = os.path.join(
      yr_dir, f"PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx.{dt_str}.gz"
  )
  candidate_plain = os.path.join(
      yr_dir, f"PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx.{dt_str}"
  )
  candidate_rt = os.path.join(
      yr_dir, f"PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx.{dt_str}.RT"
  )
  candidate_rt_gz = os.path.join(
      yr_dir, f"PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx.{dt_str}.RT.gz"
  )

  def _exists(path: str) -> bool:
    return os.path.exists(path)

  for cand in [candidate_gz, candidate_plain, candidate_rt, candidate_rt_gz]:
    if _exists(cand):
      return cand
  return None


class CPCExtractor(BaseExtractor):
  """Extractor for NOAA CPC Global Unified Daily Precipitation."""

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "auto",
      cache_dir: str = "/tmp/cpc_cache",
  ):
    super().__init__(Product.CPC, data_dir)
    self.data_dir = (
        data_dir
        if data_dir is not None
        else DEFAULT_STORAGE_PATHS[Product.CPC]["psl_netcdf"]
    )
    self.cache_dir = cache_dir

    if source in ("auto", "default", "psl", "public", "netcdf"):
      self.source = "psl"
    elif source in ("binary", "local"):
      self.source = "binary"
    else:
      self.source = source

    # Standard CPC 0.5 deg grid coordinates
    # Latitudes: -89.75 to 89.75 (south to north)
    self.lats = np.linspace(-89.75, 89.75, 360, dtype=np.float64)
    # Longitudes: -179.75 to 179.75
    self.lons = np.linspace(-179.75, 179.75, 720, dtype=np.float64)
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.5, cell_res_lon=0.5
    )

  @staticmethod
  def parse_cpc_file(file_path: str) -> np.ndarray:
    """Reads a daily CPC binary file and returns a 2D array of (lat, lon) in mm/day."""
    with open(file_path, 'rb') as f:
      content = f.read()

    if file_path.endswith(".gz"):
      content = gzip.decompress(content)

    raw_array = np.frombuffer(content, dtype="<f4")
    # Shape is (2, 360, 720): field 0 is precip (0.1mm), field 1 is num_stations
    if len(raw_array) >= 360 * 720 * 2:
      precip_field = raw_array[: 360 * 720].reshape((360, 720))
    elif len(raw_array) == 360 * 720:
      precip_field = raw_array.reshape((360, 720))
    else:
      raise ValueError(
          f"Unexpected binary size {len(raw_array)} for CPC file {file_path}"
      )

    # Set missing / negative values to NaN, convert 0.1 mm -> mm/day
    precip_mm = np.where(precip_field < 0, np.nan, precip_field * 0.1)

    # Shift longitudes from (0.25 .. 359.75) to (-179.75 .. 179.75)
    precip_shifted = np.concatenate(
        [precip_mm[:, 360:], precip_mm[:, :360]], axis=1
    )

    return precip_shifted.astype(np.float32)

  def extract_day_from_cpc_file(
      self,
      cpc_file: str,
      basin_ids: List[str],
      weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of CPC precipitation across all basins."""
    num_basins = len(basin_ids)
    res = np.full(num_basins, np.nan, dtype=np.float32)
    grid_2d = self.parse_cpc_file(cpc_file)
    for b_idx, b_id in enumerate(basin_ids):
      if b_id not in weights_dict:
        continue
      lat_idx, lon_idx, w = weights_dict[b_id]
      res[b_idx] = _weighted_mean_valid(grid_2d[lat_idx, lon_idx], w)
    return {"cpc_precipitation": res}

  def extract_for_basins_psl(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_dt: pd.Timestamp,
      end_dt: pd.Timestamp,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Extracts CPC daily precipitation using public NOAA PSL yearly NetCDF files."""
    basin_ids = list(basins_gdf.index)
    date_idx = pd.date_range(start_dt, end_dt, freq="D")

    lat_idx = None
    lon_idx = None
    if use_bounding_box:
      sub_lats, sub_lons, lat_idx, lon_idx = slice_coordinates_by_bounds(
          self.lats, self.lons, bounds=basins_gdf, buffer_degrees=0.5
      )
      if weights_matrix is not None:
        if (
            weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
            and np.allclose(weights_matrix.lats, sub_lats)
            and np.allclose(weights_matrix.lons, sub_lons)
        ):
          matrix = weights_matrix
        else:
          matrix = weights_matrix.crop_to_coords(sub_lats, sub_lons)
      else:
        matrix = ZonalWeightMatrix.from_geodataframe(
            basins_gdf, sub_lats, sub_lons, cell_res_lat=0.5, cell_res_lon=0.5
        )
    else:
      if weights_matrix is not None and (
          weights_matrix.grid_shape == (len(self.lats), len(self.lons))
          and np.allclose(weights_matrix.lats, self.lats)
          and np.allclose(weights_matrix.lons, self.lons)
      ):
        matrix = weights_matrix
      else:
        matrix = ZonalWeightMatrix.from_geodataframe(
            basins_gdf, self.lats, self.lons, cell_res_lat=0.5, cell_res_lon=0.5
        )

    precip_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )

    years = sorted(list(set(d.year for d in date_idx)))
    for yr in years:
      nc_path = ensure_psl_cpc_netcdf(yr, cache_dir=self.cache_dir)
      days_in_year = [d for d in date_idx if d.year == yr]

      if netCDF4 is not None:
        with netCDF4.Dataset(nc_path, "r") as nc:
          precip_var = nc.variables["precip"]
          t_var = nc.variables["time"]
          dates = netCDF4.num2date(t_var[:], units=t_var.units)
          date_map = {
              pd.to_datetime(str(d)[:10]): i for i, d in enumerate(dates)
          }

          for dt in tqdm.tqdm(
              days_in_year,
              desc=f"CPC PSL {yr} [{len(days_in_year)} days]",
              unit="day",
          ):
            if dt not in date_map:
              continue
            d_idx = list(date_idx).index(dt)
            t_idx = date_map[dt]
            day_slice = precip_var[t_idx, :, :]
            day_lat_inv = day_slice[::-1, :]
            day_shifted = np.concatenate(
                [day_lat_inv[:, 360:], day_lat_inv[:, :360]], axis=1
            )
            day_shifted = np.where(day_shifted < 0, np.nan, day_shifted)
            if use_bounding_box and lat_idx is not None and lon_idx is not None:
              day_shifted = day_shifted[lat_idx, :][:, lon_idx]

            precip_matrix[:, d_idx] = matrix.reduce_2d(day_shifted)
      else:
        with xr.open_dataset(nc_path) as ds:
          t_series = pd.to_datetime(ds.time.values)
          date_map = {
              pd.to_datetime(str(d)[:10]): i for i, d in enumerate(t_series)
          }
          precip_da = ds["precip"].values

          for dt in tqdm.tqdm(
              days_in_year,
              desc=f"CPC PSL {yr} [{len(days_in_year)} days]",
              unit="day",
          ):
            if dt not in date_map:
              continue
            d_idx = list(date_idx).index(dt)
            t_idx = date_map[dt]
            day_slice = precip_da[t_idx, :, :]
            day_lat_inv = day_slice[::-1, :]
            day_shifted = np.concatenate(
                [day_lat_inv[:, 360:], day_lat_inv[:, :360]], axis=1
            )
            day_shifted = np.where(day_shifted < 0, np.nan, day_shifted)
            if use_bounding_box and lat_idx is not None and lon_idx is not None:
              day_shifted = day_shifted[lat_idx, :][:, lon_idx]

            precip_matrix[:, d_idx] = matrix.reduce_2d(day_shifted)

    return xr.Dataset(
        data_vars={
            "cpc_precipitation": (["basin", "date"], precip_matrix),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )

  def extract_day_from_psl(
      self,
      dt: pd.Timestamp,
      matrix: ZonalWeightMatrix,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of CPC precipitation from PSL NetCDF."""
    dt = pd.to_datetime(dt)
    nc_path = ensure_psl_cpc_netcdf(dt.year, cache_dir=self.cache_dir)
    num_basins = matrix.matrix.shape[0]
    res = np.full(num_basins, np.nan, dtype=np.float32)

    if netCDF4 is not None:
      with netCDF4.Dataset(nc_path, "r") as nc:
        precip_var = nc.variables["precip"]
        t_var = nc.variables["time"]
        dates = netCDF4.num2date(t_var[:], units=t_var.units)
        date_map = {
            pd.to_datetime(str(d)[:10]): i for i, d in enumerate(dates)
        }
        if dt not in date_map:
          return {"cpc_precipitation": res}
        t_idx = date_map[dt]
        day_slice = precip_var[t_idx, :, :]
        day_lat_inv = day_slice[::-1, :]
        day_shifted = np.concatenate(
            [day_lat_inv[:, 360:], day_lat_inv[:, :360]], axis=1
        )
        day_shifted = np.where(day_shifted < 0, np.nan, day_shifted)
        res = matrix.reduce_2d(day_shifted)
    else:
      with xr.open_dataset(nc_path) as ds:
        t_series = pd.to_datetime(ds.time.values)
        date_map = {
            pd.to_datetime(str(d)[:10]): i for i, d in enumerate(t_series)
        }
        if dt not in date_map:
          return {"cpc_precipitation": res}
        t_idx = date_map[dt]
        day_slice = ds["precip"].values[t_idx, :, :]
        day_lat_inv = day_slice[::-1, :]
        day_shifted = np.concatenate(
            [day_lat_inv[:, 360:], day_lat_inv[:, :360]], axis=1
        )
        day_shifted = np.where(day_shifted < 0, np.nan, day_shifted)
        res = matrix.reduce_2d(day_shifted)

    return {"cpc_precipitation": res}

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      matrix: Optional[ZonalWeightMatrix] = None,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of CPC precipitation across basins."""
    dt = pd.to_datetime(dt)
    if self.source == "binary":
      cpc_file = resolve_date_to_cpc_file(self.data_dir, dt)
      if cpc_file and weights_dict is not None:
        return self.extract_day_from_cpc_file(
            cpc_file, list(basins_gdf.index), weights_dict
        )
      num_basins = len(basins_gdf)
      return {
          "cpc_precipitation": np.full(num_basins, np.nan, dtype=np.float32)
      }
    if matrix is None:
      matrix = ZonalWeightMatrix.from_geodataframe(
          basins_gdf, self.lats, self.lons, cell_res_lat=0.5, cell_res_lon=0.5
      )
    return self.extract_day_from_psl(dt, matrix)

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts CPC daily precipitation for given basin geometries."""
    basin_ids = list(basins_gdf.index)

    if start_date is not None:
      start_dt = pd.to_datetime(start_date)
    else:
      start_dt = pd.to_datetime("1979-01-01")

    if end_date is not None:
      end_dt = pd.to_datetime(end_date)
    else:
      end_dt = pd.to_datetime("today")

    if self.source == "psl":
      return self.extract_for_basins_psl(
          basins_gdf,
          start_dt,
          end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=use_bounding_box,
      )

    date_idx = pd.date_range(start_dt, end_dt, freq="D")

    # Pre-reduce weights for basins
    weights_dict = {}
    for b_id in basin_ids:
      geom = basins_gdf.loc[b_id].geometry
      w = self.zonal_calc.compute_weights(b_id, geom)
      if w is not None:
        weights_dict[b_id] = w

    precip_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )

    for d_idx, dt in enumerate(
        tqdm.tqdm(
            date_idx,
            desc=(
                f"CPC [{start_dt.strftime('%Y-%m-%d')} to"
                f" {end_dt.strftime('%Y-%m-%d')}]"
            ),
            unit="day",
            leave=True,
        )
    ):
      fpath = resolve_date_to_cpc_file(self.data_dir, dt)
      if fpath:
        day_res = self.extract_day_from_cpc_file(fpath, basin_ids, weights_dict)
        precip_matrix[:, d_idx] = day_res["cpc_precipitation"]

    ds = xr.Dataset(
        data_vars={
            "cpc_precipitation": (["basin", "date"], precip_matrix),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    return ds
