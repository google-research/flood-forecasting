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


def _weighted_mean_valid_with_coverage(
    vals: np.ndarray, weights: np.ndarray
) -> Tuple[float, float]:
  """Computes weighted mean over non-NaN grid cells and missing weight fraction."""
  if len(weights) == 0:
    return np.nan, 1.0
  total_w = float(np.sum(weights))
  if total_w <= 0.0:
    return np.nan, 1.0
  valid = ~np.isnan(vals)
  if not np.any(valid):
    return np.nan, 1.0
  w_valid = weights[valid]
  sum_w = float(np.sum(w_valid))
  if sum_w <= 0.0:
    return np.nan, 1.0
  missing_frac = float(np.clip((total_w - sum_w) / total_w, 0.0, 1.0))
  return float(np.sum(vals[valid] * w_valid) / sum_w), missing_frac


def _weighted_mean_valid(vals: np.ndarray, weights: np.ndarray) -> float:
  """Computes weighted mean over non-NaN grid cells, normalizing by valid weights."""
  mean_val, _ = _weighted_mean_valid_with_coverage(vals, weights)
  return mean_val


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
  """Extractor for NOAA CPC Global Unified Daily Precipitation and Gauge Count."""

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "auto",
      cache_dir: str = "/tmp/cpc_cache",
  ):
    super().__init__(Product.CPC, data_dir)
    self.cache_dir = cache_dir

    source_lower = source.lower()
    if source_lower in ("archive", "gridded_archive", "zarr", "zarr_archive"):
      self.source = "archive"
      if data_dir is None or not str(data_dir).strip():
        raise ValueError(
            "CPCExtractor with source='archive' requires an explicit Zarr "
            "store URI or path via data_dir."
        )
      self.data_dir = str(data_dir)
    elif source_lower in ("auto", "default"):
      if data_dir is not None and (
          str(data_dir).startswith(("gs://", "gcs://"))
          or str(data_dir).rstrip("/").endswith(".zarr")
      ):
        self.source = "archive"
        self.data_dir = str(data_dir)
      else:
        self.source = "psl"
        self.data_dir = (
            str(data_dir)
            if data_dir is not None
            else DEFAULT_STORAGE_PATHS[Product.CPC]["psl_netcdf"]
        )
    elif source_lower in ("psl", "public", "netcdf", "upstream"):
      self.source = "psl"
      self.data_dir = (
          str(data_dir)
          if data_dir is not None
          else DEFAULT_STORAGE_PATHS[Product.CPC]["psl_netcdf"]
      )
    elif source_lower in ("binary", "local"):
      self.source = "binary"
      if data_dir is None or not str(data_dir).strip():
        raise ValueError(
            "CPCExtractor with source='binary' requires an explicit local "
            "directory via data_dir."
        )
      self.data_dir = str(data_dir)
    else:
      self.source = source_lower
      self.data_dir = str(data_dir) if data_dir is not None else ""

    # Standard CPC 0.5 deg grid coordinates
    # Latitudes: -89.75 to 89.75 (south to north)
    self.lats = np.linspace(-89.75, 89.75, 360, dtype=np.float64)
    # Longitudes: -179.75 to 179.75
    self.lons = np.linspace(-179.75, 179.75, 720, dtype=np.float64)
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.5, cell_res_lon=0.5
    )

  @staticmethod
  def parse_cpc_file_fields(
      file_path: str,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a daily CPC binary file and returns (precip_mm, num_stations).

    Both arrays have shape (360, 720) on (-89.75..89.75, -179.75..179.75).
    """
    with open(file_path, "rb") as f:
      content = f.read()

    if file_path.endswith(".gz"):
      content = gzip.decompress(content)

    raw_array = np.frombuffer(content, dtype="<f4")
    # Shape is (2, 360, 720): field 0 is precip (0.1mm), field 1 is num_stations
    if len(raw_array) >= 360 * 720 * 2:
      precip_field = raw_array[: 360 * 720].reshape((360, 720))
      stations_field = raw_array[360 * 720 : 360 * 720 * 2].reshape((360, 720))
      stations_clean = np.where(stations_field < 0, np.nan, stations_field)
      stations_shifted = np.concatenate(
          [stations_clean[:, 360:], stations_clean[:, :360]], axis=1
      ).astype(np.float32)
    elif len(raw_array) == 360 * 720:
      precip_field = raw_array.reshape((360, 720))
      stations_shifted = np.full((360, 720), np.nan, dtype=np.float32)
    else:
      raise ValueError(
          f"Unexpected binary size {len(raw_array)} for CPC file {file_path}"
      )

    # Set missing / negative values to NaN, convert 0.1 mm -> mm/day
    precip_mm = np.where(precip_field < 0, np.nan, precip_field * 0.1)

    # Shift longitudes from (0.25 .. 359.75) to (-179.75 .. 179.75)
    precip_shifted = np.concatenate(
        [precip_mm[:, 360:], precip_mm[:, :360]], axis=1
    ).astype(np.float32)

    return precip_shifted, stations_shifted

  @staticmethod
  def parse_cpc_file(file_path: str) -> np.ndarray:
    """Reads a daily CPC binary file and returns a 2D array of (lat, lon) in mm/day."""
    precip_shifted, _ = CPCExtractor.parse_cpc_file_fields(file_path)
    return precip_shifted

  def extract_day_from_cpc_file(
      self,
      cpc_file: str,
      basin_ids: List[str],
      weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of CPC precipitation and gauge counts across all basins."""
    num_basins = len(basin_ids)
    res_precip = np.full(num_basins, np.nan, dtype=np.float32)
    res_stations = np.full(num_basins, np.nan, dtype=np.float32)
    res_missing = np.ones(num_basins, dtype=np.float32)
    precip_2d, stations_2d = self.parse_cpc_file_fields(cpc_file)
    for b_idx, b_id in enumerate(basin_ids):
      if b_id not in weights_dict:
        continue
      lat_idx, lon_idx, w = weights_dict[b_id]
      if len(w) == 0:
        continue
      p_val, miss_frac = _weighted_mean_valid_with_coverage(
          precip_2d[lat_idx, lon_idx], w
      )
      s_val, _ = _weighted_mean_valid_with_coverage(
          stations_2d[lat_idx, lon_idx], w
      )
      res_precip[b_idx] = p_val
      res_stations[b_idx] = s_val
      res_missing[b_idx] = miss_frac
    return {
        "cpc_precipitation": res_precip,
        "cpc_num_stations": res_stations,
        "cpc_missing_fraction": res_missing,
    }

  def extract_for_basins_psl(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_dt: pd.Timestamp,
      end_dt: pd.Timestamp,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Extracts CPC daily precipitation using public NOAA PSL yearly NetCDF files."""
    from multimet.gridded_archive import _warn_missing_variables_once

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
    stations_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )
    missing_matrix = np.ones(
        (len(basin_ids), len(date_idx)), dtype=np.float32
    )
    _warn_missing_variables_once(
        self.data_dir, Product.CPC, ["cpc_num_stations"]
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

            vals, miss = matrix.reduce_2d_with_coverage(day_shifted)
            precip_matrix[:, d_idx] = vals
            missing_matrix[:, d_idx] = miss
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

            vals, miss = matrix.reduce_2d_with_coverage(day_shifted)
            precip_matrix[:, d_idx] = vals
            missing_matrix[:, d_idx] = miss

    return xr.Dataset(
        data_vars={
            "cpc_precipitation": (["basin", "date"], precip_matrix),
            "cpc_num_stations": (["basin", "date"], stations_matrix),
            "cpc_missing_fraction": (["basin", "date"], missing_matrix),
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
    from multimet.gridded_archive import _warn_missing_variables_once

    dt = pd.to_datetime(dt)
    nc_path = ensure_psl_cpc_netcdf(dt.year, cache_dir=self.cache_dir)
    num_basins = matrix.matrix.shape[0]
    res = np.full(num_basins, np.nan, dtype=np.float32)
    stations = np.full(num_basins, np.nan, dtype=np.float32)
    missing = np.ones(num_basins, dtype=np.float32)
    _warn_missing_variables_once(
        self.data_dir, Product.CPC, ["cpc_num_stations"]
    )

    if netCDF4 is not None:
      with netCDF4.Dataset(nc_path, "r") as nc:
        precip_var = nc.variables["precip"]
        t_var = nc.variables["time"]
        dates = netCDF4.num2date(t_var[:], units=t_var.units)
        date_map = {
            pd.to_datetime(str(d)[:10]): i for i, d in enumerate(dates)
        }
        if dt not in date_map:
          return {
              "cpc_precipitation": res,
              "cpc_num_stations": stations,
              "cpc_missing_fraction": missing,
          }
        t_idx = date_map[dt]
        day_slice = precip_var[t_idx, :, :]
        day_lat_inv = day_slice[::-1, :]
        day_shifted = np.concatenate(
            [day_lat_inv[:, 360:], day_lat_inv[:, :360]], axis=1
        )
        day_shifted = np.where(day_shifted < 0, np.nan, day_shifted)
        res, missing = matrix.reduce_2d_with_coverage(day_shifted)
    else:
      with xr.open_dataset(nc_path) as ds:
        t_series = pd.to_datetime(ds.time.values)
        date_map = {
            pd.to_datetime(str(d)[:10]): i for i, d in enumerate(t_series)
        }
        if dt not in date_map:
          return {
              "cpc_precipitation": res,
              "cpc_num_stations": stations,
              "cpc_missing_fraction": missing,
          }
        t_idx = date_map[dt]
        day_slice = ds["precip"].values[t_idx, :, :]
        day_lat_inv = day_slice[::-1, :]
        day_shifted = np.concatenate(
            [day_lat_inv[:, 360:], day_lat_inv[:, :360]], axis=1
        )
        day_shifted = np.where(day_shifted < 0, np.nan, day_shifted)
        res, missing = matrix.reduce_2d_with_coverage(day_shifted)

    return {
        "cpc_precipitation": res,
        "cpc_num_stations": stations,
        "cpc_missing_fraction": missing,
    }

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
    if self.source == "archive":
      from multimet.gridded_archive import extract_nowcast_from_archive

      ds_day = extract_nowcast_from_archive(
          Product.CPC,
          self.data_dir,
          basins_gdf,
          start_date=dt,
          end_date=dt,
          weights_matrix=matrix,
          use_bounding_box=True,
      )
      return {
          var: ds_day[var].values[:, 0].astype(np.float32)
          for var in ds_day.data_vars
      }
    if self.source == "binary":
      cpc_file = resolve_date_to_cpc_file(self.data_dir, dt)
      if cpc_file and weights_dict is not None:
        return self.extract_day_from_cpc_file(
            cpc_file, list(basins_gdf.index), weights_dict
        )
      num_basins = len(basins_gdf)
      return {
          "cpc_precipitation": np.full(num_basins, np.nan, dtype=np.float32),
          "cpc_num_stations": np.full(num_basins, np.nan, dtype=np.float32),
          "cpc_missing_fraction": np.ones(num_basins, dtype=np.float32),
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
    """Extracts CPC daily precipitation and station count for given basin geometries."""
    if start_date is None or end_date is None:
      raise ValueError(
          "CPCExtractor.extract_for_basins requires both start_date and "
          "end_date to be explicitly provided."
      )
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
      raise ValueError(
          f"end_date ({end_dt}) must be >= start_date ({start_dt})."
      )

    if self.source == "archive":
      from multimet.gridded_archive import extract_nowcast_from_archive

      return extract_nowcast_from_archive(
          Product.CPC,
          self.data_dir,
          basins_gdf,
          start_date=start_dt,
          end_date=end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=use_bounding_box,
      )

    if self.source == "psl":
      return self.extract_for_basins_psl(
          basins_gdf,
          start_dt,
          end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=use_bounding_box,
      )

    basin_ids = list(basins_gdf.index)
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
    stations_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )
    missing_matrix = np.ones(
        (len(basin_ids), len(date_idx)), dtype=np.float32
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
        stations_matrix[:, d_idx] = day_res["cpc_num_stations"]
        missing_matrix[:, d_idx] = day_res["cpc_missing_fraction"]

    ds = xr.Dataset(
        data_vars={
            "cpc_precipitation": (["basin", "date"], precip_matrix),
            "cpc_num_stations": (["basin", "date"], stations_matrix),
            "cpc_missing_fraction": (["basin", "date"], missing_matrix),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    return ds
