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
import os
from typing import Dict, List, Mapping, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import fsspec
import xarray as xr

try:
  import gcsfs
except ImportError:
  gcsfs = None

from googlehydrology.multimet.base import BaseExtractor
from googlehydrology.multimet.config import (
    DEFAULT_STORAGE_PATHS,
    FORECAST_LEAD_DAYS,
    PRODUCT_BANDS,
    Product,
)
from googlehydrology.multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix

WB2_GRAPHCAST_URLS = {
    2018: (
        "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr"
    ),
    2020: (
        "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr"
    ),
}


def get_wb2_graphcast_url(dt: pd.Timestamp) -> str:
  """Returns WeatherBench 2 GCS Zarr store path for a given date."""
  if dt < pd.Timestamp("2019-06-01"):
    return WB2_GRAPHCAST_URLS[2018]
  return WB2_GRAPHCAST_URLS[2020]


def open_wb2_dataset(zarr_url: str) -> xr.Dataset:
  """Opens WeatherBench 2 GraphCast Zarr store."""
  if gcsfs is not None:
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(zarr_url)
    return xr.open_zarr(mapper, decode_timedelta=False)
  return xr.open_zarr(zarr_url, decode_timedelta=False)


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


def _compute_basin_steps(
    grid_40: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
  """Computes weighted 40-step forecast timeseries for a basin polygon."""
  vals = grid_40[:, lat_indices, lon_indices]
  if not np.isnan(vals).any():
    return (vals * weights).sum(axis=-1)
  res = np.empty(grid_40.shape[0], dtype=np.float32)
  for step in range(grid_40.shape[0]):
    res[step] = _weighted_mean_valid(vals[step], weights)
  return res


def extract_day_from_graphcast(
    graphcast_zarr_path: str,
    dt: pd.Timestamp,
    basin_ids: List[str],
    weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    sort_lon_idx: np.ndarray,
    lat_flip: bool = False,
) -> Dict[str, np.ndarray]:
  """Extracts 1 forecast initialization date across 10 lead days for GraphCast.

  Returns a dict mapping band names to 2D numpy arrays of shape (num_basins,
  10). If any required forecast timestep is missing, the lead evaluates to NaN.
  """
  dt = pd.to_datetime(dt)
  num_basins = len(basin_ids)
  res_dict = {
      band: np.full((num_basins, 10), np.nan, dtype=np.float32)
      for band in PRODUCT_BANDS[Product.GRAPHCAST]
  }

  try:
    if graphcast_zarr_path.startswith("gs://"):
      ds_raw = open_wb2_dataset(graphcast_zarr_path)
    else:
      ds_raw = xr.open_zarr(graphcast_zarr_path, decode_timedelta=False)

    time_target = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')}T00:00:00")
    times_raw = pd.to_datetime(ds_raw.time.values)
    if time_target in times_raw:
      t_slice = ds_raw.sel(time=time_target)
    elif dt in times_raw:
      t_slice = ds_raw.sel(time=dt)
    else:
      return res_dict

    for lt_day in range(1, 11):
      lt_pos = lt_day - 1
      step_start = (lt_day - 1) * 4
      step_end = lt_day * 4
      if step_end > len(ds_raw.prediction_timedelta):
        continue
      p_steps = slice(step_start, step_end)

      if "2m_temperature" in ds_raw:
        raw_grid = (
            t_slice["2m_temperature"]
            .isel(prediction_timedelta=p_steps)
            .mean(dim="prediction_timedelta", skipna=False)
            .values
            - 273.15
        )
        if lat_flip:
          raw_grid = raw_grid[::-1, :]
        sorted_grid = raw_grid[:, sort_lon_idx]
        for b_idx, b_id in enumerate(basin_ids):
          if b_id in weights_dict:
            lat_i, lon_i, w = weights_dict[b_id]
            res_dict["graphcast_temperature_2m"][b_idx, lt_pos] = (
                _weighted_mean_valid(sorted_grid[lat_i, lon_i], w)
            )

      if "10m_u_component_of_wind" in ds_raw:
        raw_grid = (
            t_slice["10m_u_component_of_wind"]
            .isel(prediction_timedelta=p_steps)
            .mean(dim="prediction_timedelta", skipna=False)
            .values
        )
        if lat_flip:
          raw_grid = raw_grid[::-1, :]
        sorted_grid = raw_grid[:, sort_lon_idx]
        for b_idx, b_id in enumerate(basin_ids):
          if b_id in weights_dict:
            lat_i, lon_i, w = weights_dict[b_id]
            res_dict["graphcast_u_component_of_wind_10m"][b_idx, lt_pos] = (
                _weighted_mean_valid(sorted_grid[lat_i, lon_i], w)
            )

      if "10m_v_component_of_wind" in ds_raw:
        raw_grid = (
            t_slice["10m_v_component_of_wind"]
            .isel(prediction_timedelta=p_steps)
            .mean(dim="prediction_timedelta", skipna=False)
            .values
        )
        if lat_flip:
          raw_grid = raw_grid[::-1, :]
        sorted_grid = raw_grid[:, sort_lon_idx]
        for b_idx, b_id in enumerate(basin_ids):
          if b_id in weights_dict:
            lat_i, lon_i, w = weights_dict[b_id]
            res_dict["graphcast_v_component_of_wind_10m"][b_idx, lt_pos] = (
                _weighted_mean_valid(sorted_grid[lat_i, lon_i], w)
            )

      if "total_precipitation_6hr" in ds_raw:
        raw_grid = (
            t_slice["total_precipitation_6hr"]
            .isel(prediction_timedelta=p_steps)
            .sum(dim="prediction_timedelta", skipna=False)
            .values
            * 1000.0
        )
        if lat_flip:
          raw_grid = raw_grid[::-1, :]
        sorted_grid = raw_grid[:, sort_lon_idx]
        for b_idx, b_id in enumerate(basin_ids):
          if b_id in weights_dict:
            lat_i, lon_i, w = weights_dict[b_id]
            res_dict["graphcast_total_precipitation"][b_idx, lt_pos] = (
                _weighted_mean_valid(sorted_grid[lat_i, lon_i], w)
            )
  except Exception:
    return res_dict

  return res_dict


class GraphCastExtractor(BaseExtractor):
  """Extractor for DeepMind GraphCast 10-Day Forecasts (0.25 deg, 4 bands)."""

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "auto",
  ):
    super().__init__(Product.GRAPHCAST, data_dir)
    source_lower = source.lower()
    if source_lower in ("auto", "default"):
      if data_dir is not None and not data_dir.startswith("gs://"):
        self.source = "zarr"
      else:
        self.source = "wb2"
    elif source_lower in ("wb2", "public", "gcs"):
      self.source = "wb2"
    elif source_lower in ("local", "zarr", "archive"):
      self.source = "zarr"
    else:
      self.source = source_lower

    if self.source == "wb2":
      self.data_dir = (
          data_dir
          if data_dir is not None
          else DEFAULT_STORAGE_PATHS[Product.GRAPHCAST]["wb2_zarr"]
      )
    else:
      self.data_dir = data_dir if data_dir is not None else ""

    # Standard GraphCast 0.25 deg grid in [-180, 180] longitude convention
    self.lats = np.linspace(90.0, -90.0, 721, dtype=np.float64)
    lons_raw = np.linspace(0.0, 359.75, 1440, dtype=np.float64)
    lons_shifted = np.where(lons_raw > 180.0, lons_raw - 360.0, lons_raw)
    self.sort_lon_idx = np.argsort(lons_shifted)
    self.lons = lons_shifted[self.sort_lon_idx]
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.25, cell_res_lon=0.25
    )
    self._opened_stores: Dict[str, xr.Dataset] = {}
    self._opened_subsets: Dict[str, Tuple[xr.Dataset, ZonalWeightMatrix]] = {}

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> xr.Dataset:
    """Extracts 10-day GraphCast forecasts for given basin geometries."""
    if self.source == "wb2":
      return self.extract_for_basins_wb2(
          basins_gdf,
          start_date=start_date,
          end_date=end_date,
          weights_matrix=weights_matrix,
      )
    return self.extract_for_basins_zarr(
        basins_gdf, start_date=start_date, end_date=end_date
    )

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 forecast initialization date across 10 lead days for GraphCast."""
    dt = pd.to_datetime(dt)
    basin_ids = list(basins_gdf.index)
    num_basins = len(basin_ids)
    expected_bands = PRODUCT_BANDS[Product.GRAPHCAST]
    res_dict = {
        band: np.full((num_basins, 10), np.nan, dtype=np.float32)
        for band in expected_bands
    }

    if self.source == "wb2":
      store_url = (
          self.data_dir
          if self.data_dir.startswith("gs://")
          and "weatherbench2" not in self.data_dir
          else get_wb2_graphcast_url(dt)
      )

      if store_url not in self._opened_subsets:
        if store_url not in self._opened_stores:
          self._opened_stores[store_url] = open_wb2_dataset(store_url)
        ds_raw = self._opened_stores[store_url]

        bounds = basins_gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        lat_slice = slice(max(-90.0, miny - 0.5), min(90.0, maxy + 0.5))

        if minx < 0 and maxx < 0:
          min_lon_wb2 = minx % 360
          max_lon_wb2 = maxx % 360
          lon_slice = slice(min_lon_wb2 - 0.5, max_lon_wb2 + 0.5)
          is_split = False
        elif minx >= 0 and maxx >= 0:
          lon_slice = slice(max(0.0, minx - 0.5), min(360.0, maxx + 0.5))
          is_split = False
        else:
          is_split = True

        if not is_split:
          sub = ds_raw.sel(lat=lat_slice, lon=lon_slice)
        else:
          sub1 = ds_raw.sel(lat=lat_slice, lon=slice((minx % 360) - 0.5, 360.0))
          sub2 = ds_raw.sel(lat=lat_slice, lon=slice(0.0, maxx + 0.5))
          sub = xr.concat([sub1, sub2], dim="lon")

        sub_lons = sub.lon.values
        converted_lons = np.where(sub_lons > 180.0, sub_lons - 360.0, sub_lons)
        sub = sub.assign_coords(lon=converted_lons).sortby("lon")

        if weights_matrix is not None and (
            weights_matrix.grid_shape == (len(sub.lat), len(sub.lon))
            and np.allclose(weights_matrix.lats, sub.lat.values)
            and np.allclose(weights_matrix.lons, sub.lon.values)
        ):
          matrix = weights_matrix
        else:
          matrix = ZonalWeightMatrix.from_geodataframe(
              basins_gdf,
              sub.lat.values,
              sub.lon.values,
              cell_res_lat=0.25,
              cell_res_lon=0.25,
          )
        self._opened_subsets[store_url] = (sub, matrix)

      sub, matrix = self._opened_subsets[store_url]
      time_target = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')}T00:00:00")

      if time_target not in pd.to_datetime(sub.time.values):
        return res_dict

      target_vars = [
          "2m_temperature",
          "total_precipitation_6hr",
          "10m_u_component_of_wind",
          "10m_v_component_of_wind",
      ]
      try:
        day_sub = sub.sel(time=time_target)[target_vars].compute()
      except Exception:
        return res_dict

      t2m_grid = day_sub["2m_temperature"].values - 273.15
      tp_grid = day_sub["total_precipitation_6hr"].values * 1000.0
      u_grid = day_sub["10m_u_component_of_wind"].values
      v_grid = day_sub["10m_v_component_of_wind"].values

      t2m_reduced = matrix.reduce_3d(t2m_grid)
      tp_reduced = matrix.reduce_3d(tp_grid)
      u_reduced = matrix.reduce_3d(u_grid)
      v_reduced = matrix.reduce_3d(v_grid)

      for lt_day in range(1, 11):
        lt_pos = lt_day - 1
        p_start = (lt_day - 1) * 4
        p_end = lt_day * 4
        if p_end > t2m_reduced.shape[1]:
          continue
        res_dict["graphcast_temperature_2m"][:, lt_pos] = np.mean(
            t2m_reduced[:, p_start:p_end], axis=1
        )
        res_dict["graphcast_total_precipitation"][:, lt_pos] = np.sum(
            tp_reduced[:, p_start:p_end], axis=1
        )
        res_dict["graphcast_u_component_of_wind_10m"][:, lt_pos] = np.mean(
            u_reduced[:, p_start:p_end], axis=1
        )
        res_dict["graphcast_v_component_of_wind_10m"][:, lt_pos] = np.mean(
            v_reduced[:, p_start:p_end], axis=1
        )
      return res_dict

    if weights_dict is None:
      weights_dict = {}
      for b_id in basin_ids:
        geom = basins_gdf.loc[b_id].geometry
        w = self.zonal_calc.compute_weights(b_id, geom)
        if w is not None:
          weights_dict[b_id] = w
    lat_flip = bool(self.lats[0] < self.lats[-1])
    return extract_day_from_graphcast(
        self.data_dir,
        dt,
        basin_ids,
        weights_dict,
        self.sort_lon_idx,
        lat_flip=lat_flip,
    )

  def extract_for_basins_wb2(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> xr.Dataset:
    """Extracts 10-day GraphCast forecasts from WeatherBench 2 on GCS."""
    basin_ids = list(basins_gdf.index)

    if start_date is not None:
      start_dt = pd.to_datetime(start_date)
    else:
      start_dt = pd.to_datetime("2020-01-01")

    if end_date is not None:
      end_dt = pd.to_datetime(end_date)
    else:
      end_dt = pd.to_datetime("2020-01-02")

    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    lead_steps = FORECAST_LEAD_DAYS[Product.GRAPHCAST]  # 10 days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps + 1), unit="D")
    expected_bands = PRODUCT_BANDS[Product.GRAPHCAST]

    shape = (len(basin_ids), len(date_idx), len(lead_time_idx))
    data_dict = {
        band: np.full(shape, np.nan, dtype=np.float32)
        for band in expected_bands
    }

    for d_pos, dt in enumerate(
        tqdm.tqdm(
            date_idx,
            desc=(
                f"GRAPHCAST WB2 [{start_dt.strftime('%Y-%m-%d')} to"
                f" {end_dt.strftime('%Y-%m-%d')}]"
            ),
            unit="init_date",
            leave=True,
        )
    ):
      day_res = self.extract_day(
          dt, basins_gdf, weights_matrix=weights_matrix
      )
      for band in expected_bands:
        data_dict[band][:, d_pos, :] = day_res[band]

    for store_ds in self._opened_stores.values():
      try:
        store_ds.close()
      except Exception:
        pass
    self._opened_stores.clear()
    self._opened_subsets.clear()

    data_vars = {
        band: (["basin", "date", "lead_time"], data_dict[band])
        for band in expected_bands
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
            "lead_time": lead_time_idx.values,
        },
    )
    return ds

  def extract_for_basins_zarr(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
  ) -> xr.Dataset:
    """Extracts 10-day GraphCast forecasts from local or archived Zarr store."""
    basin_ids = list(basins_gdf.index)

    if start_date is not None:
      start_dt = pd.to_datetime(start_date)
    else:
      start_dt = pd.to_datetime("2016-01-02")

    if end_date is not None:
      end_dt = pd.to_datetime(end_date)
    else:
      end_dt = pd.to_datetime("2023-12-21")

    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    lead_steps = FORECAST_LEAD_DAYS[Product.GRAPHCAST]  # 10 days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps + 1), unit="D")
    expected_bands = PRODUCT_BANDS[Product.GRAPHCAST]

    # Pre-reduce weights for basins
    weights_dict = {}
    for b_id in basin_ids:
      geom = basins_gdf.loc[b_id].geometry
      w = self.zonal_calc.compute_weights(b_id, geom)
      if w is not None:
        weights_dict[b_id] = w

    shape = (len(basin_ids), len(date_idx), len(lead_time_idx))
    data_dict = {
        band: np.full(shape, np.nan, dtype=np.float32)
        for band in expected_bands
    }

    # Setup coordinate indices
    try:
      ds_raw = xr.open_zarr(self.data_dir, decode_timedelta=False)
      raw_lons = ds_raw.lon.values
      converted_lons = np.where(raw_lons > 180.0, raw_lons - 360.0, raw_lons)
      sort_lon_idx = np.argsort(converted_lons)
      raw_lats = ds_raw.lat.values
      lat_flip = bool(raw_lats[0] < raw_lats[-1])
    except Exception:
      sort_lon_idx = np.arange(1440)
      lat_flip = False

    for d_pos, dt in enumerate(
        tqdm.tqdm(
            date_idx,
            desc=(
                f"GRAPHCAST CNS [{start_dt.strftime('%Y-%m-%d')} to"
                f" {end_dt.strftime('%Y-%m-%d')}]"
            ),
            unit="init_date",
            leave=True,
        )
    ):
      day_res = extract_day_from_graphcast(
          self.data_dir, dt, basin_ids, weights_dict, sort_lon_idx, lat_flip
      )
      for band in expected_bands:
        data_dict[band][:, d_pos, :] = day_res[band]

    data_vars = {
        band: (["basin", "date", "lead_time"], data_dict[band])
        for band in expected_bands
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
            "lead_time": lead_time_idx.values,
        },
    )
    return ds

# Alias for backward compatibility
GraphCastExtractor.extract_for_basins_cns = GraphCastExtractor.extract_for_basins_zarr
