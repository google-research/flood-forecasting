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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import logging
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import fsspec
import xarray as xr
import zarr

import gcsfs

from multimet.base import BaseExtractor
from multimet.config import (
    DEFAULT_STORAGE_PATHS,
    FORECAST_LEAD_DAYS,
    PRODUCT_BANDS,
    PRODUCT_METADATA_ATTRS,
    Product,
)
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix


def open_wb2_hres_dataset(zarr_url: Union[str, xr.Dataset]) -> xr.Dataset:
  """Opens WeatherBench 2 HRES Zarr store."""
  if isinstance(zarr_url, xr.Dataset):
    return zarr_url
  if gcsfs is not None:
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(zarr_url)
    return xr.open_zarr(mapper, decode_timedelta=False)
  return xr.open_zarr(zarr_url, decode_timedelta=False)


def _compute_zonal_mean(
    raster_2d: np.ndarray,
    basin_ids: List[str],
    weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
  """Computes weighted zonal mean over non-NaN grid cells for each basin."""
  out = np.full(len(basin_ids), np.nan, dtype=np.float32)
  for b_idx, b_id in enumerate(basin_ids):
    if b_id in weights_dict:
      lat_i, lon_i, w = weights_dict[b_id]
      vals = raster_2d[lat_i, lon_i]
      valid = ~np.isnan(vals)
      if np.any(valid):
        w_valid = w[valid]
        sum_w = np.sum(w_valid)
        if sum_w > 0.0:
          out[b_idx] = float(np.sum(vals[valid] * w_valid) / sum_w)
  return out


def _extract_instantaneous_lead(
    z_root: zarr.Group,
    var_name: str,
    h_start: int,
    h_end: int,
    sort_lon_idx: np.ndarray,
    basin_ids: List[str],
    weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scale: float = 1.0,
    offset: float = 0.0,
) -> Optional[np.ndarray]:
  """Extracts and averages instantaneous slices over a 24h lead window."""
  if var_name not in z_root:
    return None
  var_slice = z_root[var_name][h_start:h_end, :, :]
  expected_len = h_end - h_start
  if var_slice.shape[0] != expected_len:
    # Incomplete hourly slice: return None to propagate NaN
    return None
  var_mean = np.mean(var_slice, axis=0)[:, sort_lon_idx]
  return (
      _compute_zonal_mean(var_mean, basin_ids, weights_dict) * scale + offset
  )


def _extract_accumulated_lead(
    z_root: zarr.Group,
    var_name: str,
    h_start: int,
    h_end: int,
    t0_hours: int,
    sort_lon_idx: np.ndarray,
    basin_ids: List[str],
    weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scale: float = 1.0,
    is_strictly_positive: bool = True,
) -> Optional[np.ndarray]:
  """Extracts differenced accumulations or hourly flux sums between lead end and start."""
  if var_name not in z_root:
    return None
  is_hourly_flux = "_1hr" in var_name.lower() or "tprate" in var_name.lower()
  if is_hourly_flux:
    slice_data = z_root[var_name][h_start:h_end, :, :]
    val_diff = np.sum(slice_data, axis=0)[:, sort_lon_idx] * scale
  else:
    val_end = z_root[var_name][h_end, :, :][:, sort_lon_idx]
    val_start = (
        z_root[var_name][h_start, :, :][:, sort_lon_idx]
        if h_start > t0_hours
        else 0.0
    )
    val_diff = (val_end - val_start) * scale

  if is_strictly_positive:
    val_diff = np.where(val_diff < -1e-4, np.nan, np.maximum(0.0, val_diff))
  return _compute_zonal_mean(val_diff, basin_ids, weights_dict)


def extract_day_from_hres(
    hres_zarr_path: str,
    dt: pd.Timestamp,
    basin_ids: List[str],
    weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    sort_lon_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
  """Extracts 1 forecast initialization date across 10 lead days for HRES.

  Returns a dict mapping band names to 2D numpy arrays of shape (num_basins,
  10).
  """
  dt = pd.to_datetime(dt)
  base_dt = pd.to_datetime("2016-01-01 00:00:00")
  t0_hours = int((dt - base_dt).total_seconds() // 3600)

  num_basins = len(basin_ids)
  res_dict = {
      band: np.full((num_basins, 10), np.nan, dtype=np.float32)
      for band in PRODUCT_BANDS[Product.HRES]
  }

  z_root = zarr.open_group(hres_zarr_path, mode="r")

  for d in range(1, 11):
    lead_idx = d - 1
    h_end = t0_hours + 24 * d
    h_start = t0_hours + 24 * (d - 1)

    t2m = _extract_instantaneous_lead(
        z_root,
        "2m_temperature",
        h_start,
        h_end,
        sort_lon_idx,
        basin_ids,
        weights_dict,
        offset=-273.15,
    )
    if t2m is not None:
      res_dict["hres_temperature_2m"][:, lead_idx] = t2m

    sp = _extract_instantaneous_lead(
        z_root,
        "surface_pressure",
        h_start,
        h_end,
        sort_lon_idx,
        basin_ids,
        weights_dict,
        scale=0.001,
    )
    if sp is not None:
      res_dict["hres_surface_pressure"][:, lead_idx] = sp

    tp = _extract_accumulated_lead(
        z_root,
        "hres_fc_total_precipitation_1hr",
        h_start,
        h_end,
        t0_hours,
        sort_lon_idx,
        basin_ids,
        weights_dict,
        scale=1000.0,
    )
    if tp is not None:
      res_dict["hres_total_precipitation"][:, lead_idx] = tp

    ssr = _extract_accumulated_lead(
        z_root,
        "surface_net_solar_radiation_1hr",
        h_start,
        h_end,
        t0_hours,
        sort_lon_idx,
        basin_ids,
        weights_dict,
        scale=1.0 / 86400.0,
    )
    if ssr is not None:
      res_dict["hres_surface_net_solar_radiation"][:, lead_idx] = ssr

    str_val = _extract_accumulated_lead(
        z_root,
        "surface_net_thermal_radiation",
        h_start,
        h_end,
        t0_hours,
        sort_lon_idx,
        basin_ids,
        weights_dict,
        scale=1.0 / 86400.0,
        is_strictly_positive=False,
    )
    if str_val is not None:
      res_dict["hres_surface_net_thermal_radiation"][:, lead_idx] = str_val

  return res_dict


def _weighted_mean_1d(vals: np.ndarray, weights: np.ndarray) -> float:
  """Computes weighted mean over non-NaN grid cells."""
  valid = ~np.isnan(vals)
  if not np.any(valid):
    return np.nan
  w_v = weights[valid]
  sum_w = np.sum(w_v)
  if sum_w <= 0.0:
    return np.nan
  return float(np.sum(vals[valid] * w_v) / sum_w)


class HRESExtractor(BaseExtractor):
  """Extractor for ECMWF IFS High Resolution (HRES) 10-Day Operational Forecasts."""

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "auto",
  ):
    super().__init__(Product.HRES, data_dir)
    source_lower = source.lower()
    if source_lower in ("auto", "default"):
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
          else DEFAULT_STORAGE_PATHS[Product.HRES]["wb2_zarr"]
      )
    else:
      self.data_dir = data_dir if data_dir is not None else ""

    if self.source == "wb2":
      # WeatherBench 2 HRES 0.25 deg grid: 721 lats x 1440 lons
      self.lats = np.linspace(-90.0, 90.0, 721, dtype=np.float64)
      self.lons = np.linspace(-180.0, 179.75, 1440, dtype=np.float64)
      self.sort_lon_idx = np.arange(1440)
      self.zonal_calc = ZonalWeightCalculator(
          self.lats, self.lons, cell_res_lat=0.25, cell_res_lon=0.25
      )
    else:
      # Operational 0.1 deg grid: 1801 lats x 3600 lons
      self.lats = np.linspace(90.0, -90.0, 1801, dtype=np.float64)
      lons_raw = np.linspace(0.0, 359.9, 3600, dtype=np.float64)
      lons_shifted = np.where(lons_raw > 180.0, lons_raw - 360.0, lons_raw)
      self.sort_lon_idx = np.argsort(lons_shifted)
      self.lons = lons_shifted[self.sort_lon_idx]
      self.zonal_calc = ZonalWeightCalculator(
          self.lats, self.lons, cell_res_lat=0.1, cell_res_lon=0.1
      )

    self._cached_ds_raw = None
    self._cached_sub = None
    self._cached_matrix = None
    self._cached_basin_keys = None

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts HRES forecast dataset for given basin geometries."""
    if self.source == "wb2":
      return self.extract_for_basins_wb2(
          basins_gdf,
          start_date=start_date,
          end_date=end_date,
          weights_matrix=weights_matrix,
          use_bounding_box=use_bounding_box,
      )
    return self.extract_for_basins_zarr(
        basins_gdf, start_date=start_date, end_date=end_date
    )

  def _extract_day_wb2(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 forecast initialization date across 10 lead days from WeatherBench 2 HRES."""
    num_basins = len(basins_gdf)
    res_dict = {
        band: np.full((num_basins, 10), np.nan, dtype=np.float32)
        for band in PRODUCT_BANDS[Product.HRES]
    }

    basin_keys = tuple(basins_gdf.index)
    if (
        self._cached_sub is None
        or self._cached_matrix is None
        or self._cached_basin_keys != basin_keys
    ):
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

      if self._cached_ds_raw is None:
        if isinstance(self.data_dir, xr.Dataset):
          self._cached_ds_raw = self.data_dir
        else:
          self._cached_ds_raw = open_wb2_hres_dataset(self.data_dir)

      ds_raw = self._cached_ds_raw
      if not is_split:
        sub = ds_raw.sel(latitude=lat_slice, longitude=lon_slice)
      else:
        sub1 = ds_raw.sel(
            latitude=lat_slice, longitude=slice((minx % 360) - 0.5, 360.0)
        )
        sub2 = ds_raw.sel(latitude=lat_slice, longitude=slice(0.0, maxx + 0.5))
        sub = xr.concat([sub1, sub2], dim="longitude")

      sub_lons = sub.longitude.values
      converted_lons = np.where(sub_lons > 180.0, sub_lons - 360.0, sub_lons)
      sub = sub.assign_coords(longitude=converted_lons).sortby("longitude")

      matrix = ZonalWeightMatrix.from_geodataframe(
          basins_gdf,
          sub.latitude.values,
          sub.longitude.values,
          cell_res_lat=0.25,
          cell_res_lon=0.25,
      )
      self._cached_sub = sub
      self._cached_matrix = matrix
      self._cached_basin_keys = basin_keys

    sub = self._cached_sub
    matrix = self._cached_matrix

    time_target = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')}T00:00:00")
    if time_target not in pd.to_datetime(sub.time.values):
      return res_dict

    target_vars = ["2m_temperature", "surface_pressure"]
    if "total_precipitation_24hr" in sub:
      target_vars.append("total_precipitation_24hr")
    elif "total_precipitation" in sub:
      target_vars.append("total_precipitation")

    day_sub = sub.sel(time=time_target)[target_vars].compute()

    t2m_raw = day_sub["2m_temperature"].values - 273.15
    sp_raw = day_sub["surface_pressure"].values * 0.001
    has_tp24 = "total_precipitation_24hr" in day_sub
    if has_tp24:
      tp_24_raw = day_sub["total_precipitation_24hr"].values * 1000.0
    else:
      tp_raw = day_sub["total_precipitation"].values * 1000.0

    t2m_reduced = matrix.reduce_3d(t2m_raw)
    sp_reduced = matrix.reduce_3d(sp_raw)
    if has_tp24:
      tp_reduced = matrix.reduce_3d(tp_24_raw)
    else:
      tp_reduced = matrix.reduce_3d(tp_raw)

    for lt_day in range(1, 11):
      lt_pos = lt_day - 1
      step_slice = slice((lt_day - 1) * 4 + 1, lt_day * 4 + 1)
      if lt_day * 4 >= t2m_raw.shape[0]:
        continue

      mean_t2m = np.nanmean(t2m_reduced[:, step_slice], axis=1)
      mean_sp = np.nanmean(sp_reduced[:, step_slice], axis=1)
      if has_tp24:
        p_tp = tp_reduced[:, lt_day * 4]
      else:
        p_tp = (
            tp_reduced[:, lt_day * 4]
            - tp_reduced[:, (lt_day - 1) * 4]
        )

      res_dict["hres_temperature_2m"][:, lt_pos] = mean_t2m
      res_dict["hres_surface_pressure"][:, lt_pos] = mean_sp
      res_dict["hres_total_precipitation"][:, lt_pos] = np.maximum(0.0, p_tp)

    # Note: hres_surface_net_solar_radiation and hres_surface_net_thermal_radiation
    # remain np.nan as they are unavailable in WeatherBench 2 HRES archive.
    return res_dict

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 forecast initialization date across 10 lead days for HRES."""
    dt = pd.to_datetime(dt)
    if self.source == "wb2":
      return self._extract_day_wb2(dt, basins_gdf)

    basin_ids = list(basins_gdf.index)
    if weights_dict is None:
      weights_dict = {}
      for b_id in basin_ids:
        geom = basins_gdf.loc[b_id].geometry
        w = self.zonal_calc.compute_weights(b_id, geom)
        if w is not None:
          weights_dict[b_id] = w
    return extract_day_from_hres(
        self.data_dir, dt, basin_ids, weights_dict, self.sort_lon_idx
    )

  def extract_for_basins_wb2(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Extracts 10-day HRES forecasts from WeatherBench 2 on GCS."""
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
    lead_steps = FORECAST_LEAD_DAYS[Product.HRES]  # 10 days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps + 1), unit="D")
    expected_bands = PRODUCT_BANDS[Product.HRES]

    shape = (len(basin_ids), len(date_idx), len(lead_time_idx))
    data_dict = {
        band: np.full(shape, np.nan, dtype=np.float32)
        for band in expected_bands
    }

    ds_raw = open_wb2_hres_dataset(self.data_dir)

    if use_bounding_box:
      bounds = basins_gdf.total_bounds  # (minx, miny, maxx, maxy)
      minx, miny, maxx, maxy = bounds
      lat_slice = slice(max(-90.0, miny - 0.5), min(90.0, maxy + 0.5))

      # Spatial slicing in WB2 [0, 360) longitude convention
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
        sub = ds_raw.sel(latitude=lat_slice, longitude=lon_slice)
      else:
        sub1 = ds_raw.sel(
            latitude=lat_slice, longitude=slice((minx % 360) - 0.5, 360.0)
        )
        sub2 = ds_raw.sel(latitude=lat_slice, longitude=slice(0.0, maxx + 0.5))
        sub = xr.concat([sub1, sub2], dim="longitude")

      sub_lons = sub.longitude.values
      converted_lons = np.where(sub_lons > 180.0, sub_lons - 360.0, sub_lons)
      sub = sub.assign_coords(longitude=converted_lons).sortby("longitude")
    else:
      sub_lons = ds_raw.longitude.values
      converted_lons = np.where(sub_lons > 180.0, sub_lons - 360.0, sub_lons)
      sub = ds_raw.assign_coords(longitude=converted_lons).sortby("longitude")

    if weights_matrix is not None and (
        weights_matrix.grid_shape == (len(sub.latitude), len(sub.longitude))
        and np.allclose(weights_matrix.lats, sub.latitude.values)
        and np.allclose(weights_matrix.lons, sub.longitude.values)
    ):
      matrix = weights_matrix
    else:
      matrix = ZonalWeightMatrix.from_geodataframe(
          basins_gdf,
          sub.latitude.values,
          sub.longitude.values,
          cell_res_lat=0.25,
          cell_res_lon=0.25,
      )

    target_vars = [
        "2m_temperature",
        "surface_pressure",
    ]
    if "total_precipitation_24hr" in sub:
      target_vars.append("total_precipitation_24hr")
    elif "total_precipitation" in sub:
      target_vars.append("total_precipitation")

    for d_pos, dt in enumerate(
        tqdm.tqdm(
            date_idx,
            desc=(
                f"HRES WB2 [{start_dt.strftime('%Y-%m-%d')} to"
                f" {end_dt.strftime('%Y-%m-%d')}]"
            ),
            unit="init_date",
            leave=True,
        )
    ):
      time_target = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')}T00:00:00")
      if time_target not in pd.to_datetime(sub.time.values):
        continue

      day_sub = sub.sel(time=time_target)[target_vars].compute()

      t2m_raw = day_sub["2m_temperature"].values - 273.15
      sp_raw = day_sub["surface_pressure"].values * 0.001
      has_tp24 = "total_precipitation_24hr" in day_sub
      if has_tp24:
        tp_24_raw = day_sub["total_precipitation_24hr"].values * 1000.0
      else:
        tp_raw = day_sub["total_precipitation"].values * 1000.0

      # Vectorized sparse matrix reduction across all basins simultaneously:
      # shape (N_basins, steps)
      t2m_reduced = matrix.reduce_3d(t2m_raw)
      sp_reduced = matrix.reduce_3d(sp_raw)
      if has_tp24:
        tp_reduced = matrix.reduce_3d(tp_24_raw)
      else:
        tp_reduced = matrix.reduce_3d(tp_raw)

      for lt_day in range(1, 11):
        lt_pos = lt_day - 1
        step_slice = slice((lt_day - 1) * 4 + 1, lt_day * 4 + 1)
        if lt_day * 4 >= t2m_raw.shape[0]:
          continue

        mean_t2m = np.nanmean(t2m_reduced[:, step_slice], axis=1)
        mean_sp = np.nanmean(sp_reduced[:, step_slice], axis=1)
        if has_tp24:
          p_tp = tp_reduced[:, lt_day * 4]
        else:
          p_tp = (
              tp_reduced[:, lt_day * 4]
              - tp_reduced[:, (lt_day - 1) * 4]
          )

        data_dict["hres_temperature_2m"][:, d_pos, lt_pos] = mean_t2m
        data_dict["hres_surface_pressure"][:, d_pos, lt_pos] = mean_sp
        data_dict["hres_total_precipitation"][:, d_pos, lt_pos] = np.maximum(
            0.0, p_tp
        )

    ds_raw.close()

    data_vars = {}
    for band in expected_bands:
      var_attrs = {}
      if band in (
          "hres_surface_net_solar_radiation",
          "hres_surface_net_thermal_radiation",
      ):
        var_attrs = {
            "status": "unavailable",
            "comment": (
                "Surface radiation flux variables are unavailable in"
                " WeatherBench 2 HRES archive."
            ),
        }
      data_vars[band] = (
          ["basin", "date", "lead_time"],
          data_dict[band],
          var_attrs,
      )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
            "lead_time": lead_time_idx.values,
        },
        attrs=dict(PRODUCT_METADATA_ATTRS.get(Product.HRES, {})),
    )
    return ds

  def extract_for_basins_zarr(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
  ) -> xr.Dataset:
    """Extracts 10-day HRES forecast from local or archived Zarr store."""
    basin_ids = list(basins_gdf.index)

    if start_date is not None:
      start_dt = pd.to_datetime(start_date)
    else:
      start_dt = pd.to_datetime("2016-01-01")

    if end_date is not None:
      end_dt = pd.to_datetime(end_date)
    else:
      end_dt = pd.to_datetime("today")

    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    lead_steps = FORECAST_LEAD_DAYS[Product.HRES]  # 10 days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps + 1), unit="D")
    expected_bands = PRODUCT_BANDS[Product.HRES]

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

    for d_pos, dt in enumerate(
        tqdm.tqdm(
            date_idx,
            desc=(
                f"HRES CNS [{start_dt.strftime('%Y-%m-%d')} to"
                f" {end_dt.strftime('%Y-%m-%d')}]"
            ),
            unit="init_date",
            leave=True,
        )
    ):
      day_res = extract_day_from_hres(
          self.data_dir, dt, basin_ids, weights_dict, self.sort_lon_idx
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
HRESExtractor.extract_for_basins_cns = HRESExtractor.extract_for_basins_zarr
