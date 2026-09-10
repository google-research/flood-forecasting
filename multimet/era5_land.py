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

"""ERA5-Land hourly GRIB extractor and Caravan variable reducer."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import fsspec
import xarray as xr

from multimet.base import BaseExtractor
from multimet.config import DEFAULT_STORAGE_PATHS
from multimet.config import Product
from multimet.config import PRODUCT_BANDS
from multimet.pet import calculate_fao56_penman_monteith_pet
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix

try:
  import gcsfs
except ImportError:
  gcsfs = None


def open_wb2_era5_dataset(zarr_path: str) -> xr.Dataset:
  """Opens WeatherBench 2 ERA5 Zarr store."""
  if gcsfs is not None and zarr_path.startswith("gs://"):
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(zarr_path.replace("gs://", ""))
    return xr.open_zarr(store, consolidated=True)
  return xr.open_zarr(zarr_path, consolidated=True)


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


def resolve_date_to_era5_grib_files(
    storage_dir: str, dt: pd.Timestamp
) -> List[str]:
  """Resolves a UTC calendar date to the 24 hourly GRIB files.

  According to ECMWF convention:
  Hours 01..23 are located in YYYY/MM/DD/
  Hour 00 is located in next_day YYYY/MM/DD/

  Args:
    storage_dir: Directory containing hourly GRIB archives.
    dt: Target date to resolve.

  Returns:
    List of 24 resolved file paths.
  """
  current_date = pd.to_datetime(dt)
  next_date = current_date + pd.Timedelta(days=1)

  curr_dir = os.path.join(
      storage_dir,
      f"{current_date.year:04d}",
      f"{current_date.month:02d}",
      f"{current_date.day:02d}",
  )
  next_dir = os.path.join(
      storage_dir,
      f"{next_date.year:04d}",
      f"{next_date.month:02d}",
      f"{next_date.day:02d}",
  )

  curr_tag = (
      f"{current_date.year:04d}{current_date.month:02d}{current_date.day:02d}"
  )
  next_tag = f"{next_date.year:04d}{next_date.month:02d}{next_date.day:02d}"

  files = [
      os.path.join(
          curr_dir, f"ERA5_Land_Hourly_{curr_tag}_default_{hh:02d}.grib"
      )
      for hh in range(1, 24)
  ]
  files.append(
      os.path.join(next_dir, f"ERA5_Land_Hourly_{next_tag}_default_00.grib")
  )
  return files


class ERA5LandExtractor(BaseExtractor):
  """Extractor for ECMWF ERA5-Land Reanalysis (0.1 deg, 15 bands)."""

  def __init__(self, data_dir: Optional[str] = None, source: str = "auto"):
    super().__init__(Product.ERA5_LAND, data_dir)
    source_lower = source.lower()
    if source_lower in ("auto", "default"):
      if data_dir is not None:
        if data_dir.startswith("gs://") or data_dir.endswith(".zarr"):
          self.source = "wb2"
        else:
          self.source = "grib"
      else:
        self.source = "wb2"
    elif source_lower in ("wb2", "public", "gcs", "arco"):
      self.source = "wb2"
    elif source_lower in ("grib", "local", "files"):
      self.source = "grib"
    else:
      self.source = source_lower

    if self.source == "wb2":
      self.data_dir = (
          data_dir
          if data_dir is not None
          else DEFAULT_STORAGE_PATHS[Product.ERA5_LAND]["wb2_s2s_zarr"]
      )
    else:
      self.data_dir = data_dir if data_dir is not None else ""

    # Standard ERA5-Land grid: 1801 lats x 3600 lons (0.1 deg) for GRIB files
    self.lats = np.linspace(90.0, -90.0, 1801, dtype=np.float64)
    lons_raw = np.linspace(0.0, 359.9, 3600, dtype=np.float64)
    lons_shifted = np.where(lons_raw > 180.0, lons_raw - 360.0, lons_raw)
    self.sort_lon_idx = np.argsort(lons_shifted)
    self.lons = lons_shifted[self.sort_lon_idx]
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.1, cell_res_lon=0.1
    )

  def _read_hourly_grib(self, grib_path: str) -> Dict[str, np.ndarray]:
    """Reads 2D meteorological fields from an ERA5-Land hourly GRIB file.

    Args:
      grib_path: Absolute path to the hourly GRIB file on CNS or local disk.

    Returns:
      Dictionary mapping variable short names ('2t', 'tp', etc.) to 2D numpy
      arrays of shape (1801, 3600) with coordinates sorted (-180 to +180).
    """
    fields = {}
    local_path = grib_path
    temp_download = False

    if not os.path.exists(grib_path):
      return fields
    try:
      # pylint: disable=g-import-not-at-top
      import eccodes
    except ImportError:
      return fields

    try:
      with open(local_path, "rb") as f:
        while True:
          try:
            gid = eccodes.codes_grib_new_from_file(f)
          except (eccodes.CodesInternalError, OSError, EOFError):
            break
          if gid is None:
            break
          try:
            short_name = eccodes.codes_get(gid, "shortName")
            if short_name in [
                "2t",
                "2d",
                "sp",
                "10u",
                "10v",
                "tp",
                "ssr",
                "str",
                "pev",
                "sd",
                "swvl1",
                "swvl2",
                "swvl3",
                "swvl4",
            ]:
              vals = eccodes.codes_get_values(gid).reshape((1801, 3600))
              # Mask missing values (ECMWF land mask is 9999.0)
              vals = np.where(np.isclose(vals, 9999.0, atol=1e-2), np.nan, vals)
              vals_shifted = vals[:, self.sort_lon_idx]
              fields[short_name] = vals_shifted
          finally:
            eccodes.codes_release(gid)
    finally:
      if temp_download and os.path.exists(local_path):
        try:
          os.remove(local_path)
        except OSError:
          pass

    return fields

  def extract_day_from_grib_files(
      self,
      grib_files: List[str],
      basin_ids: List[str],
      weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
  ) -> Dict[str, np.ndarray]:
    """Extracts and aggregates 1 day of ERA5-Land across 24 hourly GRIB files.

    Args:
      grib_files: List of 24 GRIB file paths (sorted 01:00 -> 00:00 next day).
      basin_ids: Ordered list of basin IDs.
      weights_dict: Precomputed zonal weights per basin.

    Returns:
      Dictionary mapping Caravan variable names to 1D float32 arrays of shape
      (num_basins,). If any file or hour is missing, returns all NaNs.
    """
    num_basins = len(basin_ids)
    res_dict = {
        band: np.full(num_basins, np.nan, dtype=np.float32)
        for band in PRODUCT_BANDS[Product.ERA5_LAND]
    }

    # Strict validation: require exactly 24 hourly files for a complete day.
    if len(grib_files) != 24:
      return res_dict

    hourly_t2m = {b: [] for b in basin_ids}
    hourly_d2m = {b: [] for b in basin_ids}
    hourly_sp = {b: [] for b in basin_ids}
    hourly_u10 = {b: [] for b in basin_ids}
    hourly_v10 = {b: [] for b in basin_ids}
    hourly_tp = {b: [] for b in basin_ids}
    hourly_ssr = {b: [] for b in basin_ids}
    hourly_str = {b: [] for b in basin_ids}
    hourly_pev = {b: [] for b in basin_ids}
    hourly_sd = {b: [] for b in basin_ids}
    hourly_sw1 = {b: [] for b in basin_ids}
    hourly_sw2 = {b: [] for b in basin_ids}
    hourly_sw3 = {b: [] for b in basin_ids}
    hourly_sw4 = {b: [] for b in basin_ids}

    for grib_file in grib_files:
      fields = self._read_hourly_grib(grib_file)
      if not fields:
        # Missing or unreadable hourly file: invalidate the entire day.
        return res_dict

      for b_id in basin_ids:
        if b_id not in weights_dict:
          continue
        lat_idx, lon_idx, w = weights_dict[b_id]

        if "2t" in fields:
          hourly_t2m[b_id].append(
              _weighted_mean_valid(fields["2t"][lat_idx, lon_idx], w)
          )
        if "2d" in fields:
          hourly_d2m[b_id].append(
              _weighted_mean_valid(fields["2d"][lat_idx, lon_idx], w)
          )
        if "sp" in fields:
          hourly_sp[b_id].append(
              _weighted_mean_valid(fields["sp"][lat_idx, lon_idx], w)
          )
        if "10u" in fields:
          hourly_u10[b_id].append(
              _weighted_mean_valid(fields["10u"][lat_idx, lon_idx], w)
          )
        if "10v" in fields:
          hourly_v10[b_id].append(
              _weighted_mean_valid(fields["10v"][lat_idx, lon_idx], w)
          )
        if "tp" in fields:
          hourly_tp[b_id].append(
              _weighted_mean_valid(fields["tp"][lat_idx, lon_idx], w)
          )
        if "ssr" in fields:
          hourly_ssr[b_id].append(
              _weighted_mean_valid(fields["ssr"][lat_idx, lon_idx], w)
          )
        if "str" in fields:
          hourly_str[b_id].append(
              _weighted_mean_valid(fields["str"][lat_idx, lon_idx], w)
          )
        if "pev" in fields:
          hourly_pev[b_id].append(
              _weighted_mean_valid(fields["pev"][lat_idx, lon_idx], w)
          )
        if "sd" in fields:
          hourly_sd[b_id].append(
              _weighted_mean_valid(fields["sd"][lat_idx, lon_idx], w)
          )
        if "swvl1" in fields:
          hourly_sw1[b_id].append(
              _weighted_mean_valid(fields["swvl1"][lat_idx, lon_idx], w)
          )
        if "swvl2" in fields:
          hourly_sw2[b_id].append(
              _weighted_mean_valid(fields["swvl2"][lat_idx, lon_idx], w)
          )
        if "swvl3" in fields:
          hourly_sw3[b_id].append(
              _weighted_mean_valid(fields["swvl3"][lat_idx, lon_idx], w)
          )
        if "swvl4" in fields:
          hourly_sw4[b_id].append(
              _weighted_mean_valid(fields["swvl4"][lat_idx, lon_idx], w)
          )

    for b_idx, b_id in enumerate(basin_ids):
      # Must have exactly 24 valid hourly records for each variable.
      if len(hourly_t2m[b_id]) != 24 or any(
          np.isnan(x) for x in hourly_t2m[b_id]
      ):
        continue

      t2m_k_mean = float(np.mean(hourly_t2m[b_id]))
      d2m_k_mean = float(np.mean(hourly_d2m[b_id]))
      sp_pa_mean = float(np.mean(hourly_sp[b_id]))
      u10_mean = float(np.mean(hourly_u10[b_id]))
      v10_mean = float(np.mean(hourly_v10[b_id]))
      # Accumulated fields: file 24 contains daily cumulative
      # total
      tp_m_total = float(hourly_tp[b_id][-1])
      ssr_jm2_total = float(hourly_ssr[b_id][-1])
      str_jm2_total = float(hourly_str[b_id][-1])
      pev_m_total = float(hourly_pev[b_id][-1])

      sd_m_mean = float(np.mean(hourly_sd[b_id]))
      sw1_mean = float(np.mean(hourly_sw1[b_id]))
      sw2_mean = float(np.mean(hourly_sw2[b_id]))
      sw3_mean = float(np.mean(hourly_sw3[b_id]))
      sw4_mean = float(np.mean(hourly_sw4[b_id]))

      # Exact Caravan Aggregations & Units:
      res_dict["era5land_temperature_2m"][b_idx] = t2m_k_mean - 273.15
      res_dict["era5land_dewpoint_temperature_2m"][b_idx] = d2m_k_mean - 273.15
      res_dict["era5land_surface_pressure"][b_idx] = sp_pa_mean / 1000.0
      res_dict["era5land_u_component_of_wind_10m"][b_idx] = u10_mean
      res_dict["era5land_v_component_of_wind_10m"][b_idx] = v10_mean
      res_dict["era5land_total_precipitation"][b_idx] = tp_m_total * 1000.0
      res_dict["era5land_surface_net_solar_radiation"][b_idx] = (
          ssr_jm2_total / 86400.0
      )
      res_dict["era5land_surface_net_thermal_radiation"][b_idx] = (
          str_jm2_total / 86400.0
      )
      res_dict["era5land_potential_evaporation_DEPRECATED"][b_idx] = (
          abs(pev_m_total) * 1000.0
      )
      res_dict["era5land_snow_depth_water_equivalent"][b_idx] = (
          sd_m_mean * 1000.0
      )
      res_dict["era5land_volumetric_soil_water_layer_1"][b_idx] = sw1_mean
      res_dict["era5land_volumetric_soil_water_layer_2"][b_idx] = sw2_mean
      res_dict["era5land_volumetric_soil_water_layer_3"][b_idx] = sw3_mean
      res_dict["era5land_volumetric_soil_water_layer_4"][b_idx] = sw4_mean

      # FAO-56 Penman-Monteith PET
      if not np.isnan(t2m_k_mean) and not np.isnan(d2m_k_mean):
        pet_val = calculate_fao56_penman_monteith_pet(
            np.array([t2m_k_mean]),
            np.array([d2m_k_mean]),
            np.array([sp_pa_mean]),
            np.array([ssr_jm2_total]),
            np.array([str_jm2_total]),
            np.array([u10_mean]),
            np.array([v10_mean]),
        )[0]
        res_dict["era5land_potential_evaporation_FAO_PENMAN_MONTEITH"][
            b_idx
        ] = pet_val

    return res_dict

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> xr.Dataset:
    """Extracts ERA5-Land 15 daily variables for given basin geometries."""
    if self.source == "wb2":
      return self.extract_for_basins_wb2(
          basins_gdf,
          start_date=start_date,
          end_date=end_date,
          weights_matrix=weights_matrix,
      )
    return self.extract_for_basins_grib(
        basins_gdf, start_date=start_date, end_date=end_date
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
    """Extracts 1 day of ERA5-Land (15 Caravan variables) across basins."""
    dt = pd.to_datetime(dt)
    if self.source == "wb2":
      ds = self.extract_for_basins_wb2(
          basins_gdf, start_date=dt, end_date=dt, weights_matrix=matrix
      )
      return {
          var: ds[var].values[:, 0].astype(np.float32) for var in ds.data_vars
      }
    else:
      grib_files = resolve_date_to_era5_grib_files(self.data_dir, dt)
      basin_ids = list(basins_gdf.index)
      if weights_dict is None:
        weights_dict = {}
        for b_id in basin_ids:
          geom = basins_gdf.loc[b_id].geometry
          w = self.zonal_calc.compute_weights(b_id, geom)
          if w is not None:
            weights_dict[b_id] = w
      return self.extract_day_from_grib_files(
          grib_files, basin_ids, weights_dict
      )

  def extract_for_basins_wb2(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> xr.Dataset:
    """Extracts ERA5 daily reanalysis for 15 Caravan variables from WeatherBench 2."""
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
    expected_bands = PRODUCT_BANDS[Product.ERA5_LAND]

    data_dict = {
        band: np.full((len(basin_ids), len(date_idx)), np.nan, dtype=np.float32)
        for band in expected_bands
    }

    bounds = basins_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    lat_slice = slice(min(90.0, maxy + 0.5), max(-90.0, miny - 0.5))

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

    ds_raw = open_wb2_era5_dataset(self.data_dir)

    time_slice = slice(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    if not is_split:
      sub = ds_raw.sel(latitude=lat_slice, longitude=lon_slice, time=time_slice)
    else:
      sub1 = ds_raw.sel(
          latitude=lat_slice,
          longitude=slice((minx % 360) - 0.5, 360.0),
          time=time_slice,
      )
      sub2 = ds_raw.sel(
          latitude=lat_slice,
          longitude=slice(0.0, maxx + 0.5),
          time=time_slice,
      )
      sub = xr.concat([sub1, sub2], dim="longitude")

    sub_lons = sub.longitude.values
    converted_lons = np.where(sub_lons > 180.0, sub_lons - 360.0, sub_lons)
    sub = sub.assign_coords(longitude=converted_lons).sortby("longitude")
    sub = sub.sortby("latitude", ascending=False)

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

    vars_needed = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation_24hr",
        "mean_surface_net_short_wave_radiation_flux",
        "mean_surface_net_long_wave_radiation_flux",
        "snow_depth",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
    ]
    avail_vars = [v for v in vars_needed if v in sub]

    sub_data = sub[avail_vars].compute()
    sub_times = pd.to_datetime(sub_data.time.values)

    # Vectorized sparse matrix reduction across all variables and basins simultaneously
    reduced_vars = {
        v: matrix.reduce_3d(sub_data[v].values) for v in avail_vars
    }

    sub_dates = [pd.to_datetime(t.strftime("%Y-%m-%d")) for t in sub_times]
    sub_mask = [d in date_idx for d in sub_dates]
    valid_sub_indices = np.where(sub_mask)[0]
    valid_target_indices = [
        date_idx.get_loc(sub_dates[i]) for i in valid_sub_indices
    ]

    if "2m_temperature" in reduced_vars:
      t2m_k = reduced_vars["2m_temperature"]
      data_dict["era5land_temperature_2m"][:, valid_target_indices] = (
          t2m_k[:, valid_sub_indices] - 273.15
      )
    if "2m_dewpoint_temperature" in reduced_vars:
      d2m_k = reduced_vars["2m_dewpoint_temperature"]
      data_dict["era5land_dewpoint_temperature_2m"][:, valid_target_indices] = (
          d2m_k[:, valid_sub_indices] - 273.15
      )
    if "surface_pressure" in reduced_vars:
      sp_pa = reduced_vars["surface_pressure"]
      data_dict["era5land_surface_pressure"][:, valid_target_indices] = (
          sp_pa[:, valid_sub_indices] / 1000.0
      )
    if "10m_u_component_of_wind" in reduced_vars:
      u10 = reduced_vars["10m_u_component_of_wind"]
      data_dict["era5land_u_component_of_wind_10m"][:, valid_target_indices] = (
          u10[:, valid_sub_indices]
      )
    if "10m_v_component_of_wind" in reduced_vars:
      v10 = reduced_vars["10m_v_component_of_wind"]
      data_dict["era5land_v_component_of_wind_10m"][:, valid_target_indices] = (
          v10[:, valid_sub_indices]
      )
    if "total_precipitation_24hr" in reduced_vars:
      tp_m = reduced_vars["total_precipitation_24hr"]
      data_dict["era5land_total_precipitation"][:, valid_target_indices] = (
          np.maximum(0.0, tp_m[:, valid_sub_indices] * 1000.0)
      )
    if "mean_surface_net_short_wave_radiation_flux" in reduced_vars:
      ssr_wm2 = reduced_vars["mean_surface_net_short_wave_radiation_flux"]
      data_dict["era5land_surface_net_solar_radiation"][
          :, valid_target_indices
      ] = ssr_wm2[:, valid_sub_indices]
    if "mean_surface_net_long_wave_radiation_flux" in reduced_vars:
      str_wm2 = reduced_vars["mean_surface_net_long_wave_radiation_flux"]
      data_dict["era5land_surface_net_thermal_radiation"][
          :, valid_target_indices
      ] = str_wm2[:, valid_sub_indices]
    if "snow_depth" in reduced_vars:
      sd_m = reduced_vars["snow_depth"]
      data_dict["era5land_snow_depth_water_equivalent"][
          :, valid_target_indices
      ] = np.maximum(0.0, sd_m[:, valid_sub_indices] * 1000.0)
    for sw_idx in range(1, 5):
      sw_name = f"volumetric_soil_water_layer_{sw_idx}"
      if sw_name in reduced_vars:
        data_dict[f"era5land_{sw_name}"][:, valid_target_indices] = (
            reduced_vars[sw_name][:, valid_sub_indices]
        )

    # Vectorized FAO-56 Penman-Monteith PET across all basins and dates simultaneously
    if (
        "2m_temperature" in reduced_vars
        and "2m_dewpoint_temperature" in reduced_vars
        and "surface_pressure" in reduced_vars
        and "mean_surface_net_short_wave_radiation_flux" in reduced_vars
        and "mean_surface_net_long_wave_radiation_flux" in reduced_vars
        and "10m_u_component_of_wind" in reduced_vars
        and "10m_v_component_of_wind" in reduced_vars
    ):
      pet_matrix = calculate_fao56_penman_monteith_pet(
          t2m_k=t2m_k[:, valid_sub_indices],
          d2m_k=d2m_k[:, valid_sub_indices],
          sp_pa=sp_pa[:, valid_sub_indices],
          ssr_jm2=ssr_wm2[:, valid_sub_indices] * 86400.0,
          str_jm2=str_wm2[:, valid_sub_indices] * 86400.0,
          u10_ms=u10[:, valid_sub_indices],
          v10_ms=v10[:, valid_sub_indices],
      )
      data_dict["era5land_potential_evaporation_FAO_PENMAN_MONTEITH"][
          :, valid_target_indices
      ] = pet_matrix

    try:
      ds_raw.close()
    except Exception:
      pass

    data_vars = {
        band: (["basin", "date"], data_dict[band]) for band in expected_bands
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    return ds

  def extract_for_basins_grib(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
  ) -> xr.Dataset:
    """Extracts ERA5-Land 15 daily variables from local/archived GRIB files."""
    basin_ids = list(basins_gdf.index)

    if start_date is not None:
      start_dt = pd.to_datetime(start_date)
    else:
      start_dt = pd.to_datetime("1950-01-01")

    if end_date is not None:
      end_dt = pd.to_datetime(end_date)
    else:
      end_dt = pd.to_datetime("today")

    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    expected_bands = PRODUCT_BANDS[Product.ERA5_LAND]

    # Pre-reduce weights for basins
    weights_dict = {}
    for b_id in basin_ids:
      geom = basins_gdf.loc[b_id].geometry
      w = self.zonal_calc.compute_weights(b_id, geom)
      if w is not None:
        weights_dict[b_id] = w

    data_dict = {
        band: np.full((len(basin_ids), len(date_idx)), np.nan, dtype=np.float32)
        for band in expected_bands
    }

    for d_idx, dt in enumerate(
        tqdm.tqdm(
            date_idx,
            desc=(
                f"ERA5_LAND [{start_dt.strftime('%Y-%m-%d')} to"
                f" {end_dt.strftime('%Y-%m-%d')}]"
            ),
            unit="day",
            leave=True,
        )
    ):
      grib_files = resolve_date_to_era5_grib_files(self.data_dir, dt)
      day_res = self.extract_day_from_grib_files(
          grib_files, basin_ids, weights_dict
      )
      for band in expected_bands:
        data_dict[band][:, d_idx] = day_res[band]

    data_vars = {
        band: (["basin", "date"], data_dict[band]) for band in expected_bands
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    return ds

# Alias for backward compatibility
ERA5LandExtractor.extract_for_basins_cns = ERA5LandExtractor.extract_for_basins_grib
