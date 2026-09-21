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

"""Extractor for ECMWF ERA5-Land Daily Surface Reanalysis (0.1 deg, 17 bands).

By design, ``ERA5LandExtractor`` reads **exclusively** from a user-supplied
ERA5-Land gridded Zarr archive (e.g., ``gs://open-multimet/data/era5_land/daily_surface.zarr``).
Direct extraction from third-party/upstream sources (such as WeatherBench 2's
0.25 deg atmospheric ERA5 store) is intentionally disabled to prevent silent
model/resolution substitution.

Also provides :meth:`ERA5LandExtractor.extract_day_from_grib_files` for
computing daily mean, daily minimum (``era5land_temperature_2m_min``), and
daily maximum (``era5land_temperature_2m_max``) across 24 sub-daily hourly
ERA5-Land GRIB files.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from multimet.base import BaseExtractor
from multimet.config import MISSING_FRACTION_VAR, PRODUCT_BANDS, Product
from multimet.gridded_archive import extract_nowcast_from_archive
from multimet.pet import calculate_fao56_penman_monteith_pet
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix

logger = logging.getLogger(__name__)

_ALLOWED_ERA5_SOURCES = frozenset(
    {"archive", "gridded_archive", "zarr", "zarr_archive", "auto", "default"}
)


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


def resolve_date_to_era5_land_grib_files(
    archive_dir: str, dt: pd.Timestamp
) -> List[str]:
  """Resolves a UTC calendar date to the 24 hourly ERA5-Land GRIB files.

  Hours 01..23 are located in ``YYYY/MM/DD/``; hour 00 (end-of-day cumulative
  step) is located in ``next_day YYYY/MM/DD/``.
  """
  dt = pd.to_datetime(dt)
  next_dt = dt + pd.Timedelta(days=1)

  day_dir = os.path.join(
      archive_dir,
      f"{dt.year:04d}",
      f"{dt.month:02d}",
      f"{dt.day:02d}",
  )
  next_day_dir = os.path.join(
      archive_dir,
      f"{next_dt.year:04d}",
      f"{next_dt.month:02d}",
      f"{next_dt.day:02d}",
  )

  files: List[str] = []
  date_str = dt.strftime("%Y%m%d")
  for hr in range(1, 24):
    cand = os.path.join(day_dir, f"era5_land_{date_str}_{hr:02d}00.grib")
    if os.path.exists(cand):
      files.append(cand)

  next_str = next_dt.strftime("%Y%m%d")
  cand_00 = os.path.join(next_day_dir, f"era5_land_{next_str}_0000.grib")
  if os.path.exists(cand_00):
    files.append(cand_00)
  return files


class ERA5LandExtractor(BaseExtractor):
  """Extractor for ECMWF ERA5-Land Reanalysis (0.1 deg, 17 bands).

  Reads exclusively from a user-supplied ERA5-Land gridded Zarr archive.
  """

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "archive",
  ):
    super().__init__(Product.ERA5_LAND, data_dir)
    source_lower = source.lower().strip()
    if source_lower not in _ALLOWED_ERA5_SOURCES:
      raise ValueError(
          f"ERA5LandExtractor does not support third-party source={source!r}. "
          "ERA5-Land can only be extracted from a user-supplied gridded "
          "archive (source='archive')."
      )
    self.source = "archive"
    self.data_dir = str(data_dir).strip() if data_dir is not None else ""

    # Standard ERA5-Land 0.1 deg grid: 1801 lats (90 -> -90) x 3600 lons (-180 -> 179.9)
    self.lats = np.linspace(90.0, -90.0, 1801, dtype=np.float64)
    lons_raw = np.linspace(0.0, 359.9, 3600, dtype=np.float64)
    lons_shifted = np.where(lons_raw >= 180.0, lons_raw - 360.0, lons_raw)
    self.sort_lon_idx = np.argsort(lons_shifted)
    self.lons = lons_shifted[self.sort_lon_idx]
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.1, cell_res_lon=0.1
    )

  def _require_archive_uri(self) -> str:
    if not self.data_dir:
      raise ValueError(
          "ERA5LandExtractor requires an explicit gridded archive Zarr URI "
          "or path via data_dir (or --archive-store ERA5_LAND=<URI>). "
          "Default or fallback paths are not permitted."
      )
    return self.data_dir

  def _read_hourly_grib(self, grib_path: str) -> Dict[str, np.ndarray]:
    """Reads 2D meteorological fields from an ERA5-Land hourly GRIB file."""
    fields: Dict[str, np.ndarray] = {}
    if not os.path.exists(grib_path):
      return fields
    import eccodes

    with open(grib_path, "rb") as f:
      while True:
        gid = eccodes.codes_grib_new_from_file(f)
        if gid is None:
          break
        short_name = eccodes.codes_get(gid, "shortName")
        if short_name in (
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
        ):
          vals = eccodes.codes_get_values(gid).reshape((1801, 3600))
          vals = np.where(np.isclose(vals, 9999.0, atol=1e-2), np.nan, vals)
          fields[short_name] = vals[:, self.sort_lon_idx]
        eccodes.codes_release(gid)

    return fields

  def extract_day_from_grib_files(
      self,
      grib_files: List[str],
      basin_ids: List[str],
      weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
  ) -> Dict[str, np.ndarray]:
    """Aggregates 1 day of ERA5-Land across 24 sub-daily hourly GRIB files.

    Computes daily mean, daily minimum (``era5land_temperature_2m_min``), and
    daily maximum (``era5land_temperature_2m_max``) from the 24 hourly steps
    (01:00 Day D through 00:00 Day D+1). If any of the 24 hourly files is
    missing or unreadable, returns all NaNs (missing data in -> missing data out).
    """
    num_basins = len(basin_ids)
    res_dict = {
        band: np.full(num_basins, np.nan, dtype=np.float32)
        for band in PRODUCT_BANDS[Product.ERA5_LAND]
    }
    res_dict[MISSING_FRACTION_VAR[Product.ERA5_LAND]] = np.ones(
        num_basins, dtype=np.float32
    )

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
    hourly_missing = {b: [] for b in basin_ids}

    for grib_file in grib_files:
      fields = self._read_hourly_grib(grib_file)
      if not fields:
        return res_dict

      for b_id in basin_ids:
        if b_id not in weights_dict:
          continue
        lat_idx, lon_idx, w = weights_dict[b_id]
        if len(w) == 0:
          continue

        if "2t" in fields:
          v_t2m, m_t2m = _weighted_mean_valid_with_coverage(
              fields["2t"][lat_idx, lon_idx], w
          )
          hourly_t2m[b_id].append(v_t2m)
          hourly_missing[b_id].append(m_t2m)
        if "2d" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["2d"][lat_idx, lon_idx], w
          )
          hourly_d2m[b_id].append(v)
        if "sp" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["sp"][lat_idx, lon_idx], w
          )
          hourly_sp[b_id].append(v)
        if "10u" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["10u"][lat_idx, lon_idx], w
          )
          hourly_u10[b_id].append(v)
        if "10v" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["10v"][lat_idx, lon_idx], w
          )
          hourly_v10[b_id].append(v)
        if "tp" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["tp"][lat_idx, lon_idx], w
          )
          hourly_tp[b_id].append(v)
        if "ssr" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["ssr"][lat_idx, lon_idx], w
          )
          hourly_ssr[b_id].append(v)
        if "str" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["str"][lat_idx, lon_idx], w
          )
          hourly_str[b_id].append(v)
        if "pev" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["pev"][lat_idx, lon_idx], w
          )
          hourly_pev[b_id].append(v)
        if "sd" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["sd"][lat_idx, lon_idx], w
          )
          hourly_sd[b_id].append(v)
        if "swvl1" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["swvl1"][lat_idx, lon_idx], w
          )
          hourly_sw1[b_id].append(v)
        if "swvl2" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["swvl2"][lat_idx, lon_idx], w
          )
          hourly_sw2[b_id].append(v)
        if "swvl3" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["swvl3"][lat_idx, lon_idx], w
          )
          hourly_sw3[b_id].append(v)
        if "swvl4" in fields:
          v, _ = _weighted_mean_valid_with_coverage(
              fields["swvl4"][lat_idx, lon_idx], w
          )
          hourly_sw4[b_id].append(v)

    for idx, b_id in enumerate(basin_ids):
      if len(hourly_t2m[b_id]) != 24 or np.any(np.isnan(hourly_t2m[b_id])):
        continue
      t2m_arr = np.asarray(hourly_t2m[b_id], dtype=np.float32)
      t2m_mean_k = float(np.mean(t2m_arr))
      d2m_k = float(np.mean(hourly_d2m[b_id]))
      sp_pa = float(np.mean(hourly_sp[b_id]))
      u10_ms = float(np.mean(hourly_u10[b_id]))
      v10_ms = float(np.mean(hourly_v10[b_id]))
      # 24th file (00:00 Day D+1) holds the full 24h cumulative total
      tp_m = float(hourly_tp[b_id][-1])
      ssr_jm2 = float(hourly_ssr[b_id][-1])
      str_jm2 = float(hourly_str[b_id][-1])
      pev_m = float(hourly_pev[b_id][-1])

      fao_pet = float(
          calculate_fao56_penman_monteith_pet(
              np.asarray([t2m_mean_k], dtype=np.float32),
              np.asarray([d2m_k], dtype=np.float32),
              np.asarray([sp_pa], dtype=np.float32),
              np.asarray([ssr_jm2], dtype=np.float32),
              np.asarray([str_jm2], dtype=np.float32),
              np.asarray([u10_ms], dtype=np.float32),
              np.asarray([v10_ms], dtype=np.float32),
          )[0]
      )

      res_dict["era5land_temperature_2m"][idx] = t2m_mean_k - 273.15
      res_dict["era5land_dewpoint_temperature_2m"][idx] = d2m_k - 273.15
      res_dict["era5land_surface_pressure"][idx] = sp_pa / 1000.0
      res_dict["era5land_u_component_of_wind_10m"][idx] = u10_ms
      res_dict["era5land_v_component_of_wind_10m"][idx] = v10_ms
      res_dict["era5land_total_precipitation"][idx] = max(0.0, tp_m * 1000.0)
      res_dict["era5land_surface_net_solar_radiation"][idx] = ssr_jm2 / 86400.0
      res_dict["era5land_surface_net_thermal_radiation"][idx] = (
          str_jm2 / 86400.0
      )
      res_dict["era5land_potential_evaporation_DEPRECATED"][idx] = (
          -pev_m * 1000.0
      )
      res_dict["era5land_potential_evaporation_FAO_PENMAN_MONTEITH"][idx] = (
          fao_pet
      )
      res_dict["era5land_snow_depth_water_equivalent"][idx] = max(
          0.0, float(np.mean(hourly_sd[b_id])) * 1000.0
      )
      res_dict["era5land_volumetric_soil_water_layer_1"][idx] = float(
          np.mean(hourly_sw1[b_id])
      )
      res_dict["era5land_volumetric_soil_water_layer_2"][idx] = float(
          np.mean(hourly_sw2[b_id])
      )
      res_dict["era5land_volumetric_soil_water_layer_3"][idx] = float(
          np.mean(hourly_sw3[b_id])
      )
      res_dict["era5land_volumetric_soil_water_layer_4"][idx] = float(
          np.mean(hourly_sw4[b_id])
      )
      res_dict[MISSING_FRACTION_VAR[Product.ERA5_LAND]][idx] = float(
          np.mean(hourly_missing[b_id])
      )

    return res_dict

  def extract_day(
      self,
      dt: Union[str, pd.Timestamp],
      basins_gdf: gpd.GeoDataFrame,
      matrix: Optional[ZonalWeightMatrix] = None,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of ERA5-Land across all basins from the gridded archive."""
    del weights_dict
    uri = self._require_archive_uri()
    dt_ts = pd.to_datetime(dt)
    ds_day = extract_nowcast_from_archive(
        Product.ERA5_LAND,
        uri,
        basins_gdf,
        start_date=dt_ts,
        end_date=dt_ts,
        weights_matrix=matrix,
        use_bounding_box=True,
    )
    return {
        var: ds_day[var].values[:, 0].astype(np.float32)
        for var in ds_day.data_vars
    }

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts ERA5-Land daily surface variables from the gridded archive."""
    del kwargs
    uri = self._require_archive_uri()
    if start_date is None or end_date is None:
      raise ValueError(
          "ERA5LandExtractor.extract_for_basins requires both start_date and "
          "end_date to be explicitly provided."
      )
    return extract_nowcast_from_archive(
        Product.ERA5_LAND,
        uri,
        basins_gdf,
        start_date=start_date,
        end_date=end_date,
        weights_matrix=weights_matrix,
        use_bounding_box=use_bounding_box,
    )
