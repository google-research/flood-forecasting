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

import concurrent.futures
import datetime
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import gc
import logging
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import dask
import fsspec
import xarray as xr
import zarr

import gcsfs

from multimet.base import BaseExtractor
from multimet.build_hres_archive import (
    ECMWF_OPEN_DATA_BUCKET,
    decode_grib2_message,
    deaccumulate,
)
from multimet.config import (
    DEFAULT_STORAGE_PATHS,
    FORECAST_LEAD_DAYS,
    PRODUCT_BANDS,
    PRODUCT_METADATA_ATTRS,
    Product,
)
from multimet.spatial import slice_coordinates_by_bounds
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix

logger = logging.getLogger(__name__)

OPEN_DATA_LEAD_STEPS = (24, 48, 72, 96, 120, 144, 168, 192, 216, 240)
OPEN_DATA_GRID_SHAPE = (721, 1440)


def find_latest_hres_open_data_date(
    bucket: str = ECMWF_OPEN_DATA_BUCKET,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    max_lookback_days: int = 7,
    require_full_10d: bool = True,
    fs: Optional[Any] = None,
) -> pd.Timestamp:
  """Finds the latest published ECMWF Open Data 00z HRES initialization date.

  Walks backwards from ``reference_date`` (default: current UTC date) up to
  ``max_lookback_days`` until it finds an initialization date whose 00z 0.25-deg
  IFS forecast index file is published on ``gs://ecmwf-open-data``.
  """
  if fs is None:
    if gcsfs is None:
      raise ImportError("gcsfs is required to query ECMWF Open Data on GCS.")
    fs = gcsfs.GCSFileSystem(token="anon")

  bucket_clean = bucket.removeprefix("gs://").strip("/")
  if reference_date is None or str(reference_date).strip().lower() == "latest":
    ref_dt = pd.Timestamp(datetime.datetime.now(datetime.timezone.utc).date())
  else:
    ref_dt = pd.to_datetime(reference_date).floor("D")

  check_step = 240 if require_full_10d else 24
  for offset in range(max_lookback_days + 1):
    cand_dt = ref_dt - pd.Timedelta(days=offset)
    date_str = cand_dt.strftime("%Y%m%d")
    idx_path = (
        f"{bucket_clean}/{date_str}/00z/ifs/0p25/oper/"
        f"{date_str}000000-{check_step}h-oper-fc.index"
    )
    try:
      if fs.exists(idx_path):
        return cand_dt
    except Exception as err:  # noqa: BLE001
      logger.debug("Error checking %s: %s", idx_path, err)
      continue

  raise FileNotFoundError(
      f"No published ECMWF Open Data 00z HRES forecast found within "
      f"{max_lookback_days} days of {ref_dt.strftime('%Y-%m-%d')} "
      f"in bucket gs://{bucket_clean}."
  )


def _fetch_open_data_day_grids(
    fs: Any,
    bucket: str,
    dt: pd.Timestamp,
    lead_steps: Sequence[int] = OPEN_DATA_LEAD_STEPS,
    lat_idx: Optional[np.ndarray] = None,
    lon_idx: Optional[np.ndarray] = None,
    max_retries: int = 3,
) -> Optional[Dict[str, np.ndarray]]:
  """Fetches and decodes 1 initialization date of ECMWF Open Data GRIB2 slices.

  Uses batched ``fs.cat`` on ``.index`` files and ``fs.cat_ranges`` on ``.grib2``
  byte ranges for high-throughput concurrent GCS reads, then converts units to
  the canonical Caravan MultiMet ``hres_*`` schema on an ascending latitude
  ``[-90, 90]`` and longitude ``[-180, 179.75]`` grid (optionally cropped to
  ``lat_idx`` / ``lon_idx``).

  Returns ``None`` if the initialization date is not published upstream.
  """
  bucket_clean = bucket.removeprefix("gs://").strip("/")
  date_str = pd.to_datetime(dt).strftime("%Y%m%d")
  prefix = f"{bucket_clean}/{date_str}/00z/ifs/0p25/oper/{date_str}000000"
  idx_paths = [f"{prefix}-{step}h-oper-fc.index" for step in lead_steps]
  target_params = ("2t", "sp", "tp", "ssr", "str")

  last_err: Optional[Exception] = None
  for attempt in range(max_retries):
    try:
      idx_contents: Dict[str, bytes] = {}
      if hasattr(fs, "cat"):
        raw_cat = fs.cat(idx_paths, on_error="return")
        for p, val in raw_cat.items():
          if isinstance(val, FileNotFoundError):
            return None
          if isinstance(val, Exception):
            raise val
          idx_contents[p] = (
              val.encode("utf-8") if isinstance(val, str) else bytes(val)
          )
      else:
        for p in idx_paths:
          if not fs.exists(p):
            return None
          with fs.open(p, "rb") as f:
            idx_contents[p] = f.read()

      grib_paths: List[str] = []
      starts: List[int] = []
      ends: List[int] = []
      step_param_order: List[Tuple[int, str]] = []

      for step in lead_steps:
        idx_p = f"{prefix}-{step}h-oper-fc.index"
        grib_p = f"{prefix}-{step}h-oper-fc.grib2"
        if idx_p not in idx_contents:
          return None
        offsets: Dict[str, Tuple[int, int]] = {}
        for line in idx_contents[idx_p].decode("utf-8").splitlines():
          if not line.strip():
            continue
          msg = json.loads(line)
          param = msg.get("param")
          if msg.get("levtype") == "sfc" and param in target_params:
            offsets[param] = (int(msg["_offset"]), int(msg["_length"]))

        missing = [p for p in target_params if p not in offsets]
        if missing:
          raise RuntimeError(
              f"Missing surface parameters {missing} in {idx_p}"
          )
        for param in target_params:
          off, length = offsets[param]
          grib_paths.append(grib_p)
          starts.append(off)
          ends.append(off + length)
          step_param_order.append((step, param))

      if hasattr(fs, "cat_ranges"):
        raw_blobs = fs.cat_ranges(grib_paths, starts, ends)
      else:
        raw_blobs = []
        for g_p, s_off, e_off in zip(grib_paths, starts, ends):
          with fs.open(g_p, "rb") as f:
            f.seek(s_off)
            raw_blobs.append(f.read(e_off - s_off))

      raw_steps: Dict[str, List[np.ndarray]] = {p: [] for p in target_params}
      for (step, param), blob in zip(step_param_order, raw_blobs):
        arr = decode_grib2_message(
            blob,
            OPEN_DATA_GRID_SHAPE,
            param=param,
            context=f"{date_str} +{step}h",
        )
        # Flip latitude from [+90..-90] to [-90..+90] to match HRESExtractor.lats
        arr = arr[::-1, :]
        if lat_idx is not None and lon_idx is not None:
          arr = arr[lat_idx, :][:, lon_idx]
        raw_steps[param].append(arr)

      t2m = np.stack(raw_steps["2t"], axis=0) - np.float32(273.15)
      sp = np.stack(raw_steps["sp"], axis=0) * np.float32(1e-3)
      tp = deaccumulate(
          np.stack(raw_steps["tp"], axis=0), clip_negative=True
      ) * np.float32(1000.0)
      ssr = deaccumulate(
          np.stack(raw_steps["ssr"], axis=0), clip_negative=False
      ) / np.float32(86400.0)
      str_rad = deaccumulate(
          np.stack(raw_steps["str"], axis=0), clip_negative=False
      ) / np.float32(86400.0)

      return {
          "hres_temperature_2m": t2m.astype(np.float32),
          "hres_surface_pressure": sp.astype(np.float32),
          "hres_total_precipitation": tp.astype(np.float32),
          "hres_surface_net_solar_radiation": ssr.astype(np.float32),
          "hres_surface_net_thermal_radiation": str_rad.astype(np.float32),
      }
    except FileNotFoundError:
      return None
    except ImportError:
      raise
    except Exception as err:  # noqa: BLE001
      last_err = err
      if attempt < max_retries - 1:
        time.sleep(1.5 * (2**attempt))
        continue
      raise RuntimeError(
          f"Failed to fetch ECMWF Open Data HRES for {date_str} after "
          f"{max_retries} attempts: {last_err}"
      ) from last_err
  return None


def open_wb2_hres_dataset(zarr_url: Union[str, xr.Dataset]) -> xr.Dataset:
  """Opens WeatherBench 2 HRES Zarr store, pruning unneeded 3D atmospheric levels."""
  if isinstance(zarr_url, xr.Dataset):
    ds = zarr_url
  elif gcsfs is not None:
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(zarr_url)
    ds = xr.open_zarr(mapper, decode_timedelta=False)
  else:
    ds = xr.open_zarr(zarr_url, decode_timedelta=False)

  target_surface_vars = [
      "2m_temperature",
      "surface_pressure",
      "total_precipitation_24hr",
      "total_precipitation",
      "total_precipitation_6hr",
  ]
  avail = [v for v in target_surface_vars if v in ds.data_vars]
  if avail:
    ds = ds[avail]
  return ds


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
      fs: Optional[Any] = None,
  ):
    super().__init__(Product.HRES, data_dir)
    self._fs = fs
    source_lower = source.lower().strip()
    if source_lower in ("archive", "gridded_archive", "zarr_archive"):
      self.source = "archive"
      if data_dir is None or not str(data_dir).strip():
        raise ValueError(
            "HRESExtractor with source='archive' requires an explicit Zarr "
            "store URI or path via data_dir."
        )
      self.data_dir = str(data_dir)
    elif source_lower in ("auto", "default"):
      self.source = "archive"
      self.data_dir = str(data_dir) if data_dir is not None else ""
    elif source_lower in ("wb2", "public", "gcs", "upstream"):
      self.source = "wb2"
      self.data_dir = str(data_dir) if data_dir is not None else ""
    elif source_lower in (
        "open_data",
        "ecmwf_open_data",
        "realtime",
        "ecmwf",
    ):
      self.source = "open_data"
      self.data_dir = (
          str(data_dir).strip()
          if data_dir is not None and str(data_dir).strip()
          else ECMWF_OPEN_DATA_BUCKET
      )
    elif source_lower in ("local", "cns", "zarr"):
      self.source = "zarr"
      self.data_dir = str(data_dir) if data_dir is not None else ""
    else:
      self.source = source_lower
      self.data_dir = str(data_dir) if data_dir is not None else ""

    if self.source in ("wb2", "archive", "open_data"):
      # HRES 0.25 deg grid: 721 lats x 1440 lons
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

  def _get_open_data_fs(self) -> Any:
    """Returns the GCS filesystem client for ECMWF Open Data."""
    if self._fs is not None:
      return self._fs
    if gcsfs is None:
      raise ImportError("gcsfs is required to read ECMWF Open Data from GCS.")
    self._fs = gcsfs.GCSFileSystem(token="anon")
    return self._fs

  def extract_day(
      self,
      dt: Union[str, pd.Timestamp],
      basins_gdf: gpd.GeoDataFrame,
      matrix: Optional[ZonalWeightMatrix] = None,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
      lead_steps: Sequence[int] = OPEN_DATA_LEAD_STEPS,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 forecast initialization date across 10 lead days."""
    dt_ts = pd.to_datetime(dt)
    if self.source == "archive":
      if not self.data_dir:
        raise ValueError(
            "HRESExtractor requires an explicit data_dir URI for archive extraction."
        )
      from multimet.gridded_archive import extract_forecast_from_archive

      ds_day = extract_forecast_from_archive(
          Product.HRES,
          self.data_dir,
          basins_gdf,
          start_date=dt_ts,
          end_date=dt_ts,
          weights_matrix=matrix,
          use_bounding_box=True,
      )
      return {
          var: ds_day[var].values[:, 0, :].astype(np.float32)
          for var in ds_day.data_vars
      }
    if self.source == "wb2":
      if not self.data_dir:
        raise ValueError(
            "HRESExtractor requires an explicit data_dir URI; default gs:// "
            "bucket paths are not permitted."
        )
      return self._extract_day_wb2(dt_ts, basins_gdf)
    if self.source == "open_data":
      ds_day = self.extract_for_basins_open_data(
          basins_gdf,
          start_date=dt_ts,
          end_date=dt_ts,
          weights_matrix=matrix,
          use_bounding_box=True,
          spinup_only_before=(
              dt_ts + pd.Timedelta(days=1)
              if tuple(lead_steps) == (24,)
              else None
          ),
      )
      return {
          var: ds_day[var].values[:, 0, :].astype(np.float32)
          for var in ds_day.data_vars
      }
    if not self.data_dir:
      raise ValueError(
          "HRESExtractor requires an explicit data_dir path for local Zarr extraction."
      )
    basin_ids = list(basins_gdf.index)
    if weights_dict is None:
      weights_dict = {}
      for b_id in basin_ids:
        geom = basins_gdf.loc[b_id].geometry
        w = self.zonal_calc.compute_weights(b_id, geom)
        if w is not None:
          weights_dict[b_id] = w
    return extract_day_from_hres(
        self.data_dir, dt_ts, basin_ids, weights_dict, self.sort_lon_idx
    )

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
      spinup_only_before = kwargs.get("spinup_only_before")
      if start_date is None or end_date is None:
        raise ValueError(
            "HRESExtractor.extract_for_basins requires both start_date and "
            "end_date to be explicitly provided."
        )
      if self.source == "open_data":
        return self.extract_for_basins_open_data(
            basins_gdf,
            start_date=start_date,
            end_date=end_date,
            weights_matrix=weights_matrix,
            use_bounding_box=use_bounding_box,
            spinup_only_before=spinup_only_before,
        )
      if not self.data_dir:
        raise ValueError(
            "HRESExtractor requires an explicit data_dir Zarr URI or path; "
            "hardcoded default bucket paths are not permitted."
        )
      if self.source == "archive":
        from multimet.gridded_archive import extract_forecast_from_archive

        return extract_forecast_from_archive(
            Product.HRES,
            self.data_dir,
            basins_gdf,
            start_date=start_date,
            end_date=end_date,
            weights_matrix=weights_matrix,
            use_bounding_box=use_bounding_box,
        )
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

  def extract_for_basins_open_data(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      spinup_only_before: Optional[Union[str, pd.Timestamp]] = None,
      max_workers: int = 16,
  ) -> xr.Dataset:
    """Extracts 10-day (or 1-day spin-up) HRES forecasts from ECMWF Open Data on GCS.

    Args:
      basins_gdf: Catchment geometries indexed by basin ID.
      start_date: First forecast initialization date (inclusive).
      end_date: Last forecast initialization date (inclusive).
      weights_matrix: Optional precomputed :class:`ZonalWeightMatrix`.
      use_bounding_box: Whether to crop decoded GRIB2 rasters to the catchment
        bounding box before zonal reduction.
      spinup_only_before: Optional cutoff date. For initialization dates
        strictly before ``spinup_only_before``, only ``step=24h`` (``lead_time=1D``)
        is downloaded and decoded (leaving ``lead_time=2D..10D`` as ``NaN``),
        cutting Cold-Start 365-day spin-up download volume by 10x. For dates
        ``>= spinup_only_before`` (or when ``spinup_only_before is None``), all
        10 lead days (``24h..240h``) are extracted.
      max_workers: Maximum number of concurrent GCS day-fetch workers.
    """
    if start_date is None or end_date is None:
      raise ValueError(
          "HRESExtractor.extract_for_basins_open_data requires both start_date "
          "and end_date to be explicitly provided."
      )
    start_dt = pd.to_datetime(start_date).floor("D")
    end_dt = pd.to_datetime(end_date).floor("D")
    if end_dt < start_dt:
      raise ValueError(
          f"end_date ({end_dt}) must be >= start_date ({start_dt})."
      )

    spinup_cutoff = (
        pd.to_datetime(spinup_only_before).floor("D")
        if spinup_only_before is not None
        else None
    )

    basin_ids = [str(b) for b in basins_gdf.index]
    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    lead_steps_total = FORECAST_LEAD_DAYS[Product.HRES]  # 10 days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps_total + 1), unit="D")
    expected_bands = PRODUCT_BANDS[Product.HRES]

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
            basins_gdf, sub_lats, sub_lons, cell_res_lat=0.25, cell_res_lon=0.25
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
            basins_gdf,
            self.lats,
            self.lons,
            cell_res_lat=0.25,
            cell_res_lon=0.25,
        )

    shape = (len(basin_ids), len(date_idx), len(lead_time_idx))
    data_dict = {
        band: np.full(shape, np.nan, dtype=np.float32)
        for band in expected_bands
    }
    missing_fraction = np.ones(shape, dtype=np.float32)

    fs = self._get_open_data_fs()
    bucket = self.data_dir or ECMWF_OPEN_DATA_BUCKET

    def _process_one_day(
        d_pos: int, dt: pd.Timestamp
    ) -> Tuple[
        int,
        pd.Timestamp,
        int,
        Optional[Dict[str, np.ndarray]],
        Optional[np.ndarray],
    ]:
      if spinup_cutoff is not None and dt < spinup_cutoff:
        day_lead_steps: Sequence[int] = (24,)
      else:
        day_lead_steps = OPEN_DATA_LEAD_STEPS

      day_grids = _fetch_open_data_day_grids(
          fs,
          bucket,
          dt,
          lead_steps=day_lead_steps,
          lat_idx=lat_idx,
          lon_idx=lon_idx,
      )
      if day_grids is None:
        return (d_pos, dt, len(day_lead_steps), None, None)

      reduced_by_band: Dict[str, np.ndarray] = {}
      day_miss: Optional[np.ndarray] = None
      for band in expected_bands:
        if band not in day_grids:
          continue
        reduced_vals, reduced_miss = matrix.reduce_3d_with_coverage(
            day_grids[band]
        )
        reduced_by_band[band] = reduced_vals
        if day_miss is None:
          day_miss = reduced_miss
        else:
          day_miss = np.maximum(day_miss, reduced_miss)
      return (d_pos, dt, len(day_lead_steps), reduced_by_band, day_miss)

    n_workers = max(1, min(int(max_workers), len(date_idx)))
    with tqdm.tqdm(
        total=len(date_idx),
        desc=(
            f"HRES OpenData [{start_dt.strftime('%Y-%m-%d')} to"
            f" {end_dt.strftime('%Y-%m-%d')}]"
        ),
        unit="init_date",
        leave=True,
    ) as pbar:
      if n_workers == 1:
        for d_pos, dt in enumerate(date_idx):
          _, _, n_leads_fetched, reduced_by_band, day_miss = _process_one_day(
              d_pos, dt
          )
          if reduced_by_band is None:
            logger.warning(
                "ECMWF Open Data HRES not published for %s; leaving slice as NaN.",
                dt.strftime("%Y-%m-%d"),
            )
          else:
            for band, reduced_vals in reduced_by_band.items():
              data_dict[band][:, d_pos, :n_leads_fetched] = reduced_vals
            if day_miss is not None:
              missing_fraction[:, d_pos, :n_leads_fetched] = day_miss
          pbar.update(1)
      else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers
        ) as pool:
          futures = [
              pool.submit(_process_one_day, d_pos, dt)
              for d_pos, dt in enumerate(date_idx)
          ]
          for fut in concurrent.futures.as_completed(futures):
            d_pos, dt, n_leads_fetched, reduced_by_band, day_miss = fut.result()
            if reduced_by_band is None:
              logger.warning(
                  "ECMWF Open Data HRES not published for %s; leaving slice as NaN.",
                  dt.strftime("%Y-%m-%d"),
              )
            else:
              for band, reduced_vals in reduced_by_band.items():
                data_dict[band][:, d_pos, :n_leads_fetched] = reduced_vals
              if day_miss is not None:
                missing_fraction[:, d_pos, :n_leads_fetched] = day_miss
            pbar.update(1)

    data_vars = {
        band: (["basin", "date", "lead_time"], data_dict[band])
        for band in expected_bands
    }
    data_vars["hres_missing_fraction"] = (
        ["basin", "date", "lead_time"],
        missing_fraction,
    )

    attrs = dict(PRODUCT_METADATA_ATTRS.get(Product.HRES, {}))
    attrs["upstream_source"] = f"gs://{bucket.removeprefix('gs://').strip('/')}"

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
            "lead_time": lead_time_idx.values,
        },
        attrs=attrs,
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

    time_target = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')}T00:00:00")
    if self._cached_ds_raw is None:
      if isinstance(self.data_dir, xr.Dataset):
        self._cached_ds_raw = self.data_dir
      else:
        self._cached_ds_raw = open_wb2_hres_dataset(self.data_dir)

    ds_raw = self._cached_ds_raw
    if time_target not in pd.to_datetime(ds_raw.time.values):
      return res_dict

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

    tp_var = (
        "total_precipitation_24hr"
        if "total_precipitation_24hr" in sub
        else "total_precipitation"
    )
    has_tp24 = (tp_var == "total_precipitation_24hr")

    # 1. 2m Temperature
    with dask.config.set(scheduler="threads"):
      t2m_arr = sub["2m_temperature"].sel(time=time_target).compute().values - 273.15
    t2m_reduced = matrix.reduce_3d(t2m_arr)
    del t2m_arr

    # 2. Surface Pressure
    with dask.config.set(scheduler="threads"):
      sp_arr = sub["surface_pressure"].sel(time=time_target).compute().values * 0.001
    sp_reduced = matrix.reduce_3d(sp_arr)
    del sp_arr

    # 3. Total Precipitation
    with dask.config.set(scheduler="threads"):
      tp_arr = sub[tp_var].sel(time=time_target).compute().values * 1000.0
    tp_reduced = matrix.reduce_3d(tp_arr)
    del tp_arr

    for lt_day in range(1, 11):
      lt_pos = lt_day - 1
      step_slice = slice((lt_day - 1) * 4 + 1, lt_day * 4 + 1)
      if lt_day * 4 >= t2m_reduced.shape[1]:
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

    del t2m_reduced, sp_reduced, tp_reduced
    gc.collect()

    return res_dict

  def extract_for_basins_wb2(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Extracts 10-day HRES forecasts from WeatherBench 2 on GCS."""
    from multimet.gridded_archive import _warn_missing_variables_once

    if start_date is None or end_date is None:
      raise ValueError(
          "HRESExtractor.extract_for_basins_wb2 requires both start_date and "
          "end_date to be explicitly provided."
      )
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
      raise ValueError(
          f"end_date ({end_dt}) must be >= start_date ({start_dt})."
      )

    basin_ids = list(basins_gdf.index)
    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    lead_steps = FORECAST_LEAD_DAYS[Product.HRES]  # 10 days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps + 1), unit="D")
    expected_bands = PRODUCT_BANDS[Product.HRES]

    shape = (len(basin_ids), len(date_idx), len(lead_time_idx))
    data_dict = {
        band: np.full(shape, np.nan, dtype=np.float32)
        for band in expected_bands
    }
    missing_fraction = np.ones(shape, dtype=np.float32)

    ds_raw = open_wb2_hres_dataset(self.data_dir)
    _warn_missing_variables_once(
        str(self.data_dir),
        Product.HRES,
        [
            "hres_surface_net_solar_radiation",
            "hres_surface_net_thermal_radiation",
        ],
    )

    # Scope to valid requested times first to keep the Dask graph minimal
    requested_times = [
        pd.Timestamp(f"{dt.strftime('%Y-%m-%d')}T00:00:00")
        for dt in date_idx
    ]
    ds_raw_times = pd.to_datetime(ds_raw.time.values)
    valid_times = [t for t in requested_times if t in ds_raw_times]

    if not valid_times:
      ds_raw.close()
      min_avail = (
          ds_raw_times.min().strftime("%Y-%m-%d")
          if len(ds_raw_times) > 0
          else "empty"
      )
      max_avail = (
          ds_raw_times.max().strftime("%Y-%m-%d")
          if len(ds_raw_times) > 0
          else "empty"
      )
      raise ValueError(
          f"Requested date range [{start_dt.strftime('%Y-%m-%d')}, "
          f"{end_dt.strftime('%Y-%m-%d')}] has no overlap with HRES store at "
          f"{self.data_dir!r} (available range: [{min_avail}, {max_avail}])."
      )

    ds_raw = ds_raw.sel(time=valid_times)

    if use_bounding_box:
      bounds = basins_gdf.total_bounds
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

    tp_var = (
        "total_precipitation_24hr"
        if "total_precipitation_24hr" in sub
        else "total_precipitation"
    )
    has_tp24 = tp_var == "total_precipitation_24hr"

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
      if time_target not in valid_times:
        continue

      # 1. 2m Temperature
      with dask.config.set(scheduler="threads"):
        t2m_arr = (
            sub["2m_temperature"].sel(time=time_target).compute().values
            - 273.15
        )
      t2m_reduced, t2m_miss = matrix.reduce_3d_with_coverage(t2m_arr)
      del t2m_arr

      # 2. Surface Pressure
      with dask.config.set(scheduler="threads"):
        sp_arr = (
            sub["surface_pressure"].sel(time=time_target).compute().values
            * 0.001
        )
      sp_reduced = matrix.reduce_3d(sp_arr)
      del sp_arr

      # 3. Total Precipitation
      with dask.config.set(scheduler="threads"):
        tp_arr = sub[tp_var].sel(time=time_target).compute().values * 1000.0
      tp_reduced = matrix.reduce_3d(tp_arr)
      del tp_arr

      for lt_day in range(1, 11):
        lt_pos = lt_day - 1
        step_slice = slice((lt_day - 1) * 4 + 1, lt_day * 4 + 1)
        if lt_day * 4 >= t2m_reduced.shape[1]:
          continue

        mean_t2m = np.nanmean(t2m_reduced[:, step_slice], axis=1)
        mean_sp = np.nanmean(sp_reduced[:, step_slice], axis=1)
        mean_miss = np.nanmean(t2m_miss[:, step_slice], axis=1)
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
        missing_fraction[:, d_pos, lt_pos] = mean_miss

      del t2m_reduced, sp_reduced, tp_reduced
      gc.collect()

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
    data_vars["hres_missing_fraction"] = (
        ["basin", "date", "lead_time"],
        missing_fraction,
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
    if start_date is None or end_date is None:
      raise ValueError(
          "HRESExtractor.extract_for_basins_zarr requires both start_date and "
          "end_date to be explicitly provided."
      )
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
      raise ValueError(
          f"end_date ({end_dt}) must be >= start_date ({start_dt})."
      )
    basin_ids = list(basins_gdf.index)
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
    data_vars["hres_missing_fraction"] = (
        ["basin", "date", "lead_time"],
        np.isnan(data_dict["hres_temperature_2m"]).astype(np.float32),
    )

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
