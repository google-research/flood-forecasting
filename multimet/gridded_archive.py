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

"""Unified reader and catchment extractor for Open-MultiMet gridded Zarr archives.

Supports extracting basin-averaged meteorological time series from user-supplied
gridded Zarr archives (local paths or ``gs://`` URIs) for:

- **CPC** (0.5 deg daily nowcast: ``cpc_precipitation``, ``cpc_num_stations``)
- **IMERG** (0.1 deg daily nowcast: ``imerg_precipitation``)
- **ERA5_LAND** (0.1 deg daily nowcast: 17 ``era5land_*`` surface bands)
- **HRES** (0.25 deg 10-day daily forecast: 5 ``hres_*`` surface bands)

Design invariants:
1. **No hardcoded paths or fallback locations:** callers must supply an explicit
   ``store_uri``. Missing or empty URIs raise :class:`GriddedArchiveError`.
2. **Missing data in = missing data out:** absent variables or dates within a
   valid archive window remain ``NaN`` and are never substituted from another
   dataset. Missing variables emit at most one warning per ``(store_uri, product)``
   session so progress bars are never spammed.
3. **Coverage auditing:** every extracted dataset includes the product's
   companion ``<prefix>_missing_fraction`` variable recording the area-weighted
   fraction ``[0.0, 1.0]`` of missing (``NaN``) pixels per basin and timestep.
4. **Transient GCS error resilience:** opening and reading from ``gs://`` stores
   retries the *same* URI with exponential backoff before raising.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import dask
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from multimet.config import (
    FORECAST_LEAD_DAYS,
    MISSING_FRACTION_VAR,
    PRODUCT_BANDS,
    PRODUCT_METADATA_ATTRS,
    PRODUCT_TYPES,
    Product,
    ProductType,
)
from multimet.gcp import configure_gcp_project
from multimet.spatial import slice_dataset_by_bounds
from multimet.zonal import ZonalWeightMatrix

logger = logging.getLogger(__name__)

# Deduplication set so missing-variable warnings are emitted at most once per
# (store_uri, product, missing_vars) tuple across an entire process.
_WARNED_MISSING_VARS: set[Tuple[str, str, Tuple[str, ...]]] = set()


class GriddedArchiveError(ValueError):
  """Raised when a gridded archive cannot be opened or violates schema requirements."""


@dataclasses.dataclass(frozen=True)
class ArchiveBandSpec:
  """Mapping and unit-conversion rule from an archive variable to a MultiMet band.

  Attributes:
    target_band: Canonical MultiMet output band name (e.g. ``hres_temperature_2m``).
    source_candidates: Ordered candidate variable names in the gridded archive
      that represent this exact physical band (e.g. ``("temperature_2m",
      "hres_temperature_2m")``).
    scale: Multiplicative scale factor applied when reading native units.
    offset: Additive offset applied after scaling (``val * scale + offset``).
    clip_min: Optional minimum physical floor (e.g. ``0.0`` for precipitation).
    already_converted_names: Source variable names that are already in Caravan
      units (skipping ``scale`` / ``offset`` if matched).
  """

  target_band: str
  source_candidates: Tuple[str, ...]
  scale: float = 1.0
  offset: float = 0.0
  clip_min: Optional[float] = None
  already_converted_names: Tuple[str, ...] = ()

  def transform(self, values: np.ndarray, matched_source_name: str) -> np.ndarray:
    """Applies unit conversion while preserving NaNs."""
    out = np.asarray(values, dtype=np.float32)
    if matched_source_name not in self.already_converted_names:
      if self.scale != 1.0 or self.offset != 0.0:
        out = out * np.float32(self.scale) + np.float32(self.offset)
    if self.clip_min is not None:
      out = np.where(
          np.isfinite(out), np.maximum(np.float32(self.clip_min), out), out
      )
    return out.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class GriddedArchiveSpec:
  """Schema specification for a product's gridded Zarr archive."""

  product: Product
  product_type: ProductType
  cell_res_lat: float
  cell_res_lon: float
  bands: Tuple[ArchiveBandSpec, ...]
  lead_days: Optional[int] = None


ARCHIVE_SPECS: Mapping[Product, GriddedArchiveSpec] = {
    Product.CPC: GriddedArchiveSpec(
        product=Product.CPC,
        product_type=ProductType.NOWCAST,
        cell_res_lat=0.5,
        cell_res_lon=0.5,
        bands=(
            ArchiveBandSpec(
                target_band="cpc_precipitation",
                source_candidates=("cpc_precipitation", "precip"),
                scale=1.0,
                offset=0.0,
                clip_min=0.0,
                already_converted_names=("cpc_precipitation", "precip"),
            ),
            ArchiveBandSpec(
                target_band="cpc_num_stations",
                source_candidates=("cpc_num_stations", "num_stations", "gcount"),
                scale=1.0,
                offset=0.0,
                clip_min=0.0,
                already_converted_names=(
                    "cpc_num_stations",
                    "num_stations",
                    "gcount",
                ),
            ),
        ),
    ),
    Product.IMERG: GriddedArchiveSpec(
        product=Product.IMERG,
        product_type=ProductType.NOWCAST,
        cell_res_lat=0.1,
        cell_res_lon=0.1,
        bands=(
            ArchiveBandSpec(
                target_band="imerg_precipitation",
                source_candidates=("imerg_precipitation", "precipitation"),
                scale=1.0,
                offset=0.0,
                clip_min=0.0,
                already_converted_names=("imerg_precipitation", "precipitation"),
            ),
        ),
    ),
    Product.ERA5_LAND: GriddedArchiveSpec(
        product=Product.ERA5_LAND,
        product_type=ProductType.NOWCAST,
        cell_res_lat=0.1,
        cell_res_lon=0.1,
        bands=tuple(
            ArchiveBandSpec(
                target_band=band,
                source_candidates=(band,),
                scale=1.0,
                offset=0.0,
                clip_min=0.0
                if band
                in (
                    "era5land_total_precipitation",
                    "era5land_snow_depth_water_equivalent",
                    "era5land_potential_evaporation_FAO_PENMAN_MONTEITH",
                )
                else None,
                already_converted_names=(band,),
            )
            for band in PRODUCT_BANDS[Product.ERA5_LAND]
        ),
    ),
    Product.HRES: GriddedArchiveSpec(
        product=Product.HRES,
        product_type=ProductType.FORECAST,
        cell_res_lat=0.25,
        cell_res_lon=0.25,
        lead_days=FORECAST_LEAD_DAYS[Product.HRES],
        bands=(
            ArchiveBandSpec(
                target_band="hres_surface_net_solar_radiation",
                source_candidates=(
                    "surface_net_solar_radiation",
                    "hres_surface_net_solar_radiation",
                ),
                scale=1.0 / 86400.0,
                offset=0.0,
                already_converted_names=("hres_surface_net_solar_radiation",),
            ),
            ArchiveBandSpec(
                target_band="hres_surface_net_thermal_radiation",
                source_candidates=(
                    "surface_net_thermal_radiation",
                    "hres_surface_net_thermal_radiation",
                ),
                scale=1.0 / 86400.0,
                offset=0.0,
                already_converted_names=("hres_surface_net_thermal_radiation",),
            ),
            ArchiveBandSpec(
                target_band="hres_surface_pressure",
                source_candidates=(
                    "surface_pressure",
                    "hres_surface_pressure",
                ),
                scale=1e-3,
                offset=0.0,
                already_converted_names=("hres_surface_pressure",),
            ),
            ArchiveBandSpec(
                target_band="hres_temperature_2m",
                source_candidates=(
                    "temperature_2m",
                    "hres_temperature_2m",
                ),
                scale=1.0,
                offset=-273.15,
                already_converted_names=("hres_temperature_2m",),
            ),
            ArchiveBandSpec(
                target_band="hres_total_precipitation",
                source_candidates=(
                    "total_precipitation",
                    "hres_total_precipitation",
                ),
                scale=1000.0,
                offset=0.0,
                clip_min=0.0,
                already_converted_names=("hres_total_precipitation",),
            ),
        ),
    ),
}


def get_archive_spec(product: Union[Product, str]) -> GriddedArchiveSpec:
  """Returns the :class:`GriddedArchiveSpec` for a MultiMet product."""
  if isinstance(product, str):
    try:
      product = Product[product.upper()]
    except KeyError as err:
      raise GriddedArchiveError(
          f"Unknown MultiMet product {product!r}. Supported archive products: "
          f"{[p.name for p in ARCHIVE_SPECS]}"
      ) from err
  if product not in ARCHIVE_SPECS:
    raise GriddedArchiveError(
        f"Product {product.value} does not have a gridded archive specification. "
        f"Supported archive products: {[p.name for p in ARCHIVE_SPECS]}"
    )
  return ARCHIVE_SPECS[product]


def _validate_store_uri(store_uri: Optional[Union[str, os.PathLike]]) -> str:
  """Validates that a store URI was explicitly provided by the caller."""
  if store_uri is None:
    raise GriddedArchiveError(
        "A gridded archive URI or path must be explicitly provided by the user "
        "(via data_dir, archive_stores, or --archive-store). Hardcoded or "
        "default archive paths are not permitted."
    )
  uri_str = str(store_uri).strip()
  if not uri_str:
    raise GriddedArchiveError(
        "Gridded archive store_uri cannot be an empty string."
    )
  return uri_str


def _retry_with_backoff(
    fn: Callable[[], object],
    *,
    description: str,
    max_retries: int = 6,
    base_delay: float = 1.0,
):
  """Executes ``fn()`` retrying transient GCS/network errors on the same URI."""
  last_err: Optional[Exception] = None
  for attempt in range(max_retries):
    try:
      return fn()
    except FileNotFoundError:
      raise
    except (OSError, RuntimeError, TimeoutError) as err:
      last_err = err
      if attempt == max_retries - 1:
        break
      wait_s = base_delay * (2**attempt) + random.uniform(0.1, 0.75)
      logger.debug(
          "Transient error during %s (attempt %d/%d): %s. Retrying in %.2fs...",
          description,
          attempt + 1,
          max_retries,
          err,
          wait_s,
      )
      time.sleep(wait_s)
  raise GriddedArchiveError(
      f"Failed {description} after {max_retries} attempts: {last_err}"
  ) from last_err


def _warn_missing_variables_once(
    store_uri: str, product: Product, missing_bands: Sequence[str]
) -> None:
  """Logs a single warning per (store_uri, product, missing_bands) tuple."""
  if not missing_bands or os.environ.get("MULTIMET_DASK_WORKER") == "1":
    return
  key = (str(store_uri).rstrip("/"), product.value, tuple(sorted(missing_bands)))
  if key in _WARNED_MISSING_VARS:
    return
  _WARNED_MISSING_VARS.add(key)
  logger.warning(
      "Gridded archive %s for product %s is missing expected variable(s) %s. "
      "Those band(s) will be populated with NaN (missing data in -> missing data out).",
      store_uri,
      product.value,
      list(missing_bands),
  )


def open_gridded_archive(
    store_uri: Union[str, os.PathLike],
    *,
    consolidated: Optional[bool] = None,
    max_retries: int = 6,
) -> xr.Dataset:
  """Opens a Zarr v2 or Zarr v3 gridded archive from local disk or GCS.

  Retries the same ``store_uri`` with exponential backoff on transient GCS
  errors (including spurious HTTP 403/5xx responses) and never falls back to
  alternative paths.

  Args:
    store_uri: Explicit path or ``gs://`` URI to the Zarr store.
    consolidated: Optional override for consolidated metadata. If ``None``,
      determined from the store's descriptor files (``zarr.json`` -> Zarr v3
      non-consolidated; ``.zmetadata`` -> consolidated; ``.zgroup`` -> Zarr v2
      non-consolidated).
    max_retries: Maximum retry attempts for transient storage errors.

  Returns:
    Lazily opened :class:`xarray.Dataset`.

  Raises:
    GriddedArchiveError: If ``store_uri`` is missing, does not exist, or cannot
      be opened after ``max_retries``.
  """
  uri = _validate_store_uri(store_uri)
  if uri.startswith(("gs://", "gcs://")):
    configure_gcp_project()

  def _open() -> xr.Dataset:
    fs, fs_path = fsspec.core.url_to_fs(uri)
    use_consolidated = consolidated
    if use_consolidated is None:
      has_v3 = fs.exists(f"{fs_path}/zarr.json")
      has_zmeta = fs.exists(f"{fs_path}/.zmetadata")
      has_zgroup = fs.exists(f"{fs_path}/.zgroup")
      if not (has_v3 or has_zmeta or has_zgroup):
        raise FileNotFoundError(
            f"No Zarr store descriptor (zarr.json, .zmetadata, or .zgroup) "
            f"found at {uri!r}."
        )
      use_consolidated = bool(has_zmeta and not has_v3)

    return xr.open_zarr(
        uri,
        consolidated=use_consolidated,
        decode_timedelta=False,
    )

  try:
    ds = _retry_with_backoff(
        _open,
        description=f"opening gridded archive at {uri}",
        max_retries=max_retries,
    )
  except FileNotFoundError as err:
    raise GriddedArchiveError(
        f"Gridded archive does not exist at {uri!r}: {err}"
    ) from err
  return ds


def _parse_required_date_range(
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
  """Validates that start_date and end_date were explicitly provided."""
  if start_date is None or end_date is None:
    raise ValueError(
        "Both start_date and end_date must be explicitly provided; "
        "default placeholder dates are not permitted."
    )
  start_dt = pd.to_datetime(start_date).floor("D")
  end_dt = pd.to_datetime(end_date).floor("D")
  if end_dt < start_dt:
    raise ValueError(
        f"end_date ({end_dt.strftime('%Y-%m-%d')}) must be >= "
        f"start_date ({start_dt.strftime('%Y-%m-%d')})."
    )
  date_idx = pd.date_range(start_dt, end_dt, freq="D")
  return start_dt, end_dt, date_idx


def _prepare_spatial_subset_and_weights(
    ds_sub: xr.Dataset,
    basins_gdf: gpd.GeoDataFrame,
    spec: GriddedArchiveSpec,
    weights_matrix: Optional[ZonalWeightMatrix],
    use_bounding_box: bool,
) -> Tuple[xr.Dataset, ZonalWeightMatrix]:
  """Slices ``ds_sub`` spatially to ``[-180, 180)`` and aligns the weight matrix."""
  full_lats = np.asarray(ds_sub["latitude"].values, dtype=np.float64)
  full_lons = np.asarray(ds_sub["longitude"].values, dtype=np.float64)
  res_lat = (
      abs(float(full_lats[1] - full_lats[0]))
      if len(full_lats) > 1
      else spec.cell_res_lat
  )
  res_lon = (
      abs(float(full_lons[1] - full_lons[0]))
      if len(full_lons) > 1
      else spec.cell_res_lon
  )

  if use_bounding_box:
    sub = slice_dataset_by_bounds(
        ds_sub,
        bounds=basins_gdf,
        buffer_degrees=max(0.5, res_lat * 2.0, res_lon * 2.0),
        lat_dim="latitude",
        lon_dim="longitude",
        target_lon_range="minus180_180",
    )
  else:
    sub = ds_sub

  # Always ensure longitude is in [-180, 180) and sorted ascending.
  lon_vals = np.asarray(sub["longitude"].values, dtype=np.float64)
  if np.any(lon_vals >= 180.0):
    converted_lons = np.where(
        lon_vals >= 180.0, lon_vals - 360.0, lon_vals
    ).astype(sub["longitude"].dtype)
    sub = sub.assign_coords(longitude=converted_lons).sortby("longitude")

  sub_lats = np.asarray(sub["latitude"].values, dtype=np.float64)
  sub_lons = np.asarray(sub["longitude"].values, dtype=np.float64)

  if weights_matrix is not None:
    wm_lat_min = float(np.min(weights_matrix.lats)) - 1e-3
    wm_lat_max = float(np.max(weights_matrix.lats)) + 1e-3
    wm_lon_min = float(np.min(weights_matrix.lons)) - 1e-3
    wm_lon_max = float(np.max(weights_matrix.lons)) + 1e-3
    lat_mask = (sub_lats >= wm_lat_min) & (sub_lats <= wm_lat_max)
    lon_mask = (sub_lons >= wm_lon_min) & (sub_lons <= wm_lon_max)
    if not np.all(lat_mask) or not np.all(lon_mask):
      sub = sub.isel(
          latitude=np.where(lat_mask)[0],
          longitude=np.where(lon_mask)[0],
      )
      sub_lats = np.asarray(sub["latitude"].values, dtype=np.float64)
      sub_lons = np.asarray(sub["longitude"].values, dtype=np.float64)

    if (
        weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
        and np.allclose(weights_matrix.lats, sub_lats, atol=1e-3)
        and np.allclose(weights_matrix.lons, sub_lons, atol=1e-3)
    ):
      matrix = weights_matrix
    else:
      matrix = weights_matrix.crop_to_coords(sub_lats, sub_lons, atol=1e-3)
  else:
    matrix = ZonalWeightMatrix.from_geodataframe(
        basins_gdf,
        sub_lats,
        sub_lons,
        cell_res_lat=res_lat,
        cell_res_lon=res_lon,
    )
  return sub, matrix


def _resolve_band_sources(
    ds: xr.Dataset,
    spec: GriddedArchiveSpec,
    store_uri: str,
) -> Tuple[Dict[str, Tuple[ArchiveBandSpec, str]], Tuple[str, ...]]:
  """Resolves available archive variables and warns once for any absent bands."""
  resolved: Dict[str, Tuple[ArchiveBandSpec, str]] = {}
  missing_bands = []
  for band_spec in spec.bands:
    matched = next(
        (cand for cand in band_spec.source_candidates if cand in ds.data_vars),
        None,
    )
    if matched is not None:
      resolved[band_spec.target_band] = (band_spec, matched)
    else:
      missing_bands.append(band_spec.target_band)

  if not resolved:
    raise GriddedArchiveError(
        f"Gridded archive at {store_uri!r} contains none of the expected "
        f"variables for {spec.product.value}. Expected one of "
        f"{[b.source_candidates for b in spec.bands]}; found "
        f"{list(ds.data_vars)}."
    )

  if missing_bands:
    _warn_missing_variables_once(store_uri, spec.product, missing_bands)

  return resolved, tuple(missing_bands)


def extract_nowcast_from_archive(
    product: Union[Product, str],
    store_uri: Union[str, os.PathLike],
    basins_gdf: gpd.GeoDataFrame,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    *,
    weights_matrix: Optional[ZonalWeightMatrix] = None,
    use_bounding_box: bool = True,
    max_retries: int = 6,
) -> xr.Dataset:
  """Extracts basin-averaged daily nowcast time series from a gridded Zarr archive."""
  spec = get_archive_spec(product)
  if spec.product_type != ProductType.NOWCAST:
    raise GriddedArchiveError(
        f"Product {spec.product.value} is a {spec.product_type.value}, not a nowcast."
    )
  uri = _validate_store_uri(store_uri)
  start_dt, end_dt, date_idx = _parse_required_date_range(start_date, end_date)
  basin_ids = [str(b) for b in basins_gdf.index]
  n_basins = len(basin_ids)
  n_dates = len(date_idx)

  ds_raw = open_gridded_archive(uri, max_retries=max_retries)
  if "time" not in ds_raw.coords and "time" not in ds_raw.dims:
    raise GriddedArchiveError(
        f"Gridded archive at {uri!r} is missing 'time' coordinate."
    )

  archive_times = pd.DatetimeIndex(pd.to_datetime(ds_raw["time"].values).floor("D"))
  if len(archive_times) == 0:
    raise GriddedArchiveError(f"Gridded archive at {uri!r} has an empty time axis.")

  if end_dt < archive_times.min() or start_dt > archive_times.max():
    raise GriddedArchiveError(
        f"Requested date range [{start_dt.strftime('%Y-%m-%d')}, "
        f"{end_dt.strftime('%Y-%m-%d')}] has no overlap with {spec.product.value} "
        f"archive at {uri!r} (available range: "
        f"[{archive_times.min().strftime('%Y-%m-%d')}, "
        f"{archive_times.max().strftime('%Y-%m-%d')}])."
    )

  resolved_bands, _ = _resolve_band_sources(ds_raw, spec, uri)

  time_slice = slice(
      start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
  )
  ds_time = ds_raw.sel(time=time_slice)
  if len(ds_time["time"]) == 0:
    raise GriddedArchiveError(
        f"No timestamps found in {spec.product.value} archive at {uri!r} "
        f"within [{start_dt.strftime('%Y-%m-%d')}, {end_dt.strftime('%Y-%m-%d')}]."
    )

  sub, matrix = _prepare_spatial_subset_and_weights(
      ds_time, basins_gdf, spec, weights_matrix, use_bounding_box
  )

  sub_dates = pd.DatetimeIndex(
      pd.to_datetime(sub["time"].values).floor("D")
  )
  valid_sub_indices = []
  valid_target_indices = []
  date_to_target = {d: idx for idx, d in enumerate(date_idx)}
  for s_idx, d in enumerate(sub_dates):
    if d in date_to_target:
      valid_sub_indices.append(s_idx)
      valid_target_indices.append(date_to_target[d])

  data_vars: Dict[str, Tuple[Sequence[str], np.ndarray]] = {}
  for band_spec in spec.bands:
    data_vars[band_spec.target_band] = (
        ["basin", "date"],
        np.full((n_basins, n_dates), np.nan, dtype=np.float32),
    )

  missing_fraction_out = np.ones((n_basins, n_dates), dtype=np.float32)

  for target_band, (band_spec, src_name) in resolved_bands.items():
    def _compute_var(v=src_name):
      with dask.config.set(scheduler="threads"):
        return np.asarray(sub[v].compute().values, dtype=np.float32)

    raw_vals = _retry_with_backoff(
        _compute_var,
        description=f"reading {spec.product.value}:{src_name} from {uri}",
        max_retries=max_retries,
    )
    converted_3d = band_spec.transform(raw_vals, src_name)
    del raw_vals
    reduced_vals, reduced_missing = matrix.reduce_3d_with_coverage(converted_3d)
    del converted_3d
    if valid_target_indices:
      data_vars[target_band][1][:, valid_target_indices] = reduced_vals[
          :, valid_sub_indices
      ]
      missing_fraction_out[:, valid_target_indices] = np.minimum(
          missing_fraction_out[:, valid_target_indices],
          reduced_missing[:, valid_sub_indices],
      )

  missing_var_name = MISSING_FRACTION_VAR[spec.product]
  data_vars[missing_var_name] = (["basin", "date"], missing_fraction_out)

  attrs = dict(PRODUCT_METADATA_ATTRS.get(spec.product, {}))
  attrs["archive_source_uri"] = uri

  return xr.Dataset(
      data_vars=data_vars,
      coords={"basin": basin_ids, "date": date_idx.values},
      attrs=attrs,
  )


def extract_forecast_from_archive(
    product: Union[Product, str],
    store_uri: Union[str, os.PathLike],
    basins_gdf: gpd.GeoDataFrame,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    *,
    weights_matrix: Optional[ZonalWeightMatrix] = None,
    use_bounding_box: bool = True,
    max_retries: int = 6,
) -> xr.Dataset:
  """Extracts basin-averaged daily forecast time series from a gridded Zarr archive."""
  spec = get_archive_spec(product)
  if spec.product_type != ProductType.FORECAST:
    raise GriddedArchiveError(
        f"Product {spec.product.value} is a {spec.product_type.value}, not a forecast."
    )
  uri = _validate_store_uri(store_uri)
  start_dt, end_dt, date_idx = _parse_required_date_range(start_date, end_date)
  basin_ids = [str(b) for b in basins_gdf.index]
  n_basins = len(basin_ids)
  n_dates = len(date_idx)
  lead_days = spec.lead_days or FORECAST_LEAD_DAYS[spec.product]
  leads_td = pd.to_timedelta(np.arange(1, lead_days + 1), unit="D")

  ds_raw = open_gridded_archive(uri, max_retries=max_retries)
  if "time" not in ds_raw.coords and "time" not in ds_raw.dims:
    raise GriddedArchiveError(
        f"Gridded archive at {uri!r} is missing 'time' coordinate."
    )
  if "lead_time" not in ds_raw.coords and "lead_time" not in ds_raw.dims:
    raise GriddedArchiveError(
        f"Forecast archive at {uri!r} is missing 'lead_time' coordinate."
    )

  archive_times = pd.DatetimeIndex(pd.to_datetime(ds_raw["time"].values).floor("D"))
  if len(archive_times) == 0:
    raise GriddedArchiveError(f"Gridded archive at {uri!r} has an empty time axis.")

  if end_dt < archive_times.min() or start_dt > archive_times.max():
    raise GriddedArchiveError(
        f"Requested date range [{start_dt.strftime('%Y-%m-%d')}, "
        f"{end_dt.strftime('%Y-%m-%d')}] has no overlap with {spec.product.value} "
        f"archive at {uri!r} (available range: "
        f"[{archive_times.min().strftime('%Y-%m-%d')}, "
        f"{archive_times.max().strftime('%Y-%m-%d')}])."
    )

  resolved_bands, _ = _resolve_band_sources(ds_raw, spec, uri)

  time_slice = slice(
      start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
  )
  ds_time = ds_raw.sel(time=time_slice)
  if len(ds_time["time"]) == 0:
    raise GriddedArchiveError(
        f"No timestamps found in {spec.product.value} archive at {uri!r} "
        f"within [{start_dt.strftime('%Y-%m-%d')}, {end_dt.strftime('%Y-%m-%d')}]."
    )

  sub, matrix = _prepare_spatial_subset_and_weights(
      ds_time, basins_gdf, spec, weights_matrix, use_bounding_box
  )

  sub_dates = pd.DatetimeIndex(
      pd.to_datetime(sub["time"].values).floor("D")
  )
  valid_sub_indices = []
  valid_target_indices = []
  date_to_target = {d: idx for idx, d in enumerate(date_idx)}
  for s_idx, d in enumerate(sub_dates):
    if d in date_to_target:
      valid_sub_indices.append(s_idx)
      valid_target_indices.append(date_to_target[d])

  raw_leads = sub["lead_time"].values
  if np.issubdtype(raw_leads.dtype, np.timedelta64):
    lead_days_arr = (pd.to_timedelta(raw_leads) / pd.Timedelta(days=1)).astype(int)
  else:
    lead_days_arr = np.asarray(raw_leads, dtype=int)

  valid_sub_leads = []
  valid_target_leads = []
  for l_idx, l_day in enumerate(lead_days_arr):
    if 1 <= int(l_day) <= lead_days:
      valid_sub_leads.append(l_idx)
      valid_target_leads.append(int(l_day) - 1)

  data_vars: Dict[str, Tuple[Sequence[str], np.ndarray]] = {}
  for band_spec in spec.bands:
    data_vars[band_spec.target_band] = (
        ["basin", "date", "lead_time"],
        np.full((n_basins, n_dates, lead_days), np.nan, dtype=np.float32),
    )

  missing_fraction_out = np.ones(
      (n_basins, n_dates, lead_days), dtype=np.float32
  )

  for target_band, (band_spec, src_name) in resolved_bands.items():
    def _compute_fvar(v=src_name):
      with dask.config.set(scheduler="threads"):
        da = sub[v].transpose("time", "lead_time", "latitude", "longitude")
        return np.asarray(da.compute().values, dtype=np.float32)

    raw_vals = _retry_with_backoff(
        _compute_fvar,
        description=f"reading {spec.product.value}:{src_name} forecast slices from {uri}",
        max_retries=max_retries,
    )
    converted_4d = band_spec.transform(raw_vals, src_name)
    del raw_vals
    reduced_vals, reduced_missing = matrix.reduce_4d_with_coverage(converted_4d)
    del converted_4d
    if valid_target_indices and valid_target_leads:
      sub_slice = reduced_vals[:, valid_sub_indices, :][:, :, valid_sub_leads]
      target_arr = data_vars[target_band][1]
      target_arr[
          :,
          np.asarray(valid_target_indices)[:, None],
          np.asarray(valid_target_leads)[None, :],
      ] = sub_slice
      miss_slice = reduced_missing[:, valid_sub_indices, :][
          :, :, valid_sub_leads
      ]
      idx_t = np.asarray(valid_target_indices)[:, None]
      idx_l = np.asarray(valid_target_leads)[None, :]
      missing_fraction_out[:, idx_t, idx_l] = np.minimum(
          missing_fraction_out[:, idx_t, idx_l],
          miss_slice,
      )

  missing_var_name = MISSING_FRACTION_VAR[spec.product]
  data_vars[missing_var_name] = (
      ["basin", "date", "lead_time"],
      missing_fraction_out,
  )

  attrs = dict(PRODUCT_METADATA_ATTRS.get(spec.product, {}))
  attrs["archive_source_uri"] = uri

  return xr.Dataset(
      data_vars=data_vars,
      coords={
          "basin": basin_ids,
          "date": date_idx.values,
          "lead_time": leads_td.values,
      },
      attrs=attrs,
  )


def extract_from_archive(
    product: Union[Product, str],
    store_uri: Union[str, os.PathLike],
    basins_gdf: gpd.GeoDataFrame,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    *,
    weights_matrix: Optional[ZonalWeightMatrix] = None,
    use_bounding_box: bool = True,
    max_retries: int = 6,
) -> xr.Dataset:
  """Dispatches to nowcast or forecast gridded archive extraction."""
  spec = get_archive_spec(product)
  if spec.product_type == ProductType.NOWCAST:
    return extract_nowcast_from_archive(
        spec.product,
        store_uri,
        basins_gdf,
        start_date,
        end_date,
        weights_matrix=weights_matrix,
        use_bounding_box=use_bounding_box,
        max_retries=max_retries,
    )
  return extract_forecast_from_archive(
      spec.product,
      store_uri,
      basins_gdf,
      start_date,
      end_date,
      weights_matrix=weights_matrix,
      use_bounding_box=use_bounding_box,
      max_retries=max_retries,
  )
