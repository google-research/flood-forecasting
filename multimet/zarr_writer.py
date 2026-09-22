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

import inspect
import logging
import os
import random
import shutil
import tarfile
import time
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from multimet.config import (
    DEFAULT_CHUNKS_FORECAST,
    DEFAULT_CHUNKS_NOWCAST,
    FORECAST_LEAD_DAYS,
    MISSING_FRACTION_VAR,
    PRODUCT_BANDS,
    PRODUCT_METADATA_ATTRS,
    PRODUCT_TYPES,
    Product,
    ProductType,
)

logger = logging.getLogger(__name__)

_OPTIONAL_SECONDARY_BANDS: Mapping[Product, Tuple[str, ...]] = {
    Product.CPC: ("cpc_num_stations",),
    Product.ERA5_LAND: tuple(
        b
        for b in PRODUCT_BANDS[Product.ERA5_LAND]
        if b.endswith(("_min", "_max"))
    ),
}


def check_zarr_store_exists(store_path: str, max_retries: int = 6) -> bool:
  """Checks whether a Zarr store exists without masking storage/permission errors.

  For local paths, checks descriptor files directly on disk. For remote URIs
  (``gs://``, ``s3://``, etc.), retries transient ``OSError`` / HTTP 403 / 5xx
  responses with exponential backoff and raises if storage access fails after
  ``max_retries`` attempts — never silently returning ``False`` on an I/O or
  permission error.
  """
  if not store_path.startswith(("gs://", "gcs://", "s3://", "abfs://", "az://")):
    return (
        os.path.exists(os.path.join(store_path, "zarr.json"))
        or os.path.exists(os.path.join(store_path, ".zgroup"))
        or os.path.exists(os.path.join(store_path, ".zmetadata"))
    )

  import fsspec

  last_err: Optional[Exception] = None
  for attempt in range(max_retries):
    try:
      fs, fs_path = fsspec.core.url_to_fs(store_path)
      return bool(
          fs.exists(f"{fs_path}/zarr.json")
          or fs.exists(f"{fs_path}/.zgroup")
          or fs.exists(f"{fs_path}/.zmetadata")
      )
    except FileNotFoundError:
      return False
    except (OSError, RuntimeError, TimeoutError) as err:
      last_err = err
      if attempt == max_retries - 1:
        break
      wait_s = (2**attempt) + random.uniform(0.1, 0.75)
      logger.warning(
          "Transient storage error checking Zarr store %s (attempt %d/%d): %s. "
          "Retrying in %.2fs...",
          store_path,
          attempt + 1,
          max_retries,
          err,
          wait_s,
      )
      time.sleep(wait_s)

  raise RuntimeError(
      f"Failed to check Zarr store existence at {store_path} after "
      f"{max_retries} attempts: {last_err}"
  ) from last_err


def _safe_to_zarr(
    ds: xr.Dataset, store_path: Union[str, os.PathLike], **kwargs
) -> None:
  """Writes dataset to Zarr, targeting Zarr format 2 for broad compatibility."""
  sig = inspect.signature(xr.Dataset.to_zarr)
  if "zarr_format" in sig.parameters:
    kwargs.setdefault("zarr_format", 2)

  max_retries = 5
  for attempt in range(max_retries):
    try:
      ds.to_zarr(store_path, **kwargs)
      return
    except Exception as e:
      if attempt < max_retries - 1:
        backoff = (2**attempt) + random.uniform(0.5, 2.0)
        logger.warning(
            "Transient error in to_zarr(%s) on attempt %d/%d: %s. Retrying in %.1fs...",
            store_path,
            attempt + 1,
            max_retries,
            e,
            backoff,
        )
        time.sleep(backoff)
      else:
        raise


class MultiMetZarrWriter:
  """Handles writing and appending MultiMet forcing datasets into Zarr stores."""

  def __init__(self, output_dir: Union[str, os.PathLike]):
    self.output_dir = str(output_dir)
    if self.output_dir.startswith(("gs://", "gcs://")):
      from multimet.gcp import configure_gcp_project
      configure_gcp_project()
    self._open_groups: Dict[Product, zarr.hierarchy.Group] = {}

  def get_store_path(self, product: Product) -> str:
    """Returns absolute path to the product's timeseries.zarr directory."""
    return os.path.join(self.output_dir, product.value, "timeseries.zarr")

  def store_exists(self, product: Product) -> bool:
    """Checks whether the product's Zarr store exists."""
    store_path = self.get_store_path(product)
    return check_zarr_store_exists(store_path)

  def get_store_info(self, product: Product) -> Optional[Dict[str, Any]]:
    """Inspects an existing Zarr store and returns its metadata structure."""
    if not self.store_exists(product):
      return None

    store_path = self.get_store_path(product)
    last_err: Optional[Exception] = None
    for attempt in range(5):
      try:
        with xr.open_zarr(store_path) as ds:
          if "basin" not in ds.coords or "date" not in ds.coords:
            return None

          basins = [str(b).rstrip("\x00") for b in ds["basin"].values]
          basin_dtype = ds["basin"].dtype
          dates = pd.DatetimeIndex(ds["date"].values)

          prod_type = PRODUCT_TYPES[product]
          is_forecast = prod_type == ProductType.FORECAST
          lead_time = (
              list(ds["lead_time"].values)
              if is_forecast and "lead_time" in ds.coords
              else None
          )

          bands = [
              band for band in PRODUCT_BANDS[product] if band in ds.data_vars
          ]
          shape = ds[bands[0]].shape if bands else None

          return {
              "store_path": store_path,
              "basins": basins,
              "basin_dtype": basin_dtype,
              "dates": dates,
              "bands": bands,
              "is_forecast": is_forecast,
              "lead_time": lead_time,
              "shape": shape,
          }
      except (OSError, RuntimeError, TimeoutError) as err:
        last_err = err
        if attempt < 4:
          time.sleep((2**attempt) + random.uniform(0.1, 0.5))
          continue
        raise

    if last_err is not None:
      raise last_err
    return None

  def _ensure_companion_variables(
      self, ds: xr.Dataset, product: Product
  ) -> xr.Dataset:
    """Ensures optional secondary bands and the companion missing_fraction band exist."""
    prod_type = PRODUCT_TYPES[product]
    expected_dims = (
        ["basin", "date"]
        if prod_type == ProductType.NOWCAST
        else ["basin", "date", "lead_time"]
    )
    shape = tuple(len(ds[d]) for d in expected_dims)

    optional_bands = set(_OPTIONAL_SECONDARY_BANDS.get(product, ()))
    required_bands = [
        b for b in PRODUCT_BANDS[product] if b not in optional_bands
    ]
    primary_band = next(
        (b for b in required_bands if b in ds.data_vars), None
    )
    if primary_band is None:
      return ds

    out = ds.copy()
    for opt_band in optional_bands:
      if opt_band not in out.data_vars:
        out[opt_band] = (
            expected_dims,
            np.full(shape, np.nan, dtype=np.float32),
        )

    missing_var = MISSING_FRACTION_VAR.get(product)
    if missing_var and missing_var not in out.data_vars:
      out[missing_var] = (
          expected_dims,
          np.isnan(out[primary_band].values).astype(np.float32),
      )
    return out

  def validate_dataset_schema(self, ds: xr.Dataset, product: Product) -> None:
    """Validates that an xarray Dataset strictly complies with MultiMet schema.

    Args:
      ds: Dataset to validate.
      product: Product enum.

    Raises:
      ValueError: If dimensions, coordinates, dtypes, or variable names
      mismatch.
    """
    prod_type = PRODUCT_TYPES[product]

    # Check dimensions
    expected_dims = (
        ("basin", "date")
        if prod_type == ProductType.NOWCAST
        else ("basin", "date", "lead_time")
    )
    for dim in expected_dims:
      if dim not in ds.dims:
        raise ValueError(
            f"Product {product.value} missing required dimension '{dim}'. "
            f"Found dims: {list(ds.dims)}"
        )

    # Check coordinates
    if "basin" not in ds.coords:
      raise ValueError("Dataset missing 'basin' coordinate.")
    if "date" not in ds.coords:
      raise ValueError("Dataset missing 'date' coordinate.")

    # Validate date dtype is datetime64
    if not np.issubdtype(ds["date"].dtype, np.datetime64):
      raise ValueError(
          "'date' coordinate must have datetime64 dtype, got"
          f" {ds['date'].dtype}"
      )

    if prod_type == ProductType.FORECAST:
      if "lead_time" not in ds.coords:
        raise ValueError("Forecast dataset missing 'lead_time' coordinate.")
      if not (
          np.issubdtype(ds["lead_time"].dtype, np.timedelta64)
          or np.issubdtype(ds["lead_time"].dtype, np.integer)
      ):
        raise ValueError(
            "'lead_time' coordinate must have integer or timedelta64 dtype, "
            f"got {ds['lead_time'].dtype}"
        )
      expected_leads = FORECAST_LEAD_DAYS[product]
      if len(ds["lead_time"]) != expected_leads:
        raise ValueError(
            f"Product {product.value} expects {expected_leads} lead time steps,"
            f" found {len(ds['lead_time'])}"
        )

    optional_bands = set(_OPTIONAL_SECONDARY_BANDS.get(product, ()))
    expected_bands = PRODUCT_BANDS[product]
    for band in expected_bands:
      if band not in ds.data_vars:
        if band in optional_bands:
          continue
        raise ValueError(
            f"Product {product.value} missing required band variable '{band}'"
        )
      if ds[band].dtype != np.float32:
        raise ValueError(
            f"Band '{band}' must have dtype float32, got {ds[band].dtype}"
        )

  def initialize_zarr_store(
      self,
      product: Product,
      basin_ids: List[str],
      dates: List[pd.Timestamp],
      extra_attrs: Optional[Mapping[str, Any]] = None,
  ) -> str:
    """Initializes the skeleton of a Zarr store on CNS or local disk.

    Creates .zgroup, .zattrs, coordinate chunks (basin, date, [lead_time]),
    and .zarray descriptors with chunk layout chunks=(num_basins, 1) or
    (num_basins, 1, 10) and fill_value=NaN.

    Args:
      product: MultiMet Product enum.
      basin_ids: List of basin ID strings.
      dates: List of pandas Timestamps or date strings.
      extra_attrs: Optional additional global attributes (e.g. upstream archive
        store URI, shapefile path, extracted date range).

    Returns:
      Store path initialized.
    """
    store_path = self.get_store_path(product)
    prod_type = PRODUCT_TYPES[product]
    is_forecast = prod_type == ProductType.FORECAST

    coords = {
        "basin": np.array(basin_ids, dtype="<U22"),
        "date": pd.to_datetime(dates).values,
    }
    dims = ["basin", "date"]
    shape = (len(basin_ids), len(dates))
    chunk_spec = {"basin": len(basin_ids), "date": 1}

    if is_forecast:
      lead_steps = FORECAST_LEAD_DAYS[product]
      coords["lead_time"] = xr.DataArray(
          np.arange(1, lead_steps + 1, dtype=np.int64),
          dims=["lead_time"],
          attrs={"units": "days", "dtype": "timedelta64[ns]"},
      )
      dims.append("lead_time")
      shape = (len(basin_ids), len(dates), lead_steps)
      chunk_spec["lead_time"] = lead_steps

    all_store_vars = list(PRODUCT_BANDS[product])
    missing_var = MISSING_FRACTION_VAR.get(product)
    if missing_var and missing_var not in all_store_vars:
      all_store_vars.append(missing_var)

    data_vars = {}
    for band in all_store_vars:
      var_attrs = {}
      if band in (
          "hres_surface_net_solar_radiation",
          "hres_surface_net_thermal_radiation",
      ):
        var_attrs = {
            "status": "unavailable",
            "comment": (
                "Surface radiation flux variables are unavailable in"
                " WeatherBench 2 HRES archive (2016-01-01 to 2023-01-10)."
            ),
        }
      data_vars[band] = (
          dims,
          np.full(shape, np.nan, dtype=np.float32),
          var_attrs,
      )

    global_attrs = dict(PRODUCT_METADATA_ATTRS.get(product, {}))
    if extra_attrs:
      global_attrs.update(extra_attrs)
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attrs)
    ds = ds.chunk(chunk_spec)

    _safe_to_zarr(ds, store_path, mode="w", compute=False, consolidated=True)
    return store_path

  def consolidate_metadata(self, product: Product) -> str:
    """Consolidates Zarr metadata (.zmetadata) after distributed chunk writes."""
    store_path = self.get_store_path(product)
    zarr.consolidate_metadata(store_path)
    return store_path

  def write_direct_chunk(
      self,
      product: Product,
      var_name: str,
      day_idx: int,
      values: np.ndarray,
      root_group: Optional[zarr.hierarchy.Group] = None,
  ) -> str:
    """Writes a chunk directly to the Zarr store using Zarr array indexing.

    Args:
      product: MultiMet Product enum.
      var_name: Variable band name.
      day_idx: Integer index along the date dimension.
      values: 1D or 2D numpy array of float32 values for this day across all
        basins.
      root_group: Optional pre-opened zarr Group handle to avoid repeated I/O.

    Returns:
      Store path.
    """
    store_path = self.get_store_path(product)
    prod_type = PRODUCT_TYPES[product]
    is_forecast = prod_type == ProductType.FORECAST

    if root_group is not None:
      z_root = root_group
    elif product in self._open_groups:
      z_root = self._open_groups[product]
    else:
      z_root = zarr.open_group(store_path, mode="r+")
      self._open_groups[product] = z_root

    if var_name not in z_root:
      raise KeyError(f"Variable {var_name} not found in store {store_path}")

    if is_forecast:
      if values.ndim == 2:
        z_root[var_name][:, day_idx, :] = values.astype(np.float32)
      elif values.ndim == 3:
        z_root[var_name][:, day_idx : day_idx + 1, :] = values.astype(
            np.float32
        )
    else:
      if values.ndim == 1:
        z_root[var_name][:, day_idx] = values.astype(np.float32)
      elif values.ndim == 2:
        z_root[var_name][:, day_idx : day_idx + 1] = values.astype(np.float32)

    return store_path

  def write_direct_chunk_range(
      self,
      product: Product,
      var_name: str,
      start_idx: int,
      end_idx: int,
      values: np.ndarray,
      root_group: Optional[zarr.hierarchy.Group] = None,
  ) -> str:
    """Writes a contiguous date range directly to the Zarr store using Zarr array indexing.

    Args:
      product: MultiMet Product enum.
      var_name: Variable band name.
      start_idx: Integer start index along the date dimension (inclusive).
      end_idx: Integer end index along the date dimension (exclusive).
      values: 2D or 3D numpy array of float32 values for this date range across
        all basins. Shape is (num_basins, num_days) for nowcast or
        (num_basins, num_days, lead_steps) for forecast.
      root_group: Optional pre-opened zarr Group handle to avoid repeated I/O.

    Returns:
      Store path.
    """
    store_path = self.get_store_path(product)
    prod_type = PRODUCT_TYPES[product]
    is_forecast = prod_type == ProductType.FORECAST

    if root_group is not None:
      z_root = root_group
    elif product in self._open_groups:
      z_root = self._open_groups[product]
    else:
      z_root = zarr.open_group(store_path, mode="r+")
      self._open_groups[product] = z_root

    if var_name not in z_root:
      raise KeyError(f"Variable {var_name} not found in store {store_path}")

    if is_forecast:
      z_root[var_name][:, start_idx:end_idx, :] = values.astype(np.float32)
    else:
      z_root[var_name][:, start_idx:end_idx] = values.astype(np.float32)

    return store_path

  def is_date_chunk_written(
      self,
      product: Product,
      day_idx: int,
      var_name: Optional[str] = None,
      root_group: Optional[zarr.hierarchy.Group] = None,
  ) -> bool:
    """Checks whether a given day index has already been populated with non-NaN data.

    Args:
      product: MultiMet Product enum.
      day_idx: Integer index along the date dimension.
      var_name: Optional variable name to check. If None, uses the first band of
        the product.
      root_group: Optional pre-opened zarr Group handle.

    Returns:
      True if the chunk exists and contains at least one non-NaN value.
    """
    store_path = self.get_store_path(product)
    if root_group is not None:
      z_root = root_group
    elif product in self._open_groups:
      z_root = self._open_groups[product]
    else:
      try:
        z_root = zarr.open_group(store_path, mode="r")
      except Exception:
        return False

    target_var = var_name or PRODUCT_BANDS[product][0]
    if target_var not in z_root:
      return False

    arr = z_root[target_var]
    if day_idx < 0 or day_idx >= arr.shape[1]:
      return False

    prod_type = PRODUCT_TYPES[product]
    if prod_type == ProductType.FORECAST:
      slice_data = arr[:, day_idx, :]
    else:
      slice_data = arr[:, day_idx]

    return bool(np.any(~np.isnan(slice_data)))

  def append_dates(
      self,
      product: Product,
      new_dates: Sequence[Union[pd.Timestamp, str, np.datetime64]],
  ) -> Tuple[int, int]:
    """Appends new dates to an existing Zarr store by resizing the array.

    Resizes the 'date' coordinate and all data variables in-place, initializing
    the newly allocated slots with NaN. Consolidates metadata post-resize.

    Args:
      product: MultiMet Product enum.
      new_dates: Sequence of timestamps or date strings to append.

    Returns:
      Tuple of (start_idx, end_idx) indicating the integer slice range of the
      appended dates.

    Raises:
      FileNotFoundError: If the store does not exist.
      ValueError: If new dates are before or overlapping non-contiguously with
        existing dates.
    """
    store_path = self.get_store_path(product)
    info = self.get_store_info(product)
    if info is None:
      raise FileNotFoundError(
          f"Cannot append dates: store does not exist at {store_path}"
      )

    existing_dates = info["dates"]
    existing_dates_set = set(existing_dates)

    parsed_dates = [pd.to_datetime(d) for d in new_dates]
    dates_to_add = [d for d in parsed_dates if d not in existing_dates_set]

    if not dates_to_add:
      logger.info("All requested dates already exist in store %s", store_path)
      return (len(existing_dates), len(existing_dates))

    # If prepending or non-contiguous, delegate to expand_date_range
    if len(existing_dates) > 0 and min(dates_to_add) <= existing_dates.max():
      all_dates, indices = self.expand_date_range(product, parsed_dates)
      return (indices[0], indices[-1] + 1)

    # Sort dates chronologically
    dates_to_add = sorted(dates_to_add)

    old_len = len(existing_dates)
    new_len = old_len + len(dates_to_add)

    prod_type = PRODUCT_TYPES[product]
    is_forecast = prod_type == ProductType.FORECAST
    num_basins = len(info["basins"])

    nan_vars = {}
    dims = ["basin", "date"]
    shape = (num_basins, len(dates_to_add))
    chunk_spec = {"basin": num_basins, "date": 1}
    basin_dt = info.get("basin_dtype", "<U22")
    coords = {
        "basin": np.array(info["basins"], dtype=basin_dt),
        "date": [d.to_datetime64() for d in dates_to_add],
    }

    if is_forecast:
      lead_steps = FORECAST_LEAD_DAYS[product]
      coords["lead_time"] = xr.DataArray(
          np.arange(1, lead_steps + 1, dtype=np.int64),
          dims=["lead_time"],
          attrs={"units": "days", "dtype": "timedelta64[ns]"},
      )
      dims.append("lead_time")
      shape = (num_basins, len(dates_to_add), lead_steps)
      chunk_spec["lead_time"] = lead_steps

    all_store_vars = list(PRODUCT_BANDS[product])
    missing_var = MISSING_FRACTION_VAR.get(product)
    if missing_var and missing_var not in all_store_vars:
      all_store_vars.append(missing_var)
    for band in all_store_vars:
      nan_vars[band] = (dims, np.full(shape, np.nan, dtype=np.float32))

    nan_ds = xr.Dataset(data_vars=nan_vars, coords=coords).chunk(chunk_spec)
    _safe_to_zarr(nan_ds, store_path, append_dim="date", consolidated=True)
    self._open_groups.pop(product, None)

    logger.info(
        "Successfully expanded %s store dates from %d to %d days (+%d days)",
        product.value,
        old_len,
        new_len,
        len(dates_to_add),
    )
    return (old_len, new_len)

  def expand_date_range(
      self,
      product: Product,
      requested_dates: Sequence[Union[pd.Timestamp, str, np.datetime64]],
  ) -> Tuple[pd.DatetimeIndex, List[int]]:
    """Expands the store's date dimension if needed to cover requested_dates.

    Supports:
    - Postpending dates (dates after existing end date).
    - Prepending dates (dates before existing start date).
    - Partial or full overlaps with existing dates.

    Args:
      product: MultiMet Product enum.
      requested_dates: Sequence of timestamps or date strings to ensure are present.

    Returns:
      Tuple of (all_store_dates, requested_date_indices_in_store).
    """
    store_path = self.get_store_path(product)
    info = self.get_store_info(product)
    if info is None:
      raise FileNotFoundError(
          f"Cannot expand date range: store does not exist at {store_path}"
      )

    existing_dates = pd.DatetimeIndex(info["dates"])
    req_dates = pd.DatetimeIndex(pd.to_datetime(requested_dates)).sort_values()

    self._open_groups.pop(product, None)

    # If all requested dates are already within existing dates, no resize is required
    if req_dates.isin(existing_dates).all():
      indices = [int(existing_dates.get_loc(d)) for d in req_dates]
      return (existing_dates, indices)

    union_dates = existing_dates.union(req_dates).sort_values()
    min_existing = existing_dates.min()
    max_existing = existing_dates.max()
    new_dates_to_add = [d for d in union_dates if d not in set(existing_dates)]

    # Fast path: all new dates are strictly after the existing store's end date
    if all(d > max_existing for d in new_dates_to_add):
      self.append_dates(product, new_dates_to_add)
      self._open_groups.pop(product, None)
      updated_info = self.get_store_info(product)
      updated_dates = pd.DatetimeIndex(updated_info["dates"])
      indices = [int(updated_dates.get_loc(d)) for d in req_dates]
      return (updated_dates, indices)

    # Prepending or internal gap: reindex existing dataset to union_dates with NaN fill
    logger.info(
        "Prepending/reindexing %s store dates from [%s..%s] to [%s..%s]...",
        product.value,
        min_existing.strftime("%Y-%m-%d"),
        max_existing.strftime("%Y-%m-%d"),
        union_dates.min().strftime("%Y-%m-%d"),
        union_dates.max().strftime("%Y-%m-%d"),
    )
    prod_type = PRODUCT_TYPES[product]
    is_forecast = prod_type == ProductType.FORECAST
    num_basins = len(info["basins"])
    chunk_spec = {"basin": num_basins, "date": 1}
    if is_forecast:
      chunk_spec["lead_time"] = FORECAST_LEAD_DAYS[product]

    with xr.open_zarr(store_path) as existing_ds:
      expanded_ds = existing_ds.reindex(
          date=union_dates.values, fill_value=np.nan
      ).chunk(chunk_spec)
      tmp_swap_dir = os.path.join(
          self.output_dir, f".tmp_swap_{product.value}_{int(time.time() * 1000)}"
      )
      tmp_writer = MultiMetZarrWriter(tmp_swap_dir)
      tmp_swap_path = tmp_writer.get_store_path(product)
      _safe_to_zarr(expanded_ds, tmp_swap_path, mode="w", consolidated=True)

    # Swap into store_path
    try:
      import fsspec
      fs, fs_path = fsspec.core.url_to_fs(store_path)
      _, tmp_fs_path = fsspec.core.url_to_fs(tmp_swap_path)
      if fs.exists(fs_path):
        fs.rm(fs_path, recursive=True)
      if hasattr(fs, "mv"):
        fs.mv(tmp_fs_path, fs_path, recursive=True)
      else:
        fs.copy(tmp_fs_path, fs_path, recursive=True)
        fs.rm(tmp_fs_path, recursive=True)
    except Exception:
      if os.path.exists(store_path):
        shutil.rmtree(store_path, ignore_errors=True)
      shutil.move(tmp_swap_path, store_path)

    try:
      if os.path.exists(tmp_swap_dir):
        shutil.rmtree(tmp_swap_dir, ignore_errors=True)
    except Exception:
      pass

    self._open_groups.pop(product, None)
    self.consolidate_metadata(product)
    indices = [int(union_dates.get_loc(d)) for d in req_dates]
    return (union_dates, indices)

  def append_basins(
      self,
      product: Product,
      new_ds: xr.Dataset,
  ) -> str:
    """Appends new basins to an existing Zarr store across existing dates.

    Args:
      product: MultiMet Product enum.
      new_ds: xarray Dataset containing new basins matching MultiMet schema.

    Returns:
      Store path written to.

    Raises:
      FileNotFoundError: If target store does not exist.
      ValueError: If dates do not match existing store dates, or if schema mismatches.
    """
    store_path = self.get_store_path(product)
    info = self.get_store_info(product)
    if info is None:
      raise FileNotFoundError(
          f"Cannot append basins: store does not exist at {store_path}"
      )

    # Validate schema
    ds_to_write = new_ds.sortby("date").copy()
    for var in ds_to_write.data_vars:
      if ds_to_write[var].dtype != np.float32:
        ds_to_write[var] = ds_to_write[var].astype(np.float32)

    self.validate_dataset_schema(ds_to_write, product)
    ds_to_write = self._ensure_companion_variables(ds_to_write, product)

    existing_dates = info["dates"]
    incoming_dates = pd.to_datetime(ds_to_write["date"].values)

    if len(existing_dates) != len(incoming_dates) or not np.array_equal(
        existing_dates.values, incoming_dates.values
    ):
      raise ValueError(
          f"Cannot append basins to {store_path}: dates mismatch. "
          f"Store has {len(existing_dates)} dates "
          f"[{existing_dates[0].strftime('%Y-%m-%d')}..{existing_dates[-1].strftime('%Y-%m-%d')}], "
          f"but incoming dataset has {len(incoming_dates)} dates "
          f"[{incoming_dates[0].strftime('%Y-%m-%d')}..{incoming_dates[-1].strftime('%Y-%m-%d')}]. "
          "New basins must be extracted across the exact existing store date range."
      )

    existing_basins_set = set(info["basins"])
    incoming_basins = [str(b) for b in ds_to_write["basin"].values]
    new_basins = [b for b in incoming_basins if b not in existing_basins_set]

    if not new_basins:
      logger.info("All incoming basins already exist in store %s", store_path)
      return store_path

    new_slice = ds_to_write.sel(basin=new_basins)
    prod_type = PRODUCT_TYPES[product]
    chunk_spec = (
        DEFAULT_CHUNKS_NOWCAST
        if prod_type == ProductType.NOWCAST
        else DEFAULT_CHUNKS_FORECAST
    )
    new_slice = new_slice.chunk(chunk_spec)

    _safe_to_zarr(
        new_slice, store_path, append_dim="basin", consolidated=True
    )
    self.consolidate_metadata(product)
    logger.info(
        "Successfully appended %d new basins to %s store (total: %d basins)",
        len(new_basins),
        product.value,
        len(existing_basins_set) + len(new_basins),
    )
    return store_path

  def write_or_append(
      self,
      ds: xr.Dataset,
      product: Product,
      overwrite_existing_basins: bool = False,
  ) -> str:
    """Writes or appends a dataset into the product's Zarr store."""
    ds_to_write = ds.copy()
    for var in ds_to_write.data_vars:
      if ds_to_write[var].dtype != np.float32:
        ds_to_write[var] = ds_to_write[var].astype(np.float32)

    self.validate_dataset_schema(ds_to_write, product)
    ds_to_write = self._ensure_companion_variables(ds_to_write, product)
    store_path = self.get_store_path(product)
    prod_type = PRODUCT_TYPES[product]
    is_forecast = prod_type == ProductType.FORECAST
    chunk_spec = (
        DEFAULT_CHUNKS_NOWCAST
        if prod_type == ProductType.NOWCAST
        else DEFAULT_CHUNKS_FORECAST
    )

    local_target = store_path

    def _exists(path: str) -> bool:
      return os.path.exists(path)

    has_metadata = (
        _exists(os.path.join(local_target, ".zmetadata"))
        or _exists(os.path.join(local_target, ".zgroup"))
        or _exists(os.path.join(local_target, "zarr.json"))
    )
    if not _exists(local_target) or not has_metadata:
      # Create new Zarr store
      if _exists(local_target):
        shutil.rmtree(local_target, ignore_errors=True)
      os.makedirs(os.path.dirname(local_target), exist_ok=True)
      ds_chunked = ds_to_write.chunk(chunk_spec)
      _safe_to_zarr(ds_chunked, local_target, mode="w", consolidated=True)
    else:
      # Existing store
      existing_ds = xr.open_zarr(local_target)
      existing_basins = set(str(b) for b in existing_ds["basin"].values)

      if not existing_basins:
        shutil.rmtree(local_target, ignore_errors=True)
        ds_chunked = ds_to_write.chunk(chunk_spec)
        _safe_to_zarr(ds_chunked, local_target, mode="w", consolidated=True)
      else:
        incoming_basins = [str(b) for b in ds_to_write["basin"].values]
        incoming_basins_set = set(incoming_basins)
        existing_basins_list = [str(b) for b in existing_ds["basin"].values]
        existing_basins_set = set(existing_basins_list)

        existing_dates_set = set(pd.to_datetime(existing_ds["date"].values))
        incoming_dates = [pd.to_datetime(d) for d in ds_to_write["date"].values]
        incoming_dates_set = set(incoming_dates)

        has_new_basins = bool(incoming_basins_set - existing_basins_set)
        has_new_dates = bool(incoming_dates_set - existing_dates_set)

        # STRICT CHECK: Disallow adding both new basins and new dates simultaneously
        if has_new_basins and has_new_dates:
          raise ValueError(
              "Cannot add both new basins and new dates to an existing Zarr store simultaneously. "
              "A Zarr store requires a dense coordinate grid; adding coordinates along two dimensions "
              "at once leaves unpopulated cross-quadrants (old basins for new dates, and new basins for old dates). "
              "Please either append dates for existing basins, or append new basins across existing dates."
          )

        # Case 1: Same basins, adding/updating along date dimension
        if (
            incoming_basins_set == existing_basins_set
            and not overwrite_existing_basins
        ):
          new_dates = [d for d in incoming_dates if d not in existing_dates_set]
          if not new_dates and not incoming_dates:
            return store_path

          # Pure postpending fast-path
          if (
              new_dates
              and existing_dates_set
              and min(new_dates) > max(existing_dates_set)
              and len(new_dates) == len(incoming_dates)
          ):
            new_dates_dt64 = [d.to_datetime64() for d in new_dates]
            new_slice = ds_to_write.sel(
                basin=existing_basins_list, date=new_dates_dt64
            )
            new_slice = new_slice.chunk(chunk_spec)
            _safe_to_zarr(
                new_slice, local_target, append_dim="date", consolidated=True
            )
            self._open_groups.pop(product, None)
            self.consolidate_metadata(product)
            return store_path
          else:
            # Prepending, overlapping, or rewriting dates
            all_dates, indices = self.expand_date_range(product, incoming_dates)
            self._open_groups.pop(product, None)
            z_root = zarr.open_group(local_target, mode="r+")
            ds_aligned = ds_to_write.sel(basin=existing_basins_list)
            for var in ds_aligned.data_vars:
              vals = ds_aligned[var].values.astype(np.float32)
              for local_idx, store_idx in enumerate(indices):
                if is_forecast:
                  z_root[var][:, store_idx, :] = vals[:, local_idx, :]
                else:
                  z_root[var][:, store_idx] = vals[:, local_idx]
            self.consolidate_metadata(product)
            return store_path
        elif overwrite_existing_basins:
          keep_basins = [
              b for b in existing_basins_list if b not in incoming_basins_set
          ]
          if keep_basins:
            existing_kept = existing_ds.sel(basin=keep_basins)
            combined = xr.concat([existing_kept, ds_to_write], dim="basin")
          else:
            combined = ds_to_write
          combined = combined.chunk(chunk_spec)
          _safe_to_zarr(combined, local_target, mode="w", consolidated=True)
        else:
          # Case 2: Append new basins along basin dimension
          return self.append_basins(product, ds_to_write)

    return store_path
