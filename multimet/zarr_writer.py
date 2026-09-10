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

import os
import shutil
import tarfile
from typing import Dict, List, Mapping, Optional, Sequence, Union

import logging
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from multimet.config import (
    DEFAULT_CHUNKS_FORECAST,
    DEFAULT_CHUNKS_NOWCAST,
    FORECAST_LEAD_DAYS,
    PRODUCT_BANDS,
    PRODUCT_METADATA_ATTRS,
    PRODUCT_TYPES,
    Product,
    ProductType,
)


class MultiMetZarrWriter:
  """Handles writing and appending MultiMet forcing datasets into Zarr stores."""

  def __init__(self, output_dir: Union[str, os.PathLike]):
    self.output_dir = str(output_dir)
    self._open_groups: Dict[Product, zarr.hierarchy.Group] = {}

  def get_store_path(self, product: Product) -> str:
    """Returns absolute path to the product's timeseries.zarr directory."""
    return os.path.join(self.output_dir, product.value, "timeseries.zarr")

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

    # Check data variables and dtypes
    expected_bands = PRODUCT_BANDS[product]
    for band in expected_bands:
      if band not in ds.data_vars:
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
  ) -> str:
    """Initializes the skeleton of a Zarr store on CNS or local disk.

    Creates .zgroup, .zattrs, coordinate chunks (basin, date, [lead_time]),
    and .zarray descriptors with chunk layout chunks=(num_basins, 1) or
    (num_basins, 1, 10) and fill_value=NaN.

    Args:
      product: MultiMet Product enum.
      basin_ids: List of basin ID strings.
      dates: List of pandas Timestamps or date strings.

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
          attrs={"units": "days"},
      )
      dims.append("lead_time")
      shape = (len(basin_ids), len(dates), lead_steps)
      chunk_spec["lead_time"] = lead_steps

    data_vars = {}
    for band in PRODUCT_BANDS[product]:
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
      data_vars[band] = (dims, np.full(shape, np.nan, dtype=np.float32), var_attrs)

    global_attrs = dict(PRODUCT_METADATA_ATTRS.get(product, {}))
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attrs)
    ds = ds.chunk(chunk_spec)

    ds.to_zarr(store_path, mode="w", consolidated=True)
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

  def write_or_append(
      self,
      ds: xr.Dataset,
      product: Product,
      overwrite_existing_basins: bool = False,
  ) -> str:
    """Writes or appends a dataset into the product's Zarr store.

    Args:
      ds: xarray Dataset matching MultiMet schema.
      product: MultiMet Product enum.
      overwrite_existing_basins: If True and store exists, existing overlapping
        basins will be replaced. If False, only new basins are appended.

    Returns:
      Store path written to.
    """
    # Ensure float32 and ensure chunks
    ds_to_write = ds.copy()
    for var in ds_to_write.data_vars:
      if ds_to_write[var].dtype != np.float32:
        ds_to_write[var] = ds_to_write[var].astype(np.float32)

    self.validate_dataset_schema(ds_to_write, product)
    store_path = self.get_store_path(product)
    prod_type = PRODUCT_TYPES[product]
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
      ds_chunked.to_zarr(local_target, mode="w", consolidated=True)
    else:
      # Append along basin dimension
      existing_ds = xr.open_zarr(local_target)
      existing_basins = set(existing_ds["basin"].values)

      if not existing_basins:
        shutil.rmtree(local_target, ignore_errors=True)
        ds_chunked = ds_to_write.chunk(chunk_spec)
        ds_chunked.to_zarr(local_target, mode="w", consolidated=True)
      else:
        incoming_basins = list(ds_to_write["basin"].values)
        incoming_basins_set = set(incoming_basins)
        existing_basins_list = list(existing_ds["basin"].values)
        existing_basins_set = set(existing_basins_list)

        existing_dates_set = set(pd.to_datetime(existing_ds["date"].values))
        incoming_dates = [pd.to_datetime(d) for d in ds_to_write["date"].values]
        incoming_dates_set = set(incoming_dates)

        # Case 1: Same basins, appending along date dimension
        if (
            incoming_basins_set == existing_basins_set
            and not overwrite_existing_basins
        ):
          new_dates = [d for d in incoming_dates if d not in existing_dates_set]
          if not new_dates:
            return store_path
          new_dates_dt64 = [d.to_datetime64() for d in new_dates]
          new_slice = ds_to_write.sel(
              basin=existing_basins_list, date=new_dates_dt64
          )
          new_slice = new_slice.chunk(chunk_spec)
          new_slice.to_zarr(local_target, append_dim="date", consolidated=True)
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
          combined.to_zarr(local_target, mode="w", consolidated=True)
        else:
          # Case 2: Append new basins along basin dimension
          new_basins = [
              b for b in incoming_basins if b not in existing_basins_set
          ]
          if not new_basins:
            return store_path
          new_slice = ds_to_write.sel(basin=new_basins)
          new_slice = new_slice.chunk(chunk_spec)
          new_slice.to_zarr(local_target, append_dim="basin", consolidated=True)



    return store_path
