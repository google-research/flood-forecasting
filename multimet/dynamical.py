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

"""Generic DataLoader and Extractor for dynamical.org datasets with Icechunk acceleration."""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
import sys
import xarray as xr

# Ensure PROJ finds its database if running in standard conda paths
if "PROJ_DATA" not in os.environ and "PROJ_LIB" not in os.environ:
  candidate_proj = Path(sys.prefix) / "share" / "proj"
  if candidate_proj.exists():
    os.environ["PROJ_DATA"] = str(candidate_proj)

try:
  import dynamical_catalog
except ImportError:
  dynamical_catalog = None

try:
  import icechunk
except ImportError:
  icechunk = None

from multimet.base import BaseExtractor
from multimet.config import (
    FORECAST_LEAD_DAYS,
    MISSING_FRACTION_VAR,
    PRODUCT_BANDS,
    PRODUCT_METADATA_ATTRS,
    Product,
    ProductType,
)
from multimet.geometry import get_bounding_box, load_basin_geometries
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix

logger = logging.getLogger(__name__)

_GLOBAL_DATASET_CACHE: Dict[str, xr.Dataset] = {}


def list_catalog_datasets() -> List[str]:
  """Returns all dataset identifiers available in the dynamical.org catalog."""
  if dynamical_catalog is None:
    raise ImportError(
        "dynamical_catalog is required to list catalog datasets. "
        "Install it via: pip install dynamical-catalog"
    )
  return dynamical_catalog.list()


def clear_catalog_cache() -> None:
  """Clears in-process cache of opened dynamical.org datasets."""
  _GLOBAL_DATASET_CACHE.clear()


@dataclasses.dataclass(frozen=True)
class DynamicalDatasetInfo:
  """Metadata describing the coordinates, grid type, and dimensions of a dynamical dataset."""

  dataset_id: str
  grid_type: str  # 'geographic_1d' or 'projected_2d'
  spatial_dims: Tuple[str, str]
  temporal_dim: str
  has_lead_time: bool
  has_ensemble: bool
  crs_wkt: Optional[str]
  variables: List[str]
  dataset_attrs: Dict[str, Any]


class DynamicalDataLoader:
  """Universal data loader for dynamical.org catalog datasets with Icechunk acceleration.

  Leverages Icechunk's transactional Zarr storage engine to perform
  geographically bounded, zero-copy byte-range reads over public cloud object storage (S3).
  Supports all datasets across the dynamical.org catalog:
    - Global 1D Geographic: IMERG, GFS, GEFS, AIFS, IFS
    - Regional 1D Geographic: MRMS, ICON-EU
    - Projected 2D Grids: HRRR (Lambert Conformal), HRDPS (Rotated Pole)
  """

  def __init__(
      self,
      dataset_id: str,
      auto_open: bool = True,
      cache_dataset: bool = True,
  ):
    """Initializes the loader for a given dynamical.org dataset.

    Args:
      dataset_id: Identifier of the dataset in dynamical.org (e.g.
        'nasa-imerg-analysis-early', 'noaa-gfs-forecast', 'noaa-hrrr-analysis').
      auto_open: Whether to immediately open the Icechunk Zarr store.
      cache_dataset: Whether to cache and reuse the opened xr.Dataset across instances.
    """
    if dynamical_catalog is None:
      raise ImportError(
          "dynamical_catalog is required. Install via: pip install dynamical-catalog icechunk"
      )

    self.dataset_id = dataset_id
    self.cache_dataset = cache_dataset
    self._ds: Optional[xr.Dataset] = None

    self.grid_type: str = "geographic_1d"
    self.spatial_dims: Tuple[str, str] = ("latitude", "longitude")
    self.lat_coord: str = "latitude"
    self.lon_coord: str = "longitude"
    self.y_coord: str = "y"
    self.x_coord: str = "x"

    self.lat_descending: bool = True
    self.y_descending: bool = False
    self.lon_ascending: bool = True
    self.x_ascending: bool = True

    self.time_dim: str = "time"
    self.has_lead_time: bool = False
    self.has_ensemble: bool = False
    self.crs_wkt: Optional[str] = None
    self._transformer: Optional[pyproj.Transformer] = None

    if auto_open:
      self.open()

  @property
  def ds(self) -> xr.Dataset:
    """Returns the opened xarray.Dataset."""
    if self._ds is None:
      self.open()
    return self._ds

  def open(self) -> xr.Dataset:
    """Opens the dataset via dynamical_catalog and analyzes its coordinate system."""
    if self.cache_dataset and self.dataset_id in _GLOBAL_DATASET_CACHE:
      self._ds = _GLOBAL_DATASET_CACHE[self.dataset_id]
    else:
      self._ds = dynamical_catalog.open(self.dataset_id)
      if self.cache_dataset:
        _GLOBAL_DATASET_CACHE[self.dataset_id] = self._ds

    self._analyze_schema()
    return self._ds

  def _analyze_schema(self) -> None:
    """Inspects coordinate dimensions, CRS, and temporal topologies."""
    ds = self._ds
    assert ds is not None

    # Temporal dimension detection
    if "init_time" in ds.coords:
      self.time_dim = "init_time"
    elif "time" in ds.coords:
      self.time_dim = "time"
    else:
      for c in ds.coords:
        if np.issubdtype(ds[c].dtype, np.datetime64):
          self.time_dim = c
          break

    self.has_lead_time = "lead_time" in ds.coords or "lead_time" in ds.dims
    self.has_ensemble = (
        "ensemble_member" in ds.coords or "ensemble_member" in ds.dims
    )

    # CRS detection
    self.crs_wkt = None
    if "spatial_ref" in ds.coords and hasattr(ds.spatial_ref, "attrs"):
      self.crs_wkt = ds.spatial_ref.attrs.get("crs_wkt")
    elif "crs" in ds.attrs:
      self.crs_wkt = ds.attrs.get("crs")

    # Spatial dimensions & coordinates detection
    coords = set(ds.coords.keys())
    dims = set(ds.sizes.keys())

    # Case 1: Projected 2D Grid with 1D (y, x) dimensions (e.g. HRRR, HRDPS)
    if ("y" in dims and "x" in dims) or ("y" in coords and "x" in coords):
      self.grid_type = "projected_2d"
      self.spatial_dims = ("y", "x")
      self.y_coord = "y"
      self.x_coord = "x"

      y_vals = ds[self.y_coord].values
      x_vals = ds[self.x_coord].values
      self.y_descending = bool(len(y_vals) > 1 and y_vals[0] > y_vals[-1])
      self.x_ascending = bool(len(x_vals) > 1 and x_vals[0] < x_vals[-1])

      if self.crs_wkt is not None:
        try:
          self._transformer = pyproj.Transformer.from_crs(
              "EPSG:4326", self.crs_wkt, always_xy=True
          )
        except Exception as e:
          logger.warning("Could not initialize pyproj Transformer: %s", e)

    # Case 2: Geographic 1D Grid with (latitude, longitude) or (lat, lon)
    else:
      self.grid_type = "geographic_1d"
      if "latitude" in coords:
        self.lat_coord = "latitude"
      elif "lat" in coords:
        self.lat_coord = "lat"
      else:
        raise ValueError(f"Could not find latitude coordinate in {list(coords)}")

      if "longitude" in coords:
        self.lon_coord = "longitude"
      elif "lon" in coords:
        self.lon_coord = "lon"
      else:
        raise ValueError(f"Could not find longitude coordinate in {list(coords)}")

      self.spatial_dims = (self.lat_coord, self.lon_coord)
      lat_vals = ds[self.lat_coord].values
      lon_vals = ds[self.lon_coord].values
      self.lat_descending = bool(len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1])
      self.lon_ascending = bool(len(lon_vals) > 1 and lon_vals[0] < lon_vals[-1])

  def get_info(self) -> DynamicalDatasetInfo:
    """Returns structured information about the dataset."""
    ds = self.ds
    return DynamicalDatasetInfo(
        dataset_id=self.dataset_id,
        grid_type=self.grid_type,
        spatial_dims=self.spatial_dims,
        temporal_dim=self.time_dim,
        has_lead_time=self.has_lead_time,
        has_ensemble=self.has_ensemble,
        crs_wkt=self.crs_wkt,
        variables=list(ds.data_vars.keys()),
        dataset_attrs=dict(ds.attrs),
    )

  def compute_spatial_slices(
      self,
      watersheds: Union[
          gpd.GeoDataFrame,
          shapely.geometry.base.BaseGeometry,
          Sequence[float],
          Tuple[float, float, float, float],
          str,
          Path,
      ],
      buffer: Optional[float] = None,
  ) -> Dict[str, slice]:
    """Computes coordinate index slices bounding the requested watersheds.

    Icechunk leverages these slices to fetch ONLY the intersecting Zarr chunks
    over S3 via byte-range requests, drastically reducing data transfer.

    Args:
      watersheds: Basin GeoDataFrame, Shapely polygon, bounding box tuple
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326, or file path.
      buffer: Optional spatial buffer around the bounding box. Defaults to 0.1
        degrees for geographic grids, or 2x grid cell size for projected grids.

    Returns:
      Dictionary mapping spatial coordinate names to Python slice objects.
    """
    ds = self.ds

    # Resolve bounding box in WGS84 EPSG:4326
    if isinstance(watersheds, (str, Path, gpd.GeoDataFrame)):
      gdf = (
          watersheds
          if isinstance(watersheds, gpd.GeoDataFrame)
          else load_basin_geometries(watersheds)
      )
      min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    elif isinstance(watersheds, shapely.geometry.base.BaseGeometry):
      min_lon, min_lat, max_lon, max_lat = watersheds.bounds
    elif isinstance(watersheds, (tuple, list)) and len(watersheds) == 4:
      min_lon, min_lat, max_lon, max_lat = watersheds
    else:
      raise TypeError(f"Unsupported watershed geometry input: {type(watersheds)}")

    # 1. Geographic 1D Grids
    if self.grid_type == "geographic_1d":
      buf = buffer if buffer is not None else 0.1
      b_min_lat = max(-90.0, float(min_lat) - buf)
      b_max_lat = min(90.0, float(max_lat) + buf)
      b_min_lon = max(-180.0, float(min_lon) - buf)
      b_max_lon = min(180.0, float(max_lon) + buf)

      if self.lat_descending:
        lat_slice = slice(b_max_lat, b_min_lat)
      else:
        lat_slice = slice(b_min_lat, b_max_lat)

      if self.lon_ascending:
        lon_slice = slice(b_min_lon, b_max_lon)
      else:
        lon_slice = slice(b_max_lon, b_min_lon)

      return {self.lat_coord: lat_slice, self.lon_coord: lon_slice}

    # 2. Projected 2D Grids (e.g. HRRR, HRDPS)
    else:
      if self._transformer is None and self.crs_wkt is not None:
        self._transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", self.crs_wkt, always_xy=True
        )

      if self._transformer is not None:
        corners_lon = [min_lon, max_lon, max_lon, min_lon]
        corners_lat = [min_lat, min_lat, max_lat, max_lat]
        px, py = self._transformer.transform(corners_lon, corners_lat)
        proj_min_x, proj_max_x = min(px), max(px)
        proj_min_y, proj_max_y = min(py), max(py)
      else:
        proj_min_x, proj_max_x = min_lon, max_lon
        proj_min_y, proj_max_y = min_lat, max_lat

      y_vals = ds[self.y_coord].values
      x_vals = ds[self.x_coord].values
      dy = abs(float(y_vals[1] - y_vals[0])) if len(y_vals) > 1 else 3000.0
      dx = abs(float(x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 3000.0
      buf_y = buffer if buffer is not None else 2.0 * dy
      buf_x = buffer if buffer is not None else 2.0 * dx

      b_min_y = proj_min_y - buf_y
      b_max_y = proj_max_y + buf_y
      b_min_x = proj_min_x - buf_x
      b_max_x = proj_max_x + buf_x

      if self.y_descending:
        y_slice = slice(b_max_y, b_min_y)
      else:
        y_slice = slice(b_min_y, b_max_y)

      if self.x_ascending:
        x_slice = slice(b_min_x, b_max_x)
      else:
        x_slice = slice(b_max_x, b_min_x)

      return {self.y_coord: y_slice, self.x_coord: x_slice}

  def load_spatial_subset(
      self,
      watersheds: Union[
          gpd.GeoDataFrame,
          shapely.geometry.base.BaseGeometry,
          Sequence[float],
          Tuple[float, float, float, float],
          str,
          Path,
      ],
      variables: Optional[Union[str, Sequence[str]]] = None,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      lead_time_slice: Optional[Union[slice, Sequence[int], pd.Timedelta]] = None,
      ensemble_members: Optional[Union[int, Sequence[int]]] = None,
      buffer: Optional[float] = None,
      compute: bool = False,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Loads a geographically bounded data cube from the Icechunk store.

    Args:
      watersheds: Basin geometries or bounding box in EPSG:4326.
      variables: List of variable names to select. If None, selects all variables.
        Pruning variables before slicing dramatically accelerates Dask graph construction.
      start_date: Optional start date filter for the time dimension.
      end_date: Optional end date filter for the time dimension.
      lead_time_slice: Optional slice or step selection for forecast lead times.
      ensemble_members: Optional selection of ensemble member indices.
      buffer: Spatial buffer in degrees (geographic) or meters (projected).
      compute: If True, executes  eagerly and loads into memory.
        If False, returns lazy xarray Dataset backed by Icechunk/Dask.
      use_bounding_box: If True, restricts spatial domain to watershed bounds.

    Returns:
      xr.Dataset spatially clipped to the watershed bounds.
    """
    ds = self.ds

    # 1. Prune variables on the lazy dataset first to minimize chunk index overhead
    if variables is not None:
      if isinstance(variables, str):
        variables = [variables]
      valid_vars = [v for v in variables if v in ds.data_vars]
      if not valid_vars:
        raise KeyError(
            f"None of requested variables {variables} found in dataset {self.dataset_id}. "
            f"Available variables: {list(ds.data_vars.keys())}"
        )
      subset = ds[valid_vars]
    else:
      subset = ds

    # 2. Compute spatial bounding slices for Icechunk
    if use_bounding_box:
      spatial_slices = self.compute_spatial_slices(watersheds, buffer=buffer)
    else:
      spatial_slices = {}

    # 3. Build temporal selection kwargs
    time_kwargs: Dict[str, Any] = {}
    if start_date is not None or end_date is not None:
      start_ts = pd.to_datetime(start_date) if start_date is not None else None
      if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        if isinstance(end_date, str) and len(end_date) <= 10 and "T" not in end_date:
          end_ts = end_ts.replace(hour=23, minute=59, second=59)
      else:
        end_ts = None
      time_kwargs[self.time_dim] = slice(start_ts, end_ts)

    isel_kwargs: Dict[str, Any] = {}
    if self.has_lead_time and lead_time_slice is not None:
      if isinstance(lead_time_slice, int):
        isel_kwargs["lead_time"] = lead_time_slice
      elif isinstance(lead_time_slice, slice) and (
          isinstance(lead_time_slice.start, int)
          or isinstance(lead_time_slice.stop, int)
      ):
        isel_kwargs["lead_time"] = lead_time_slice
      elif (
          isinstance(lead_time_slice, (list, tuple, np.ndarray))
          and len(lead_time_slice) > 0
          and isinstance(lead_time_slice[0], (int, np.integer))
      ):
        isel_kwargs["lead_time"] = lead_time_slice
      else:
        time_kwargs["lead_time"] = lead_time_slice

    if self.has_ensemble and ensemble_members is not None:
      if isinstance(ensemble_members, (int, np.integer)):
        isel_kwargs["ensemble_member"] = ensemble_members
      elif (
          isinstance(ensemble_members, (list, tuple, np.ndarray))
          and len(ensemble_members) > 0
          and isinstance(ensemble_members[0], (int, np.integer))
      ):
        isel_kwargs["ensemble_member"] = ensemble_members
      else:
        time_kwargs["ensemble_member"] = ensemble_members

    # 4. Apply selection
    sub_ds = subset.sel(**spatial_slices, **time_kwargs)
    if isel_kwargs:
      sub_ds = sub_ds.isel(**isel_kwargs)

    # 5. Eager compute if requested
    if compute:
      sub_ds = sub_ds.compute()

    return sub_ds

  def extract_basin_timeseries(
      self,
      watersheds: Union[gpd.GeoDataFrame, str, Path],
      variables: Optional[Union[str, Sequence[str]]] = None,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      lead_time_slice: Optional[slice] = None,
      ensemble_members: Optional[Union[int, Sequence[int]]] = None,
      buffer: Optional[float] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Extracts catchment-averaged forcing timeseries for basins using exact zonal weighting.

    Args:
      watersheds: Basin GeoDataFrame or file path (GeoJSON / Shapefile) in EPSG:4326.
      variables: List of variables to extract.
      start_date: Optional start date filter.
      end_date: Optional end date filter.
      lead_time_slice: Optional forecast lead time slice.
      ensemble_members: Optional ensemble member filter.
      buffer: Spatial buffer for bounding box query.
      weights_matrix: Optional pre-calculated ZonalWeightMatrix.
      use_bounding_box: If True, restricts spatial domain to watershed bounds.

    Returns:
      xr.Dataset indexed by (basin, date) or (basin, date, lead_time) containing
      catchment-averaged values.
    """
    basins_gdf = (
        watersheds
        if isinstance(watersheds, gpd.GeoDataFrame)
        else load_basin_geometries(watersheds)
    )
    basin_ids = list(basins_gdf.index)

    # 1. Fetch geographically bounded slice from Icechunk
    sub_ds = self.load_spatial_subset(
        watersheds=basins_gdf,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        lead_time_slice=lead_time_slice,
        ensemble_members=ensemble_members,
        buffer=buffer,
        compute=True,
        use_bounding_box=use_bounding_box,
    )

    data_vars_to_process = list(sub_ds.data_vars.keys())
    if not data_vars_to_process:
      raise ValueError("No data variables available to extract.")

    # 2. Build or verify ZonalWeightMatrix on the geographically clipped subgrid
    if self.grid_type == "geographic_1d":
      sub_lats = sub_ds[self.lat_coord].values
      sub_lons = sub_ds[self.lon_coord].values

      if (
          weights_matrix is not None
          and weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
          and np.allclose(weights_matrix.lats, sub_lats)
          and np.allclose(weights_matrix.lons, sub_lons)
      ):
        matrix = weights_matrix
      else:
        dlat = abs(float(sub_lats[1] - sub_lats[0])) if len(sub_lats) > 1 else 0.1
        dlon = abs(float(sub_lons[1] - sub_lons[0])) if len(sub_lons) > 1 else 0.1
        matrix = ZonalWeightMatrix.from_geodataframe(
            basins_gdf, sub_lats, sub_lons, cell_res_lat=dlat, cell_res_lon=dlon
        )

    else:
      # Projected 2D Grid: Reproject basins into native CRS where (y, x) is orthogonal Cartesian
      assert self.crs_wkt is not None
      basins_proj = basins_gdf.to_crs(self.crs_wkt)
      sub_y = sub_ds[self.y_coord].values
      sub_x = sub_ds[self.x_coord].values

      dy = abs(float(sub_y[1] - sub_y[0])) if len(sub_y) > 1 else 3000.0
      dx = abs(float(sub_x[1] - sub_x[0])) if len(sub_x) > 1 else 3000.0

      matrix = ZonalWeightMatrix.from_geodataframe(
          basins_proj, sub_y, sub_x, cell_res_lat=dy, cell_res_lon=dx
      )

    # 3. Reduce variables using sparse matrix multiplication
    reduced_data: Dict[str, np.ndarray] = {}

    for var_name in data_vars_to_process:
      var_arr = sub_ds[var_name].values
      ndim = var_arr.ndim

      # Standard nowcast 3D: (time, lat, lon) or (time, y, x)
      if ndim == 3:
        reduced = matrix.reduce_3d(var_arr)
        reduced_data[var_name] = reduced

      # Forecast 4D: (init_time, lead_time, lat, lon)
      elif ndim == 4:
        n_time, n_lead, _, _ = var_arr.shape
        flat_3d = var_arr.reshape((n_time * n_lead, len(matrix.lats), len(matrix.lons)))
        reduced_flat = matrix.reduce_3d(flat_3d)
        reduced = reduced_flat.reshape((len(basin_ids), n_time, n_lead))
        reduced_data[var_name] = reduced

      # Ensemble forecast 5D: (init_time, ensemble, lead_time, lat, lon)
      elif ndim == 5:
        n_time, n_ens, n_lead, _, _ = var_arr.shape
        flat_3d = var_arr.reshape(
            (n_time * n_ens * n_lead, len(matrix.lats), len(matrix.lons))
        )
        reduced_flat = matrix.reduce_3d(flat_3d)
        reduced = reduced_flat.reshape((len(basin_ids), n_time, n_ens, n_lead))
        reduced_data[var_name] = reduced

      else:
        logger.warning(
            "Variable %s has unsupported shape %s for zonal reduction. Skipping.",
            var_name,
            var_arr.shape,
        )

    # 4. Construct output xarray Dataset
    out_coords: Dict[str, Any] = {"basin": basin_ids}
    time_coords = sub_ds[self.time_dim].values
    out_coords["date"] = time_coords

    if self.has_lead_time and "lead_time" in sub_ds.coords:
      out_coords["lead_time"] = sub_ds.lead_time.values

    if self.has_ensemble and "ensemble_member" in sub_ds.coords:
      out_coords["ensemble_member"] = sub_ds.ensemble_member.values

    out_vars: Dict[str, Any] = {}
    for var_name, red_arr in reduced_data.items():
      if red_arr.ndim == 2:
        out_vars[var_name] = (["basin", "date"], red_arr)
      elif red_arr.ndim == 3:
        out_vars[var_name] = (["basin", "date", "lead_time"], red_arr)
      elif red_arr.ndim == 4:
        out_vars[var_name] = (
            ["basin", "date", "ensemble_member", "lead_time"],
            red_arr,
        )

    result_ds = xr.Dataset(data_vars=out_vars, coords=out_coords)
    result_ds.attrs.update(sub_ds.attrs)
    result_ds.attrs["dynamical_dataset_id"] = self.dataset_id
    return result_ds


class DynamicalExtractor(BaseExtractor):
  """MultiMet BaseExtractor adapter for any dynamical.org catalog dataset.

  Enables any dataset from dynamical.org to be used directly in the MultiMet
  extraction and Caravan harmonization pipelines.
  """

  def __init__(
      self,
      dataset_id: str,
      product: Optional[Product] = None,
      variable_map: Optional[Mapping[str, str]] = None,
      unit_conversions: Optional[Mapping[str, Any]] = None,
      loader: Optional[DynamicalDataLoader] = None,
  ):
    """Initializes the dynamical extractor.

    Args:
      dataset_id: Identifier of the dataset in dynamical.org.
      product: Optional MultiMet Product enum (if standardizing into Caravan).
      variable_map: Optional mapping from dynamical variable names to target band names.
      unit_conversions: Optional mapping of band name to conversion function.
      loader: Optional pre-configured DynamicalDataLoader.
    """
    prod = product if product is not None else Product.IMERG
    super().__init__(prod)
    self.dataset_id = dataset_id
    self.variable_map = dict(variable_map) if variable_map else {}
    self.unit_conversions = dict(unit_conversions) if unit_conversions else {}
    self.loader = (
        loader if loader is not None else DynamicalDataLoader(dataset_id)
    )

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      use_bounding_box: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts basin forcing time series from dynamical.org via Icechunk.

    Args:
      basins_gdf: Basin GeoDataFrame in EPSG:4326.
      start_date: Optional start date filter.
      end_date: Optional end date filter.

    Returns:
      xr.Dataset matching the Caravan/MultiMet schema.
    """
    selected_vars = list(self.variable_map.keys()) if self.variable_map else None
    extracted = self.loader.extract_basin_timeseries(
        watersheds=basins_gdf,
        variables=selected_vars,
        start_date=start_date,
        end_date=end_date,
        use_bounding_box=use_bounding_box,
    )

    if self.variable_map:
      rename_dict = {
          k: v for k, v in self.variable_map.items() if k in extracted.data_vars
      }
      extracted = extracted.rename(rename_dict)

    if self.unit_conversions:
      for band, conv_fn in self.unit_conversions.items():
        if band in extracted.data_vars:
          extracted[band].values = conv_fn(extracted[band].values)

    return extracted


class DynamicalIMERGExtractor(BaseExtractor):
  """MultiMet BaseExtractor for NASA IMERG Early via dynamical.org with Icechunk acceleration.

  Extracts half-hourly precipitation flux (kg m-2 s-1), converts to daily accumulated
  precipitation depth (mm/day), and applies exact zonal weighting over basin geometries.
  """

  def __init__(
      self,
      product: Optional[Product] = None,
      source: str = "dynamical",
      dataset_id: str = "nasa-imerg-analysis-early",
      loader: Optional[DynamicalDataLoader] = None,
      **kwargs,
  ):
    prod = product if product is not None else Product.DYNAMICAL_IMERG
    super().__init__(prod)
    self.source = source
    self.dataset_id = dataset_id
    self.loader = loader if loader is not None else DynamicalDataLoader(dataset_id)
    self.lats: Optional[np.ndarray] = None
    self.lons: Optional[np.ndarray] = None
    self._init_coords()

  def _init_coords(self) -> None:
    try:
      self.lats = self.loader.ds.latitude.values
      self.lons = self.loader.ds.longitude.values
    except Exception:
      pass

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      require_complete_day: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts daily accumulated IMERG precipitation for given basins."""
    basin_ids = list(basins_gdf.index)
    start_dt = (
        pd.to_datetime(start_date)
        if start_date is not None
        else pd.to_datetime("2000-06-01")
    )
    end_dt = pd.to_datetime(end_date) if end_date is not None else start_dt
    date_idx = pd.date_range(start_dt, end_dt, freq="D")

    precip_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )

    # Load geographically clipped half-hourly cube from Icechunk
    sub_ds = self.loader.load_spatial_subset(
        watersheds=basins_gdf,
        variables=["precipitation_surface"],
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        buffer=0.1,
        compute=True,
    )

    if "time" in sub_ds.dims and len(sub_ds.time) > 0:
      sub_lats = sub_ds.latitude.values
      sub_lons = sub_ds.longitude.values

      if (
          weights_matrix is not None
          and weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
          and np.allclose(weights_matrix.lats, sub_lats)
          and np.allclose(weights_matrix.lons, sub_lons)
      ):
        matrix = weights_matrix
      else:
        dlat = (
            abs(float(sub_lats[1] - sub_lats[0])) if len(sub_lats) > 1 else 0.1
        )
        dlon = (
            abs(float(sub_lons[1] - sub_lons[0])) if len(sub_lons) > 1 else 0.1
        )
        matrix = ZonalWeightMatrix.from_geodataframe(
            basins_gdf, sub_lats, sub_lons, cell_res_lat=dlat, cell_res_lon=dlon
        )

      raw_times = pd.DatetimeIndex(pd.to_datetime(sub_ds.time.values))
      day_step_counts = raw_times.floor("D").value_counts()

      # Convert half-hourly flux (kg m-2 s-1) to 30-min depth (mm), then resample to daily sum
      daily_depth = (
          (sub_ds["precipitation_surface"] * 1800.0)
          .resample(time="1D")
          .sum(dim="time", min_count=1)
      )
      reduced = matrix.reduce_3d(daily_depth.values)

      daily_times = pd.to_datetime(daily_depth.time.values)
      for t_idx, t_val in enumerate(daily_times):
        dt_day = pd.to_datetime(t_val.strftime("%Y-%m-%d"))
        if require_complete_day and day_step_counts.get(dt_day, 0) < 48:
          continue
        if dt_day in date_idx:
          d_pos = date_idx.get_loc(dt_day)
          precip_matrix[:, d_pos] = reduced[:, t_idx]

    ds = xr.Dataset(
        data_vars={
            "imerg_precipitation": (
                ["basin", "date"],
                precip_matrix.astype(np.float32),
            ),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    if self.product in PRODUCT_METADATA_ATTRS:
      ds.attrs.update(PRODUCT_METADATA_ATTRS[self.product])
    ds.attrs["dynamical_dataset_id"] = self.dataset_id
    return ds

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      **kwargs,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of IMERG precipitation across basins."""
    ds = self.extract_for_basins(
        basins_gdf, start_date=dt, end_date=dt, weights_matrix=weights_matrix
    )
    return {"imerg_precipitation": ds["imerg_precipitation"].values[:, 0]}


DYNAMICAL_FORECAST_DATASETS: Dict[str, Tuple[Product, str, str]] = {
    "AIFS": (Product.AIFS, "ecmwf-aifs-single-forecast", "aifs"),
    "GFS": (Product.GFS, "noaa-gfs-forecast", "gfs"),
    "GEFS": (Product.GEFS, "noaa-gefs-forecast-35-day", "gefs"),
    "IFS_ENS": (
        Product.IFS_ENS,
        "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "ifs_ens",
    ),
}


def _lead_times_to_seconds(lead_coord: Union[xr.DataArray, np.ndarray]) -> np.ndarray:
  """Converts a lead_time coordinate array into elapsed seconds from initialization."""
  vals = (
      lead_coord.values
      if isinstance(lead_coord, xr.DataArray)
      else np.asarray(lead_coord)
  )
  if np.issubdtype(vals.dtype, np.timedelta64):
    return (vals / np.timedelta64(1, "s")).astype(np.float64)
  return np.asarray(vals, dtype=np.float64) * 3600.0


def find_latest_dynamical_forecast_date(
    dataset_id: str = "ecmwf-aifs-single-forecast",
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    max_lookback_days: int = 7,
    require_full_10d: bool = True,
    lead_days: int = 10,
    loader: Optional[DynamicalDataLoader] = None,
) -> pd.Timestamp:
  """Finds the newest published 00z forecast initialization date in a dynamical.org store.

  Walks backwards from ``reference_date`` (defaulting to current UTC date) up to
  ``max_lookback_days`` and probes a single grid pixel at the 24h lead step (and
  at the terminal ``lead_days * 24h`` step when ``require_full_10d=True``) to
  verify that the 00:00:00 UTC initialization has been ingested and contains
  valid (finite) values.

  Args:
    dataset_id: Identifier of the forecast dataset in ``dynamical_catalog``
      (e.g. ``"ecmwf-aifs-single-forecast"``, ``"noaa-gfs-forecast"``).
    reference_date: Optional upper-bound date to start searching backwards from.
      Defaults to current UTC date.
    max_lookback_days: Maximum number of days to search backwards.
    require_full_10d: If True, requires both the 24h step and the terminal
      ``lead_days * 24h`` step to be finite.
    lead_days: Number of forecast lead days required when ``require_full_10d=True``.
    loader: Optional pre-opened :class:`DynamicalDataLoader`.

  Returns:
    Floor-normalized ``pd.Timestamp`` of the latest valid 00z forecast run.

  Raises:
    FileNotFoundError: If no valid 00z forecast initialization is found within
      the lookback window.
  """
  if reference_date is None:
    ref_dt = pd.Timestamp.now("UTC").tz_localize(None).floor("D")
  else:
    ts = pd.to_datetime(reference_date)
    if ts.tzinfo is not None:
      ts = ts.tz_convert("UTC").tz_localize(None)
    ref_dt = ts.floor("D")

  active_loader = (
      loader if loader is not None else DynamicalDataLoader(dataset_id)
  )
  ds = active_loader.ds
  time_dim = active_loader.time_dim
  if time_dim not in ds.coords:
    raise FileNotFoundError(
        f"Temporal coordinate {time_dim!r} not found in dynamical dataset {dataset_id!r}."
    )

  raw_times = pd.DatetimeIndex(
      pd.to_datetime(ds[time_dim].values).tz_localize(None)
  )
  earliest_dt = ref_dt - pd.Timedelta(days=max(0, int(max_lookback_days)))
  is_00z = (raw_times.hour == 0) & (raw_times.minute == 0)
  in_window = (raw_times >= earliest_dt) & (raw_times <= ref_dt + pd.Timedelta(hours=23, minutes=59))
  candidate_indices = np.where(is_00z & in_window)[0][::-1]

  if len(candidate_indices) == 0:
    raise FileNotFoundError(
        f"No 00z forecast initialization found in dynamical.org dataset "
        f"{dataset_id!r} within {max_lookback_days} days of "
        f"{ref_dt.strftime('%Y-%m-%d')}."
    )

  lead_sec = _lead_times_to_seconds(ds["lead_time"])
  probe_lead_indices: List[int] = [int(np.argmin(np.abs(lead_sec - 86400.0)))]
  if require_full_10d:
    target_sec = float(lead_days) * 86400.0
    idx_end = int(np.argmin(np.abs(lead_sec - target_sec)))
    if idx_end not in probe_lead_indices:
      probe_lead_indices.append(idx_end)

  probe_var = (
      "temperature_2m"
      if "temperature_2m" in ds.data_vars
      else next(iter(ds.data_vars))
  )
  sp_dim0, sp_dim1 = active_loader.spatial_dims

  for t_idx in candidate_indices:
    isel_kwargs: Dict[str, Any] = {
        time_dim: int(t_idx),
        "lead_time": probe_lead_indices,
        sp_dim0: 0,
        sp_dim1: 0,
    }
    if active_loader.has_ensemble and "ensemble_member" in ds[probe_var].dims:
      isel_kwargs["ensemble_member"] = 0
    try:
      vals = np.asarray(
          ds[probe_var].isel(**isel_kwargs).compute().values,
          dtype=np.float32,
      )
      if vals.size > 0 and bool(np.all(np.isfinite(vals))):
        return pd.Timestamp(raw_times[t_idx]).floor("D")
    except Exception as err:  # noqa: BLE001
      logger.debug(
          "Probe failed for %s at init_time=%s: %s",
          dataset_id,
          raw_times[t_idx],
          err,
      )

  raise FileNotFoundError(
      f"No populated 00z forecast run found in dynamical.org dataset "
      f"{dataset_id!r} within {max_lookback_days} days of "
      f"{ref_dt.strftime('%Y-%m-%d')} (require_full_10d={require_full_10d})."
  )


class DynamicalForecastExtractor(BaseExtractor):
  """MultiMet BaseExtractor for dynamical.org medium-range forecast datasets.

  Extracts multi-day forecasts initialized at 00:00:00 UTC from dynamical.org
  Icechunk stores (e.g. ECMWF AIFS, NOAA GFS, NOAA GEFS, ECMWF IFS ENS),
  aggregates sub-daily lead steps over each 24-hour window
  ``((d - 1) * 24h, d * 24h]`` into daily lead steps ``1..D`` weighted by step
  duration, and applies exact zonal weighting over basin geometries with
  area-weighted ``missing_fraction`` tracking.

  Supports both uniform sub-daily step spacing (e.g. 6-hourly AIFS, 3-hourly
  GEFS) and non-uniform step spacing (e.g. GFS 1-hourly steps for days 1..5
  followed by 3-hourly steps for days 6..10). Incomplete 24-hour windows or windows
  containing any ``NaN`` sub-daily step strictly evaluate to ``NaN`` (never
  imputing partial days).
  """

  DEFAULT_PRODUCT: Product = Product.AIFS
  DEFAULT_DATASET_ID: str = "ecmwf-aifs-single-forecast"
  DEFAULT_BAND_PREFIX: str = "aifs"

  def __init__(
      self,
      product: Optional[Union[Product, str]] = None,
      source: str = "dynamical",
      dataset_id: Optional[str] = None,
      band_prefix: Optional[str] = None,
      lead_days: Optional[int] = None,
      ensemble_member: Optional[Union[int, str]] = 0,
      data_dir: Optional[Union[str, os.PathLike]] = None,
      loader: Optional[DynamicalDataLoader] = None,
      **kwargs,
  ):
    if product is None:
      prod_enum = self.DEFAULT_PRODUCT
    elif isinstance(product, Product):
      prod_enum = product
    else:
      prod_enum = Product[str(product).strip().upper()]

    super().__init__(prod_enum)
    self.source = source.lower().strip() if source else "dynamical"
    self.data_dir = str(data_dir) if data_dir is not None else None

    default_meta = DYNAMICAL_FORECAST_DATASETS.get(prod_enum.value)
    self.dataset_id = (
        dataset_id
        or (default_meta[1] if default_meta else None)
        or self.DEFAULT_DATASET_ID
    )
    self.band_prefix = (
        band_prefix
        or (default_meta[2] if default_meta else None)
        or self.DEFAULT_BAND_PREFIX
    )
    self.lead_days = int(
        lead_days
        if lead_days is not None
        else FORECAST_LEAD_DAYS.get(prod_enum, 10)
    )
    self.ensemble_member = ensemble_member
    self.loader: Optional[DynamicalDataLoader] = loader
    self.lats: Optional[np.ndarray] = None
    self.lons: Optional[np.ndarray] = None

    if self.source != "archive":
      if self.loader is None:
        self.loader = DynamicalDataLoader(self.dataset_id)
      self._init_coords()

  def _init_coords(self) -> None:
    try:
      if self.loader is not None:
        self.lats = self.loader.ds.latitude.values
        self.lons = self.loader.ds.longitude.values
    except Exception:
      pass

  def _lead_slice_for_days(self, max_lead_days: int) -> slice:
    """Computes the integer lead_time index slice covering ``0h .. max_lead_days * 24h``."""
    target_sec = float(max_lead_days) * 86400.0
    try:
      if self.loader is not None and "lead_time" in self.loader.ds.coords:
        lead_sec = _lead_times_to_seconds(self.loader.ds["lead_time"])
        stop_idx = int(np.searchsorted(lead_sec, target_sec, side="right"))
        if stop_idx > 0:
          return slice(0, stop_idx)
    except Exception:
      pass
    return slice(0, max_lead_days * 4 + 1)

  def _extract_window(
      self,
      basins_gdf: gpd.GeoDataFrame,
      w_start: pd.Timestamp,
      w_end: pd.Timestamp,
      max_lead_days: int,
      date_idx: pd.DatetimeIndex,
      data_dict: Dict[str, np.ndarray],
      missing_fraction: np.ndarray,
      weights_matrix: Optional[ZonalWeightMatrix],
      use_bounding_box: bool,
  ) -> Optional[ZonalWeightMatrix]:
    """Loads and reduces a single temporal/lead window into ``data_dict`` and ``missing_fraction``."""
    assert self.loader is not None
    lead_slice = self._lead_slice_for_days(max_lead_days)

    ensemble_arg: Optional[Union[int, Sequence[int]]] = None
    if self.loader.has_ensemble and isinstance(
        self.ensemble_member, (int, np.integer)
    ):
      ensemble_arg = int(self.ensemble_member)

    sub_ds = self.loader.load_spatial_subset(
        watersheds=basins_gdf,
        variables=[
            "temperature_2m",
            "precipitation_surface",
            "wind_u_10m",
            "wind_v_10m",
        ],
        start_date=w_start.strftime("%Y-%m-%d"),
        end_date=w_end.strftime("%Y-%m-%d"),
        lead_time_slice=lead_slice,
        ensemble_members=ensemble_arg,
        buffer=0.1,
        compute=False,
        use_bounding_box=use_bounding_box,
    )

    if "init_time" in sub_ds.dims and len(sub_ds.init_time) > 0:
      init_times = pd.to_datetime(sub_ds.init_time.values)
      is_00z = (init_times.hour == 0) & (init_times.minute == 0)
      sub_ds = sub_ds.isel(init_time=is_00z)

    if "init_time" not in sub_ds.dims or len(sub_ds.init_time) == 0:
      return weights_matrix

    sub_ds = sub_ds.compute()

    if "ensemble_member" in sub_ds.dims:
      if str(self.ensemble_member).lower() == "mean":
        sub_ds = sub_ds.mean(dim="ensemble_member", skipna=False)
      else:
        sub_ds = sub_ds.isel(ensemble_member=0)

    if len(sub_ds.lead_time) < 2:
      return weights_matrix

    sub_lats = sub_ds.latitude.values
    sub_lons = sub_ds.longitude.values

    if (
        weights_matrix is not None
        and weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
        and np.allclose(weights_matrix.lats, sub_lats)
        and np.allclose(weights_matrix.lons, sub_lons)
    ):
      matrix = weights_matrix
    else:
      dlat = (
          abs(float(sub_lats[1] - sub_lats[0])) if len(sub_lats) > 1 else 0.25
      )
      dlon = (
          abs(float(sub_lons[1] - sub_lons[0])) if len(sub_lons) > 1 else 0.25
      )
      matrix = ZonalWeightMatrix.from_geodataframe(
          basins_gdf,
          sub_lats,
          sub_lons,
          cell_res_lat=dlat,
          cell_res_lon=dlon,
      )

    lead_sec = _lead_times_to_seconds(sub_ds["lead_time"])
    prev_sec = np.concatenate([[0.0], lead_sec[:-1]])
    dt_sec = lead_sec - prev_sec

    t2m_raw = np.asarray(sub_ds["temperature_2m"].values, dtype=np.float32)
    pr_raw = np.asarray(sub_ds["precipitation_surface"].values, dtype=np.float32)
    u_raw = np.asarray(sub_ds["wind_u_10m"].values, dtype=np.float32)
    v_raw = np.asarray(sub_ds["wind_v_10m"].values, dtype=np.float32)

    red_t2m, miss_t2m = matrix.reduce_4d_with_coverage(t2m_raw)
    red_pr, miss_pr = matrix.reduce_4d_with_coverage(pr_raw)
    red_u, miss_u = matrix.reduce_4d_with_coverage(u_raw)
    red_v, miss_v = matrix.reduce_4d_with_coverage(v_raw)
    step_miss = np.maximum.reduce([miss_t2m, miss_pr, miss_u, miss_v])

    t2m_band = f"{self.band_prefix}_temperature_2m"
    pr_band = f"{self.band_prefix}_total_precipitation"
    u_band = f"{self.band_prefix}_u_component_of_wind_10m"
    v_band = f"{self.band_prefix}_v_component_of_wind_10m"

    sub_init_times = pd.to_datetime(sub_ds.init_time.values)
    for lt_day in range(1, max_lead_days + 1):
      t_start = float(lt_day - 1) * 86400.0
      t_end = float(lt_day) * 86400.0
      step_idx = np.where((lead_sec > t_start) & (lead_sec <= t_end))[0]
      if len(step_idx) == 0:
        continue
      w_sec = dt_sec[step_idx].astype(np.float32)
      total_dt = float(np.sum(w_sec))
      if not np.isclose(total_dt, 86400.0, atol=1.0):
        continue
      w_norm = (w_sec / total_dt).astype(np.float32)
      lt_pos = lt_day - 1

      for t_idx, t_val in enumerate(sub_init_times):
        dt_day = pd.to_datetime(t_val.strftime("%Y-%m-%d"))
        if dt_day not in date_idx:
          continue
        d_pos = date_idx.get_loc(dt_day)

        # Strict duration-weighted aggregation (np.sum propagates NaN if any
        # sub-daily step in the 24h window is missing/NaN).
        day_t2m = np.sum(
            red_t2m[:, t_idx, step_idx] * w_norm[np.newaxis, :], axis=-1
        )
        day_pr = np.sum(
            red_pr[:, t_idx, step_idx] * w_sec[np.newaxis, :], axis=-1
        )
        day_pr = np.where(np.isnan(day_pr), np.nan, np.maximum(0.0, day_pr))
        day_u = np.sum(
            red_u[:, t_idx, step_idx] * w_norm[np.newaxis, :], axis=-1
        )
        day_v = np.sum(
            red_v[:, t_idx, step_idx] * w_norm[np.newaxis, :], axis=-1
        )
        day_miss = np.max(step_miss[:, t_idx, step_idx], axis=-1)
        any_nan = (
            np.isnan(day_t2m)
            | np.isnan(day_pr)
            | np.isnan(day_u)
            | np.isnan(day_v)
        )
        day_miss = np.where(any_nan, 1.0, day_miss).astype(np.float32)

        data_dict[t2m_band][:, d_pos, lt_pos] = day_t2m
        data_dict[pr_band][:, d_pos, lt_pos] = day_pr
        data_dict[u_band][:, d_pos, lt_pos] = day_u
        data_dict[v_band][:, d_pos, lt_pos] = day_v
        missing_fraction[:, d_pos, lt_pos] = day_miss

    return matrix

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      spinup_only_before: Optional[Union[str, pd.Timestamp]] = None,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts daily forecasts (lead days 1..D) for given basins."""
    if self.source == "archive":
      if not self.data_dir:
        raise ValueError(
            f"{self.__class__.__name__} requires an explicit data_dir URI for "
            "archive extraction."
        )
      from multimet.gridded_archive import extract_forecast_from_archive

      return extract_forecast_from_archive(
          self.product,
          self.data_dir,
          basins_gdf,
          start_date=start_date,
          end_date=end_date,
          weights_matrix=weights_matrix,
          use_bounding_box=use_bounding_box,
      )

    basin_ids = list(basins_gdf.index)
    start_dt = (
        pd.to_datetime(start_date).floor("D")
        if start_date is not None
        else pd.to_datetime("2024-05-01")
    )
    end_dt = (
        pd.to_datetime(end_date).floor("D")
        if end_date is not None
        else start_dt
    )
    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    lead_steps = self.lead_days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps + 1), unit="D")

    shape = (len(basin_ids), len(date_idx), lead_steps)
    expected_bands = PRODUCT_BANDS.get(
        self.product,
        (
            f"{self.band_prefix}_temperature_2m",
            f"{self.band_prefix}_total_precipitation",
            f"{self.band_prefix}_u_component_of_wind_10m",
            f"{self.band_prefix}_v_component_of_wind_10m",
        ),
    )
    data_dict: Dict[str, np.ndarray] = {
        band: np.full(shape, np.nan, dtype=np.float32)
        for band in expected_bands
    }
    missing_fraction = np.ones(shape, dtype=np.float32)

    windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if spinup_only_before is not None:
      cutoff_dt = pd.to_datetime(spinup_only_before).floor("D")
      if start_dt < cutoff_dt:
        spinup_end = min(end_dt, cutoff_dt - pd.Timedelta(days=1))
        windows.append((start_dt, spinup_end, 1))
      if end_dt >= cutoff_dt:
        forecast_start = max(start_dt, cutoff_dt)
        windows.append((forecast_start, end_dt, lead_steps))
    else:
      windows.append((start_dt, end_dt, lead_steps))

    active_matrix = weights_matrix
    for w_start, w_end, w_max_leads in windows:
      active_matrix = self._extract_window(
          basins_gdf=basins_gdf,
          w_start=w_start,
          w_end=w_end,
          max_lead_days=w_max_leads,
          date_idx=date_idx,
          data_dict=data_dict,
          missing_fraction=missing_fraction,
          weights_matrix=active_matrix,
          use_bounding_box=use_bounding_box,
      )

    data_vars: Dict[str, Any] = {
        band: (
            ["basin", "date", "lead_time"],
            data_dict[band].astype(np.float32),
        )
        for band in expected_bands
    }
    missing_var = MISSING_FRACTION_VAR.get(
        self.product, f"{self.band_prefix}_missing_fraction"
    )
    data_vars[missing_var] = (
        ["basin", "date", "lead_time"],
        missing_fraction.astype(np.float32),
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
            "lead_time": lead_time_idx.values,
        },
    )
    if self.product in PRODUCT_METADATA_ATTRS:
      ds.attrs.update(PRODUCT_METADATA_ATTRS[self.product])
    ds.attrs["dynamical_dataset_id"] = self.dataset_id
    return ds

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      **kwargs,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 forecast initialization date across lead days."""
    ds = self.extract_for_basins(
        basins_gdf,
        start_date=dt,
        end_date=dt,
        weights_matrix=weights_matrix,
        **kwargs,
    )
    return {band: ds[band].values[:, 0, :] for band in ds.data_vars}


class AIFSExtractor(DynamicalForecastExtractor):
  """MultiMet BaseExtractor for ECMWF AIFS single-forecast via dynamical.org.

  Extracts 10-day medium-range forecasts initialized at 00:00:00 UTC,
  aggregates 6-hourly lead steps (steps 1..40) into 10 daily lead steps,
  and applies exact zonal weighting over basin geometries.
  """

  DEFAULT_PRODUCT = Product.AIFS
  DEFAULT_DATASET_ID = "ecmwf-aifs-single-forecast"
  DEFAULT_BAND_PREFIX = "aifs"


class GFSExtractor(DynamicalForecastExtractor):
  """MultiMet BaseExtractor for NOAA GFS operational forecasts via dynamical.org.

  Extracts 10-day medium-range forecasts initialized at 00:00:00 UTC from
  ``noaa-gfs-forecast``, duration-weighting 1-hourly steps (lead days 1..5,
  ``1h..120h``) and 3-hourly steps (lead days 6..10, ``123h..240h``) into 10
  daily lead steps, and applies exact zonal weighting over basin geometries.
  """

  DEFAULT_PRODUCT = Product.GFS
  DEFAULT_DATASET_ID = "noaa-gfs-forecast"
  DEFAULT_BAND_PREFIX = "gfs"


class GEFSExtractor(DynamicalForecastExtractor):
  """MultiMet BaseExtractor for NOAA GEFS forecasts via dynamical.org.

  Extracts 10-day forecasts initialized at 00:00:00 UTC from
  ``noaa-gefs-forecast-35-day`` (selecting control member ``0`` by default, or
  ``ensemble_member="mean"`` for the 31-member ensemble mean), aggregating
  3-hourly steps into 10 daily lead steps.
  """

  DEFAULT_PRODUCT = Product.GEFS
  DEFAULT_DATASET_ID = "noaa-gefs-forecast-35-day"
  DEFAULT_BAND_PREFIX = "gefs"


class IFSEnsExtractor(DynamicalForecastExtractor):
  """MultiMet BaseExtractor for ECMWF IFS Ensemble (ENS) forecasts via dynamical.org.

  Extracts 10-day forecasts initialized at 00:00:00 UTC from
  ``ecmwf-ifs-ens-forecast-15-day-0-25-degree`` (selecting control member ``0``
  by default, or ``ensemble_member="mean"`` for the 51-member ensemble mean),
  duration-weighting 3-hourly steps (lead days 1..6) and 6-hourly steps (lead
  days 7..10) into 10 daily lead steps.
  """

  DEFAULT_PRODUCT = Product.IFS_ENS
  DEFAULT_DATASET_ID = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
  DEFAULT_BAND_PREFIX = "ifs_ens"


def load_dynamical(
    dataset_id: str,
    watersheds: Union[
        gpd.GeoDataFrame,
        shapely.geometry.base.BaseGeometry,
        Sequence[float],
        Tuple[float, float, float, float],
        str,
        Path,
    ],
    variables: Optional[Union[str, Sequence[str]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    lead_time_slice: Optional[Union[slice, Sequence[int]]] = None,
    buffer: Optional[float] = None,
    mode: str = "cube",
    compute: bool = True,
) -> xr.Dataset:
  """Convenience function to load geographically-bounded data from dynamical.org.

  Args:
    dataset_id: Name of dataset in the dynamical.org catalog.
    watersheds: Watershed geometries (GeoDataFrame, shapefile/geojson path,
      Shapely geometry, or bbox tuple).
    variables: Optional list of variables to retrieve.
    start_date: Optional start date filter.
    end_date: Optional end date filter.
    lead_time_slice: Optional forecast lead time slice.
    buffer: Optional spatial buffer around watershed bounds.
    mode: 'cube' to return the geographically-bounded gridded dataset, or
      'timeseries' to return catchment-averaged basin timeseries.
    compute: Whether to compute eagerly into memory (default True).

  Returns:
    xr.Dataset clipped to the watershed spatial domain.
  """
  loader = DynamicalDataLoader(dataset_id)
  if mode == "timeseries":
    return loader.extract_basin_timeseries(
        watersheds=watersheds,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        lead_time_slice=lead_time_slice,
        buffer=buffer,
    )
  else:
    return loader.load_spatial_subset(
        watersheds=watersheds,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        lead_time_slice=lead_time_slice,
        buffer=buffer,
        compute=compute,
    )
