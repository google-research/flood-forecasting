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
from multimet.config import Product, ProductType
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
    spatial_slices = self.compute_spatial_slices(watersheds, buffer=buffer)

    # 3. Build temporal selection kwargs
    time_kwargs: Dict[str, Any] = {}
    if start_date is not None or end_date is not None:
      start_ts = pd.to_datetime(start_date) if start_date is not None else None
      end_ts = pd.to_datetime(end_date) if end_date is not None else None
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
