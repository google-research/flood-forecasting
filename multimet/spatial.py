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

"""Spatial bounding box utilities and raster grid slicing for MultiMet.

Provides geographic bounding box abstractions, coordinate transforms,
and spatial slicing for xarray Datasets and DataArrays to optimize
geographically constrained cloud I/O.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import shapely.geometry
import xarray as xr


@dataclasses.dataclass(frozen=True)
class BoundingBox:
  """Geographic bounding box in WGS84 coordinates (EPSG:4326).

  Attributes:
    min_lon: Minimum longitude in degrees [-180.0, 180.0].
    min_lat: Minimum latitude in degrees [-90.0, 90.0].
    max_lon: Maximum longitude in degrees [-180.0, 180.0].
    max_lat: Maximum latitude in degrees [-90.0, 90.0].
  """

  min_lon: float
  min_lat: float
  max_lon: float
  max_lat: float

  def __post_init__(self):
    min_lat = float(self.min_lat)
    max_lat = float(self.max_lat)
    min_lon = float(self.min_lon)
    max_lon = float(self.max_lon)

    if min_lat > max_lat:
      raise ValueError(
          f"min_lat ({min_lat}) cannot be greater than max_lat ({max_lat})"
      )
    if min_lat < -90.0 or max_lat > 90.0:
      raise ValueError(
          f"Latitude bounds [{min_lat}, {max_lat}] out of range [-90, 90]"
      )

    # Use object.__setattr__ because class is frozen
    object.__setattr__(self, "min_lat", min_lat)
    object.__setattr__(self, "max_lat", max_lat)
    object.__setattr__(self, "min_lon", min_lon)
    object.__setattr__(self, "max_lon", max_lon)

  @property
  def crosses_antimeridian(self) -> bool:
    """Returns True if the bounding box crosses the ±180° antimeridian (dateline)."""
    return self.min_lon > self.max_lon

  @property
  def crosses_prime_meridian(self) -> bool:
    """Returns True if the bounding box spans across the 0° prime meridian."""
    return self.min_lon < 0.0 < self.max_lon and not self.crosses_antimeridian

  @property
  def lon_span(self) -> float:
    """Total longitudinal span in degrees [0, 360]."""
    if self.crosses_antimeridian:
      return (180.0 - self.min_lon) + (self.max_lon - (-180.0))
    return self.max_lon - self.min_lon

  @property
  def lat_span(self) -> float:
    """Total latitudinal span in degrees [0, 180]."""
    return self.max_lat - self.min_lat

  def is_global(self, threshold_deg: float = 350.0) -> bool:
    """Returns True if the bounding box substantially covers the entire globe."""
    return self.lon_span >= threshold_deg

  def to_tuple(self) -> Tuple[float, float, float, float]:
    """Returns (min_lon, min_lat, max_lon, max_lat)."""
    return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

  def as_dict(self) -> Dict[str, float]:
    """Returns bounds as a dictionary."""
    return {
        "min_lon": self.min_lon,
        "min_lat": self.min_lat,
        "max_lon": self.max_lon,
        "max_lat": self.max_lat,
    }

  def buffer(self, degrees: float = 0.5) -> BoundingBox:
    """Expands the bounding box by a given buffer in degrees, clamped to valid coordinates."""
    deg = abs(float(degrees))
    new_min_lat = max(-90.0, self.min_lat - deg)
    new_max_lat = min(90.0, self.max_lat + deg)

    if self.crosses_antimeridian:
      new_min_lon = self.min_lon - deg
      new_max_lon = self.max_lon + deg
      if new_min_lon <= new_max_lon or new_min_lon <= -180.0 or new_max_lon >= 180.0:
        return BoundingBox(-180.0, new_min_lat, 180.0, new_max_lat)
      return BoundingBox(new_min_lon, new_min_lat, new_max_lon, new_max_lat)

    new_min_lon = max(-180.0, self.min_lon - deg)
    new_max_lon = min(180.0, self.max_lon + deg)
    if (new_max_lon - new_min_lon) >= 360.0:
      new_min_lon, new_max_lon = -180.0, 180.0

    return BoundingBox(new_min_lon, new_min_lat, new_max_lon, new_max_lat)

  def to_0_360_ranges(self) -> List[Tuple[float, float]]:
    """Returns longitude ranges [min_lon, max_lon] mapped to the [0, 360) convention.

    If the bounding box crosses the 0° prime meridian (e.g. [-5, 5]), returns
    two non-overlapping ranges: [(355.0, 360.0), (0.0, 5.0)].
    """
    if self.is_global():
      return [(0.0, 360.0)]

    if self.crosses_antimeridian:
      # e.g., 170 to -170 in [-180, 180] maps to 170 to 190 in [0, 360]
      return [(self.min_lon % 360.0, self.max_lon % 360.0)]

    if self.crosses_prime_meridian:
      # e.g. -5 to 5 maps to [355, 360] and [0, 5]
      return [((self.min_lon % 360.0), 360.0), (0.0, self.max_lon % 360.0)]

    # Standard positive or negative range
    return [((self.min_lon % 360.0), (self.max_lon % 360.0))]

  @classmethod
  def from_geodataframe(
      cls,
      gdf: gpd.GeoDataFrame,
      buffer_degrees: float = 0.0,
  ) -> BoundingBox:
    """Computes total bounding box from a GeoDataFrame with optional buffer."""
    if gdf.empty:
      raise ValueError("Cannot compute bounding box for empty GeoDataFrame.")
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = cls(float(minx), float(miny), float(maxx), float(maxy))
    if buffer_degrees > 0.0:
      bbox = bbox.buffer(buffer_degrees)
    return bbox

  @classmethod
  def from_geometry(
      cls,
      geom: shapely.geometry.base.BaseGeometry,
      buffer_degrees: float = 0.0,
  ) -> BoundingBox:
    """Computes bounding box from a Shapely geometry with optional buffer."""
    if geom.is_empty:
      raise ValueError("Cannot compute bounding box for empty geometry.")
    minx, miny, maxx, maxy = geom.bounds
    bbox = cls(float(minx), float(miny), float(maxx), float(maxy))
    if buffer_degrees > 0.0:
      bbox = bbox.buffer(buffer_degrees)
    return bbox

  @classmethod
  def from_tuple(cls, bounds: Sequence[float]) -> BoundingBox:
    """Creates a BoundingBox from (min_lon, min_lat, max_lon, max_lat)."""
    if len(bounds) != 4:
      raise ValueError(
          "Expected 4 bounds elements (min_lon, min_lat, max_lon, max_lat), "
          f"got {len(bounds)}"
      )
    return cls(
        float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    )


def find_lat_lon_dims(
    obj: Union[xr.Dataset, xr.DataArray],
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
) -> Tuple[str, str]:
  """Detects latitude and longitude coordinate/dimension names in an xarray object."""
  if lat_dim is not None and lon_dim is not None:
    return lat_dim, lon_dim

  coords = list(obj.coords) if hasattr(obj, "coords") else list(obj.dims)
  dims = list(obj.dims)

  candidates_lat = ["latitude", "lat", "LATITUDE", "LAT", "y", "Y"]
  candidates_lon = ["longitude", "lon", "LONGITUDE", "LON", "x", "X"]

  found_lat = lat_dim
  if found_lat is None:
    for c in candidates_lat:
      if c in coords or c in dims:
        found_lat = c
        break
  if found_lat is None:
    raise ValueError(
        f"Could not automatically detect latitude dimension among {coords} / {dims}"
    )

  found_lon = lon_dim
  if found_lon is None:
    for c in candidates_lon:
      if c in coords or c in dims:
        found_lon = c
        break
  if found_lon is None:
    raise ValueError(
        f"Could not automatically detect longitude dimension among {coords} / {dims}"
    )

  return found_lat, found_lon


def slice_coordinates_by_bounds(
    lats: np.ndarray,
    lons: np.ndarray,
    bounds: Union[BoundingBox, Sequence[float], gpd.GeoDataFrame],
    buffer_degrees: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Slices 1D latitude and longitude coordinate arrays to a bounding box.

  Args:
    lats: 1D array of latitude coordinates.
    lons: 1D array of longitude coordinates.
    bounds: BoundingBox, 4-element sequence, or GeoDataFrame.
    buffer_degrees: Optional buffer degrees to expand the box.

  Returns:
    Tuple of (sub_lats, sub_lons, lat_indices, lon_indices).
  """
  if isinstance(bounds, gpd.GeoDataFrame):
    bbox = BoundingBox.from_geodataframe(bounds, buffer_degrees=buffer_degrees)
  elif isinstance(bounds, BoundingBox):
    bbox = bounds.buffer(buffer_degrees) if buffer_degrees > 0.0 else bounds
  else:
    bbox = BoundingBox.from_tuple(bounds)
    if buffer_degrees > 0.0:
      bbox = bbox.buffer(buffer_degrees)

  lats = np.asarray(lats, dtype=np.float64)
  lons = np.asarray(lons, dtype=np.float64)

  lat_mask = (lats >= bbox.min_lat) & (lats <= bbox.max_lat)
  lat_indices = np.where(lat_mask)[0]
  if len(lat_indices) == 0:
    # Nearest fallback if grid is coarser than bounds
    nearest_lat = np.argmin(np.abs(lats - (bbox.min_lat + bbox.max_lat) / 2.0))
    lat_indices = np.array([nearest_lat], dtype=np.int32)
  sub_lats = lats[lat_indices]

  is_lon_360 = bool(
      np.any(lons > 180.0)
      or (np.all(lons >= 0.0) and len(lons) > 1 and lons[-1] > 180.0)
  )

  if is_lon_360:
    ranges = bbox.to_0_360_ranges()
    lon_mask = np.zeros(len(lons), dtype=bool)
    for min_l, max_l in ranges:
      lon_mask |= (lons >= min_l) & (lons <= max_l)
    lon_indices = np.where(lon_mask)[0]
  else:
    if bbox.crosses_antimeridian:
      lon_mask = (lons >= bbox.min_lon) | (lons <= bbox.max_lon)
    else:
      lon_mask = (lons >= bbox.min_lon) & (lons <= bbox.max_lon)
    lon_indices = np.where(lon_mask)[0]

  if len(lon_indices) == 0:
    nearest_lon = np.argmin(np.abs(lons - (bbox.min_lon + bbox.max_lon) / 2.0))
    lon_indices = np.array([nearest_lon], dtype=np.int32)
  sub_lons = lons[lon_indices]

  return sub_lats, sub_lons, lat_indices, lon_indices


def slice_dataset_by_bounds(
    ds: Union[xr.Dataset, xr.DataArray],
    bounds: Union[BoundingBox, Sequence[float], gpd.GeoDataFrame],
    buffer_degrees: float = 0.0,
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
    target_lon_range: str = "auto",
) -> Union[xr.Dataset, xr.DataArray]:
  """Slices an xarray Dataset or DataArray geographically to the requested bounding box.

  Handles:
  1. Latitude direction: Automatically determines if latitude is ascending or descending
     and constructs the appropriate slice so the selection is non-empty.
  2. Longitude representations: Automatically determines if the dataset uses [0, 360)
     or [-180, 180) and handles conversions, prime-meridian crossings, and antimeridian
     crossings seamlessly.
  3. Global bypass: If the bounding box is global (spans >= 350°), preserves the
     dataset without unnecessary copying or slicing.

  Args:
    ds: xarray Dataset or DataArray to slice.
    bounds: BoundingBox instance, tuple (min_lon, min_lat, max_lon, max_lat), or GeoDataFrame.
    buffer_degrees: Optional buffer in degrees to expand the bounding box.
    lat_dim: Optional name of latitude dimension (auto-detected if None).
    lon_dim: Optional name of longitude dimension (auto-detected if None).
    target_lon_range: Convention for returned longitudes ('auto', 'minus180_180', '0_360').

  Returns:
    Spatially subsetted xarray object.
  """
  if isinstance(bounds, gpd.GeoDataFrame):
    bbox = BoundingBox.from_geodataframe(bounds, buffer_degrees=buffer_degrees)
  elif isinstance(bounds, BoundingBox):
    bbox = bounds.buffer(buffer_degrees) if buffer_degrees > 0.0 else bounds
  else:
    bbox = BoundingBox.from_tuple(bounds)
    if buffer_degrees > 0.0:
      bbox = bbox.buffer(buffer_degrees)

  lat_name, lon_name = find_lat_lon_dims(ds, lat_dim, lon_dim)
  lat_coords = np.asarray(ds[lat_name].values, dtype=np.float64)
  lon_coords = np.asarray(ds[lon_name].values, dtype=np.float64)

  # Check if global
  if bbox.is_global() and bbox.min_lat <= -89.0 and bbox.max_lat >= 89.0:
    return ds

  # 1. Determine latitude slice
  is_lat_descending = lat_coords[0] > lat_coords[-1]
  if is_lat_descending:
    lat_slice = slice(bbox.max_lat, bbox.min_lat)
  else:
    lat_slice = slice(bbox.min_lat, bbox.max_lat)

  # 2. Determine longitude slice
  is_ds_360 = bool(
      np.any(lon_coords > 180.0)
      or (np.all(lon_coords >= 0.0) and len(lon_coords) > 1 and lon_coords[-1] > 180.0)
  )

  if is_ds_360:
    ranges_360 = bbox.to_0_360_ranges()
    if len(ranges_360) == 1:
      min_l, max_l = ranges_360[0]
      sub = ds.sel({lat_name: lat_slice, lon_name: slice(min_l, max_l)})
    else:
      # Crosses 0° Prime Meridian in [0, 360): e.g. [355, 360] and [0, 5]
      sub1 = ds.sel({lat_name: lat_slice, lon_name: slice(ranges_360[0][0], ranges_360[0][1])})
      sub2 = ds.sel({lat_name: lat_slice, lon_name: slice(ranges_360[1][0], ranges_360[1][1])})
      sub = xr.concat([sub1, sub2], dim=lon_name)

    # Standardize to [-180, 180] unless caller explicitly requested '0_360'
    if target_lon_range in ("auto", "minus180_180"):
      sub_lons = sub[lon_name].values
      converted_lons = np.where(sub_lons > 180.0, sub_lons - 360.0, sub_lons)
      sub = sub.assign_coords({lon_name: converted_lons}).sortby(lon_name)

  else:
    # Dataset is [-180, 180)
    if not bbox.crosses_antimeridian:
      sub = ds.sel({lat_name: lat_slice, lon_name: slice(bbox.min_lon, bbox.max_lon)})
    else:
      # Crosses 180° dateline: e.g. [175, 180] and [-180, -175]
      sub1 = ds.sel({lat_name: lat_slice, lon_name: slice(bbox.min_lon, 180.0)})
      sub2 = ds.sel({lat_name: lat_slice, lon_name: slice(-180.0, bbox.max_lon)})
      sub = xr.concat([sub1, sub2], dim=lon_name)

    if target_lon_range == "0_360":
      sub_lons = sub[lon_name].values
      converted_lons = sub_lons % 360.0
      sub = sub.assign_coords({lon_name: converted_lons}).sortby(lon_name)

  # Maintain consistent latitude sorting matching original dataset
  if is_lat_descending:
    sub = sub.sortby(lat_name, ascending=False)
  else:
    sub = sub.sortby(lat_name, ascending=True)

  return sub
