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

"""Unit and integration tests for MultiMet spatial bounding box utilities."""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry
import xarray as xr

from multimet.geometry import load_basin_geometries
from multimet.spatial import (
    BoundingBox,
    find_lat_lon_dims,
    slice_coordinates_by_bounds,
    slice_dataset_by_bounds,
)
from multimet.zonal import ZonalWeightMatrix


@pytest.fixture
def geojson_path() -> Path:
  path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  assert path.exists(), f"Missing test geojson at {path}"
  return path


def test_bounding_box_init_and_validation():
  bbox = BoundingBox(min_lon=-88.0, min_lat=39.0, max_lon=-86.0, max_lat=41.0)
  assert bbox.min_lon == -88.0
  assert bbox.min_lat == 39.0
  assert bbox.max_lon == -86.0
  assert bbox.max_lat == 41.0
  assert bbox.lon_span == 2.0
  assert bbox.lat_span == 2.0
  assert not bbox.is_global()
  assert not bbox.crosses_prime_meridian
  assert not bbox.crosses_antimeridian

  # Invalid latitude range
  with pytest.raises(ValueError, match="cannot be greater than max_lat"):
    BoundingBox(min_lon=0.0, min_lat=50.0, max_lon=10.0, max_lat=40.0)

  with pytest.raises(ValueError, match="out of range"):
    BoundingBox(min_lon=0.0, min_lat=-95.0, max_lon=10.0, max_lat=40.0)


def test_bounding_box_meridian_crossings():
  # Prime meridian crossing
  pm_box = BoundingBox(min_lon=-3.0, min_lat=40.0, max_lon=5.0, max_lat=50.0)
  assert pm_box.crosses_prime_meridian
  assert not pm_box.crosses_antimeridian
  assert pm_box.lon_span == 8.0
  ranges_360 = pm_box.to_0_360_ranges()
  assert len(ranges_360) == 2
  assert ranges_360[0] == (357.0, 360.0)
  assert ranges_360[1] == (0.0, 5.0)

  # Antimeridian crossing (e.g. Fiji / Pacific)
  am_box = BoundingBox(min_lon=175.0, min_lat=-20.0, max_lon=-175.0, max_lat=-15.0)
  assert am_box.crosses_antimeridian
  assert not am_box.crosses_prime_meridian
  assert am_box.lon_span == 10.0
  am_ranges_360 = am_box.to_0_360_ranges()
  assert len(am_ranges_360) == 1
  assert am_ranges_360[0] == (175.0, 185.0)

  # Global box
  global_box = BoundingBox(min_lon=-180.0, min_lat=-90.0, max_lon=180.0, max_lat=90.0)
  assert global_box.is_global()
  assert global_box.to_0_360_ranges() == [(0.0, 360.0)]


def test_bounding_box_buffer_and_clamp():
  bbox = BoundingBox(min_lon=-88.0, min_lat=-89.8, max_lon=-86.0, max_lat=89.8)
  buf = bbox.buffer(0.5)
  assert buf.min_lat == -90.0  # Clamped
  assert buf.max_lat == 90.0   # Clamped
  assert buf.min_lon == -88.5
  assert buf.max_lon == -85.5


def test_bounding_box_from_geodataframe(geojson_path: Path):
  gdf = load_basin_geometries(geojson_path)
  bbox = BoundingBox.from_geodataframe(gdf, buffer_degrees=0.5)
  assert bbox.min_lon < -87.0
  assert bbox.max_lon > -86.0
  assert bbox.min_lat < 39.0
  assert bbox.max_lat > 40.0

  tup = bbox.to_tuple()
  assert len(tup) == 4
  assert tup[0] == bbox.min_lon

  d = bbox.as_dict()
  assert d["min_lon"] == bbox.min_lon


def test_find_lat_lon_dims():
  ds1 = xr.Dataset(coords={"latitude": [1, 2], "longitude": [3, 4]})
  assert find_lat_lon_dims(ds1) == ("latitude", "longitude")

  ds2 = xr.Dataset(coords={"lat": [1, 2], "lon": [3, 4]})
  assert find_lat_lon_dims(ds2) == ("lat", "lon")

  ds3 = xr.Dataset(coords={"y": [1, 2], "x": [3, 4]})
  assert find_lat_lon_dims(ds3) == ("y", "x")

  # Manual override
  assert find_lat_lon_dims(ds1, lat_dim="latitude", lon_dim="longitude") == ("latitude", "longitude")

  with pytest.raises(ValueError, match="Could not automatically detect"):
    find_lat_lon_dims(xr.Dataset(coords={"a": [1], "b": [2]}))


def test_slice_coordinates_by_bounds():
  lats_desc = np.linspace(90.0, -90.0, 181)
  lons_wgs = np.linspace(-180.0, 179.0, 360)
  bbox = BoundingBox(min_lon=-88.0, min_lat=39.0, max_lon=-86.0, max_lat=41.0)

  sub_lats, sub_lons, lat_idx, lon_idx = slice_coordinates_by_bounds(
      lats_desc, lons_wgs, bbox, buffer_degrees=0.0
  )
  assert np.all(sub_lats >= 39.0)
  assert np.all(sub_lats <= 41.0)
  assert np.all(sub_lons >= -88.0)
  assert np.all(sub_lons <= -86.0)
  assert len(sub_lats) == 3  # 41, 40, 39
  assert len(sub_lons) == 3  # -88, -87, -86


def test_slice_dataset_descending_lats_0_360():
  """Tests WeatherBench 2 style dataset (descending lats, 0..360 lons)."""
  lats = np.linspace(90.0, -90.0, 181)
  lons = np.linspace(0.0, 359.0, 360)
  data = np.arange(181 * 360, dtype=np.float32).reshape((181, 360))
  ds = xr.Dataset(
      {"temp": (["latitude", "longitude"], data)},
      coords={"latitude": lats, "longitude": lons},
  )

  # US box: [-88, -86] lon, [39, 41] lat
  bbox = BoundingBox(min_lon=-88.0, min_lat=39.0, max_lon=-86.0, max_lat=41.0)
  sliced = slice_dataset_by_bounds(ds, bbox, buffer_degrees=0.0)

  assert sliced["temp"].shape == (3, 3)
  assert np.allclose(sliced["latitude"].values, [41.0, 40.0, 39.0])
  # Coordinates converted to standard WGS84 [-180, 180]
  assert np.allclose(sliced["longitude"].values, [-88.0, -87.0, -86.0])


def test_slice_dataset_prime_meridian_crossing():
  """Tests slicing across 0° Prime Meridian on a 0..360 grid."""
  lats = np.linspace(60.0, 40.0, 21)
  lons = np.linspace(0.0, 359.0, 360)
  data = np.arange(21 * 360, dtype=np.float32).reshape((21, 360))
  ds = xr.Dataset(
      {"temp": (["latitude", "longitude"], data)},
      coords={"latitude": lats, "longitude": lons},
  )

  # Box spans across Prime Meridian: [-3, 3] lon, [48, 52] lat
  bbox = BoundingBox(min_lon=-3.0, min_lat=48.0, max_lon=3.0, max_lat=52.0)
  sliced = slice_dataset_by_bounds(ds, bbox, buffer_degrees=0.0)

  assert np.all(sliced["latitude"].values <= 52.0)
  assert np.all(sliced["latitude"].values >= 48.0)
  assert np.allclose(sliced["longitude"].values, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
  # Monotonic increasing longitudes
  assert np.all(np.diff(sliced["longitude"].values) > 0)


def test_slice_dataset_ascending_lats_wgs84():
  """Tests IMERG/CPC style dataset (ascending lats, -180..180 lons)."""
  lats = np.linspace(-90.0, 90.0, 181)
  lons = np.linspace(-180.0, 179.0, 360)
  data = np.arange(181 * 360, dtype=np.float32).reshape((181, 360))
  ds = xr.Dataset(
      {"precip": (["lat", "lon"], data)},
      coords={"lat": lats, "lon": lons},
  )

  bbox = BoundingBox(min_lon=10.0, min_lat=45.0, max_lon=15.0, max_lat=50.0)
  sliced = slice_dataset_by_bounds(ds, bbox, buffer_degrees=0.0)

  assert np.allclose(sliced["lat"].values, [45.0, 46.0, 47.0, 48.0, 49.0, 50.0])
  assert np.allclose(sliced["lon"].values, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


def test_slice_dataset_global_bypass():
  """Tests that a global bounding box bypasses slicing and preserves the dataset."""
  lats = np.linspace(90.0, -90.0, 19)
  lons = np.linspace(-180.0, 180.0, 37)
  ds = xr.Dataset(
      {"val": (["lat", "lon"], np.zeros((19, 37)))},
      coords={"lat": lats, "lon": lons},
  )
  global_box = BoundingBox(-180.0, -90.0, 180.0, 90.0)
  sliced = slice_dataset_by_bounds(ds, global_box)
  assert sliced["val"].shape == (19, 37)


def test_crop_weight_matrix_to_coords_parity():
  """Verifies mathematical parity between full grid reduction and cropped reduction."""
  lats = np.linspace(45.0, 35.0, 101)
  lons = np.linspace(-92.0, -82.0, 101)

  # Create 2 test basin polygons
  b1 = shapely.geometry.box(-88.5, 39.5, -87.5, 40.5)
  b2 = shapely.geometry.box(-85.0, 36.0, -84.0, 37.0)
  gdf = gpd.GeoDataFrame({"geometry": [b1, b2]}, index=["basin_A", "basin_B"])

  full_matrix = ZonalWeightMatrix.from_geodataframe(
      gdf, lats, lons, cell_res_lat=0.1, cell_res_lon=0.1
  )

  # Define subgrid around basin_A with 0.5 deg buffer
  sub_bbox = BoundingBox.from_geometry(b1, buffer_degrees=0.5)
  sub_lats, sub_lons, lat_idx, lon_idx = slice_coordinates_by_bounds(
      lats, lons, sub_bbox
  )

  # Fast crop of full matrix to subgrid
  cropped_matrix = full_matrix.crop_to_coords(sub_lats, sub_lons)
  assert cropped_matrix.grid_shape == (len(sub_lats), len(sub_lons))
  assert cropped_matrix.num_basins == 2

  # Test 2D reduction parity
  rng = np.random.RandomState(42)
  full_grid_2d = rng.randn(len(lats), len(lons)).astype(np.float32)
  sub_grid_2d = full_grid_2d[lat_idx[:, None], lon_idx[None, :]]

  res_full_2d = full_matrix.reduce_2d(full_grid_2d)
  res_sub_2d = cropped_matrix.reduce_2d(sub_grid_2d)

  # basin_A is fully inside subgrid -> exact match
  assert np.isclose(res_full_2d[0], res_sub_2d[0], atol=1e-6)
  # basin_B is outside subgrid -> evaluates to NaN
  assert np.isnan(res_sub_2d[1])

  # Test 3D reduction parity
  T = 5
  full_grid_3d = rng.randn(T, len(lats), len(lons)).astype(np.float32)
  sub_grid_3d = full_grid_3d[:, lat_idx[:, None], lon_idx[None, :]]

  res_full_3d = full_matrix.reduce_3d(full_grid_3d)
  res_sub_3d = cropped_matrix.reduce_3d(sub_grid_3d)

  assert np.allclose(res_full_3d[0, :], res_sub_3d[0, :], atol=1e-6)
  assert np.all(np.isnan(res_sub_3d[1, :]))


def test_crop_weight_matrix_to_bounds():
  """Verifies crop_to_bounds directly crops via a BoundingBox or GeoDataFrame."""
  lats = np.linspace(45.0, 35.0, 51)
  lons = np.linspace(-90.0, -80.0, 51)
  b1 = shapely.geometry.box(-86.0, 39.0, -85.0, 40.0)
  gdf = gpd.GeoDataFrame({"geometry": [b1]}, index=["basin_1"])

  full_matrix = ZonalWeightMatrix.from_geodataframe(
      gdf, lats, lons, cell_res_lat=0.2, cell_res_lon=0.2
  )

  cropped = full_matrix.crop_to_bounds(gdf, buffer_degrees=0.4)
  assert cropped.grid_shape[0] < full_matrix.grid_shape[0]
  assert cropped.grid_shape[1] < full_matrix.grid_shape[1]

  # Matrix reduction produces valid non-NaN result
  test_grid = np.ones(cropped.grid_shape, dtype=np.float32)
  res = cropped.reduce_2d(test_grid)
  assert np.isclose(res[0], 1.0, atol=1e-5)
