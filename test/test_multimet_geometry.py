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

"""Unit tests for MultiMet geometry utilities."""

from pathlib import Path
import pytest
import geopandas as gpd
from shapely.geometry import Polygon

from googlehydrology.multimet.geometry import (
    get_bounding_box,
    load_basin_geometries,
)

_EXPECTED_BASINS = [
    "us_03338780",
    "us_03340800",
    "us_03346000",
    "us_03364500",
    "us_03366500",
]


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


def test_load_basin_geometries(geojson_path: Path):
  gdf = load_basin_geometries(geojson_path)
  assert len(gdf) == 5
  assert gdf.crs is not None
  for b_id in _EXPECTED_BASINS:
    assert b_id in gdf.index
    geom = gdf.loc[b_id, "geometry"]
    assert not geom.is_empty
    assert geom.is_valid
    rep_pt = geom.representative_point()
    assert geom.contains(rep_pt)


def test_get_bounding_box(geojson_path: Path):
  gdf = load_basin_geometries(geojson_path)
  minx, miny, maxx, maxy = get_bounding_box(gdf, buffer_degrees=0.5)
  assert minx < -87.0
  assert maxx > -86.0
  assert miny < 39.0
  assert maxy > 40.0


def test_load_basin_geometries_from_dict():
  poly = Polygon([[-87.5, 40.0], [-87.0, 40.0], [-87.0, 40.5], [-87.5, 40.0]])
  features_dict = {
      "type": "FeatureCollection",
      "features": [{
          "type": "Feature",
          "properties": {"basin_id": "test_basin_1"},
          "geometry": poly.__geo_interface__,
      }],
  }
  gdf = load_basin_geometries(features_dict)
  assert len(gdf) == 1
  assert "test_basin_1" in gdf.index
