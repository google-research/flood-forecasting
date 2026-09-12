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

from multimet.geometry import (
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


def test_load_multiple_files_as_list(geojson_path: Path, tmp_path: Path):
  full_gdf = gpd.read_file(str(geojson_path))
  assert len(full_gdf) == 5

  # Split into two separate GeoJSON files
  p1 = tmp_path / "part1.geojson"
  p2 = tmp_path / "part2.geojson"
  full_gdf.iloc[:2].to_file(str(p1), driver="GeoJSON")
  full_gdf.iloc[2:].to_file(str(p2), driver="GeoJSON")

  combined_gdf = load_basin_geometries([str(p1), str(p2)])
  assert len(combined_gdf) == 5
  for b_id in _EXPECTED_BASINS:
    assert b_id in combined_gdf.index


def test_load_multiple_files_as_comma_separated_string(geojson_path: Path, tmp_path: Path):
  full_gdf = gpd.read_file(str(geojson_path))
  p1 = tmp_path / "part1.geojson"
  p2 = tmp_path / "part2.geojson"
  full_gdf.iloc[:2].to_file(str(p1), driver="GeoJSON")
  full_gdf.iloc[2:].to_file(str(p2), driver="GeoJSON")

  combined_gdf = load_basin_geometries(f"{p1},{p2}")
  assert len(combined_gdf) == 5
  for b_id in _EXPECTED_BASINS:
    assert b_id in combined_gdf.index


def test_load_from_directory(geojson_path: Path, tmp_path: Path):
  full_gdf = gpd.read_file(str(geojson_path))
  sub_dir = tmp_path / "basins_dir"
  sub_dir.mkdir()
  p1 = sub_dir / "part1.geojson"
  p2 = sub_dir / "part2.geojson"
  full_gdf.iloc[:3].to_file(str(p1), driver="GeoJSON")
  full_gdf.iloc[3:].to_file(str(p2), driver="GeoJSON")

  combined_gdf = load_basin_geometries(sub_dir)
  assert len(combined_gdf) == 5
  for b_id in _EXPECTED_BASINS:
    assert b_id in combined_gdf.index


def test_load_from_glob_pattern(geojson_path: Path, tmp_path: Path):
  full_gdf = gpd.read_file(str(geojson_path))
  sub_dir = tmp_path / "nested" / "shapes"
  sub_dir.mkdir(parents=True)
  p1 = sub_dir / "chunk1.geojson"
  p2 = sub_dir / "chunk2.geojson"
  full_gdf.iloc[:2].to_file(str(p1), driver="GeoJSON")
  full_gdf.iloc[2:].to_file(str(p2), driver="GeoJSON")

  combined_gdf = load_basin_geometries(str(tmp_path / "**/*.geojson"))
  assert len(combined_gdf) == 5
  for b_id in _EXPECTED_BASINS:
    assert b_id in combined_gdf.index


def test_load_basin_geometries_deduplication(geojson_path: Path, tmp_path: Path):
  full_gdf = gpd.read_file(str(geojson_path))
  p1 = tmp_path / "part1.geojson"
  p2 = tmp_path / "part2.geojson"
  # Both files include the basin at index 1
  full_gdf.iloc[:2].to_file(str(p1), driver="GeoJSON")
  full_gdf.iloc[1:3].to_file(str(p2), driver="GeoJSON")

  # Passing distinct files with overlapping basin IDs:
  # With drop_duplicates=True (default), duplicates are pruned
  gdf_dedup = load_basin_geometries([str(p1), str(p2)], drop_duplicates=True)
  assert len(gdf_dedup) == 3

  # With drop_duplicates=False, raises ValueError
  with pytest.raises(ValueError, match="duplicate basin IDs"):
    load_basin_geometries([str(p1), str(p2)], drop_duplicates=False)


def test_load_nonexistent_source_raises_error(tmp_path: Path):
  with pytest.raises(FileNotFoundError):
    load_basin_geometries(tmp_path / "nonexistent_file.geojson")

  empty_dir = tmp_path / "empty_dir"
  empty_dir.mkdir()
  with pytest.raises(FileNotFoundError, match="No supported geometry files"):
    load_basin_geometries(empty_dir)


def test_remote_geometry_resolution_and_caching(geojson_path: Path, monkeypatch, tmp_path: Path):
  import fsspec
  from unittest.mock import MagicMock
  from multimet.geometry import _resolve_geometry_sources

  mock_fs = MagicMock()
  mock_fs.isdir.return_value = True
  mock_fs.find.return_value = [
      "my-bucket/shapes/camels/camels_basin_shapes.shp",
      "my-bucket/shapes/camels/camels_gauges.shp",
      "my-bucket/shapes/hysets/hysets_basin_shapes.shp",
  ]

  monkeypatch.setattr(fsspec.core, "url_to_fs", lambda url: (mock_fs, "my-bucket/shapes/"))

  resolved = _resolve_geometry_sources("gs://my-bucket/shapes/")
  assert len(resolved) == 2
  assert "gs://my-bucket/shapes/camels/camels_basin_shapes.shp" in resolved
  assert "gs://my-bucket/shapes/hysets/hysets_basin_shapes.shp" in resolved
  # Verify gauges.shp point file was properly filtered out
  assert "gs://my-bucket/shapes/camels/camels_gauges.shp" not in resolved


