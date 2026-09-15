"""Unit and integration tests for DEM catchment delineation module."""

import csv
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from catchment_delineation import (
    DemDelineator,
    delineate_dem,
    latlon_to_tile_key,
    tile_key_to_filename,
    is_tile_available,
    list_available_tiles,
    INFLOW_MAP,
)
from catchment_delineation.cli import main, parse_coord_str, load_coords_from_csv


def test_tile_key_and_filename():
  # Point in Illinois (lat 39.68, lon -88.77) -> lat_top=40, lon_left=-90 -> n40w090.npy
  lat_top, lon_left = latlon_to_tile_key(39.6828, -88.7729)
  assert lat_top == 40
  assert lon_left == -90
  assert tile_key_to_filename(lat_top, lon_left) == "n40w090.npy"

  # Southern/Eastern hemisphere test
  lat_s, lon_e = latlon_to_tile_key(-12.4, 25.6)
  assert lat_s == -10
  assert lon_e == 25
  assert tile_key_to_filename(lat_s, lon_e) == "s10e025.npy"


def test_parse_coord_str():
  lat, lon = parse_coord_str("39.6828,-88.7729")
  assert pytest.approx(lat, abs=1e-4) == 39.6828
  assert pytest.approx(lon, abs=1e-4) == -88.7729

  lat, lon = parse_coord_str("40.5   -86.2")
  assert pytest.approx(lat, abs=1e-4) == 40.5
  assert pytest.approx(lon, abs=1e-4) == -86.2

  with pytest.raises(ValueError):
    parse_coord_str("invalid_coord")


def test_load_coords_from_csv(tmp_path):
  csv_file = tmp_path / "test_coords.csv"
  with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gauge_id", "latitude", "longitude"])
    writer.writerow(["G1", "39.6828", "-88.7729"])
    writer.writerow(["G2", "40.4172", "-86.8858"])

  coords, ids = load_coords_from_csv(csv_file)
  assert len(coords) == 2
  assert ids == ["G1", "G2"]
  assert pytest.approx(coords[0][0], abs=1e-4) == 39.6828
  assert pytest.approx(coords[0][1], abs=1e-4) == -88.7729


def test_synthetic_dem_delineator(tmp_path):
  """Tests delineation algorithm on a synthetic 5x5 degree tile."""
  # Tile n40w090 covers lat [35, 40], lon [-90, -85]
  tile_arr = np.zeros((6000, 6000), dtype=np.uint8)

  # Construct a mini drainage branch:
  # Outlet at (r=100, c=100)
  # (99, 100) flows South (4) into (100, 100)
  # (98, 100) flows South (4) into (99, 100)
  # (99, 101) flows West (16) into (99, 100)
  tile_arr[99, 100] = 4
  tile_arr[98, 100] = 4
  tile_arr[99, 101] = 16

  tile_path = tmp_path / "n40w090.npy"
  np.save(tile_path, tile_arr)

  delineator = DemDelineator(tiles_dir=tmp_path)
  # Outlet lat/lon corresponding to r=100, c=100:
  # lat = 40.0 - 100 * (1/1200) = 40.0 - 0.083333 = 39.91667
  # lon = -90.0 + 100 * (1/1200) = -90.0 + 0.083333 = -89.91667
  out_lat = 40.0 - 100.0 / 1200.0
  out_lon = -90.0 + 100.0 / 1200.0

  res = delineator.delineate(lat=out_lat, lon=out_lon, snap_window_cells=1)
  assert res["type"] == "Feature"
  props = res["properties"]
  assert props["upstream_cells_count"] == 4  # (100,100), (99,100), (98,100), (99,101)
  assert props["area_km2"] > 0
  assert "geometry" in res
  assert res["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_live_tile_delineation():
  """Tests against available real-world tiles if present on the system."""
  delineator = DemDelineator()
  if not is_tile_available(40, -90, delineator.tiles_dir):
    pytest.skip("DEM tiles not present in cache on this test system.")

  res = delineator.delineate(lat=39.6828, lon=-88.7729)
  assert res["type"] == "Feature"
  props = res["properties"]
  assert props["area_km2"] > 10.0
  assert props["upstream_cells_count"] > 1000
  assert res["geometry"]["type"] in ("Polygon", "MultiPolygon")
  assert "bbox" in props
  assert props["bbox"]["min_lon"] < props["bbox"]["max_lon"]
  assert props["bbox"]["min_lat"] < props["bbox"]["max_lat"]


def test_cli_execution(tmp_path):
  out_json = tmp_path / "output.geojson"
  exit_code = main(["--coords", "39.6828,-88.7729", "-o", str(out_json)])
  assert exit_code == 0
  assert out_json.exists()

  with open(out_json, "r") as f:
    data = json.load(f)
  assert data["type"] == "FeatureCollection"
  assert len(data["features"]) == 1
  assert data["features"][0]["properties"]["area_km2"] > 10.0


def test_is_gcs_path():
  from catchment_delineation.gcs import is_gcs_path

  assert is_gcs_path("gs://open-multimet/data/DEMs")
  assert is_gcs_path("gcs://open-multimet/data/DEMs")
  assert not is_gcs_path("/usr/local/data/DEMs")
  assert not is_gcs_path("data/dem")


def test_default_gcs_download_mocked(tmp_path, monkeypatch):
  """Verifies that with no tiles_dir supplied, DemDelineator automatically retrieves tiles from GCS into cache."""
  import catchment_delineation.gcs

  def mock_download(lat_top, lon_left, target_dir=None, source_uri=None):
    target = Path(target_dir) if target_dir else tmp_path
    arr = np.zeros((6000, 6000), dtype=np.uint8)
    tile_file = target / tile_key_to_filename(lat_top, lon_left)
    np.save(tile_file, arr)
    return tile_file

  monkeypatch.setattr(catchment_delineation.gcs, "download_tile_from_gcs", mock_download)

  # Default: tiles_dir is None, cache_dir is tmp_path
  delineator = DemDelineator(cache_dir=tmp_path)
  tile = delineator.get_tile(40, -90)
  assert tile is not None
  assert tile.shape == (6000, 6000)
  assert (tmp_path / "n40w090.npy").exists()


def test_user_supplied_tiles_dir_no_searching(tmp_path, monkeypatch):
  """Verifies that when tiles_dir is user-supplied, it ONLY checks that path with no searching or downloading."""
  import catchment_delineation.gcs

  download_called = False

  def mock_download(*args, **kwargs):
    nonlocal download_called
    download_called = True
    raise RuntimeError("Should not be called when custom tiles_dir is supplied")

  monkeypatch.setattr(catchment_delineation.gcs, "download_tile_from_gcs", mock_download)

  # Empty custom directory
  custom_dir = tmp_path / "custom_tiles"
  custom_dir.mkdir()

  delineator = DemDelineator(tiles_dir=custom_dir)
  tile = delineator.get_tile(40, -90)
  assert tile is None
  assert not download_called

  # Now put tile in custom_dir, verify it loads
  arr = np.zeros((6000, 6000), dtype=np.uint8)
  np.save(custom_dir / "n40w090.npy", arr)
  delineator_new = DemDelineator(tiles_dir=custom_dir)
  tile = delineator_new.get_tile(40, -90)
  assert tile is not None
  assert tile.shape == (6000, 6000)


def test_benchmark_execution(tmp_path):
  from catchment_delineation.benchmark import run_benchmark
  out_csv = tmp_path / "bench_test.csv"
  df = run_benchmark(samples=2, workers=1, output_path=str(out_csv))
  assert len(df) == 2
  assert out_csv.exists()
  assert "iou" in df.columns
  assert "dice" in df.columns



