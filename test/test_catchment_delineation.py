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


def test_out_of_coverage_pour_point_raises_error():
  """Verifies that requesting coordinates outside coverage raises CatchmentCoverageError."""
  from catchment_delineation import CatchmentCoverageError, DemDelineator

  delineator = DemDelineator()
  # Alaska (64.9N) is beyond HydroSHEDS 60N boundary
  with pytest.raises(CatchmentCoverageError) as excinfo:
    delineator.delineate(64.9024, -146.3594)
  assert "outside the global DEM coverage domain" in str(excinfo.value)

  # Antarctica (-70S) is beyond HydroSHEDS 56S boundary
  with pytest.raises(CatchmentCoverageError) as excinfo2:
    delineator.delineate(-70.0, 0.0)
  assert "outside the global DEM coverage domain" in str(excinfo2.value)


def test_watershed_extending_past_boundary_aborts(tmp_path):
  """Verifies that if a watershed extends past the 60N boundary during traversal,
  it raises CatchmentCoverageError and aborts without returning a partial polygon."""
  from catchment_delineation import CatchmentCoverageError, DemDelineator

  # Create a custom tile at northern boundary (lat_top = 60, lon_left = 10)
  custom_dir = tmp_path / "boundary_tiles"
  custom_dir.mkdir()

  tile = np.zeros((6000, 6000), dtype=np.uint8)
  # Stream flows South (4) from row 0 -> row 1 -> row 2 -> row 3
  tile[0, 100] = 4
  tile[1, 100] = 4
  tile[2, 100] = 4
  tile[3, 100] = 4

  np.save(custom_dir / "n60e010.npy", tile)

  delineator = DemDelineator(tiles_dir=custom_dir)
  # Pour point at row 3 (lat = 60 - 3 * (1/1200) = 59.9975, lon = 10 + 100 * (1/1200) = 10.0833)
  lat = 60.0 - 3.0 / 1200.0
  lon = 10.0 + 100.0 / 1200.0

  with pytest.raises(CatchmentCoverageError) as excinfo:
    delineator.delineate(lat, lon, snap_window_cells=1)

  assert "Watershed extends north past the DEM coverage boundary" in str(excinfo.value)


def test_cli_out_of_coverage_aborts(tmp_path, capsys):
  """Verifies that CLI halts with exit code 1 when given coordinates outside coverage."""
  out_json = tmp_path / "out_ooc.geojson"
  exit_code = main(["--lat", "65.0", "--lon", "-150.0", "-o", str(out_json)])
  assert exit_code == 1
  assert not out_json.exists()


def test_delineate_batch_omits_out_of_coverage(tmp_path):
  """Verifies that batch delineation omits out-of-coverage basins rather than outputting partial polygons."""
  from catchment_delineation import DemDelineator

  delineator = DemDelineator()
  coords = [(39.6828, -88.7729), (65.0, -150.0)]
  fc = delineator.delineate_batch(coords=coords)

  # Only the valid coordinate in Illinois (39.68N) should be in the FeatureCollection
  assert len(fc["features"]) == 1
  assert fc["features"][0]["properties"]["outlet"]["input_latitude"] == 39.6828


def test_clean_cache_flag(tmp_path):
  """Verifies that --clean-cache cleans up the cache directory after run."""
  from catchment_delineation.cli import main
  cache_dir = tmp_path / "test_cache"
  cache_dir.mkdir()
  tile = np.zeros((6000, 6000), dtype=np.uint8)
  tile[380, 1472] = 4
  np.save(cache_dir / "n40w090.npy", tile)

  out_json = tmp_path / "test_out.geojson"
  exit_code = main([
      "--lat", "39.6828",
      "--lon", "-88.7729",
      "--tiles-dir", str(cache_dir),
      "--clean-cache",
      "-o", str(out_json),
  ])
  assert exit_code == 0
  assert not cache_dir.exists()


def test_cli_parallel_workers_and_geoparquet(tmp_path):
  """Tests CLI batch delineation using multiple workers, custom column names, and parquet output."""
  from catchment_delineation.cli import main, load_coords_from_file
  import geopandas as gpd
  import pandas as pd

  # Create a CSV with Caravan-style column names
  csv_file = tmp_path / "caravan_test.csv"
  df = pd.DataFrame({
      "gauge_id": ["C1", "C2"],
      "CARAVAN:gauge_lat": [39.6828, 40.4172],
      "CARAVAN:gauge_lon": [-88.7729, -86.8858],
  })
  df.to_csv(csv_file, index=False)

  # Verify load_coords_from_file parses automatically
  coords, ids = load_coords_from_file(csv_file)
  assert len(coords) == 2
  assert ids == ["C1", "C2"]

  # Run CLI with --workers 2 and GeoParquet output
  out_parquet = tmp_path / "delineated.geoparquet"
  exit_code = main([
      "--csv", str(csv_file),
      "--workers", "2",
      "-o", str(out_parquet),
  ])
  assert exit_code == 0
  assert out_parquet.exists()
  gdf = gpd.read_parquet(out_parquet)
  assert len(gdf) == 2
  assert "catchment_id" in gdf.columns
  assert "geometry" in gdf.columns


def test_cli_preserve_caravan_dirs(tmp_path):
  """Verifies that --preserve-caravan-dirs partitions output into caravan/<ds>/ hierarchy."""
  from catchment_delineation.cli import main
  import geopandas as gpd
  import pandas as pd

  csv_file = tmp_path / "caravan_multi_ds.csv"
  df = pd.DataFrame({
      "gauge_id": ["CARAVAN_CAMELS_01013500", "CARAVAN_CAMELSAUS_102101A", "CARAVAN_GRDC_1234567"],
      "CARAVAN:gauge_lat": [39.6828, 39.6828, 39.6828],
      "CARAVAN:gauge_lon": [-88.7729, -88.7729, -88.7729],
  })
  df.to_csv(csv_file, index=False)

  out_dir = tmp_path / "partitioned_out"
  exit_code = main([
      "--csv", str(csv_file),
      "--output-dir", str(out_dir),
      "--preserve-caravan-dirs",
      "--format", "geoparquet",
      "--workers", "1",
  ])
  assert exit_code == 0
  assert (out_dir / "caravan" / "camels" / "camels_delineated_catchments.geoparquet").exists()
  assert (out_dir / "caravan" / "camelsaus" / "camelsaus_delineated_catchments.geoparquet").exists()
  assert (out_dir / "caravan_extensions" / "grdc" / "grdc_delineated_catchments.geoparquet").exists()

  gdf_camels = gpd.read_parquet(out_dir / "caravan" / "camels" / "camels_delineated_catchments.geoparquet")
  assert len(gdf_camels) == 1
  assert gdf_camels["catchment_id"].iloc[0] == "CARAVAN_CAMELS_01013500"


def test_gcs_helpers():
  """Tests GCS path detection and helpers."""
  from catchment_delineation.gcs import is_gcs_path

  assert is_gcs_path("gs://open-multimet/data/caravan/coordinates.csv")
  assert is_gcs_path("gcs://bucket/path/file.parquet")
  assert not is_gcs_path("/local/path/file.csv")
  assert not is_gcs_path("relative/file.csv")


def test_load_coords_gcs(monkeypatch):
  """Tests loading coordinates from GCS URI and shorthand."""
  from catchment_delineation.cli import load_coords_from_file
  import pandas as pd

  orig_read_csv = pd.read_csv

  def mock_read_csv(filepath_or_buffer, *args, **kwargs):
    if str(filepath_or_buffer).startswith("gs://"):
      return pd.DataFrame({
          "gauge_id": ["GCS_TEST_1", "GCS_TEST_2"],
          "CARAVAN:gauge_lat": [39.5, 40.0],
          "CARAVAN:gauge_lon": [-88.5, -86.5],
      })
    return orig_read_csv(filepath_or_buffer, *args, **kwargs)

  monkeypatch.setattr(pd, "read_csv", mock_read_csv)

  # Direct GCS URI
  coords, ids = load_coords_from_file("gs://test-bucket/coords.csv")
  assert len(coords) == 2
  assert ids == ["GCS_TEST_1", "GCS_TEST_2"]

  # Shorthand 'caravan'
  coords2, ids2 = load_coords_from_file("caravan")
  assert len(coords2) == 2
  assert ids2 == ["GCS_TEST_1", "GCS_TEST_2"]


def test_cli_gcs_direct_output(monkeypatch, tmp_path):
  """Tests CLI saving output directly to GCS path."""
  from catchment_delineation.cli import main
  import pandas as pd
  import geopandas as gpd

  csv_file = tmp_path / "test_coords.csv"
  df = pd.DataFrame({
      "gauge_id": ["CARAVAN_CAMELS_01013500"],
      "CARAVAN:gauge_lat": [39.6828],
      "CARAVAN:gauge_lon": [-88.7729],
  })
  df.to_csv(csv_file, index=False)

  saved_targets = []

  def mock_to_parquet(self, path, *args, **kwargs):
    saved_targets.append(str(path))

  monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", mock_to_parquet)

  # Test partitioned GCS output-dir
  exit_code = main([
      "--csv", str(csv_file),
      "--output-dir", "gs://open-multimet/data/catchment_polygons/caravan",
      "--preserve-caravan-dirs",
      "--format", "geoparquet",
      "--workers", "1",
  ])
  assert exit_code == 0
  assert any("gs://open-multimet/data/catchment_polygons/caravan/caravan/camels/camels_delineated_catchments.geoparquet" in t for t in saved_targets)

  # Test single file GCS output
  exit_code2 = main([
      "--csv", str(csv_file),
      "-o", "gs://open-multimet/data/catchment_polygons/single.geoparquet",
      "--workers", "1",
  ])
  assert exit_code2 == 0
  assert "gs://open-multimet/data/catchment_polygons/single.geoparquet" in saved_targets







