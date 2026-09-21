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

"""Unit tests for gridded Zarr archive extraction and strict data-quality invariants."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import xarray as xr

from multimet.config import (
    DEFAULT_STORAGE_PATHS,
    MISSING_FRACTION_VAR,
    PRODUCT_BANDS,
    Product,
)
from multimet.cpc import CPCExtractor
from multimet.era5_land import ERA5LandExtractor
from multimet.geometry import load_basin_geometries
from multimet.gridded_archive import (
    _WARNED_MISSING_VARS,
    GriddedArchiveError,
    extract_from_archive,
    open_gridded_archive,
)
from multimet.hres import HRESExtractor
from multimet.imerg import IMERGExtractor
from multimet.runner import extract_multimet_serial
from multimet.zarr_writer import check_zarr_store_exists
from multimet.zonal import ZonalWeightCalculator, ZonalWeightMatrix

pytestmark = pytest.mark.unit


@pytest.fixture
def basins_gdf() -> gpd.GeoDataFrame:
  path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  return load_basin_geometries(path)


def test_no_hardcoded_gs_bucket_paths_in_config():
  """Verifies that all hardcoded gs:// bucket paths have been stripped from DEFAULT_STORAGE_PATHS."""
  for prod, paths_map in DEFAULT_STORAGE_PATHS.items():
    for key, val in paths_map.items():
      assert not str(val).startswith(("gs://", "gcs://")), (
          f"Found prohibited hardcoded GCS bucket URI in "
          f"DEFAULT_STORAGE_PATHS[{prod}][{key!r}] = {val!r}"
      )


def test_out_of_domain_basin_never_snaps_to_nearest_cell():
  """Verifies that a basin outside the grid domain gets zero weight and NaN, not a distant pixel."""
  # Grid covering North America only: lats [35..45], lons [-95..-80]
  lats = np.linspace(35.0, 45.0, 21, dtype=np.float64)
  lons = np.linspace(-95.0, -80.0, 31, dtype=np.float64)

  in_domain_poly = shapely.geometry.box(-88.0, 39.0, -87.0, 40.0)
  out_of_domain_poly = shapely.geometry.box(10.0, 48.0, 11.0, 49.0)  # Europe

  gdf = gpd.GeoDataFrame(
      {"geometry": [in_domain_poly, out_of_domain_poly]},
      index=["basin_us", "basin_europe"],
      crs="EPSG:4326",
  )

  calc = ZonalWeightCalculator(lats, lons, cell_res_lat=0.5, cell_res_lon=0.5)
  lat_eu, lon_eu, w_eu = calc.compute_weights(
      "basin_europe", out_of_domain_poly
  )
  assert len(w_eu) == 0
  assert len(lat_eu) == 0
  assert len(lon_eu) == 0

  matrix = ZonalWeightMatrix.from_geodataframe(
      gdf, lats, lons, cell_res_lat=0.5, cell_res_lon=0.5
  )
  assert bool(matrix.has_weights[0]) is True
  assert bool(matrix.has_weights[1]) is False

  grid_2d = np.full((len(lats), len(lons)), 25.0, dtype=np.float32)
  vals, missing = matrix.reduce_2d_with_coverage(grid_2d)
  assert np.isclose(vals[0], 25.0)
  assert np.isclose(missing[0], 0.0)
  assert np.isnan(vals[1])
  assert np.isclose(missing[1], 1.0)


def test_partial_coverage_missing_fraction_matches_cookie_cutter():
  """Verifies that partial NaN coverage renormalizes valid cells and records missing_fraction."""
  lats = np.array([39.5, 40.5], dtype=np.float64)
  lons = np.array([-88.5, -87.5], dtype=np.float64)
  # Basin covering all 4 cells equally
  poly = shapely.geometry.box(-89.0, 39.0, -87.0, 41.0)
  gdf = gpd.GeoDataFrame(
      {"geometry": [poly]}, index=["coastal_basin"], crs="EPSG:4326"
  )
  matrix = ZonalWeightMatrix.from_geodataframe(
      gdf, lats, lons, cell_res_lat=1.0, cell_res_lon=1.0
  )

  # Top row (lat=40.5) is valid=10.0, bottom row (lat=39.5) is NaN (ocean)
  grid_2d = np.array([[np.nan, np.nan], [10.0, 10.0]], dtype=np.float32)
  vals, missing = matrix.reduce_2d_with_coverage(grid_2d)
  assert np.isclose(vals[0], 10.0, atol=1e-4)
  # Approximately 50% of the cos(lat)-weighted area is missing
  assert 0.48 <= float(missing[0]) <= 0.52


def test_check_zarr_store_exists_does_not_mask_storage_errors(monkeypatch):
  """Verifies that transient GCS 403s retry and persistent errors raise rather than returning False."""
  import fsspec

  attempts = {"count": 0}

  class DummyFS:
    def exists(self, path: str) -> bool:
      attempts["count"] += 1
      if attempts["count"] < 3:
        raise OSError("403 Forbidden: transient GCS error")
      return path.endswith("zarr.json")

  monkeypatch.setattr(
      fsspec.core,
      "url_to_fs",
      lambda url: (DummyFS(), url.removeprefix("gs://")),
  )
  monkeypatch.setattr("time.sleep", lambda _: None)

  assert check_zarr_store_exists("gs://open-multimet/test.zarr", max_retries=4) is True
  assert attempts["count"] == 3

  # Persistent 403 must raise RuntimeError, NEVER return False
  class AlwaysFailingFS:
    def exists(self, path: str) -> bool:
      raise OSError("403 Forbidden: persistent permission failure")

  monkeypatch.setattr(
      fsspec.core,
      "url_to_fs",
      lambda url: (AlwaysFailingFS(), url.removeprefix("gs://")),
  )
  with pytest.raises(RuntimeError, match="Failed to check Zarr store existence"):
    check_zarr_store_exists("gs://open-multimet/test.zarr", max_retries=2)


def test_cpc_archive_extraction(tmp_path, basins_gdf):
  """Tests CPC gridded archive extraction."""
  lats = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
  lons = np.linspace(-179.75, 179.75, 720, dtype=np.float32)
  times = pd.date_range("2022-05-01", "2022-05-03", freq="D")

  precip = np.full((len(times), len(lats), len(lons)), 12.5, dtype=np.float32)
  ds_archive = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["time", "latitude", "longitude"], precip),
      },
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  )
  store_path = tmp_path / "cpc_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = CPCExtractor(data_dir=str(store_path), source="archive")
  ds_out = ext.extract_for_basins(
      basins_gdf, start_date="2022-05-01", end_date="2022-05-03"
  )

  assert "cpc_precipitation" in ds_out.data_vars
  assert "cpc_missing_fraction" in ds_out.data_vars
  assert np.allclose(ds_out["cpc_precipitation"].values, 12.5, atol=1e-3)
  assert np.allclose(ds_out["cpc_missing_fraction"].values, 0.0)


def test_era5_land_archive_extraction_and_deduplicated_warning(
    tmp_path, basins_gdf, caplog
):
  """Tests ERA5-Land archive extraction on descending latitudes and single-warning behavior when a band is absent."""
  _WARNED_MISSING_VARS.clear()
  lats = np.linspace(90.0, -90.0, 181, dtype=np.float32)  # descending
  lons = np.linspace(-180.0, 179.0, 360, dtype=np.float32)
  times = pd.date_range("2021-07-01", "2021-07-02", freq="D")

  data_vars = {}
  for idx, band in enumerate(PRODUCT_BANDS[Product.ERA5_LAND]):
    if band == "era5land_snow_depth_water_equivalent":
      continue  # Intentionally omit one band to test deduplicated warning + NaN output
    if band == "era5land_temperature_2m_min":
      val = 14.0
    elif band == "era5land_temperature_2m":
      val = 20.0
    elif band == "era5land_temperature_2m_max":
      val = 26.0
    else:
      val = 10.0 + float(idx)
    data_vars[band] = (
        ["time", "latitude", "longitude"],
        np.full((len(times), len(lats), len(lons)), val, dtype=np.float32),
    )

  ds_archive = xr.Dataset(
      data_vars=data_vars,
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  )
  store_path = tmp_path / "era5_land_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = ERA5LandExtractor(data_dir=str(store_path), source="archive")
  with caplog.at_level(logging.WARNING):
    ds_out = ext.extract_for_basins(
        basins_gdf, start_date="2021-07-01", end_date="2021-07-02"
    )
    _ = ext.extract_for_basins(
        basins_gdf, start_date="2021-07-01", end_date="2021-07-02"
    )
  for band in PRODUCT_BANDS[Product.ERA5_LAND]:
    assert band in ds_out.data_vars
  assert MISSING_FRACTION_VAR[Product.ERA5_LAND] in ds_out.data_vars
  assert np.allclose(ds_out["era5land_temperature_2m_min"].values, 14.0, atol=1e-3)
  assert np.allclose(ds_out["era5land_temperature_2m"].values, 20.0, atol=1e-3)
  assert np.allclose(ds_out["era5land_temperature_2m_max"].values, 26.0, atol=1e-3)
  assert np.all(np.isnan(ds_out["era5land_snow_depth_water_equivalent"].values))
  assert np.allclose(ds_out["era5land_missing_fraction"].values, 0.0)

  snow_warnings = [
      rec
      for rec in caplog.records
      if "era5land_snow_depth_water_equivalent" in rec.getMessage()
  ]
  assert len(snow_warnings) == 1


def test_hres_archive_extraction_unit_conversion_and_lon_wrapping(
    tmp_path, basins_gdf
):
  """Tests HRES archive extraction from [0, 360) longitudes and native ECMWF units."""
  lats = np.linspace(-90.0, 90.0, 181, dtype=np.float32)
  lons = np.linspace(0.0, 359.0, 360, dtype=np.float32)  # [0, 360) convention
  times = pd.date_range("2023-03-01", "2023-03-02", freq="D")
  leads = np.arange(1, 11, dtype=np.int32)
  shape = (len(times), len(leads), len(lats), len(lons))

  ds_archive = xr.Dataset(
      data_vars={
          "temperature_2m": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 293.15, dtype=np.float32),  # 20.0 degC
          ),
          "surface_pressure": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 101325.0, dtype=np.float32),  # 101.325 kPa
          ),
          "total_precipitation": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 0.015, dtype=np.float32),  # 15.0 mm
          ),
          "surface_net_solar_radiation": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 86400.0 * 200.0, dtype=np.float32),  # 200.0 W/m^2
          ),
          "surface_net_thermal_radiation": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, -86400.0 * 50.0, dtype=np.float32),  # -50.0 W/m^2
          ),
      },
      coords={
          "time": times.values,
          "lead_time": leads,
          "latitude": lats,
          "longitude": lons,
      },
  )
  store_path = tmp_path / "hres_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = HRESExtractor(data_dir=str(store_path), source="archive")
  ds_out = ext.extract_for_basins(
      basins_gdf, start_date="2023-03-01", end_date="2023-03-02"
  )
  assert ds_out["hres_temperature_2m"].shape == (len(basins_gdf), 2, 10)
  assert np.issubdtype(ds_out["lead_time"].dtype, np.timedelta64)
  assert np.allclose(ds_out["hres_temperature_2m"].values, 20.0, atol=1e-2)
  assert np.allclose(ds_out["hres_surface_pressure"].values, 101.325, atol=1e-2)
  assert np.allclose(ds_out["hres_total_precipitation"].values, 15.0, atol=1e-2)
  assert np.allclose(
      ds_out["hres_surface_net_solar_radiation"].values, 200.0, atol=1e-2
  )
  assert np.allclose(
      ds_out["hres_surface_net_thermal_radiation"].values, -50.0, atol=1e-2
  )
  assert np.allclose(ds_out["hres_missing_fraction"].values, 0.0)


def test_strict_validation_missing_dates_and_non_overlapping_window(
    tmp_path, basins_gdf
):
  """Verifies that omitting dates or requesting dates outside the archive raises."""
  lats = np.linspace(-89.75, 89.75, 36, dtype=np.float32)
  lons = np.linspace(-179.75, 179.75, 72, dtype=np.float32)
  times = pd.date_range("2020-01-01", "2020-01-05", freq="D")
  ds_archive = xr.Dataset(
      data_vars={
          "imerg_precipitation": (
              ["time", "latitude", "longitude"],
              np.full((len(times), len(lats), len(lons)), 4.0, dtype=np.float32),
          ),
      },
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  )
  store_path = tmp_path / "imerg_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = IMERGExtractor(data_dir=str(store_path), source="archive")
  with pytest.raises(ValueError, match="start_date and end_date"):
    ext.extract_for_basins(basins_gdf)

  with pytest.raises(GriddedArchiveError, match="no overlap"):
    ext.extract_for_basins(
        basins_gdf, start_date="2025-01-01", end_date="2025-01-05"
    )


def test_serial_runner_with_archive_stores(tmp_path, basins_gdf):
  """Tests extract_multimet_serial end-to-end using explicit archive_stores."""
  lats = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
  lons = np.linspace(-179.75, 179.75, 720, dtype=np.float32)
  times = pd.date_range("2022-01-01", "2022-01-02", freq="D")

  cpc_store = tmp_path / "cpc_archive.zarr"
  xr.Dataset(
      data_vars={
          "cpc_precipitation": (
              ["time", "latitude", "longitude"],
              np.full((2, 360, 720), 6.0, dtype=np.float32),
          ),
      },
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  ).to_zarr(cpc_store)

  out_dir = tmp_path / "output_stores"
  stores = extract_multimet_serial(
      basins=basins_gdf,
      output_dir=out_dir,
      products=["CPC"],
      start_date="2022-01-01",
      end_date="2022-01-02",
      source="archive",
      archive_stores={"CPC": str(cpc_store)},
  )
  assert "CPC" in stores
  ds_written = xr.open_zarr(stores["CPC"])
  assert np.allclose(ds_written["cpc_precipitation"].values, 6.0, atol=1e-3)
  assert np.allclose(ds_written["cpc_missing_fraction"].values, 0.0, atol=1e-3)
