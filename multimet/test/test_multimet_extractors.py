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

"""Unit tests for the 5 MultiMet extractors in local serial execution."""

from pathlib import Path
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from multimet.config import (
    PRODUCT_BANDS,
    Product,
)
from multimet.cpc import CPCExtractor
from multimet.era5_land import ERA5LandExtractor
from multimet.geometry import load_basin_geometries
from multimet.graphcast import (
    GraphCastExtractor,
    _compute_basin_steps,
)
from multimet.hres import (
    HRESExtractor,
    _extract_accumulated_lead,
    _extract_instantaneous_lead,
)
from multimet.imerg import IMERGExtractor
from multimet.pet import calculate_fao56_penman_monteith_pet
from multimet.runner import extract_multimet_serial
from multimet.zarr_writer import MultiMetZarrWriter


@pytest.fixture
def basins_gdf():
  path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  return load_basin_geometries(path)


def test_fao56_penman_monteith_pet():
  """Tests FAO-56 Penman-Monteith potential evapotranspiration formula."""
  t2m = np.array([293.15, 303.15], dtype=np.float32)  # 20 C, 30 C
  d2m = np.array([288.15, 293.15], dtype=np.float32)  # 15 C, 20 C
  sp = np.array([101325.0, 101325.0], dtype=np.float32)
  ssr = np.array([1.5e7, 2.0e7], dtype=np.float32)  # J/m^2
  str_flux = np.array([-4.0e6, -5.0e6], dtype=np.float32)
  u10 = np.array([2.0, 3.0], dtype=np.float32)
  v10 = np.array([1.0, 2.0], dtype=np.float32)

  pet = calculate_fao56_penman_monteith_pet(
      t2m, d2m, sp, ssr, str_flux, u10, v10
  )
  assert pet.shape == (2,)
  assert np.all(pet > 0.0)
  assert pet[1] > pet[0]  # Warmer, higher radiation -> higher PET


def test_cpc_extractor_binary_parsing(tmp_path):
  """Tests CPCExtractor binary reading and scaling logic."""
  # Create synthetic CPC daily binary file (360 lat x 720 lon)
  # Values in 0.1 mm
  shape = (360, 720)
  precip_data = np.full(shape, 150.0, dtype="<f4")  # 15.0 mm/day
  num_stations = np.ones(shape, dtype="<f4")
  binary_content = (
      precip_data.tobytes() + num_stations.tobytes()
  )

  cpc_file = tmp_path / "PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx.20200101"
  cpc_file.write_bytes(binary_content)

  grid = CPCExtractor.parse_cpc_file(str(cpc_file))
  assert grid.shape == (360, 720)
  # Should be scaled to 15.0 mm/day
  assert np.isclose(np.nanmean(grid), 15.0, atol=1e-3)


def test_era5_land_strict_missing_day_handling(basins_gdf):
  """Verifies that incomplete hourly file lists return all NaNs for ERA5-Land."""
  extractor = ERA5LandExtractor()
  basin_ids = list(basins_gdf.index)
  weights_dict = {
      b_id: extractor.zonal_calc.compute_weights(
          b_id, basins_gdf.loc[b_id].geometry
      )
      for b_id in basin_ids
  }

  # Passing incomplete file list (e.g. 5 hours instead of 24)
  res = extractor.extract_day_from_grib_files(
      ["file1", "file2", "file3", "file4", "file5"],
      basin_ids,
      weights_dict,
  )
  for band in PRODUCT_BANDS[Product.ERA5_LAND]:
    assert np.all(np.isnan(res[band]))


def test_imerg_daily_extraction(tmp_path, basins_gdf):
  """Tests IMERG extractor on synthetic daily NetCDF file."""
  basin_ids = list(basins_gdf.index)
  extractor = IMERGExtractor()

  # Create a small synthetic NetCDF4 file matching IMERG coordinate layout
  ds_mock = xr.Dataset(
      data_vars={
          "precipitation": (
              ["lat", "lon"],
              np.full((1800, 3600), 8.5, dtype=np.float32),
          )
      },
      coords={
          "lat": extractor.lats,
          "lon": extractor.lons,
      },
  )
  nc_path = tmp_path / "3B-DAY-E.MS.MRG.3IMERG.20200101-S000000-E235959.V07B.nc4"
  ds_mock.to_netcdf(nc_path)

  res = extractor.extract_day_from_nc4(str(nc_path), basins_gdf)
  assert "imerg_precipitation" in res
  vals = res["imerg_precipitation"]
  assert len(vals) == len(basin_ids)
  assert np.allclose(vals, 8.5, atol=1e-3)


def test_hres_lead_differencing(tmp_path, basins_gdf):
  """Tests HRES accumulation differencing for precipitation."""
  extractor = HRESExtractor()
  basin_ids = list(basins_gdf.index)
  weights_dict = {
      b_id: extractor.zonal_calc.compute_weights(
          b_id, basins_gdf.loc[b_id].geometry
      )
      for b_id in basin_ids
  }

  # Create mock Zarr store for HRES
  store_path = tmp_path / "mock_hres.zarr"
  z_root = zarr.open_group(str(store_path), mode="w")

  # 24h lead accumulation steps:
  # step 24: 0.010 m (10 mm)
  # step 48: 0.025 m (15 mm increment)
  hours = 72
  data_tp = np.zeros((hours, len(extractor.lats), len(extractor.lons)), dtype=np.float32)
  data_tp[24, :, :] = 0.010
  data_tp[48, :, :] = 0.025
  arr = z_root.create_array("total_precipitation", shape=data_tp.shape, dtype=data_tp.dtype)
  arr[:] = data_tp


  # Day 1: 0 to 24h
  lead1 = _extract_accumulated_lead(
      z_root,
      "total_precipitation",
      h_start=0,
      h_end=24,
      t0_hours=0,
      sort_lon_idx=extractor.sort_lon_idx,
      basin_ids=basin_ids,
      weights_dict=weights_dict,
      scale=1000.0,
  )
  assert np.allclose(lead1, 10.0, atol=1e-3)

  # Day 2: 24h to 48h
  lead2 = _extract_accumulated_lead(
      z_root,
      "total_precipitation",
      h_start=24,
      h_end=48,
      t0_hours=0,
      sort_lon_idx=extractor.sort_lon_idx,
      basin_ids=basin_ids,
      weights_dict=weights_dict,
      scale=1000.0,
  )
  assert np.allclose(lead2, 15.0, atol=1e-3)


def test_graphcast_step_aggregation(basins_gdf):
  """Tests DeepMind GraphCast 40-step forecast reduction."""
  extractor = GraphCastExtractor()
  b_id = basins_gdf.index[0]
  geom = basins_gdf.loc[b_id].geometry
  lat_idx, lon_idx, w = extractor.zonal_calc.compute_weights(b_id, geom)

  # 40 forecast steps (6-hourly up to 240 hours)
  grid_40 = np.full((40, len(extractor.lats), len(extractor.lons)), 2.5, dtype=np.float32)
  basin_steps = _compute_basin_steps(grid_40, lat_idx, lon_idx, w)

  assert basin_steps.shape == (40,)
  assert np.allclose(basin_steps, 2.5, atol=1e-4)


def test_serial_runner_end_to_end(tmp_path, basins_gdf):
  """Tests extract_multimet_serial on synthetic data."""
  out_dir = tmp_path / "multimet_serial_out"
  geojson_path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )

  # Mock CPCExtractor.extract_for_basins to return synthetic dataset
  dates = pd.date_range("2020-01-01", periods=2, freq="D")
  basin_ids = list(basins_gdf.index)
  mock_cpc_ds = xr.Dataset(
      data_vars={
          "cpc_precipitation": (
              ["basin", "date"],
              np.full((len(basin_ids), len(dates)), 7.0, dtype=np.float32),
          ),
      },
      coords={"basin": basin_ids, "date": dates.values},
  )

  with mock.patch.object(
      CPCExtractor, "extract_for_basins", return_value=mock_cpc_ds
  ):
    stores = extract_multimet_serial(
        basins=geojson_path,
        output_dir=out_dir,
        products=["CPC"],
        start_date="2020-01-01",
        end_date="2020-01-02",
        source="public",
    )
    assert "CPC" in stores
    store_path = stores["CPC"]
    assert Path(store_path).exists()

    ds_read = xr.open_zarr(store_path)
    assert "cpc_precipitation" in ds_read.data_vars
    assert len(ds_read["date"]) == 2
    assert len(ds_read["basin"]) == len(basin_ids)
    assert np.allclose(ds_read["cpc_precipitation"].values, 7.0)
