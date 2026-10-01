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

"""Unit tests for generic DynamicalDataLoader and DynamicalExtractor with Icechunk."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray as xr

from multimet.config import Product
from multimet.dynamical import (
    AIFSExtractor,
    DynamicalDataLoader,
    DynamicalDatasetInfo,
    DynamicalExtractor,
    DynamicalIMERGExtractor,
    clear_catalog_cache,
    list_catalog_datasets,
    load_dynamical,
)
from multimet.geometry import load_basin_geometries
from multimet.zarr_writer import MultiMetZarrWriter


pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def basins_gdf():
  path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  return load_basin_geometries(path)


def test_list_catalog_datasets():
  """Verifies that dynamical.org catalog datasets can be listed."""
  datasets = list_catalog_datasets()
  assert isinstance(datasets, list)
  assert len(datasets) > 0
  assert "nasa-imerg-analysis-early" in datasets
  assert "noaa-gfs-forecast" in datasets
  assert "noaa-hrrr-analysis" in datasets


def test_loader_schema_analysis_geographic_1d():
  """Tests schema and coordinate inspection on 1D geographic dataset (IMERG)."""
  loader = DynamicalDataLoader("nasa-imerg-analysis-early")
  assert loader.grid_type == "geographic_1d"
  assert loader.spatial_dims == ("latitude", "longitude")
  assert loader.lat_coord == "latitude"
  assert loader.lon_coord == "longitude"
  assert loader.lat_descending is True
  assert loader.lon_ascending is True
  assert loader.time_dim == "time"
  assert loader.has_lead_time is False

  info = loader.get_info()
  assert isinstance(info, DynamicalDatasetInfo)
  assert info.dataset_id == "nasa-imerg-analysis-early"
  assert "precipitation_surface" in info.variables


def test_spatial_slice_computation_geographic(basins_gdf):
  """Verifies spatial bounding slice calculation for 1D geographic datasets."""
  loader = DynamicalDataLoader("nasa-imerg-analysis-early")

  # Test from GeoDataFrame
  slices = loader.compute_spatial_slices(basins_gdf, buffer=0.2)
  lat_slice = slices["latitude"]
  lon_slice = slices["longitude"]

  minx, miny, maxx, maxy = basins_gdf.total_bounds
  # Latitudes are descending, so slice.start > slice.stop
  assert lat_slice.start > lat_slice.stop
  assert np.isclose(lat_slice.start, maxy + 0.2, atol=1e-3)
  assert np.isclose(lat_slice.stop, miny - 0.2, atol=1e-3)
  assert np.isclose(lon_slice.start, minx - 0.2, atol=1e-3)
  assert np.isclose(lon_slice.stop, maxx + 0.2, atol=1e-3)

  # Test from tuple (min_lon, min_lat, max_lon, max_lat)
  bbox = (-88.0, 39.0, -85.0, 41.0)
  slices_bbox = loader.compute_spatial_slices(bbox, buffer=0.1)
  assert slices_bbox["latitude"].start > slices_bbox["latitude"].stop
  assert np.isclose(slices_bbox["latitude"].start, 41.1)
  assert np.isclose(slices_bbox["latitude"].stop, 38.9)


def test_loader_schema_analysis_projected_2d():
  """Tests schema inspection on 2D projected dataset (HRRR)."""
  loader = DynamicalDataLoader("noaa-hrrr-analysis")
  assert loader.grid_type == "projected_2d"
  assert loader.spatial_dims == ("y", "x")
  assert loader.y_coord == "y"
  assert loader.x_coord == "x"
  assert loader.crs_wkt is not None
  assert "Lambert_Conformal_Conic" in loader.crs_wkt or "PROJCS" in loader.crs_wkt

  info = loader.get_info()
  assert info.grid_type == "projected_2d"
  assert "precipitation_surface" in info.variables


def test_spatial_slice_computation_projected(basins_gdf):
  """Verifies spatial bounding slice calculation with pyproj reprojection for HRRR."""
  loader = DynamicalDataLoader("noaa-hrrr-analysis")
  slices = loader.compute_spatial_slices(basins_gdf)
  assert "y" in slices
  assert "x" in slices
  y_slice = slices["y"]
  x_slice = slices["x"]

  # Slices should be in projected meters (> 100,000 m)
  assert abs(y_slice.start) > 10000.0 or abs(y_slice.stop) > 10000.0
  assert abs(x_slice.start) > 10000.0 or abs(x_slice.stop) > 10000.0


def test_load_spatial_subset_icechunk_acceleration(basins_gdf):
  """Tests that load_spatial_subset retrieves only the target spatial bounding box via Icechunk."""
  loader = DynamicalDataLoader("nasa-imerg-analysis-early")

  # Request small 2-hour window and prune to only 1 variable
  sub = loader.load_spatial_subset(
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T02:00:00",
      buffer=0.1,
      compute=True,
  )

  assert isinstance(sub, xr.Dataset)
  assert "precipitation_surface" in sub.data_vars
  assert "precipitation_quality_index_surface" not in sub.data_vars
  assert sub.sizes["time"] == 5  # 00:00, 00:30, 01:00, 01:30, 02:00
  # Verify spatial clipping: instead of 1800x3600 global grid, small local window
  assert sub.sizes["latitude"] < 50
  assert sub.sizes["longitude"] < 50


def test_extract_basin_timeseries_geographic(basins_gdf):
  """Tests exact catchment zonal averaging for 1D geographic datasets."""
  loader = DynamicalDataLoader("nasa-imerg-analysis-early")

  ts_ds = loader.extract_basin_timeseries(
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T01:00:00",
      buffer=0.1,
  )

  assert isinstance(ts_ds, xr.Dataset)
  assert "basin" in ts_ds.dims
  assert "date" in ts_ds.dims
  assert list(ts_ds.basin.values) == list(basins_gdf.index)
  assert ts_ds["precipitation_surface"].shape == (len(basins_gdf), 3)
  assert not np.isnan(ts_ds["precipitation_surface"].values).all()


def test_forecast_dataset_slicing(basins_gdf):
  """Tests loading and slicing a forecast dataset with lead_time (GFS)."""
  loader = DynamicalDataLoader("noaa-gfs-forecast")
  assert loader.has_lead_time is True
  assert loader.time_dim == "init_time"

  sub = loader.load_spatial_subset(
      watersheds=basins_gdf,
      variables=["downward_short_wave_radiation_flux_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T00:00:00",
      lead_time_slice=slice(0, 3),
      buffer=0.2,
      compute=True,
  )

  assert sub.sizes["init_time"] == 1
  assert sub.sizes["lead_time"] == 3
  assert sub.sizes["latitude"] < 25
  assert sub.sizes["longitude"] < 25


def test_dynamical_extractor_adapter(basins_gdf):
  """Tests DynamicalExtractor adapter conforming to BaseExtractor."""
  extractor = DynamicalExtractor(
      dataset_id="nasa-imerg-analysis-early",
      variable_map={"precipitation_surface": "imerg_precipitation"},
      unit_conversions={"imerg_precipitation": lambda x: x * 1000.0},
  )

  res = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T01:00:00",
  )

  assert "imerg_precipitation" in res.data_vars
  assert "basin" in res.dims
  assert "date" in res.dims
  assert len(res.basin) == len(basins_gdf)


def test_load_dynamical_convenience_function(basins_gdf):
  """Tests top-level load_dynamical convenience function in cube and timeseries modes."""
  # Mode 1: cube
  cube = load_dynamical(
      dataset_id="nasa-imerg-analysis-early",
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T00:30:00",
      mode="cube",
      compute=True,
  )
  assert "latitude" in cube.dims
  assert "longitude" in cube.dims

  # Mode 2: timeseries
  ts = load_dynamical(
      dataset_id="nasa-imerg-analysis-early",
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T00:30:00",
      mode="timeseries",
  )
  assert "basin" in ts.dims
  assert "date" in ts.dims


def test_extract_basin_timeseries_projected(basins_gdf):
  """Tests exact catchment zonal averaging on projected 2D datasets (HRRR)."""
  loader = DynamicalDataLoader("noaa-hrrr-analysis")
  ts_ds = loader.extract_basin_timeseries(
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T01:00:00",
  )
  assert isinstance(ts_ds, xr.Dataset)
  assert "basin" in ts_ds.dims
  assert "date" in ts_ds.dims
  assert list(ts_ds.basin.values) == list(basins_gdf.index)
  assert ts_ds["precipitation_surface"].shape == (len(basins_gdf), 2)
  assert not np.isnan(ts_ds["precipitation_surface"].values).all()


def test_dynamical_imerg_extractor(basins_gdf, tmp_path):
  """Tests DynamicalIMERGExtractor produces schema-compliant daily precipitation."""
  extractor = DynamicalIMERGExtractor()
  ds = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2020-01-01",
      end_date="2020-01-01",
  )

  assert isinstance(ds, xr.Dataset)
  assert ds.dims == {"basin": len(basins_gdf), "date": 1}
  assert "imerg_precipitation" in ds.data_vars
  assert ds["imerg_precipitation"].dtype == np.float32
  assert not np.isnan(ds["imerg_precipitation"].values).any()
  assert (ds["imerg_precipitation"].values >= 0.0).all()

  # Test single-day helper
  day_res = extractor.extract_day(pd.Timestamp("2020-01-01"), basins_gdf)
  assert "imerg_precipitation" in day_res
  assert len(day_res["imerg_precipitation"]) == len(basins_gdf)

  # Validate against MultiMetZarrWriter schema
  writer = MultiMetZarrWriter(tmp_path)
  writer.validate_dataset_schema(ds, Product.DYNAMICAL_IMERG)


def test_aifs_extractor(basins_gdf, tmp_path):
  """Tests AIFSExtractor produces schema-compliant 10-day daily forecasts."""
  extractor = AIFSExtractor()
  ds = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2024-05-01",
      end_date="2024-05-01",
  )

  assert isinstance(ds, xr.Dataset)
  assert ds.dims == {"basin": len(basins_gdf), "date": 1, "lead_time": 10}
  expected_bands = [
      "aifs_temperature_2m",
      "aifs_total_precipitation",
      "aifs_u_component_of_wind_10m",
      "aifs_v_component_of_wind_10m",
  ]
  for band in expected_bands:
    assert band in ds.data_vars
    assert ds[band].dtype == np.float32
    assert not np.isnan(ds[band].values).any()

  # Test single-day helper
  day_res = extractor.extract_day(pd.Timestamp("2024-05-01"), basins_gdf)
  for band in expected_bands:
    assert band in day_res
    assert day_res[band].shape == (len(basins_gdf), 10)

  # Validate against MultiMetZarrWriter schema
  writer = MultiMetZarrWriter(tmp_path)
  writer.validate_dataset_schema(ds, Product.AIFS)


def test_gfs_extractor(basins_gdf, tmp_path):
  """Tests GFSExtractor produces schema-compliant 10-day daily forecasts across 1h and 3h steps."""
  from multimet.dynamical import GFSExtractor

  extractor = GFSExtractor()
  ds = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2023-01-01",
      end_date="2023-01-01",
  )

  assert isinstance(ds, xr.Dataset)
  assert ds.dims == {"basin": len(basins_gdf), "date": 1, "lead_time": 10}
  expected_bands = [
      "gfs_temperature_2m",
      "gfs_total_precipitation",
      "gfs_u_component_of_wind_10m",
      "gfs_v_component_of_wind_10m",
      "gfs_missing_fraction",
  ]
  for band in expected_bands:
    assert band in ds.data_vars
    assert ds[band].dtype == np.float32
    assert not np.isnan(ds[band].values).any()
  assert np.allclose(ds["gfs_missing_fraction"].values, 0.0)
  assert (ds["gfs_total_precipitation"].values >= 0.0).all()

  writer = MultiMetZarrWriter(tmp_path)
  writer.validate_dataset_schema(ds, Product.GFS)


class _SyntheticForecastLoader:
  """In-memory synthetic dynamical forecast loader for unit testing."""

  def __init__(self, ds: xr.Dataset, has_ensemble: bool = False):
    self.ds = ds
    self.time_dim = "init_time"
    self.spatial_dims = ("latitude", "longitude")
    self.has_lead_time = True
    self.has_ensemble = has_ensemble
    self.calls = []

  def load_spatial_subset(
      self,
      watersheds,
      variables=None,
      start_date=None,
      end_date=None,
      lead_time_slice=None,
      ensemble_members=None,
      buffer=None,
      compute=False,
      use_bounding_box=True,
  ) -> xr.Dataset:
    self.calls.append((start_date, end_date, lead_time_slice))
    sub = self.ds[variables] if variables else self.ds
    if start_date is not None or end_date is not None:
      s_ts = pd.to_datetime(start_date)
      e_ts = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
      sub = sub.sel(init_time=slice(s_ts, e_ts))
    if isinstance(lead_time_slice, slice):
      sub = sub.isel(lead_time=lead_time_slice)
    if ensemble_members is not None and "ensemble_member" in sub.dims:
      sub = sub.isel(ensemble_member=ensemble_members)
    return sub


def test_dynamical_forecast_strict_nan_and_spinup_1d(basins_gdf):
  """Verifies strict NaN propagation on partial 24h windows and spinup_only_before 1D optimization."""
  from multimet.dynamical import (
      GFSExtractor,
      find_latest_dynamical_forecast_date,
  )

  minx, miny, maxx, maxy = basins_gdf.total_bounds
  lats = np.linspace(maxy + 0.5, miny - 0.5, 6, dtype=np.float64)
  lons = np.linspace(minx - 0.5, maxx + 0.5, 6, dtype=np.float64)
  init_times = pd.to_datetime([
      "2026-04-01T00:00:00",
      "2026-04-01T06:00:00",  # non-00z should be ignored
      "2026-04-02T00:00:00",
      "2026-04-03T00:00:00",
  ])
  # GFS-style mixed lead times: 1h for 0..120h, 3h for 123..240h (161 steps)
  lead_hours = list(range(0, 121, 1)) + list(range(123, 241, 3))
  lead_td = pd.to_timedelta(lead_hours, unit="h")

  shape = (len(init_times), len(lead_td), len(lats), len(lons))
  t2m = np.full(shape, 15.0, dtype=np.float32)
  pr = np.full(shape, 1.0 / 86400.0, dtype=np.float32)  # 1 mm/day
  pr[:, 0, :, :] = np.nan  # step 0h is NaN in upstream
  u10 = np.full(shape, 2.5, dtype=np.float32)
  v10 = np.full(shape, -1.5, dtype=np.float32)

  # Inject a single NaN step at lead_time=18h (index 18) on 2026-04-02 (init index 2)
  pr[2, 18, :, :] = np.nan
  # Inject NaN at terminal 240h step on 2026-04-03 (init index 3 -> lead day 10 incomplete)
  t2m[3, -1, :, :] = np.nan

  synth_ds = xr.Dataset(
      data_vars={
          "temperature_2m": (["init_time", "lead_time", "latitude", "longitude"], t2m),
          "precipitation_surface": (["init_time", "lead_time", "latitude", "longitude"], pr),
          "wind_u_10m": (["init_time", "lead_time", "latitude", "longitude"], u10),
          "wind_v_10m": (["init_time", "lead_time", "latitude", "longitude"], v10),
      },
      coords={
          "init_time": init_times,
          "lead_time": lead_td,
          "latitude": lats,
          "longitude": lons,
      },
  )
  loader = _SyntheticForecastLoader(synth_ds)

  # 1. Auto-discovery with require_full_10d=True skips 2026-04-03 (240h is NaN) and returns 2026-04-02
  latest_full = find_latest_dynamical_forecast_date(
      dataset_id="noaa-gfs-forecast",
      reference_date="2026-04-03",
      require_full_10d=True,
      loader=loader,
  )
  assert latest_full == pd.Timestamp("2026-04-02")

  # 2. Extract with spinup_only_before="2026-04-02"
  extractor = GFSExtractor(loader=loader)
  ds_out = extractor.extract_for_basins(
      basins_gdf,
      start_date="2026-04-01",
      end_date="2026-04-03",
      spinup_only_before="2026-04-02",
  )

  # Spin-up date 2026-04-01 (index 0): only lead_time=1D (index 0) is populated; leads 2D..10D are NaN
  assert np.allclose(ds_out["gfs_temperature_2m"].values[:, 0, 0], 15.0)
  assert np.allclose(ds_out["gfs_total_precipitation"].values[:, 0, 0], 1.0, atol=1e-4)
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 0, 0], 0.0)
  assert np.all(np.isnan(ds_out["gfs_temperature_2m"].values[:, 0, 1:]))
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 0, 1:], 1.0)

  # Forecast date 2026-04-02 (index 1): lead day 1 had 1 NaN sub-step in pr -> pr is NaN and missing_fraction is 1.0!
  assert np.all(np.isnan(ds_out["gfs_total_precipitation"].values[:, 1, 0]))
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 1, 0], 1.0)
  # Lead days 2..10 on 2026-04-02 (spanning both 1h steps in days 2..5 and 3h steps in days 6..10) are 100% valid!
  assert np.allclose(ds_out["gfs_total_precipitation"].values[:, 1, 1:], 1.0, atol=1e-4)
  assert np.allclose(ds_out["gfs_temperature_2m"].values[:, 1, 1:], 15.0)
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 1, 1:], 0.0)

  # Forecast date 2026-04-03 (index 2): lead day 10 (index 9) had NaN at 240h -> strictly NaN!
  assert np.all(np.isnan(ds_out["gfs_temperature_2m"].values[:, 2, 9]))
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 2, 9], 1.0)
  assert np.allclose(ds_out["gfs_temperature_2m"].values[:, 2, :9], 15.0)



