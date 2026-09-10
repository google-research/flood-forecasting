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

from multimet.dynamical import (
    DynamicalDataLoader,
    DynamicalDatasetInfo,
    DynamicalExtractor,
    clear_catalog_cache,
    list_catalog_datasets,
    load_dynamical,
)
from multimet.geometry import load_basin_geometries


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

