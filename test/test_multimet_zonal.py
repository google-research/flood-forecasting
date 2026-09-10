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

"""Unit tests for MultiMet zonal weight calculation and sparse matrix vectorization."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from googlehydrology.multimet.geometry import load_basin_geometries
from googlehydrology.multimet.zonal import (
    ZonalWeightCalculator,
    ZonalWeightMatrix,
)


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


def test_zonal_weights_normalization(basins_gdf):
  lats = np.arange(42.0, 37.0, -0.1)
  lons = np.arange(-89.0, -84.0, 0.1)
  calc = ZonalWeightCalculator(lats, lons)

  for b_id, row in basins_gdf.iterrows():
    geom = row["geometry"]
    lat_idx, lon_idx, weights = calc.compute_weights(b_id, geom)

    assert len(lat_idx) > 0
    assert len(lat_idx) == len(lon_idx) == len(weights)
    assert np.all(weights >= 0.0)
    assert np.isclose(float(np.sum(weights)), 1.0, atol=1e-4)


def test_zonal_weights_caching(basins_gdf):
  lats = np.arange(42.0, 37.0, -0.1)
  lons = np.arange(-89.0, -84.0, 0.1)
  calc = ZonalWeightCalculator(lats, lons)

  b_id = basins_gdf.index[0]
  geom = basins_gdf.loc[b_id, "geometry"]

  res1 = calc.compute_weights(b_id, geom)
  res2 = calc.compute_weights(b_id, geom)
  assert res1 is res2


def test_reduce_grid_uniform_field(basins_gdf):
  lats = np.arange(42.0, 37.0, -0.1)
  lons = np.arange(-89.0, -84.0, 0.1)
  calc = ZonalWeightCalculator(lats, lons)

  field_2d = np.full((len(lats), len(lons)), 42.0, dtype=np.float32)
  for b_id, row in basins_gdf.iterrows():
    val = calc.reduce_grid(field_2d, b_id, row["geometry"])
    assert np.isclose(val, 42.0, atol=1e-4)


def test_reduce_grid_partial_nans(basins_gdf):
  lats = np.arange(42.0, 37.0, -0.1)
  lons = np.arange(-89.0, -84.0, 0.1)
  calc = ZonalWeightCalculator(lats, lons)

  field_2d = np.full((len(lats), len(lons)), 10.0, dtype=np.float32)
  field_2d[:, : len(lons) // 2] = np.nan

  for b_id, row in basins_gdf.iterrows():
    val = calc.reduce_grid(field_2d, b_id, row["geometry"])
    if not np.isnan(val):
      assert np.isclose(val, 10.0, atol=1e-4)


def test_zonal_weight_matrix_sparse_reduction(basins_gdf):
  lats = np.arange(42.0, 37.0, -0.1)
  lons = np.arange(-89.0, -84.0, 0.1)
  matrix = ZonalWeightMatrix.from_geodataframe(
      basins_gdf, lats, lons, cell_res_lat=0.1, cell_res_lon=0.1
  )

  assert matrix.num_basins == len(basins_gdf)
  assert matrix.grid_shape == (len(lats), len(lons))

  # 2D test
  field_2d = np.full((len(lats), len(lons)), 25.0, dtype=np.float32)
  res_2d = matrix.reduce_2d(field_2d)
  assert res_2d.shape == (len(basins_gdf),)
  assert np.allclose(res_2d, 25.0, atol=1e-4)

  # 3D test (time, lat, lon)
  T = 5
  field_3d = np.zeros((T, len(lats), len(lons)), dtype=np.float32)
  for t in range(T):
    field_3d[t, :, :] = 10.0 + t
  res_3d = matrix.reduce_3d(field_3d)
  assert res_3d.shape == (len(basins_gdf), T)
  for t in range(T):
    assert np.allclose(res_3d[:, t], 10.0 + t, atol=1e-4)

  # 4D test (time, lead_time, lat, lon)
  K = 10
  field_4d = np.zeros((T, K, len(lats), len(lons)), dtype=np.float32)
  for t in range(T):
    for k in range(K):
      field_4d[t, k, :, :] = t * 10.0 + k
  res_4d = matrix.reduce_4d(field_4d)
  assert res_4d.shape == (len(basins_gdf), T, K)
  for t in range(T):
    for k in range(K):
      assert np.allclose(res_4d[:, t, k], t * 10.0 + k, atol=1e-4)


def test_zonal_weight_matrix_serialization(basins_gdf, tmp_path):
  lats = np.arange(42.0, 37.0, -0.1)
  lons = np.arange(-89.0, -84.0, 0.1)
  matrix = ZonalWeightMatrix.from_geodataframe(
      basins_gdf, lats, lons, cell_res_lat=0.1, cell_res_lon=0.1
  )

  save_path = tmp_path / "weights.npz"
  matrix.save(save_path)
  assert save_path.exists()

  loaded = ZonalWeightMatrix.load(save_path)
  assert loaded.num_basins == matrix.num_basins
  assert loaded.basin_ids == matrix.basin_ids
  assert np.allclose(loaded.lats, matrix.lats)
  assert np.allclose(loaded.lons, matrix.lons)

  # Test subset
  sub_ids = matrix.basin_ids[:2]
  sub_matrix = loaded.subset(sub_ids)
  assert sub_matrix.num_basins == 2
  assert sub_matrix.basin_ids == sub_ids
