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

from __future__ import annotations

import concurrent.futures
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
import shapely.geometry
import xarray as xr


class ZonalWeightCalculator:
  """Computes and caches intersection weights between basin polygons and regular lat/lon grids."""

  def __init__(
      self,
      lats: np.ndarray,
      lons: np.ndarray,
      cell_res_lat: Optional[float] = None,
      cell_res_lon: Optional[float] = None,
  ):
    """Initializes the weight calculator for a given 1D lat/lon coordinate grid."""
    self.lats = np.asarray(lats, dtype=np.float64)
    self.lons = np.asarray(lons, dtype=np.float64)

    if cell_res_lat is None:
      self.dlat = (
          abs(float(self.lats[1] - self.lats[0])) if len(self.lats) > 1 else 0.1
      )
    else:
      self.dlat = abs(float(cell_res_lat))

    if cell_res_lon is None:
      self.dlon = (
          abs(float(self.lons[1] - self.lons[0])) if len(self.lons) > 1 else 0.1
      )
    else:
      self.dlon = abs(float(cell_res_lon))

    self._weights_cache: Dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

  def compute_weights(
      self, basin_id: str, polygon: shapely.geometry.base.BaseGeometry
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes (lat_indices, lon_indices, normalized_weights) for a polygon.

    Returns:
      lat_idx: 1D array of latitude indices
      lon_idx: 1D array of longitude indices
      weights: 1D array of normalized area weights (summing to 1.0)
    """
    if basin_id in self._weights_cache:
      return self._weights_cache[basin_id]

    minx, miny, maxx, maxy = polygon.bounds

    # Handle both ascending and descending latitudes
    lat_mask = (self.lats >= miny - self.dlat) & (self.lats <= maxy + self.dlat)
    lon_mask = (self.lons >= minx - self.dlon) & (self.lons <= maxx + self.dlon)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lat_list = []
    lon_list = []
    w_list = []

    half_lat = self.dlat / 2.0
    half_lon = self.dlon / 2.0

    poly_area = polygon.area
    if poly_area <= 0.0:
      poly_area = 1.0

    for li in lat_indices:
      lat_val = self.lats[li]
      cell_miny = lat_val - half_lat
      cell_maxy = lat_val + half_lat

      # Cosine weighting for latitude cell area distortion
      lat_cos = np.cos(np.radians(lat_val))

      for lj in lon_indices:
        lon_val = self.lons[lj]
        cell_minx = lon_val - half_lon
        cell_maxx = lon_val + half_lon

        cell_box = shapely.geometry.box(
            cell_minx, cell_miny, cell_maxx, cell_maxy
        )
        if polygon.intersects(cell_box):
          inter = polygon.intersection(cell_box)
          inter_area = inter.area * lat_cos
          if inter_area > 0.0:
            lat_list.append(li)
            lon_list.append(lj)
            w_list.append(inter_area)

    weights = np.array(w_list, dtype=np.float32)
    if len(weights) > 0 and weights.sum() > 0:
      weights /= weights.sum()
    else:
      # Fallback to nearest representative point (guaranteed inside polygon)
      rep_point = polygon.representative_point()
      nearest_lat_idx = int(np.argmin(np.abs(self.lats - rep_point.y)))
      nearest_lon_idx = int(np.argmin(np.abs(self.lons - rep_point.x)))
      lat_list = [nearest_lat_idx]
      lon_list = [nearest_lon_idx]
      weights = np.array([1.0], dtype=np.float32)

    res = (
        np.array(lat_list, dtype=np.int32),
        np.array(lon_list, dtype=np.int32),
        weights,
    )
    self._weights_cache[basin_id] = res
    return res

  def reduce_grid(
      self,
      grid_data: np.ndarray,
      basin_id: str,
      polygon: shapely.geometry.base.BaseGeometry,
  ) -> float:
    """Reduces a 2D (lat, lon) slice over the polygon using weighted areal mean."""
    lat_idx, lon_idx, weights = self.compute_weights(basin_id, polygon)
    vals = grid_data[lat_idx, lon_idx]
    # Filter out NaNs if partial coverage
    valid = ~np.isnan(vals)
    if not np.any(valid):
      return np.nan
    valid_weights = weights[valid]
    if valid_weights.sum() == 0:
      return np.nan
    return float(np.sum(vals[valid] * valid_weights) / valid_weights.sum())


class ZonalWeightMatrix:
  """Sparse weight matrix for vectorized batch areal reduction of gridded datasets.

  Represents the intersection weights between N basin polygons and a regular 2D
  (lat, lon) grid of shape (H, W) as a SciPy CSR sparse matrix of shape (N, H * W).

  Areal mean reduction for any variable field X (2D, 3D, or 4D) is evaluated via
  sparse BLAS matrix multiplication (``Y = W * X_vec``), taking milliseconds even
  for thousands of basins across multi-decade records.
  """

  def __init__(
      self,
      basin_ids: Sequence[str],
      lats: np.ndarray,
      lons: np.ndarray,
      matrix: sp.csr_matrix,
  ):
    self.basin_ids = list(basin_ids)
    self.lats = np.asarray(lats, dtype=np.float64)
    self.lons = np.asarray(lons, dtype=np.float64)
    self.matrix = matrix.tocsr().astype(np.float32)
    expected_shape = (len(self.basin_ids), len(self.lats) * len(self.lons))
    if self.matrix.shape != expected_shape:
      raise ValueError(
          f"Matrix shape {self.matrix.shape} does not match expected "
          f"{expected_shape} (N_basins={len(self.basin_ids)}, "
          f"H={len(self.lats)}, W={len(self.lons)})"
      )
    self.basin_id_to_row: Dict[str, int] = {
        b_id: i for i, b_id in enumerate(self.basin_ids)
    }

  @property
  def num_basins(self) -> int:
    return len(self.basin_ids)

  @property
  def grid_shape(self) -> Tuple[int, int]:
    return (len(self.lats), len(self.lons))

  @classmethod
  def from_geodataframe(
      cls,
      basins_gdf: gpd.GeoDataFrame,
      lats: np.ndarray,
      lons: np.ndarray,
      cell_res_lat: Optional[float] = None,
      cell_res_lon: Optional[float] = None,
      num_workers: int = 1,
  ) -> ZonalWeightMatrix:
    """Builds a ZonalWeightMatrix from a basin GeoDataFrame and coordinate arrays."""
    basin_ids = list(basins_gdf.index)
    calc = ZonalWeightCalculator(
        lats, lons, cell_res_lat=cell_res_lat, cell_res_lon=cell_res_lon
    )
    n_lons = len(calc.lons)

    def _calc_basin(b_idx: int, b_id: str):
      geom = basins_gdf.loc[b_id].geometry
      lat_idx, lon_idx, w = calc.compute_weights(b_id, geom)
      col_idx = lat_idx * n_lons + lon_idx
      return b_idx, col_idx, w

    rows = []
    cols = []
    data = []

    if num_workers > 1 and len(basin_ids) > 10:
      with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_calc_basin, idx, b_id)
            for idx, b_id in enumerate(basin_ids)
        ]
        for f in concurrent.futures.as_completed(futures):
          b_idx, col_idx, w = f.result()
          rows.extend([b_idx] * len(col_idx))
          cols.extend(col_idx)
          data.extend(w)
    else:
      for idx, b_id in enumerate(basin_ids):
        b_idx, col_idx, w = _calc_basin(idx, b_id)
        rows.extend([b_idx] * len(col_idx))
        cols.extend(col_idx)
        data.extend(w)

    csr = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(basin_ids), len(calc.lats) * len(calc.lons)),
        dtype=np.float32,
    )
    return cls(basin_ids, calc.lats, calc.lons, csr)

  def get_basin_weights(
      self, basin_id: str
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lat_idx, lon_idx, weights) for a specific basin."""
    if basin_id not in self.basin_id_to_row:
      raise KeyError(f"Basin '{basin_id}' not found in weight matrix.")
    row = self.basin_id_to_row[basin_id]
    col_idx = self.matrix.indices[
        self.matrix.indptr[row] : self.matrix.indptr[row + 1]
    ]
    weights = self.matrix.data[
        self.matrix.indptr[row] : self.matrix.indptr[row + 1]
    ]
    n_lons = len(self.lons)
    lat_idx = col_idx // n_lons
    lon_idx = col_idx % n_lons
    return lat_idx, lon_idx, weights

  def reduce_2d(self, grid_2d: np.ndarray) -> np.ndarray:
    """Reduces a 2D (lat, lon) grid across all basins.

    Args:
      grid_2d: Array of shape (H, W).

    Returns:
      1D array of shape (N_basins,) with areal weighted means.
    """
    if grid_2d.shape != self.grid_shape:
      raise ValueError(
          f"Grid shape {grid_2d.shape} does not match expected {self.grid_shape}"
      )
    flat = grid_2d.reshape(-1)
    if not np.isnan(flat).any():
      return self.matrix.dot(flat).astype(np.float32)

    valid = ~np.isnan(flat)
    val_0 = np.nan_to_num(flat, nan=0.0)
    S = self.matrix.dot(val_0)
    V = self.matrix.dot(valid.astype(np.float32))
    with np.errstate(divide="ignore", invalid="ignore"):
      res = np.where(V > 0, S / V, np.nan)
    return res.astype(np.float32)

  def reduce_3d(self, grid_3d: np.ndarray) -> np.ndarray:
    """Reduces a 3D (time, lat, lon) grid across all basins.

    Args:
      grid_3d: Array of shape (T, H, W).

    Returns:
      2D array of shape (N_basins, T) with areal weighted means.
    """
    T, H, W = grid_3d.shape
    if (H, W) != self.grid_shape:
      raise ValueError(
          f"Grid spatial shape ({H}, {W}) does not match expected {self.grid_shape}"
      )
    X = grid_3d.reshape(T, -1)
    if not np.isnan(X).any():
      # (N, C) @ (C, T) -> (N, T)
      return self.matrix.dot(X.T).astype(np.float32)

    valid = ~np.isnan(X)
    X0 = np.nan_to_num(X, nan=0.0)
    S = self.matrix.dot(X0.T)
    V = self.matrix.dot(valid.astype(np.float32).T)
    with np.errstate(divide="ignore", invalid="ignore"):
      res = np.where(V > 0, S / V, np.nan)
    return res.astype(np.float32)

  def reduce_4d(self, grid_4d: np.ndarray) -> np.ndarray:
    """Reduces a 4D (time, lead_time/member, lat, lon) grid across all basins.

    Args:
      grid_4d: Array of shape (T, K, H, W).

    Returns:
      3D array of shape (N_basins, T, K) with areal weighted means.
    """
    T, K, H, W = grid_4d.shape
    flat_3d = grid_4d.reshape(T * K, H, W)
    res_2d = self.reduce_3d(flat_3d)  # (N_basins, T * K)
    return res_2d.reshape(len(self.basin_ids), T, K).astype(np.float32)

  def subset(self, basin_ids: Sequence[str]) -> ZonalWeightMatrix:
    """Returns a new ZonalWeightMatrix containing only the requested basins."""
    row_indices = []
    valid_ids = []
    for b_id in basin_ids:
      if b_id in self.basin_id_to_row:
        row_indices.append(self.basin_id_to_row[b_id])
        valid_ids.append(b_id)
      else:
        raise KeyError(f"Basin '{b_id}' not found in weight matrix.")
    sub_matrix = self.matrix[row_indices]
    return ZonalWeightMatrix(valid_ids, self.lats, self.lons, sub_matrix)

  def save(self, path: Union[str, os.PathLike]) -> str:
    """Serializes the weight matrix to a compressed .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        data=self.matrix.data,
        indices=self.matrix.indices,
        indptr=self.matrix.indptr,
        shape=self.matrix.shape,
        basin_ids=np.array(self.basin_ids, dtype=object),
        lats=self.lats,
        lons=self.lons,
    )
    return str(path)

  @classmethod
  def load(cls, path: Union[str, os.PathLike]) -> ZonalWeightMatrix:
    """Loads a serialized weight matrix from a compressed .npz file."""
    data = np.load(path, allow_pickle=True)
    matrix = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"]),
        dtype=np.float32,
    )
    basin_ids = [str(b) for b in data["basin_ids"]]
    lats = data["lats"]
    lons = data["lons"]
    return cls(basin_ids, lats, lons, matrix)
