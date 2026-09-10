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

import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import shapely.validation



def load_basin_geometries(
    source: Union[str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any]],
    id_column: Optional[str] = None,
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
  """Loads and standardizes basin geometries into a WGS84 GeoDataFrame.

  Args:
    source: Path to geometry file (Shapefile, GeoJSON, etc.), an existing
      GeoDataFrame, or GeoJSON dict.
    id_column: Optional column name containing the basin/gauge identifier. If
      None, will search for common candidate names ('basin_id', 'gauge_id',
      'id', 'HYBAS_ID', etc.) or default to the DataFrame index.
    target_crs: Coordinate reference system to reproject to (default
      'EPSG:4326').

  Returns:
    gpd.GeoDataFrame indexed by string basin IDs with geometry in EPSG:4326.
  """
  if isinstance(source, gpd.GeoDataFrame):
    gdf = source.copy()
  elif isinstance(source, (str, os.PathLike)):
    source_str = str(source)
    gdf = gpd.read_file(source_str)
  elif isinstance(source, dict):
    gdf = gpd.GeoDataFrame.from_features(source)
  else:
    raise TypeError(f"Unsupported geometry source type: {type(source)}")

  if gdf.empty:
    raise ValueError("Input geometry dataset is empty.")

  # Reproject if CRS is set and differs from target_crs.
  if gdf.crs is not None:
    if gdf.crs.to_string() != target_crs:
      gdf = gdf.to_crs(target_crs)
  else:
    # Assume target_crs if not specified.
    gdf = gdf.set_crs(target_crs)

  # Identify ID column.
  if id_column is not None:
    if id_column not in gdf.columns:
      raise KeyError(f"Specified id_column '{id_column}' not found in dataset.")
    basin_ids = gdf[id_column].astype(str)
  else:
    candidates = [
        "basin_id",
        "basin",
        "gauge_id",
        "gauge",
        "HYBAS_ID",
        "hybas_id",
        "station_id",
        "ID",
        "id",
        "name",
    ]
    found = None
    for cand in candidates:
      if cand in gdf.columns:
        found = cand
        break
    if found is not None:
      basin_ids = gdf[found].astype(str)
    else:
      basin_ids = pd.Series([f"basin_{i}" for i in range(len(gdf))])

  gdf["basin_id"] = basin_ids.values
  gdf = gdf.set_index("basin_id")

  # Ensure valid polygon / multipolygon geometries.
  def _ensure_valid(geom):
    if geom.is_valid:
      return geom
    try:
      return shapely.validation.make_valid(geom)
    except Exception:
      return geom.buffer(0)

  gdf["geometry"] = gdf["geometry"].apply(_ensure_valid)
  return gdf


def get_bounding_box(
    gdf: gpd.GeoDataFrame,
    buffer_degrees: float = 0.1,
) -> Tuple[float, float, float, float]:
  """Returns total bounding box (min_lon, min_lat, max_lon, max_lat)."""
  minx, miny, maxx, maxy = gdf.total_bounds
  return (
      max(-180.0, minx - buffer_degrees),
      max(-90.0, miny - buffer_degrees),
      min(180.0, maxx + buffer_degrees),
      min(90.0, maxy + buffer_degrees),
  )
