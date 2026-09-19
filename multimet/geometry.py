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

import glob
import io
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import fsspec
import geopandas as gpd
import pandas as pd
import shapely.validation

from multimet.spatial import BoundingBox

logger = logging.getLogger(__name__)

SUPPORTED_GEOMETRY_EXTENSIONS = (".shp", ".geojson", ".gpkg")


def _discover_geometry_files(directory: Union[str, os.PathLike]) -> List[str]:
  """Recursively discovers all shapefiles, geojsons, and geopackages in a directory."""
  found = []
  for root, _, files in os.walk(directory):
    for f in files:
      f_lower = f.lower()
      ext = os.path.splitext(f_lower)[1]
      if ext in SUPPORTED_GEOMETRY_EXTENSIONS:
        # Avoid point gauge files when scanning directories of watershed boundaries
        if f_lower.endswith(("_gauges.shp", "_gauges.geojson", "_gauges.gpkg", "_gauge.shp")):
          continue
        found.append(os.path.join(root, f))
  return sorted(found)


def _resolve_geometry_sources(
    source: Any,
) -> List[Union[str, gpd.GeoDataFrame, Dict[str, Any]]]:
  """Resolves single paths, comma-separated paths, lists, globs, or directories into discrete sources."""
  if isinstance(source, (gpd.GeoDataFrame, dict)):
    return [source]

  if isinstance(source, (list, tuple, set)):
    resolved = []
    for item in source:
      resolved.extend(_resolve_geometry_sources(item))
    seen = set()
    deduped = []
    for item in resolved:
      if isinstance(item, (str, os.PathLike)):
        item_str = str(item)
        proto, _ = fsspec.core.split_protocol(item_str)
        norm = item_str if proto else os.path.abspath(item_str)
        if norm not in seen:
          seen.add(norm)
          deduped.append(item_str)
      else:
        deduped.append(item)
    return deduped

  if isinstance(source, (str, os.PathLike)):
    raw_s = str(source).strip()
    if not raw_s:
      return []

    protocol, path_in_fs = fsspec.core.split_protocol(raw_s)
    if protocol in ("gs", "gcs", "s3"):
      if protocol in ("gs", "gcs"):
        from multimet.gcp import configure_gcp_project
        configure_gcp_project()
      fs, path = fsspec.core.url_to_fs(raw_s)
      # Check if remote glob pattern
      if any(char in path for char in ("*", "?", "[")):
        matches = sorted(fs.glob(path))
        if not matches:
          raise FileNotFoundError(f"No files matched remote glob pattern: {raw_s}")
        resolved = []
        for m in matches:
          if fs.isdir(m):
            sub_files = fs.find(m)
            for sf in sub_files:
              sf_l = sf.lower()
              if sf_l.endswith(SUPPORTED_GEOMETRY_EXTENSIONS) and not sf_l.endswith(
                  ("_gauges.shp", "_gauges.geojson", "_gauges.gpkg", "_gauge.shp")
              ):
                resolved.append(f"{protocol}://{sf}")
          elif fs.isfile(m):
            m_l = m.lower()
            if m_l.endswith(SUPPORTED_GEOMETRY_EXTENSIONS) and not m_l.endswith(
                ("_gauges.shp", "_gauges.geojson", "_gauges.gpkg", "_gauge.shp")
            ):
              resolved.append(f"{protocol}://{m}")
        if not resolved:
          raise FileNotFoundError(f"No supported geometry files found for: {raw_s}")
        return resolved

      # Check if remote directory
      if fs.isdir(path):
        all_remote = sorted(fs.find(path))
        resolved = []
        for f in all_remote:
          f_l = f.lower()
          if f_l.endswith(SUPPORTED_GEOMETRY_EXTENSIONS) and not f_l.endswith(
              ("_gauges.shp", "_gauges.geojson", "_gauges.gpkg", "_gauge.shp")
          ):
            resolved.append(f"{protocol}://{f}")
        if not resolved:
          raise FileNotFoundError(
              f"No supported geometry files ({', '.join(SUPPORTED_GEOMETRY_EXTENSIONS)}) found in remote directory: {raw_s}"
          )
        return resolved

      # Single remote file
      if fs.isfile(path):
        return [raw_s]

      raise FileNotFoundError(f"Remote geometry path does not exist: {raw_s}")

    s = os.path.expanduser(os.path.expandvars(raw_s))

    # If it's an existing directory, search recursively
    if os.path.isdir(s):
      files = _discover_geometry_files(s)
      if not files:
        raise FileNotFoundError(
            f"No supported geometry files ({', '.join(SUPPORTED_GEOMETRY_EXTENSIONS)}) found in directory: {s}"
        )
      return files

    # If it's an existing file, return directly
    if os.path.isfile(s):
      return [s]

    # If it contains commas, split and resolve each component
    if "," in s:
      parts = [p.strip() for p in s.split(",") if p.strip()]
      resolved = []
      for p in parts:
        resolved.extend(_resolve_geometry_sources(p))
      return resolved

    # Check if it's a glob pattern
    if any(char in s for char in ("*", "?", "[")):
      matches = sorted(glob.glob(s, recursive=True))
      if not matches:
        raise FileNotFoundError(f"No files matched glob pattern: {s}")
      resolved = []
      for m in matches:
        if os.path.isdir(m):
          resolved.extend(_discover_geometry_files(m))
        elif os.path.isfile(m):
          ext = os.path.splitext(m)[1].lower()
          if ext in SUPPORTED_GEOMETRY_EXTENSIONS:
            resolved.append(m)
      if not resolved:
        raise FileNotFoundError(
            f"No supported geometry files matched glob pattern: {s}"
        )
      return resolved

    raise FileNotFoundError(f"Geometry source path does not exist: {s}")

  raise TypeError(f"Unsupported geometry source type: {type(source)}")


def _load_single_basin_geometry(
    source: Union[str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any]],
    id_column: Optional[str] = None,
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
  """Loads and standardizes a single geometry source."""
  if isinstance(source, gpd.GeoDataFrame):
    gdf = source.copy()
  elif isinstance(source, (str, os.PathLike)):
    source_str = str(source)
    proto, path_in_fs = fsspec.core.split_protocol(source_str)
    if proto in ("gs", "gcs", "s3"):
      if proto in ("gs", "gcs"):
        from multimet.gcp import configure_gcp_project
        configure_gcp_project()
      fs, path = fsspec.core.url_to_fs(source_str)
      ext = os.path.splitext(path)[1].lower()
      if ext in (".geojson", ".json"):
        with fsspec.open(source_str, "r") as f:
          gdf = gpd.read_file(f)
      else:
        # For multi-file formats (.shp, .gpkg), cache locally to tempdir
        cache_dir = os.path.join(
            tempfile.gettempdir(),
            "multimet_geometries",
            os.path.dirname(path).strip("/").replace("/", "_"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        local_target = os.path.join(cache_dir, os.path.basename(path))
        if not os.path.exists(local_target):
          parent_dir = os.path.dirname(path)
          stem = os.path.splitext(os.path.basename(path))[0]
          companion_files = fs.glob(f"{parent_dir}/{stem}.*")
          logger.info("Caching remote geometry bundle from %s to %s", source_str, local_target)
          fs.get(companion_files, cache_dir)
        gdf = gpd.read_file(local_target)
    else:
      gdf = gpd.read_file(source_str)
  elif isinstance(source, dict):
    gdf = gpd.GeoDataFrame.from_features(source)
  else:
    raise TypeError(f"Unsupported geometry source type: {type(source)}")

  if gdf.empty:
    return gpd.GeoDataFrame(columns=["geometry"], crs=target_crs)

  # Reproject if CRS is set and differs from target_crs.
  if gdf.crs is not None:
    if gdf.crs.to_string() != target_crs:
      gdf = gdf.to_crs(target_crs)
  else:
    gdf = gdf.set_crs(target_crs)

  # Identify ID column.
  if id_column is not None:
    if id_column not in gdf.columns:
      raise KeyError(f"Specified id_column '{id_column}' not found in dataset: {source}")
    basin_ids = gdf[id_column].astype(str)
  else:
    candidates = [
        "basin_id",
        "basin",
        "gauge_id",
        "gauge",
        "station_id",
        "HYBAS_ID",
        "hybas_id",
        "Official_ID",
        "official_id",
        "watershed_id",
        "provider_id",
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
    elif gdf.index.name in candidates or (not gdf.index.empty and not isinstance(gdf.index, pd.RangeIndex)):
      basin_ids = gdf.index.astype(str)
    else:
      stem = ""
      if isinstance(source, (str, os.PathLike)):
        stem = f"{Path(source).stem}_"
      basin_ids = pd.Series([f"{stem}basin_{i}" for i in range(len(gdf))])

  gdf["basin_id"] = basin_ids.values
  gdf = gdf.set_index("basin_id")

  # Ensure valid polygon / multipolygon geometries.
  def _ensure_valid(geom):
    if geom is None or geom.is_empty:
      return geom
    if geom.is_valid:
      return geom
    return shapely.validation.make_valid(geom)

  gdf["geometry"] = gdf["geometry"].apply(_ensure_valid)
  gdf = gdf[gdf["geometry"].notna() & (~gdf["geometry"].is_empty)]
  return gdf


def load_basin_geometries(
    source: Union[
        str,
        os.PathLike,
        gpd.GeoDataFrame,
        Dict[str, Any],
        Sequence[Union[str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any]]],
    ],
    id_column: Optional[str] = None,
    target_crs: str = "EPSG:4326",
    drop_duplicates: bool = True,
) -> gpd.GeoDataFrame:
  """Loads and standardizes basin geometries into a WGS84 GeoDataFrame.

  Supports single files, comma-separated files, lists of files/globs/dirs,
  directory paths containing shapefiles or geojsons, and glob expressions.
  Standardizes CRS to EPSG:4326, identifies basin IDs, makes geometries valid,
  and combines all inputs into a single GeoDataFrame indexed by basin_id.

  Args:
    source: Path to geometry file (Shapefile, GeoJSON, etc.), directory of
      geometry files, glob pattern, sequence of paths, an existing
      GeoDataFrame, or GeoJSON dict.
    id_column: Optional column name containing the basin/gauge identifier. If
      None, will search for common candidate names ('basin_id', 'gauge_id',
      'id', 'HYBAS_ID', etc.) or default to the DataFrame index.
    target_crs: Coordinate reference system to reproject to (default
      'EPSG:4326').
    drop_duplicates: Whether to drop duplicate basin IDs across combined files
      (retaining first occurrence). If False and duplicates exist, raises ValueError.

  Returns:
    gpd.GeoDataFrame indexed by string basin IDs with geometry in EPSG:4326.
  """
  discrete_sources = _resolve_geometry_sources(source)
  if not discrete_sources:
    raise ValueError(f"No geometry sources found for: {source}")

  gdfs = []
  for src in discrete_sources:
    single_gdf = _load_single_basin_geometry(
        src, id_column=id_column, target_crs=target_crs
    )
    if not single_gdf.empty:
      gdfs.append(single_gdf)

  if not gdfs:
    raise ValueError("Input geometry dataset is empty or contains no valid geometries.")

  if len(gdfs) == 1:
    combined = gdfs[0]
  else:
    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, axis=0),
        crs=target_crs,
        geometry="geometry",
    )

  if combined.index.duplicated().any():
    num_dups = int(combined.index.duplicated().sum())
    if drop_duplicates:
      logger.warning(
          "Dropping %d duplicate basin IDs from combined geometries.",
          num_dups,
      )
      combined = combined[~combined.index.duplicated(keep="first")]
    else:
      dups = list(combined.index[combined.index.duplicated()].unique())
      raise ValueError(
          f"Found {num_dups} duplicate basin IDs across sources (samples: {dups[:5]})."
      )

  return combined


def get_bounding_box(
    gdf: gpd.GeoDataFrame,
    buffer_degrees: float = 0.1,
) -> Tuple[float, float, float, float]:
  """Returns total bounding box (min_lon, min_lat, max_lon, max_lat)."""
  return BoundingBox.from_geodataframe(
      gdf, buffer_degrees=buffer_degrees
  ).to_tuple()
