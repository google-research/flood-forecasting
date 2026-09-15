"""Pure DEM Flow-Direction Watershed Delineator.

Performs authentic reverse-flow tree traversal on high-resolution (90m / 3 arc-second)
D8 flow direction rasters (HydroSHEDS DIR / MERIT flwdir) with seamless multi-tile
boundary traversal, delineating completely natural, curving watershed boundaries
across arbitrary 5x5 degree tile boundaries without edge artifacts.
"""

from collections import deque
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from catchment_delineation.config import (
    GCS_TILES_URI,
    INFLOW_MAP,
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
    get_default_cache_dir,
)
from catchment_delineation.tiles import tile_key_to_filename


class DemDelineator:
  """High-performance multi-tile DEM watershed delineator using D8 flow direction matrices."""

  def __init__(
      self,
      tiles_dir: Optional[Union[str, Path]] = None,
      cache_tiles: bool = True,
      gcs_uri: str = GCS_TILES_URI,
      cache_dir: Optional[Union[str, Path]] = None,
  ):
    """Initializes the DEM delineator.

    Args:
        tiles_dir: Optional user-supplied directory containing DEM .npy tiles.
          If provided, tiles are loaded exclusively from this path (no searching).
          If None, tiles are loaded exclusively from the gs bucket (GCS_TILES_URI).
        cache_tiles: If True, caches memory-mapped tile references in memory.
        gcs_uri: GCS bucket URI for DEM tiles (default: gs://open-multimet/data/DEMs/tiles_5deg).
        cache_dir: Local cache directory for tiles retrieved from GCS.
    """
    if tiles_dir is not None:
      self.tiles_dir = Path(tiles_dir).expanduser().resolve()
    else:
      self.tiles_dir = None

    self.gcs_uri = gcs_uri
    self.cache_dir = (
        Path(cache_dir).expanduser() if cache_dir else get_default_cache_dir()
    )
    self.cache_tiles = cache_tiles
    self._tile_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

  def get_tile(self, lat_top: float, lon_left: float) -> Optional[np.ndarray]:
    """Loads a 5x5 degree tile array with memory-mapping for instant retrieval.

    Args:
        lat_top: Northern boundary of the 5x5 degree tile (integer multiple of 5).
        lon_left: Western boundary of the 5x5 degree tile (integer multiple of 5).

    Returns:
        Memory-mapped 2D numpy array of shape (6000, 6000), or None if tile not found.
    """
    key = (int(round(lat_top)), int(round(lon_left)))
    if self.cache_tiles and key in self._tile_cache:
      return self._tile_cache[key]

    tile_name = tile_key_to_filename(key[0], key[1])

    if self.tiles_dir is not None:
      # User supplied their own path: ONLY load from this path, no searching or falling back
      tile_path = self.tiles_dir / tile_name
      if not tile_path.exists():
        if self.cache_tiles:
          self._tile_cache[key] = None
        return None
    else:
      # Default: ONLY load from the gs bucket (cached locally)
      tile_path = self.cache_dir / tile_name
      if not tile_path.exists():
        try:
          from catchment_delineation.gcs import download_tile_from_gcs

          tile_path = download_tile_from_gcs(
              key[0], key[1], target_dir=self.cache_dir, source_uri=self.gcs_uri
          )
        except Exception as e:
          print(f"Error downloading DEM tile {tile_name} from {self.gcs_uri}: {e}")
          if self.cache_tiles:
            self._tile_cache[key] = None
          return None

    try:
      arr = np.load(tile_path, mmap_mode="r")
      if self.cache_tiles:
        self._tile_cache[key] = arr
      return arr
    except Exception as e:
      print(f"Error loading DEM tile {tile_path}: {e}")
      if self.cache_tiles:
        self._tile_cache[key] = None
      return None


  def snap_outlet(
      self,
      lat: float,
      lon: float,
      snap_window_cells: int = 4,
  ) -> Tuple[float, float, int, int, Tuple[int, int], float]:
    """Snaps input coordinates to the nearest channel outlet cell within search window.

    Args:
        lat: Target latitude.
        lon: Target longitude.
        snap_window_cells: Half-width of search box in grid cells.

    Returns:
        Tuple of (snapped_lat, snapped_lon, best_r, best_c, start_key, snap_distance_m).
    """
    lat_top = float(math.ceil(lat / TILE_DEG) * TILE_DEG)
    lon_left = float(math.floor(lon / TILE_DEG) * TILE_DEG)

    r0 = int(round((lat_top - lat) / RES_DEG))
    c0 = int(round((lon - lon_left) / RES_DEG))

    r0 = max(0, min(TILE_CELLS - 1, r0))
    c0 = max(0, min(TILE_CELLS - 1, c0))

    start_key = (int(round(lat_top)), int(round(lon_left)))
    start_grid = self.get_tile(*start_key)

    if start_grid is None:
      raise FileNotFoundError(
          f"DEM tile not found for coordinates ({lat}, {lon}) in {self.tiles_dir}."
      )

    best_r, best_c = r0, c0
    best_cnt = -1

    for dr in range(-snap_window_cells, snap_window_cells + 1):
      for dc in range(-snap_window_cells, snap_window_cells + 1):
        tr, tc = r0 + dr, c0 + dc
        if 0 <= tr < TILE_CELLS and 0 <= tc < TILE_CELLS:
          # Bounded local search to measure upstream channel connectivity
          sq = deque([(tr, tc, 0)])
          svis = {(tr, tc)}
          cnt = 0
          while sq and cnt < 300:
            cr, cc, depth = sq.popleft()
            cnt += 1
            if depth >= 25:
              continue
            for d_r, d_c, req in INFLOW_MAP:
              nr, nc = cr + d_r, cc + d_c
              if 0 <= nr < TILE_CELLS and 0 <= nc < TILE_CELLS:
                if (nr, nc) not in svis and start_grid[nr, nc] == req:
                  svis.add((nr, nc))
                  sq.append((nr, nc, depth + 1))
          dist_penalty = int((dr * dr + dc * dc) * 0.2)
          score = cnt - dist_penalty
          if score > best_cnt:
            best_cnt = score
            best_r, best_c = tr, tc

    outlet_lat = lat_top - best_r * RES_DEG
    outlet_lon = lon_left + best_c * RES_DEG
    snap_dist_m = float(
        math.hypot(
            (outlet_lat - lat) * 111000.0,
            (outlet_lon - lon) * 111000.0 * math.cos(math.radians(lat)),
        )
    )

    return outlet_lat, outlet_lon, best_r, best_c, start_key, snap_dist_m

  def delineate(
      self,
      lat: float,
      lon: float,
      snap_window_cells: int = 4,
      max_cells: int = 5000000,
      simplify_tolerance: Optional[float] = None,
      catchment_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Delineates the upstream catchment basin draining to (lat, lon) on the DEM grid.

    Seamlessly traverses across contiguous 5x5 degree tile boundaries.

    Args:
        lat: Target latitude.
        lon: Target longitude.
        snap_window_cells: Half-width of search box in grid cells to find local channel.
        max_cells: Safety limit for total cells traversed.
        simplify_tolerance: Geometry simplification tolerance in degrees (default: 0.4 * RES_DEG).
        catchment_id: Custom catchment identifier string (auto-generated if None).

    Returns:
        GeoJSON Feature dict with natural multi-tile polygon geometry and properties.
    """
    (
        outlet_lat,
        outlet_lon,
        best_r,
        best_c,
        start_key,
        snap_dist_m,
    ) = self.snap_outlet(lat, lon, snap_window_cells=snap_window_cells)

    # 1. Multi-Tile BFS Reverse Flow Traversal
    q = deque([(start_key, best_r, best_c)])
    visited_tiles: Dict[Tuple[int, int], np.ndarray] = {
        start_key: np.zeros((TILE_CELLS, TILE_CELLS), dtype=bool)
    }
    visited_tiles[start_key][best_r, best_c] = True
    total_accum = 0

    while q and total_accum < max_cells:
      (t_lat, t_lon), cr, cc = q.popleft()
      total_accum += 1

      for dr, dc, req_val in INFLOW_MAP:
        nr, nc = cr + dr, cc + dc
        nt_lat, nt_lon = t_lat, t_lon

        if nr < 0:
          nr += TILE_CELLS
          nt_lat += int(TILE_DEG)
        elif nr >= TILE_CELLS:
          nr -= TILE_CELLS
          nt_lat -= int(TILE_DEG)

        if nc < 0:
          nc += TILE_CELLS
          nt_lon -= int(TILE_DEG)
        elif nc >= TILE_CELLS:
          nc -= TILE_CELLS
          nt_lon += int(TILE_DEG)

        nkey = (nt_lat, nt_lon)
        if nkey not in visited_tiles:
          ngrid = self.get_tile(*nkey)
          if ngrid is None:
            continue
          visited_tiles[nkey] = np.zeros((TILE_CELLS, TILE_CELLS), dtype=bool)
        else:
          ngrid = self.get_tile(*nkey)
          if ngrid is None:
            continue

        v_mask = visited_tiles[nkey]
        if not v_mask[nr, nc] and ngrid[nr, nc] == req_val:
          v_mask[nr, nc] = True
          q.append((nkey, nr, nc))

    # 2. Calculate accurate ground area summed across all visited tiles
    total_area_km2 = 0.0
    for (t_lat, _), mask in visited_tiles.items():
      cell_cnt = int(mask.sum())
      if cell_cnt > 0:
        mid_lat = t_lat - 2.5
        lat_scale = RES_DEG * 111.0
        lon_scale = RES_DEG * 111.0 * math.cos(math.radians(mid_lat))
        total_area_km2 += cell_cnt * lat_scale * lon_scale
    if total_area_km2 < 0.1:
      total_area_km2 = round(float(total_area_km2), 4)
    else:
      total_area_km2 = round(float(total_area_km2), 1)

    # 3. Vectorize boolean raster masks across all visited tiles into unified polygon
    if simplify_tolerance is None:
      simplify_tolerance = RES_DEG * 0.4

    tile_polys = []
    for (t_lat, t_lon), mask in visited_tiles.items():
      if mask.any():
        p = self._vectorize_tile_mask(
            mask, float(t_lat), float(t_lon), simplify_tolerance
        )
        if p and not p.is_empty:
          tile_polys.append(p)

    if not tile_polys:
      p0 = (outlet_lon, outlet_lat)
      poly = Polygon([
          (p0[0], p0[1]),
          (p0[0] + RES_DEG, p0[1]),
          (p0[0] + RES_DEG, p0[1] - RES_DEG),
          (p0[0], p0[1] - RES_DEG),
      ])
    elif len(tile_polys) == 1:
      poly = tile_polys[0]
    else:
      poly = unary_union(tile_polys)
      if not poly.is_valid:
        poly = poly.buffer(0)
      poly = poly.simplify(simplify_tolerance)

    # Compute bounding box
    bounds = poly.bounds
    bbox_dict = {
        "min_lon": round(float(bounds[0]), 5),
        "min_lat": round(float(bounds[1]), 5),
        "max_lon": round(float(bounds[2]), 5),
        "max_lat": round(float(bounds[3]), 5),
    }

    if catchment_id is None:
      catchment_id = (
          f"catchment_dem_{abs(int(outlet_lat * 10000))}_{abs(int(outlet_lon * 10000))}"
      )

    geojson_feature = {
        "type": "Feature",
        "properties": {
            "catchment_id": catchment_id,
            "area_km2": total_area_km2,
            "upstream_cells_count": int(total_accum),
            "tiles_spanned_count": len(
                [k for k, m in visited_tiles.items() if m.any()]
            ),
            "grid_resolution": "90m (3 arc-second)",
            "outlet": {
                "input_latitude": lat,
                "input_longitude": lon,
                "latitude": round(outlet_lat, 5),
                "longitude": round(outlet_lon, 5),
                "reach_id": f"DEM_{int(best_r)}_{int(best_c)}",
                "snap_distance_m": round(snap_dist_m, 1),
            },
            "reach_attributes": {
                "reach_id": f"DEM_CELL_{int(best_r)}_{int(best_c)}",
                "dataset": "dem_flow_direction",
                "river_name": (
                    f"DEM Flow Path ({outlet_lat:.4f}°N, {outlet_lon:.4f}°E)"
                ),
                "stream_order": (
                    1
                    if total_area_km2 < 50
                    else (2 if total_area_km2 < 500 else 3)
                ),
                "upstream_area_km2": total_area_km2,
            },
            "bbox": bbox_dict,
            "delineation_method": (
                "DEM Digital Elevation Flow-Routing (90m HydroSHEDS Multi-Tile Seamless Grid)"
            ),
            "delineation_mode": "dem_flow_direction",
        },
        "geometry": mapping(poly),
    }

    return geojson_feature

  def delineate_batch(
      self,
      coords: Iterable[Tuple[float, float]],
      ids: Optional[Iterable[str]] = None,
      snap_window_cells: int = 4,
      max_cells: int = 5000000,
      simplify_tolerance: Optional[float] = None,
  ) -> Dict[str, Any]:
    """Delineates catchments for multiple coordinate pairs and returns a GeoJSON FeatureCollection.

    Args:
        coords: List or iterable of (lat, lon) coordinate tuples.
        ids: Optional list of catchment IDs corresponding to coords.
        snap_window_cells: Half-width of search box in grid cells.
        max_cells: Maximum cells to traverse per catchment.
        simplify_tolerance: Geometry simplification tolerance in degrees.

    Returns:
        GeoJSON FeatureCollection dict containing features for each delineated catchment.
    """
    coords_list = list(coords)
    ids_list = list(ids) if ids is not None else [None] * len(coords_list)
    features = []

    for (lat, lon), cid in zip(coords_list, ids_list):
      feat = self.delineate(
          lat=lat,
          lon=lon,
          snap_window_cells=snap_window_cells,
          max_cells=max_cells,
          simplify_tolerance=simplify_tolerance,
          catchment_id=cid,
      )
      features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features,
    }

  def _vectorize_tile_mask(
      self,
      mask: np.ndarray,
      lat_top: float,
      lon_left: float,
      simplify_tolerance: float,
  ) -> Optional[Polygon]:
    """Converts a 2D boolean tile mask into a simplified Shapely Polygon using row run-length fusion."""
    if not mask.any():
      return None
    min_r, max_r = np.where(mask.any(axis=1))[0][[0, -1]]
    min_c, max_c = np.where(mask.any(axis=0))[0][[0, -1]]
    sub_mask = mask[min_r : max_r + 1, min_c : max_c + 1]

    sub_lat_top = lat_top - min_r * RES_DEG
    sub_lon_left = lon_left + min_c * RES_DEG

    boxes = []
    for r in range(sub_mask.shape[0]):
      row = sub_mask[r]
      if not np.any(row):
        continue
      diff = np.diff(np.pad(row.astype(int), 1))
      starts = np.where(diff == 1)[0]
      ends = np.where(diff == -1)[0]
      y_top = sub_lat_top - r * RES_DEG
      y_bot = sub_lat_top - (r + 1) * RES_DEG
      for s, e in zip(starts, ends):
        x_left = sub_lon_left + s * RES_DEG
        x_right = sub_lon_left + e * RES_DEG
        boxes.append(
            Polygon([
                (x_left, y_bot),
                (x_right, y_bot),
                (x_right, y_top),
                (x_left, y_top),
            ])
        )

    if not boxes:
      p0 = (lon_left + min_c * RES_DEG, lat_top - min_r * RES_DEG)
      return Polygon([
          (p0[0], p0[1]),
          (p0[0] + RES_DEG, p0[1]),
          (p0[0] + RES_DEG, p0[1] - RES_DEG),
          (p0[0], p0[1] - RES_DEG),
      ])

    poly = unary_union(boxes)
    if not poly.is_valid:
      poly = poly.buffer(0)
    return poly.simplify(simplify_tolerance)


def delineate_dem(
    lat: float,
    lon: float,
    tiles_dir: Optional[Union[str, Path]] = None,
    snap_window_cells: int = 4,
    max_cells: int = 5000000,
    catchment_id: Optional[str] = None,
) -> Dict[str, Any]:
  """Convenience function to delineate a catchment from (lat, lon) coordinates using DEM flow direction."""
  delineator = DemDelineator(tiles_dir=tiles_dir)
  return delineator.delineate(
      lat=lat,
      lon=lon,
      snap_window_cells=snap_window_cells,
      max_cells=max_cells,
      catchment_id=catchment_id,
  )


# Alias
delineate_catchment = delineate_dem


def delineate_coordinates(
    coords: Iterable[Tuple[float, float]],
    tiles_dir: Optional[Union[str, Path]] = None,
    ids: Optional[Iterable[str]] = None,
    snap_window_cells: int = 4,
    max_cells: int = 5000000,
) -> Dict[str, Any]:
  """Convenience function to delineate multiple catchments from coordinate tuples."""
  delineator = DemDelineator(tiles_dir=tiles_dir)
  return delineator.delineate_batch(
      coords=coords,
      ids=ids,
      snap_window_cells=snap_window_cells,
      max_cells=max_cells,
  )


