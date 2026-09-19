# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure DEM Flow-Direction Watershed Delineator.

Performs reverse-flow tree traversal on high-resolution (90m / 3 arc-second)
D8 flow-direction rasters (HydroSHEDS DIR / MERIT flwdir) with seamless
multi-tile boundary traversal across 5x5 degree tile boundaries.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import rasterio.features
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from catchment_delineation.config import (
    DEM_MAX_LAT,
    DEM_MAX_LON,
    DEM_MIN_LAT,
    DEM_MIN_LON,
    INFLOW_MAP,
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
)
from catchment_delineation.gcs import (
    download_tile_from_gcs,
    is_gcs_path,
    normalize_gcs_path,
)
from catchment_delineation.tiles import (
    MIN_TILE_LAT_TOP,
    is_coord_in_coverage,
    is_tile_in_coverage,
    tile_key_to_filename,
)

logger = logging.getLogger(__name__)

_SNAP_BFS_MAX_NODES: int = 5000
_SNAP_BFS_MAX_DEPTH: int = 100
_SNAP_DIST_PENALTY_WEIGHT: float = 2.0
_METERS_PER_DEGREE: float = 111000.0
_KM_PER_DEGREE: float = 111.0
_SMALL_AREA_THRESHOLD_KM2: float = 0.1
_STREAM_ORDER_1_MAX_KM2: float = 50.0
_STREAM_ORDER_2_MAX_KM2: float = 500.0
_COORD_SCALE: int = 10000


class CatchmentCoverageError(FileNotFoundError, ValueError):
    """Raised when a pour point or its watershed extends outside DEM bounds."""


def build_missing_feature(
    lat: float,
    lon: float,
    catchment_id: str | None,
    reason: str,
) -> dict[str, Any]:
    """Build an explicit NaN/None Feature for missing or out-of-domain input."""
    nan_val = float('nan')
    return {
        'type': 'Feature',
        'properties': {
            'catchment_id': catchment_id,
            'gauge_id': catchment_id,
            'area_km2': nan_val,
            'area': nan_val,
            'upstream_cells_count': None,
            'tiles_spanned_count': None,
            'grid_resolution': '90m (3 arc-second)',
            'outlet': {
                'input_latitude': lat,
                'input_longitude': lon,
                'latitude': nan_val,
                'longitude': nan_val,
                'reach_id': None,
                'snap_distance_m': nan_val,
            },
            'reach_attributes': None,
            'bbox': None,
            'delineation_method': (
                'DEM Digital Elevation Flow-Routing '
                '(90m HydroSHEDS Multi-Tile Seamless Grid)'
            ),
            'delineation_mode': 'dem_flow_direction',
            'status': f'MISSING_DATA: {reason}',
        },
        'geometry': None,
    }


class DemDelineator:
    """Multi-tile DEM watershed delineator using D8 flow-direction rasters."""

    def __init__(
        self,
        tiles_dir: str | Path | None = None,
        *,
        cache_tiles: bool = True,
        gcs_uri: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the DEM delineator with explicit user-supplied I/O paths.

        Args:
            tiles_dir: Local directory containing 5x5 degree DEM .npy tiles
                (or a gs:// URI if gcs_uri is not set).
            cache_tiles: Whether to memoize memory-mapped tile arrays in RAM.
            gcs_uri: Explicit GCS URI containing 5x5 degree DEM .npy tiles.
                Requires cache_dir to also be provided.
            cache_dir: Explicit local directory used to store tiles downloaded
                from gcs_uri.
        """
        if tiles_dir is not None and is_gcs_path(tiles_dir):
            if gcs_uri is not None:
                raise ValueError(
                    'Provide either a gs:// tiles_dir or gcs_uri, not both.'
                )
            gcs_uri = normalize_gcs_path(tiles_dir)
            tiles_dir = None

        if tiles_dir is not None and gcs_uri is not None:
            raise ValueError(
                'Provide either tiles_dir (local) or gcs_uri (GCS), not both.'
            )

        if tiles_dir is None and gcs_uri is None:
            raise ValueError(
                'An explicit tile source is required: pass tiles_dir '
                '(local directory) or gcs_uri + cache_dir.'
            )

        if tiles_dir is not None:
            resolved_tiles = Path(tiles_dir).expanduser().resolve()
            if not resolved_tiles.is_dir():
                raise FileNotFoundError(
                    f'DEM tiles_dir does not exist: {resolved_tiles}'
                )
            self.tiles_dir: Path | None = resolved_tiles
            self.gcs_uri: str | None = None
            self.cache_dir: Path | None = (
                Path(cache_dir).expanduser().resolve() if cache_dir else None
            )
            self._created_cache_dir = False
        else:
            if cache_dir is None:
                raise ValueError(
                    'An explicit cache_dir is required when using gcs_uri.'
                )
            self.tiles_dir = None
            self.gcs_uri = normalize_gcs_path(gcs_uri)  # type: ignore[arg-type]
            resolved_cache = Path(cache_dir).expanduser().resolve()
            self._created_cache_dir = not resolved_cache.exists()
            self.cache_dir = resolved_cache

        self.cache_tiles = cache_tiles
        self._tile_cache: dict[tuple[int, int], np.ndarray] = {}
        self.created_cache_files: set[Path] = set()

    def clean_created_cache(self) -> None:
        """Delete only the tile files created in cache_dir by this instance."""
        import gc

        for arr in self._tile_cache.values():
            mmap_obj = getattr(arr, '_mmap', None)
            if mmap_obj is not None:
                with contextlib.suppress(Exception):
                    mmap_obj.close()
        self._tile_cache.clear()
        gc.collect()
        for file_path in list(self.created_cache_files):
            if file_path.is_file():
                with contextlib.suppress(OSError):
                    file_path.unlink()
            self.created_cache_files.discard(file_path)
        if (
            self._created_cache_dir
            and self.cache_dir is not None
            and self.cache_dir.is_dir()
            and not any(self.cache_dir.iterdir())
        ):
            with contextlib.suppress(OSError):
                self.cache_dir.rmdir()

    def get_tile(self, lat_top: float, lon_left: float) -> np.ndarray:
        """Load a 5x5 degree (6000, 6000) uint8 tile array or raise on failure.

        Args:
            lat_top: Northern boundary of the 5x5 degree tile.
            lon_left: Western boundary of the 5x5 degree tile.

        Returns:
            Memory-mapped 2D uint8 numpy array of shape (6000, 6000).

        Raises:
            CatchmentCoverageError: If the tile coordinates are out of domain.
            FileNotFoundError: If the tile does not exist in tiles_dir.
            RuntimeError: If downloading the tile from GCS fails.
            ValueError: If the loaded tile array has invalid shape or dtype.
        """
        key = (int(round(lat_top)), int(round(lon_left)))
        if self.cache_tiles and key in self._tile_cache:
            return self._tile_cache[key]

        tile_name = tile_key_to_filename(key[0], key[1])
        if not is_tile_in_coverage(key[0], key[1]):
            raise CatchmentCoverageError(
                f'DEM tile {tile_name} is outside the global DEM coverage '
                f'domain ({DEM_MIN_LAT}° to {DEM_MAX_LAT}° latitude, '
                f'{DEM_MIN_LON}° to {DEM_MAX_LON}° longitude).'
            )

        if self.tiles_dir is not None:
            tile_path = self.tiles_dir / tile_name
            if not tile_path.is_file():
                raise CatchmentCoverageError(
                    f'Required DEM tile {tile_name} not found in '
                    f'user-supplied tiles_dir ({self.tiles_dir}).'
                )
        else:
            assert self.cache_dir is not None
            assert self.gcs_uri is not None
            tile_path = self.cache_dir / tile_name
            if not tile_path.is_file():
                tile_path = download_tile_from_gcs(
                    key[0],
                    key[1],
                    target_dir=self.cache_dir,
                    source_uri=self.gcs_uri,
                    created_files=self.created_cache_files,
                )

        arr = np.load(tile_path, mmap_mode='r')
        if arr.shape != (TILE_CELLS, TILE_CELLS) or arr.dtype != np.uint8:
            raise ValueError(
                f'Invalid DEM tile {tile_path}: expected shape '
                f'({TILE_CELLS}, {TILE_CELLS}) and dtype uint8, '
                f'got shape {arr.shape} and dtype {arr.dtype}.'
            )

        if self.cache_tiles:
            self._tile_cache[key] = arr
        return arr

    def snap_outlet(
        self,
        lat: float,
        lon: float,
        snap_window_cells: int = 12,
    ) -> tuple[float, float, int, int, tuple[int, int], float]:
        """Snap input coordinates to the nearest channel outlet cell."""
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise CatchmentCoverageError(
                f'Pour point coordinates ({lat}, {lon}) are missing or '
                'non-finite. Catchment cannot be delineated.'
            )
        if not is_coord_in_coverage(lat, lon):
            raise CatchmentCoverageError(
                f'Pour point coordinates ({lat:.4f}, {lon:.4f}) are outside '
                f'the global DEM coverage domain ({DEM_MIN_LAT}° to '
                f'{DEM_MAX_LAT}° latitude, {DEM_MIN_LON}° to {DEM_MAX_LON}° '
                'longitude). Catchment cannot be delineated.'
            )

        lat_top = float(math.ceil(lat / TILE_DEG) * TILE_DEG)
        lon_left = float(math.floor(lon / TILE_DEG) * TILE_DEG)

        r0 = int(round((lat_top - lat) / RES_DEG))
        c0 = int(round((lon - lon_left) / RES_DEG))
        r0 = max(0, min(TILE_CELLS - 1, r0))
        c0 = max(0, min(TILE_CELLS - 1, c0))

        start_key = (int(round(lat_top)), int(round(lon_left)))
        start_grid = self.get_tile(*start_key)

        best_r, best_c = r0, c0
        best_score = -1.0

        for dr in range(-snap_window_cells, snap_window_cells + 1):
            for dc in range(-snap_window_cells, snap_window_cells + 1):
                tr, tc = r0 + dr, c0 + dc
                if 0 <= tr < TILE_CELLS and 0 <= tc < TILE_CELLS:
                    sq = deque([(tr, tc, 0)])
                    svis = {(tr, tc)}
                    cnt = 0
                    while sq and cnt < _SNAP_BFS_MAX_NODES:
                        cr, cc, depth = sq.popleft()
                        cnt += 1
                        if depth >= _SNAP_BFS_MAX_DEPTH:
                            continue
                        for d_r, d_c, req in INFLOW_MAP:
                            nr, nc = cr + d_r, cc + d_c
                            if (
                                0 <= nr < TILE_CELLS
                                and 0 <= nc < TILE_CELLS
                                and (nr, nc) not in svis
                                and start_grid[nr, nc] == req
                            ):
                                svis.add((nr, nc))
                                sq.append((nr, nc, depth + 1))
                    dist_penalty = (
                        float(math.hypot(dr, dc)) * _SNAP_DIST_PENALTY_WEIGHT
                    )
                    score = cnt - dist_penalty
                    if score > best_score:
                        best_score = score
                        best_r, best_c = tr, tc

        outlet_lat = lat_top - best_r * RES_DEG
        outlet_lon = lon_left + best_c * RES_DEG
        snap_dist_m = float(
            math.hypot(
                (outlet_lat - lat) * _METERS_PER_DEGREE,
                (outlet_lon - lon)
                * _METERS_PER_DEGREE
                * math.cos(math.radians(lat)),
            )
        )
        return outlet_lat, outlet_lon, best_r, best_c, start_key, snap_dist_m

    def _traverse_upstream_bfs(
        self,
        start_key: tuple[int, int],
        best_r: int,
        best_c: int,
        *,
        max_cells: int | None = None,
    ) -> tuple[dict[tuple[int, int], np.ndarray], int]:
        """Run 1D-vectorized NumPy reverse-flow BFS across 5x5 degree tiles."""
        offsets_1d = [
            (dr, dc, dr * TILE_CELLS + dc, np.uint8(req))
            for dr, dc, req in INFLOW_MAP
        ]
        visited_tiles: dict[tuple[int, int], np.ndarray] = {
            start_key: np.zeros((TILE_CELLS, TILE_CELLS), dtype=bool)
        }
        visited_tiles[start_key][best_r, best_c] = True
        frontiers: dict[tuple[int, int], np.ndarray] = {
            start_key: np.array(
                [best_r * TILE_CELLS + best_c], dtype=np.int32
            )
        }
        total_accum = 0

        while frontiers:
            next_frontiers: dict[tuple[int, int], np.ndarray] = {}
            for (t_lat, t_lon), idx_arr in frontiers.items():
                n_cur = int(idx_arr.size)
                if max_cells is not None and total_accum + n_cur > max_cells:
                    raise CatchmentCoverageError(
                        f'Watershed exceeded max_cells={max_cells} with '
                        f'{n_cur} upstream cells still queued. Delineation '
                        'aborted to prevent returning a truncated catchment.'
                    )
                total_accum += n_cur

                grid_1d = self.get_tile(t_lat, t_lon).ravel()
                v_mask = visited_tiles[(t_lat, t_lon)]
                v_1d = v_mask.ravel()

                r_arr = idx_arr // TILE_CELLS
                c_arr = idx_arr % TILE_CELLS
                all_interior = (
                    int(r_arr.min()) > 0
                    and int(r_arr.max()) < TILE_CELLS - 1
                    and int(c_arr.min()) > 0
                    and int(c_arr.max()) < TILE_CELLS - 1
                )

                in_idx_list: list[np.ndarray] = []
                if all_interior:
                    for _, _, d_idx, req_val in offsets_1d:
                        n_idx = idx_arr + d_idx
                        hit = (grid_1d[n_idx] == req_val) & (~v_1d[n_idx])
                        if np.any(hit):
                            h_idx = n_idx[hit]
                            v_1d[h_idx] = True
                            in_idx_list.append(h_idx)
                else:
                    for dr, dc, d_idx, req_val in offsets_1d:
                        nr = r_arr + dr
                        nc = c_arr + dc
                        inside = (
                            (nr >= 0)
                            & (nr < TILE_CELLS)
                            & (nc >= 0)
                            & (nc < TILE_CELLS)
                        )
                        if np.any(inside):
                            n_idx = idx_arr[inside] + d_idx
                            hit = (grid_1d[n_idx] == req_val) & (~v_1d[n_idx])
                            if np.any(hit):
                                h_idx = n_idx[hit]
                                v_1d[h_idx] = True
                                in_idx_list.append(h_idx)
                        if not np.all(inside):
                            br = nr[~inside]
                            bc = nc[~inside]
                            orig_c = c_arr[~inside]
                            for k in range(br.size):
                                rr = int(br[k])
                                cc = int(bc[k])
                                cc_src = int(orig_c[k])
                                nt_lat, nt_lon = t_lat, t_lon
                                if rr < 0:
                                    rr += TILE_CELLS
                                    nt_lat += int(TILE_DEG)
                                elif rr >= TILE_CELLS:
                                    rr -= TILE_CELLS
                                    nt_lat -= int(TILE_DEG)
                                if cc < 0:
                                    cc += TILE_CELLS
                                    nt_lon -= int(TILE_DEG)
                                elif cc >= TILE_CELLS:
                                    cc -= TILE_CELLS
                                    nt_lon += int(TILE_DEG)

                                if not is_tile_in_coverage(nt_lat, nt_lon):
                                    if nt_lat > int(DEM_MAX_LAT):
                                        raise CatchmentCoverageError(
                                            'Watershed extends north past the '
                                            'DEM coverage boundary '
                                            f'({DEM_MAX_LAT}°N) at longitude '
                                            f'{t_lon + cc_src * RES_DEG:.4f}°.'
                                            ' Delineation stopped to prevent '
                                            'returning a partial catchment.'
                                        )
                                    if nt_lat < MIN_TILE_LAT_TOP:
                                        raise CatchmentCoverageError(
                                            'Watershed extends south past the '
                                            'DEM coverage boundary '
                                            f'({DEM_MIN_LAT}°S) at longitude '
                                            f'{t_lon + cc_src * RES_DEG:.4f}°.'
                                            ' Delineation stopped to prevent '
                                            'returning a partial catchment.'
                                        )
                                    raise CatchmentCoverageError(
                                        'Watershed extends past the longitude '
                                        f'domain boundary into tile '
                                        f'({nt_lat}, {nt_lon}). Delineation '
                                        'stopped to prevent returning a '
                                        'partial catchment.'
                                    )

                                nkey = (nt_lat, nt_lon)
                                ngrid = self.get_tile(*nkey)
                                if nkey not in visited_tiles:
                                    visited_tiles[nkey] = np.zeros(
                                        (TILE_CELLS, TILE_CELLS), dtype=bool
                                    )
                                nv_mask = visited_tiles[nkey]
                                if (
                                    not nv_mask[rr, cc]
                                    and ngrid[rr, cc] == req_val
                                ):
                                    nv_mask[rr, cc] = True
                                    single = np.array(
                                        [rr * TILE_CELLS + cc], dtype=np.int32
                                    )
                                    prev = next_frontiers.get(nkey)
                                    next_frontiers[nkey] = (
                                        single
                                        if prev is None
                                        else np.concatenate((prev, single))
                                    )

                if in_idx_list:
                    new_idx = (
                        np.concatenate(in_idx_list)
                        if len(in_idx_list) > 1
                        else in_idx_list[0]
                    )
                    prev = next_frontiers.get((t_lat, t_lon))
                    next_frontiers[(t_lat, t_lon)] = (
                        new_idx
                        if prev is None
                        else np.concatenate((prev, new_idx))
                    )
            frontiers = next_frontiers

        return visited_tiles, total_accum

    def delineate(
        self,
        lat: float,
        lon: float,
        snap_window_cells: int = 12,
        max_cells: int | None = None,
        simplify_tolerance: float | None = None,
        catchment_id: str | None = None,
    ) -> dict[str, Any]:
        """Delineate the upstream catchment draining to (lat, lon).

        Args:
            lat: Target latitude.
            lon: Target longitude.
            snap_window_cells: Half-width of local channel snap window.
            max_cells: Optional hard cell cap. Defaults to None (no cap). If
                specified and exceeded, raises CatchmentCoverageError rather
                than returning a truncated catchment.
            simplify_tolerance: Geometry simplification tolerance in degrees.
            catchment_id: Optional identifier for the catchment feature.

        Returns:
            GeoJSON Feature dict with multi-tile polygon geometry.
        """
        (
            outlet_lat,
            outlet_lon,
            best_r,
            best_c,
            start_key,
            snap_dist_m,
        ) = self.snap_outlet(lat, lon, snap_window_cells=snap_window_cells)

        visited_tiles, total_accum = self._traverse_upstream_bfs(
            start_key, best_r, best_c, max_cells=max_cells
        )

        total_area_km2 = _compute_visited_area_km2(visited_tiles)

        if simplify_tolerance is None:
            simplify_tolerance = RES_DEG * 0.4

        tile_polys: list[Polygon] = []
        for (t_lat, t_lon), mask in visited_tiles.items():
            if mask.any():
                poly_part = self._vectorize_tile_mask(
                    mask, float(t_lat), float(t_lon), simplify_tolerance
                )
                tile_polys.append(poly_part)

        if not tile_polys:
            raise RuntimeError(
                'Failed to vectorize delineated catchment mask into a polygon.'
            )
        if len(tile_polys) == 1:
            poly = tile_polys[0]
        else:
            poly = unary_union(tile_polys)
            if not poly.is_valid:
                poly = poly.buffer(0)
            poly = poly.simplify(simplify_tolerance)

        bounds = poly.bounds
        bbox_dict = {
            'min_lon': round(float(bounds[0]), 5),
            'min_lat': round(float(bounds[1]), 5),
            'max_lon': round(float(bounds[2]), 5),
            'max_lat': round(float(bounds[3]), 5),
        }

        if catchment_id is None:
            lat_tag = abs(int(outlet_lat * _COORD_SCALE))
            lon_tag = abs(int(outlet_lon * _COORD_SCALE))
            catchment_id = f'catchment_dem_{lat_tag}_{lon_tag}'

        if total_area_km2 < _STREAM_ORDER_1_MAX_KM2:
            stream_order = 1
        elif total_area_km2 < _STREAM_ORDER_2_MAX_KM2:
            stream_order = 2
        else:
            stream_order = 3

        return {
            'type': 'Feature',
            'properties': {
                'catchment_id': catchment_id,
                'gauge_id': catchment_id,
                'area_km2': total_area_km2,
                'area': total_area_km2,
                'upstream_cells_count': int(total_accum),
                'tiles_spanned_count': sum(
                    1 for m in visited_tiles.values() if m.any()
                ),
                'grid_resolution': '90m (3 arc-second)',
                'outlet': {
                    'input_latitude': lat,
                    'input_longitude': lon,
                    'latitude': round(outlet_lat, 5),
                    'longitude': round(outlet_lon, 5),
                    'reach_id': f'DEM_{int(best_r)}_{int(best_c)}',
                    'snap_distance_m': round(snap_dist_m, 1),
                },
                'reach_attributes': {
                    'reach_id': f'DEM_CELL_{int(best_r)}_{int(best_c)}',
                    'dataset': 'dem_flow_direction',
                    'river_name': (
                        f'DEM Flow Path ({outlet_lat:.4f}°N, '
                        f'{outlet_lon:.4f}°E)'
                    ),
                    'stream_order': stream_order,
                    'upstream_area_km2': total_area_km2,
                },
                'bbox': bbox_dict,
                'delineation_method': (
                    'DEM Digital Elevation Flow-Routing '
                    '(90m HydroSHEDS Multi-Tile Seamless Grid)'
                ),
                'delineation_mode': 'dem_flow_direction',
                'status': 'SUCCESS',
            },
            'geometry': mapping(poly),
        }

    def delineate_batch(
        self,
        coords: Iterable[tuple[float, float]],
        ids: Iterable[str | None] | None = None,
        snap_window_cells: int = 12,
        max_cells: int | None = None,
        simplify_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Delineate catchments for multiple coordinates.

        Missing or out-of-coverage inputs emit explicit NaN/None Feature records
        so missing data in always produces missing data out. Missing tiles or
        I/O failures raise immediately.
        """
        coords_list = list(coords)
        ids_list = list(ids) if ids is not None else [None] * len(coords_list)
        features: list[dict[str, Any]] = []

        for (lat, lon), cid in zip(coords_list, ids_list, strict=True):
            try:
                feat = self.delineate(
                    lat=lat,
                    lon=lon,
                    snap_window_cells=snap_window_cells,
                    max_cells=max_cells,
                    simplify_tolerance=simplify_tolerance,
                    catchment_id=cid,
                )
                features.append(feat)
            except CatchmentCoverageError as err:
                logger.warning(
                    'Catchment %s at (%s, %s) out of coverage or missing: %s',
                    cid or f'{lat},{lon}',
                    lat,
                    lon,
                    err,
                )
                features.append(build_missing_feature(lat, lon, cid, str(err)))

        return {
            'type': 'FeatureCollection',
            'features': features,
        }

    def _vectorize_tile_mask(
        self,
        mask: np.ndarray,
        lat_top: float,
        lon_left: float,
        simplify_tolerance: float,
    ) -> Polygon:
        """Convert a 2D boolean tile mask into a simplified Shapely Polygon."""
        row_indices = np.where(mask.any(axis=1))[0]
        col_indices = np.where(mask.any(axis=0))[0]
        if row_indices.size == 0 or col_indices.size == 0:
            raise ValueError('Cannot vectorize an empty tile mask.')

        min_r, max_r = int(row_indices[0]), int(row_indices[-1])
        min_c, max_c = int(col_indices[0]), int(col_indices[-1])
        sub_mask = mask[min_r : max_r + 1, min_c : max_c + 1]

        sub_lat_top = lat_top - min_r * RES_DEG
        sub_lon_left = lon_left + min_c * RES_DEG

        transform = from_origin(sub_lon_left, sub_lat_top, RES_DEG, RES_DEG)
        shapes = rasterio.features.shapes(
            sub_mask.astype(np.uint8), mask=sub_mask, transform=transform
        )
        polys = [shape(geom) for geom, val in shapes if val == 1]
        if not polys:
            raise RuntimeError(
                'rasterio.features.shapes produced no polygons for non-empty '
                'mask.'
            )

        poly = polys[0] if len(polys) == 1 else unary_union(polys)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if simplify_tolerance:
            poly = poly.simplify(simplify_tolerance)
        return poly


def _compute_visited_area_km2(
    visited_tiles: dict[tuple[int, int], np.ndarray],
) -> float:
    """Compute surface area in km2 using per-row latitude cosine scaling."""
    lat_scale = RES_DEG * _KM_PER_DEGREE
    row_offsets = (np.arange(TILE_CELLS, dtype=np.float64) + 0.5) * RES_DEG
    total_area_km2 = 0.0
    for (t_lat, _), mask in visited_tiles.items():
        row_counts = mask.sum(axis=1)
        if not np.any(row_counts):
            continue
        row_lats_rad = np.radians(float(t_lat) - row_offsets)
        row_lon_scales = RES_DEG * _KM_PER_DEGREE * np.cos(row_lats_rad)
        total_area_km2 += float(np.sum(row_counts * lat_scale * row_lon_scales))

    if total_area_km2 < _SMALL_AREA_THRESHOLD_KM2:
        return round(float(total_area_km2), 4)
    return round(float(total_area_km2), 1)


def delineate_dem(
    lat: float,
    lon: float,
    tiles_dir: str | Path | None = None,
    *,
    gcs_uri: str | None = None,
    cache_dir: str | Path | None = None,
    snap_window_cells: int = 12,
    max_cells: int | None = None,
    catchment_id: str | None = None,
) -> dict[str, Any]:
    """Delineate a catchment from (lat, lon) using explicit DEM tile paths."""
    delineator = DemDelineator(
        tiles_dir=tiles_dir, gcs_uri=gcs_uri, cache_dir=cache_dir
    )
    return delineator.delineate(
        lat=lat,
        lon=lon,
        snap_window_cells=snap_window_cells,
        max_cells=max_cells,
        catchment_id=catchment_id,
    )


delineate_catchment = delineate_dem


def delineate_coordinates(
    coords: Iterable[tuple[float, float]],
    tiles_dir: str | Path | None = None,
    *,
    gcs_uri: str | None = None,
    cache_dir: str | Path | None = None,
    ids: Iterable[str | None] | None = None,
    snap_window_cells: int = 12,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Delineate multiple catchments from coordinate tuples."""
    delineator = DemDelineator(
        tiles_dir=tiles_dir, gcs_uri=gcs_uri, cache_dir=cache_dir
    )
    return delineator.delineate_batch(
        coords=coords,
        ids=ids,
        snap_window_cells=snap_window_cells,
        max_cells=max_cells,
    )
