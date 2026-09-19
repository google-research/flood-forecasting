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
multi-tile boundary traversal across 5x5 degree tile boundaries and optional
expected-area-guided pour-point snapping.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import contextlib
import ctypes
import gc
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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
_AREA_HINT_MAX_WINDOW_CELLS: int = 80
_AREA_HINT_PROBE_CAP: int = 40000
_DOWNSTREAM_TRACE_STEPS: int = 250
_MAX_BOUNDARY_CROSSINGS: int = 200000

_OUTFLOW_OFFSET: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

_C_BFS_SOURCE = r"""
#include <stdint.h>

static const int DR[8]  = {-1, -1,  0,  1,  1,  1,  0, -1};
static const int DC[8]  = { 0,  1,  1,  1,  0, -1, -1, -1};
static const uint8_t REQ[8] = {4, 8, 16, 32, 64, 128, 1, 2};

int64_t bfs_tile_c(
    const uint8_t* grid,
    uint8_t* visited,
    int32_t* queue,
    int32_t q_tail,
    int32_t* b_r,
    int32_t* b_c,
    uint8_t* b_req,
    int32_t max_b,
    int32_t* out_b_count,
    int64_t remaining_budget
) {
    const int TILE = 6000;
    int32_t head = 0;
    int32_t b_cnt = 0;
    int64_t popped = 0;

    while (head < q_tail) {
        if (remaining_budget >= 0 && popped >= remaining_budget) {
            *out_b_count = -1;
            return popped;
        }
        int32_t idx = queue[head++];
        popped++;
        int r = idx / TILE;
        int c = idx % TILE;

        if (r > 0 && r < TILE - 1 && c > 0 && c < TILE - 1) {
            for (int d = 0; d < 8; d++) {
                int nidx = idx + DR[d] * TILE + DC[d];
                if (!visited[nidx] && grid[nidx] == REQ[d]) {
                    visited[nidx] = 1;
                    queue[q_tail++] = nidx;
                }
            }
        } else {
            for (int d = 0; d < 8; d++) {
                int nr = r + DR[d];
                int nc = c + DC[d];
                if (nr >= 0 && nr < TILE && nc >= 0 && nc < TILE) {
                    int nidx = nr * TILE + nc;
                    if (!visited[nidx] && grid[nidx] == REQ[d]) {
                        visited[nidx] = 1;
                        queue[q_tail++] = nidx;
                    }
                } else if (b_cnt < max_b) {
                    b_r[b_cnt] = nr;
                    b_c[b_cnt] = nc;
                    b_req[b_cnt] = REQ[d];
                    b_cnt++;
                }
            }
        }
    }
    *out_b_count = b_cnt;
    return popped;
}
"""

_C_LIB: ctypes.CDLL | None = None
_C_LIB_INITIALIZED: bool = False


def _get_c_bfs_lib() -> ctypes.CDLL | None:
    """Compile and load the C intra-tile BFS kernel if a C compiler exists."""
    global _C_LIB, _C_LIB_INITIALIZED
    if _C_LIB_INITIALIZED:
        return _C_LIB
    _C_LIB_INITIALIZED = True

    compiler = (
        shutil.which('gcc') or shutil.which('clang') or shutil.which('cc')
    )
    if not compiler or sys.platform == 'win32':
        return None

    try:
        so_dir = Path(tempfile.gettempdir()) / f'gh_dem_bfs_{os.getuid()}'
        so_dir.mkdir(parents=True, exist_ok=True)
        so_path = so_dir / 'bfs_tile_v1.so'
        if not so_path.is_file():
            tmp_c = so_dir / f'bfs_{os.getpid()}.c'
            tmp_so = so_dir / f'bfs_{os.getpid()}.so'
            tmp_c.write_text(_C_BFS_SOURCE, encoding='utf-8')
            subprocess.run(
                [
                    compiler,
                    '-O3',
                    '-shared',
                    '-fPIC',
                    str(tmp_c),
                    '-o',
                    str(tmp_so),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            tmp_so.replace(so_path)
            with contextlib.suppress(OSError):
                tmp_c.unlink()
        lib = ctypes.CDLL(str(so_path))
        lib.bfs_tile_c.restype = ctypes.c_int64
        lib.bfs_tile_c.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
        ]
        _C_LIB = lib
    except Exception:  # noqa: BLE001
        _C_LIB = None
    return _C_LIB


class CatchmentCoverageError(FileNotFoundError, ValueError):
    """Raised when a pour point or watershed fails coverage or area checks."""


class CatchmentAreaMismatchError(CatchmentCoverageError):
    """Raised when no channel matches a user-supplied expected_area_km2 hint."""


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


def _trace_downstream_chain(
    grid: np.ndarray, tr: int, tc: int, max_steps: int = _DOWNSTREAM_TRACE_STEPS
) -> list[tuple[int, int]]:
    """Follow D8 flow directions downstream from (tr, tc) within the tile."""
    cr, cc = tr, tc
    chain = [(cr, cc)]
    seen = {(cr, cc)}
    for _ in range(max_steps):
        off = _OUTFLOW_OFFSET.get(int(grid[cr, cc]))
        if off is None:
            break
        nr, nc = cr + off[0], cc + off[1]
        if (
            not (0 <= nr < TILE_CELLS and 0 <= nc < TILE_CELLS)
            or (nr, nc) in seen
        ):
            break
        seen.add((nr, nc))
        chain.append((nr, nc))
        cr, cc = nr, nc
    return chain


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
        """Initialize the DEM delineator with explicit user-supplied I/O paths."""
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
        self._q_buf: np.ndarray | None = None
        self._b_r: np.ndarray | None = None
        self._b_c: np.ndarray | None = None
        self._b_req: np.ndarray | None = None

    def clean_created_cache(self) -> None:
        """Delete only the tile files created in cache_dir by this instance."""
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
        """Load a 5x5 degree (6000, 6000) uint8 tile array or raise on failure."""
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

    def _raise_out_of_coverage(
        self, nt_lat: int, nt_lon: int, t_lon: int, cc_src: int
    ) -> None:
        if nt_lat > int(DEM_MAX_LAT):
            raise CatchmentCoverageError(
                'Watershed extends north past the DEM coverage boundary '
                f'({DEM_MAX_LAT}°N) at longitude '
                f'{t_lon + cc_src * RES_DEG:.4f}°. Delineation stopped to '
                'prevent returning a partial catchment.'
            )
        if nt_lat < MIN_TILE_LAT_TOP:
            raise CatchmentCoverageError(
                'Watershed extends south past the DEM coverage boundary '
                f'({DEM_MIN_LAT}°S) at longitude '
                f'{t_lon + cc_src * RES_DEG:.4f}°. Delineation stopped to '
                'prevent returning a partial catchment.'
            )
        raise CatchmentCoverageError(
            'Watershed extends past the longitude domain boundary into tile '
            f'({nt_lat}, {nt_lon}). Delineation stopped to prevent returning '
            'a partial catchment.'
        )

    def _traverse_upstream_bfs_c(
        self,
        lib: ctypes.CDLL,
        start_key: tuple[int, int],
        best_r: int,
        best_c: int,
        *,
        max_cells: int | None = None,
    ) -> tuple[dict[tuple[int, int], np.ndarray], int]:
        """Run C-accelerated multi-tile reverse-flow BFS."""
        if self._q_buf is None:
            self._q_buf = np.empty(TILE_CELLS * TILE_CELLS, dtype=np.int32)
            self._b_r = np.empty(_MAX_BOUNDARY_CROSSINGS, dtype=np.int32)
            self._b_c = np.empty(_MAX_BOUNDARY_CROSSINGS, dtype=np.int32)
            self._b_req = np.empty(_MAX_BOUNDARY_CROSSINGS, dtype=np.uint8)
        assert self._q_buf is not None
        assert self._b_r is not None
        assert self._b_c is not None
        assert self._b_req is not None

        visited_tiles: dict[tuple[int, int], np.ndarray] = {
            start_key: np.zeros((TILE_CELLS, TILE_CELLS), dtype=bool)
        }
        visited_tiles[start_key][best_r, best_c] = True
        seeds: dict[tuple[int, int], list[int]] = {
            start_key: [best_r * TILE_CELLS + best_c]
        }
        total_accum = 0

        while seeds:
            next_seeds: dict[tuple[int, int], list[int]] = {}
            for (t_lat, t_lon), seed_list in seeds.items():
                grid = self.get_tile(t_lat, t_lon)
                v_mask = visited_tiles[(t_lat, t_lon)]
                q_len = len(seed_list)
                self._q_buf[:q_len] = seed_list
                out_b = ctypes.c_int32(0)
                rem = -1 if max_cells is None else (max_cells - total_accum)
                popped = lib.bfs_tile_c(
                    grid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    v_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    self._q_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    q_len,
                    self._b_r.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    self._b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    self._b_req.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    _MAX_BOUNDARY_CROSSINGS,
                    ctypes.byref(out_b),
                    rem,
                )
                total_accum += int(popped)
                if out_b.value < 0:
                    raise CatchmentCoverageError(
                        f'Watershed exceeded max_cells={max_cells}. '
                        'Delineation aborted to prevent returning a '
                        'truncated catchment.'
                    )
                for k in range(out_b.value):
                    rr = int(self._b_r[k])
                    cc = int(self._b_c[k])
                    req_val = int(self._b_req[k])
                    cc_src = max(0, min(TILE_CELLS - 1, cc))
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
                        self._raise_out_of_coverage(
                            nt_lat, nt_lon, t_lat, cc_src
                        )
                    nkey = (nt_lat, nt_lon)
                    ngrid = self.get_tile(*nkey)
                    if nkey not in visited_tiles:
                        visited_tiles[nkey] = np.zeros(
                            (TILE_CELLS, TILE_CELLS), dtype=bool
                        )
                    nv_mask = visited_tiles[nkey]
                    if not nv_mask[rr, cc] and ngrid[rr, cc] == req_val:
                        nv_mask[rr, cc] = True
                        next_seeds.setdefault(nkey, []).append(
                            rr * TILE_CELLS + cc
                        )
            seeds = next_seeds

        return visited_tiles, total_accum

    def _traverse_upstream_bfs(
        self,
        start_key: tuple[int, int],
        best_r: int,
        best_c: int,
        *,
        max_cells: int | None = None,
    ) -> tuple[dict[tuple[int, int], np.ndarray], int]:
        """Run multi-tile reverse-flow BFS (C-accelerated with NumPy fallback)."""
        c_lib = _get_c_bfs_lib()
        if c_lib is not None:
            return self._traverse_upstream_bfs_c(
                c_lib, start_key, best_r, best_c, max_cells=max_cells
            )

        offsets_1d = [
            (dr, dc, dr * TILE_CELLS + dc, np.uint8(req))
            for dr, dc, req in INFLOW_MAP
        ]
        visited_tiles: dict[tuple[int, int], np.ndarray] = {
            start_key: np.zeros((TILE_CELLS, TILE_CELLS), dtype=bool)
        }
        visited_tiles[start_key][best_r, best_c] = True
        frontiers: dict[tuple[int, int], np.ndarray] = {
            start_key: np.array([best_r * TILE_CELLS + best_c], dtype=np.int32)
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
                                    self._raise_out_of_coverage(
                                        nt_lat, nt_lon, t_lon, cc_src
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

    def _delineate_with_area_hint(
        self,
        lat: float,
        lon: float,
        expected_area_km2: float,
        area_tolerance: float,
        snap_window_cells: int,
        max_cells: int | None,
        catchment_id: str | None,
    ) -> tuple[
        float,
        float,
        int,
        int,
        tuple[int, int],
        float,
        dict[tuple[int, int], np.ndarray],
        int,
    ]:
        """Find the nearest pour-point cell matching expected_area_km2 or fail loudly."""
        if not math.isfinite(expected_area_km2) or expected_area_km2 <= 0.0:
            msg = (
                f'[AREA HINT FAILURE] Catchment {catchment_id or ""} at '
                f'({lat}, {lon}): expected_area_km2={expected_area_km2} must '
                'be a positive finite number.'
            )
            logger.error(msg)
            sys.stderr.write(msg + '\n')
            raise CatchmentAreaMismatchError(msg)

        # First try the default snap cell at snap_window_cells:
        (
            outlet_lat,
            outlet_lon,
            best_r,
            best_c,
            start_key,
            snap_dist_m,
        ) = self.snap_outlet(lat, lon, snap_window_cells=snap_window_cells)

        min_area = expected_area_km2 * max(0.05, 1.0 - area_tolerance)
        max_area = expected_area_km2 * (1.0 + area_tolerance)

        cell_area = (RES_DEG * _KM_PER_DEGREE) * (
            RES_DEG * _KM_PER_DEGREE * max(0.05, math.cos(math.radians(lat)))
        )
        exp_cells = expected_area_km2 / max(cell_area, 1e-6)
        min_cells = max(
            1, int(exp_cells * max(0.04, 0.95 * (1.0 - area_tolerance)))
        )
        hint_cap = int(exp_cells * (1.15 + area_tolerance)) + 100
        if max_cells is not None:
            hint_cap = min(hint_cap, max_cells)

        closest_area_seen: float = 0.0
        try:
            vt0, cnt0 = self._traverse_upstream_bfs(
                start_key, best_r, best_c, max_cells=hint_cap
            )
            area0 = _compute_visited_area_km2(vt0)
            closest_area_seen = area0
            if min_area <= area0 <= max_area:
                return (
                    outlet_lat,
                    outlet_lon,
                    best_r,
                    best_c,
                    start_key,
                    snap_dist_m,
                    vt0,
                    cnt0,
                )
        except CatchmentCoverageError:
            vt0 = None
            area0 = -1.0

        # Default snap did not match expected_area_km2: expand search window
        lat_top = float(start_key[0])
        lon_left = float(start_key[1])
        r0 = max(0, min(TILE_CELLS - 1, int(round((lat_top - lat) / RES_DEG))))
        c0 = max(0, min(TILE_CELLS - 1, int(round((lon - lon_left) / RES_DEG))))
        grid = self.get_tile(*start_key)

        search_w = max(snap_window_cells, _AREA_HINT_MAX_WINDOW_CELLS)
        r_min = max(0, r0 - search_w)
        r_max = min(TILE_CELLS - 1, r0 + search_w)
        c_min = max(0, c0 - search_w)
        c_max = min(TILE_CELLS - 1, c0 + search_w)

        candidates: list[tuple[float, int, int]] = []
        for dr in range(-search_w, search_w + 1):
            for dc in range(-search_w, search_w + 1):
                tr, tc = r0 + dr, c0 + dc
                if (
                    r_min <= tr <= r_max
                    and c_min <= tc <= c_max
                    and grid[tr, tc] > 0
                ):
                    candidates.append((math.hypot(dr, dc), tr, tc))
        candidates.sort(key=lambda item: item[0])

        rejected = np.zeros((TILE_CELLS, TILE_CELLS), dtype=bool)
        if vt0 is not None and 0.0 <= area0 < min_area:
            rejected |= vt0[start_key]

        probe_cap = min(min_cells, _AREA_HINT_PROBE_CAP)

        for _, tr, tc in candidates:
            if rejected[tr, tc]:
                continue
            chain = _trace_downstream_chain(grid, tr, tc)
            er, ec = chain[-1]
            if rejected[er, ec]:
                for rr, cc in chain:
                    rejected[rr, cc] = True
                continue

            if probe_cap > 1:
                try:
                    vt_p, _ = self._traverse_upstream_bfs(
                        start_key, er, ec, max_cells=probe_cap
                    )
                    area_p = _compute_visited_area_km2(vt_p)
                    if abs(area_p - expected_area_km2) < abs(
                        closest_area_seen - expected_area_km2
                    ):
                        closest_area_seen = area_p
                    rejected |= vt_p[start_key]
                    for rr, cc in chain:
                        rejected[rr, cc] = True
                    continue
                except CatchmentCoverageError:
                    pass

            # Full check on downstream exit (er, ec):
            try:
                vt_e, cnt_e = self._traverse_upstream_bfs(
                    start_key, er, ec, max_cells=hint_cap
                )
                area_e = _compute_visited_area_km2(vt_e)
                if abs(area_e - expected_area_km2) < abs(
                    closest_area_seen - expected_area_km2
                ):
                    closest_area_seen = area_e
                if area_e < min_area:
                    rejected |= vt_e[start_key]
                    for rr, cc in chain:
                        rejected[rr, cc] = True
                    continue
                if min_area <= area_e <= max_area:
                    # Binary search along chain to find the first cell >= min_area
                    chosen_r, chosen_c, chosen_vt, chosen_cnt = (
                        er,
                        ec,
                        vt_e,
                        cnt_e,
                    )
                    lo, hi = 0, len(chain) - 1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        mr, mc = chain[mid]
                        try:
                            vt_m, cnt_m = self._traverse_upstream_bfs(
                                start_key, mr, mc, max_cells=hint_cap
                            )
                            area_m = _compute_visited_area_km2(vt_m)
                            if min_area <= area_m <= max_area:
                                chosen_r, chosen_c, chosen_vt, chosen_cnt = (
                                    mr,
                                    mc,
                                    vt_m,
                                    cnt_m,
                                )
                                hi = mid - 1
                            elif area_m < min_area:
                                lo = mid + 1
                            else:
                                hi = mid - 1
                        except CatchmentCoverageError:
                            hi = mid - 1
                    out_lat = lat_top - chosen_r * RES_DEG
                    out_lon = lon_left + chosen_c * RES_DEG
                    dist_m = float(
                        math.hypot(
                            (out_lat - lat) * _METERS_PER_DEGREE,
                            (out_lon - lon)
                            * _METERS_PER_DEGREE
                            * math.cos(math.radians(lat)),
                        )
                    )
                    return (
                        out_lat,
                        out_lon,
                        chosen_r,
                        chosen_c,
                        start_key,
                        dist_m,
                        chosen_vt,
                        chosen_cnt,
                    )
            except CatchmentCoverageError:
                pass

            # (er, ec) exceeded hint_cap: check (tr, tc) directly
            try:
                vt_t, cnt_t = self._traverse_upstream_bfs(
                    start_key, tr, tc, max_cells=hint_cap
                )
                area_t = _compute_visited_area_km2(vt_t)
                if abs(area_t - expected_area_km2) < abs(
                    closest_area_seen - expected_area_km2
                ):
                    closest_area_seen = area_t
                if min_area <= area_t <= max_area:
                    out_lat = lat_top - tr * RES_DEG
                    out_lon = lon_left + tc * RES_DEG
                    dist_m = float(
                        math.hypot(
                            (out_lat - lat) * _METERS_PER_DEGREE,
                            (out_lon - lon)
                            * _METERS_PER_DEGREE
                            * math.cos(math.radians(lat)),
                        )
                    )
                    return (
                        out_lat,
                        out_lon,
                        tr,
                        tc,
                        start_key,
                        dist_m,
                        vt_t,
                        cnt_t,
                    )
                if area_t < min_area:
                    rejected |= vt_t[start_key]
            except CatchmentCoverageError:
                for rr, cc in chain:
                    rejected[rr, cc] = True

        msg = (
            f'[AREA HINT FAILURE] Catchment {catchment_id or "unnamed"} at '
            f'({lat:.4f}, {lon:.4f}) failed expected_area_km2={expected_area_km2:.2f} km2 '
            f'(allowed range [{min_area:.2f}, {max_area:.2f}] km2; '
            f'closest candidate found={closest_area_seen:.2f} km2 within '
            f'{search_w}-cell search window). Refusing to output a polygon.'
        )
        logger.error(msg)
        sys.stderr.write(msg + '\n')
        raise CatchmentAreaMismatchError(msg)

    def delineate(
        self,
        lat: float,
        lon: float,
        snap_window_cells: int = 12,
        max_cells: int | None = None,
        simplify_tolerance: float | None = None,
        catchment_id: str | None = None,
        expected_area_km2: float | None = None,
        area_tolerance: float = 0.50,
    ) -> dict[str, Any]:
        """Delineate the upstream catchment draining to (lat, lon)."""
        if expected_area_km2 is not None:
            (
                outlet_lat,
                outlet_lon,
                best_r,
                best_c,
                start_key,
                snap_dist_m,
                visited_tiles,
                total_accum,
            ) = self._delineate_with_area_hint(
                lat=lat,
                lon=lon,
                expected_area_km2=expected_area_km2,
                area_tolerance=area_tolerance,
                snap_window_cells=snap_window_cells,
                max_cells=max_cells,
                catchment_id=catchment_id,
            )
        else:
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
        expected_areas_km2: Iterable[float | None] | None = None,
        area_tolerance: float = 0.50,
    ) -> dict[str, Any]:
        """Delineate catchments for multiple coordinates."""
        coords_list = list(coords)
        ids_list = list(ids) if ids is not None else [None] * len(coords_list)
        areas_list = (
            list(expected_areas_km2)
            if expected_areas_km2 is not None
            else [None] * len(coords_list)
        )
        features: list[dict[str, Any]] = []

        for (lat, lon), cid, exp_area in zip(
            coords_list, ids_list, areas_list, strict=True
        ):
            try:
                feat = self.delineate(
                    lat=lat,
                    lon=lon,
                    snap_window_cells=snap_window_cells,
                    max_cells=max_cells,
                    simplify_tolerance=simplify_tolerance,
                    catchment_id=cid,
                    expected_area_km2=exp_area,
                    area_tolerance=area_tolerance,
                )
                features.append(feat)
            except CatchmentCoverageError as err:
                logger.warning(
                    'Catchment %s at (%s, %s) failed coverage or area check: %s',
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
    expected_area_km2: float | None = None,
    area_tolerance: float = 0.50,
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
        expected_area_km2=expected_area_km2,
        area_tolerance=area_tolerance,
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
    expected_areas_km2: Iterable[float | None] | None = None,
    area_tolerance: float = 0.50,
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
        expected_areas_km2=expected_areas_km2,
        area_tolerance=area_tolerance,
    )
