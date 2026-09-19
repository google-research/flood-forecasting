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

"""Slice continental HydroSHEDS D8 GeoTIFFs into 5x5 degree .npy tiles."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from catchment_delineation.config import RES_DEG, TILE_CELLS, TILE_DEG
from catchment_delineation.tiles import tile_key_to_filename


def _slice_single_tile(
    task: tuple[Path, Path, int, int, float, float],
) -> tuple[str, bool, int]:
    """Slice a single 5x5 degree tile window from a continental GeoTIFF."""
    tif_path, out_path, lat_top, lon_left, tif_left, tif_top = task
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path.name, False, 0

    col_off = int(round((lon_left - tif_left) / RES_DEG))
    row_off = int(round((tif_top - lat_top) / RES_DEG))

    with rasterio.open(tif_path) as src:
        win = Window(col_off, row_off, TILE_CELLS, TILE_CELLS)
        data = src.read(1, window=win)

    valid_mask = (data > 0) & (data <= 128)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        return out_path.name, False, 0

    arr = np.where(valid_mask, data, 0).astype(np.uint8)
    np.save(out_path, arr)
    return out_path.name, True, valid_count


def main(argv: list[str] | None = None) -> int:
    """Slice user-supplied continental D8 GeoTIFF files into 5x5 .npy tiles."""
    parser = argparse.ArgumentParser(
        description='Slice continental D8 GeoTIFFs into 5x5 degree .npy tiles.'
    )
    parser.add_argument(
        '--input-tifs',
        nargs='+',
        required=True,
        help='Explicit paths to input continental D8 flow-direction GeoTIFFs.',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Explicit output directory for 5x5 degree .npy tiles.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of worker processes.',
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_paths = [Path(p).expanduser().resolve() for p in args.input_tifs]
    for tif_p in tif_paths:
        if not tif_p.is_file():
            raise FileNotFoundError(f'Input GeoTIFF not found: {tif_p}')

    tasks: list[tuple[Path, Path, int, int, float, float]] = []
    for tif_p in tif_paths:
        with rasterio.open(tif_p) as src:
            b = src.bounds
            min_lon = int(math.floor(b.left / TILE_DEG) * TILE_DEG)
            max_lon = int(math.ceil(b.right / TILE_DEG) * TILE_DEG)
            min_lat = int(math.floor(b.bottom / TILE_DEG) * TILE_DEG)
            max_lat = int(math.ceil(b.top / TILE_DEG) * TILE_DEG)

            for lat_top in range(
                min_lat + int(TILE_DEG), max_lat + int(TILE_DEG), int(TILE_DEG)
            ):
                for lon_left in range(min_lon, max_lon, int(TILE_DEG)):
                    if (
                        lon_left < b.left - 1e-6
                        or lon_left + TILE_DEG > b.right + 1e-6
                        or lat_top - TILE_DEG < b.bottom - 1e-6
                        or lat_top > b.top + 1e-6
                    ):
                        continue
                    fname = tile_key_to_filename(lat_top, lon_left)
                    tasks.append(
                        (
                            tif_p,
                            out_dir / fname,
                            lat_top,
                            lon_left,
                            b.left,
                            b.top,
                        )
                    )

    written = 0
    mp_ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp_ctx
    ) as pool:
        futures = [pool.submit(_slice_single_tile, t) for t in tasks]
        for fut in as_completed(futures):
            _, did_write, _ = fut.result()
            if did_write:
                written += 1

    sys.stdout.write(f'Sliced {written} non-empty 5x5 tiles to {out_dir}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
