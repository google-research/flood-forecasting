#!/usr/bin/env python3
"""Slices continental HydroSHEDS 3 arc-second flow direction GeoTIFFs into 5x5 degree .npy tiles."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from catchment_delineation.tiles import tile_key_to_filename

VALID_D8_VALS = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)


def process_single_tile(
    tif_path: str,
    lat_top: int,
    lon_left: int,
    col_off: int,
    row_off: int,
    out_dir: str,
) -> Tuple[str, bool, int]:
  """Extracts a 5x5 degree window and saves it as .npy if land exists."""
  out_path = Path(out_dir) / tile_key_to_filename(lat_top, lon_left)
  try:
    with rasterio.open(tif_path) as src:
      win = Window(col_off, row_off, 6000, 6000)
      data = src.read(1, window=win)

    # Check for valid flow direction cells
    is_valid = np.isin(data, VALID_D8_VALS)
    valid_count = int(is_valid.sum())

    if valid_count < 50:
      return out_path.name, False, 0

    if out_path.exists():
      existing = np.load(out_path)
      # Merge new valid cells onto existing
      existing[is_valid] = data[is_valid]
      np.save(out_path, existing)
      return out_path.name, True, valid_count
    else:
      np.save(out_path, data)
      return out_path.name, True, valid_count

  except Exception as e:
    return f"{out_path.name}: {e}", False, -1


def slice_geotiff(
    tif_path: Path,
    out_dir: Path,
    max_workers: int = 16,
) -> int:
  """Slices a continental GeoTIFF into 5x5 degree tiles."""
  print(f"\n--- Processing {tif_path.name} ---")
  start_time = time.time()

  with rasterio.open(tif_path) as src:
    left, bottom, right, top = src.bounds
    width, height = src.width, src.height
    print(f"Bounds: [{left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f}], Dimensions: {width}x{height}")

    min_lat = int(math.floor(bottom / 5.0) * 5)
    max_lat = int(math.ceil(top / 5.0) * 5)
    min_lon = int(math.floor(left / 5.0) * 5)
    max_lon = int(math.ceil(right / 5.0) * 5)

    tasks = []
    for lat_top in range(min_lat + 5, max_lat + 5, 5):
      for lon_left in range(min_lon, max_lon, 5):
        col_off = int(round((lon_left - left) * 1200))
        row_off = int(round((top - lat_top) * 1200))

        if 0 <= col_off and col_off + 6000 <= width and 0 <= row_off and row_off + 6000 <= height:
          tasks.append((str(tif_path), lat_top, lon_left, col_off, row_off, str(out_dir)))

  print(f"Candidate 5x5 tiles to check: {len(tasks)}")
  saved_count = 0

  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_single_tile, *task) for task in tasks]
    for fut in as_completed(futures):
      name, saved, cnt = fut.result()
      if saved:
        saved_count += 1

  elapsed = time.time() - start_time
  print(f"Finished {tif_path.name}: {saved_count} valid tiles saved/merged in {elapsed:.1f}s")
  return saved_count


def main():
  parser = argparse.ArgumentParser(description="Slice continental HydroSHEDS GeoTIFFs to 5x5 degree .npy tiles")
  parser.add_argument("--tifs", nargs="+", help="Paths to continental GeoTIFF files")
  parser.add_argument("--out-dir", default="/usr/local/google/home/gsnearing/data/DEMs/tiles_5deg", help="Output directory for .npy tiles")
  parser.add_argument("--workers", type=int, default=16, help="Worker processes")
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  tifs = [Path(p) for p in args.tifs] if args.tifs else [
      Path("/usr/local/google/home/gsnearing/data/DEMs/continental/au_dir_3s.tif"),
      Path("/usr/local/google/home/gsnearing/data/DEMs/continental/sa_dir_3s.tif"),
      Path("/usr/local/google/home/gsnearing/data/DEMs/continental/eu_dir_3s.tif"),
      Path("/usr/local/google/home/gsnearing/data/DEMs/continental/af_dir_3s.tif"),
      Path("/usr/local/google/home/gsnearing/data/DEMs/continental/as_dir_3s.tif"),
  ]

  total_saved = 0
  for tif in tifs:
    if not tif.exists():
      print(f"Warning: {tif} not found, skipping.")
      continue
    total_saved += slice_geotiff(tif, out_dir, max_workers=args.workers)

  print(f"\n==========================================")
  print(f"All continental grids sliced successfully!")
  print(f"Total tiles in {out_dir}: {len(list(out_dir.glob('*.npy')))}")
  print(f"==========================================")


if __name__ == "__main__":
  main()
