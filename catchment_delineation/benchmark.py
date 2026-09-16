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

"""Automated global benchmarking suite for DEM catchment delineation against reference polygons."""

from __future__ import annotations

import os

# Configure environment before NumPy / C-extensions initialize to ensure fork and thread safety
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import shape
import shapely.wkt

from catchment_delineation.config import (
    GCS_BENCHMARK_URI,
    GCS_TILES_URI,
    LOCAL_BENCHMARK_FILE,
    LOCAL_BENCHMARKS_DIR,
    get_default_cache_dir,
)
from catchment_delineation.delineator import CatchmentCoverageError, DemDelineator
from catchment_delineation.gcs import download_tile_from_gcs
from catchment_delineation.tiles import (
    is_coord_in_coverage,
    is_tile_in_coverage,
    latlon_to_tile_key,
    tile_key_to_filename,
)

logger = logging.getLogger("catchment_delineation.benchmark")

DEFAULT_PACKAGE_BENCHMARK_PATH = (
    Path(__file__).parent / "data" / "benchmark_basins_1000.parquet"
)
DEFAULT_BENCHMARK_PATH = LOCAL_BENCHMARK_FILE


def compute_iou_and_metrics(
    del_geom,
    ref_geom,
    ref_area_km2: float,
    del_area_km2: float,
) -> Tuple[float, float, float, float]:
  """Computes spatial overlap and error metrics between delineated and reference geometries."""
  try:
    if not del_geom.is_valid:
      del_geom = del_geom.buffer(0)
    if not ref_geom.is_valid:
      ref_geom = ref_geom.buffer(0)

    intersection = del_geom.intersection(ref_geom).area
    union = del_geom.union(ref_geom).area

    iou = float(intersection / union) if union > 0 else 0.0
    dice = (
        float((2.0 * intersection) / (del_geom.area + ref_geom.area))
        if (del_geom.area + ref_geom.area) > 0
        else 0.0
    )

    area_rel_diff_pct = (
        float((del_area_km2 - ref_area_km2) / ref_area_km2 * 100.0)
        if ref_area_km2 > 0
        else 0.0
    )
    abs_area_err_pct = abs(area_rel_diff_pct)

    return iou, dice, area_rel_diff_pct, abs_area_err_pct
  except Exception as e:
    logger.warning("Error computing spatial metrics: %s", e)
    return 0.0, 0.0, 0.0, 0.0


def _evaluate_single_basin(
    row_dict: Dict[str, Any],
    tiles_dir: Optional[str] = None,
    gcs_uri: str = GCS_TILES_URI,
    cache_dir: Optional[str] = None,
    snap_window_cells: int = 12,
) -> Dict[str, Any]:
  """Worker function to evaluate a single basin."""
  gauge_id = row_dict["gauge_id"]
  lat = float(row_dict["latitude"])
  lon = float(row_dict["longitude"])
  ref_area_km2 = float(row_dict["reference_area_km2"])
  ref_wkt = row_dict["geometry_wkt"]
  continent = row_dict.get("continent", "Unknown")
  hemisphere = row_dict.get("hemisphere", "Unknown")
  size_tier = row_dict.get("size_tier", "Unknown")

  t0 = time.time()
  try:
    delineator = DemDelineator(
        tiles_dir=tiles_dir,
        gcs_uri=gcs_uri,
        cache_dir=cache_dir,
        cache_tiles=True,
    )
    res = delineator.delineate(
        lat=lat,
        lon=lon,
        catchment_id=gauge_id,
        snap_window_cells=snap_window_cells,
    )
    elapsed = time.time() - t0

    props = res["properties"]
    del_area_km2 = float(props["area_km2"])
    tiles_spanned = int(props.get("tiles_spanned_count", 1))
    snap_dist_m = float(props.get("outlet", {}).get("snap_distance_m", 0.0))

    del_geom = shape(res["geometry"])
    ref_geom = shapely.wkt.loads(ref_wkt)

    iou, dice, area_rel_diff, abs_area_err = compute_iou_and_metrics(
        del_geom, ref_geom, ref_area_km2, del_area_km2
    )

    return {
        "gauge_id": gauge_id,
        "continent": continent,
        "hemisphere": hemisphere,
        "size_tier": size_tier,
        "latitude": lat,
        "longitude": lon,
        "ref_area_km2": ref_area_km2,
        "del_area_km2": del_area_km2,
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "area_bias_pct": round(area_rel_diff, 2),
        "abs_area_err_pct": round(abs_area_err, 2),
        "snap_dist_m": round(snap_dist_m, 1),
        "tiles_spanned": tiles_spanned,
        "elapsed_sec": round(elapsed, 3),
        "status": "SUCCESS",
    }
  except CatchmentCoverageError as e:
    elapsed = time.time() - t0
    return {
        "gauge_id": gauge_id,
        "continent": continent,
        "hemisphere": hemisphere,
        "size_tier": size_tier,
        "latitude": lat,
        "longitude": lon,
        "ref_area_km2": ref_area_km2,
        "del_area_km2": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "area_bias_pct": -100.0,
        "abs_area_err_pct": 100.0,
        "snap_dist_m": 0.0,
        "tiles_spanned": 0,
        "elapsed_sec": round(elapsed, 3),
        "status": f"OUT_OF_COVERAGE: {e}",
    }
  except Exception as e:
    elapsed = time.time() - t0
    return {
        "gauge_id": gauge_id,
        "continent": continent,
        "hemisphere": hemisphere,
        "size_tier": size_tier,
        "latitude": lat,
        "longitude": lon,
        "ref_area_km2": ref_area_km2,
        "del_area_km2": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "area_bias_pct": -100.0,
        "abs_area_err_pct": 100.0,
        "snap_dist_m": 0.0,
        "tiles_spanned": 0,
        "elapsed_sec": round(elapsed, 3),
        "status": f"ERROR: {e}",
    }


def print_summary_table(df: pd.DataFrame, title: str, group_col: str):
  """Prints formatted summary statistics for a given grouping column."""
  print(f"\n--- {title} ---")
  header = (
      f"{'Group':<18} | {'Count':>6} | {'Median IoU':>10} | {'Median Dice':>11}"
      f" | {'IoU >= 0.80':>11} | {'Med |ΔArea| %':>13} | {'Mean Time':>9}"
  )
  print(header)
  print("-" * len(header))

  for grp, grp_df in df.groupby(group_col):
    n = len(grp_df)
    med_iou = grp_df["iou"].median()
    med_dice = grp_df["dice"].median()
    pct_80 = (grp_df["iou"] >= 0.80).mean() * 100.0
    med_area_err = grp_df["abs_area_err_pct"].median()
    mean_time = grp_df["elapsed_sec"].mean()
    print(
        f"{str(grp):<18} | {n:>6d} | {med_iou:>10.3f} | {med_dice:>11.3f} |"
        f" {pct_80:>10.1f}% | {med_area_err:>12.1f}% | {mean_time:>8.3f}s"
    )


def run_benchmark(
    dataset_path: Optional[Path] = None,
    samples: Optional[int] = None,
    continents: Optional[List[str]] = None,
    size_tiers: Optional[List[str]] = None,
    workers: int = 8,
    tiles_dir: Optional[str] = None,
    gcs_uri: str = GCS_TILES_URI,
    cache_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    snap_window_cells: int = 12,
    clean_cache: bool = False,
) -> pd.DataFrame:
  """Executes the global catchment delineation benchmark across test basins."""
  if dataset_path:
    # Support direct gs:// URIs (e.g. gs://open-multimet/ancillary-data/benchmarks/benchmark_basins_1000.parquet)
    if str(dataset_path).startswith("gs://"):
      gcs_src = str(dataset_path)
      filename = gcs_src.split("/")[-1]
      ds_path = LOCAL_BENCHMARKS_DIR / filename
      if not ds_path.exists():
        print(f"Downloading benchmark dataset from {gcs_src} to {ds_path}...")
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        import subprocess
        if shutil.which("gcloud"):
          subprocess.run(["gcloud", "storage", "cp", gcs_src, str(ds_path)], check=True)
        else:
          raise FileNotFoundError(f"Cannot download {gcs_src}: gcloud CLI not found.")
    else:
      ds_path = Path(dataset_path).expanduser().resolve()
  else:
    # Preferred resolution order:
    # 1. Local canonical directory: ~/ancillary-data/benchmarks/benchmark_basins_1000.parquet
    # 2. Bundled package fallback: catchment_delineation/data/benchmark_basins_1000.parquet
    if LOCAL_BENCHMARK_FILE.exists():
      ds_path = LOCAL_BENCHMARK_FILE
    elif DEFAULT_PACKAGE_BENCHMARK_PATH.exists():
      ds_path = DEFAULT_PACKAGE_BENCHMARK_PATH.resolve()
    else:
      ds_path = LOCAL_BENCHMARK_FILE

  if not ds_path.exists():
    # Attempt to download from GCS canonical URI if dataset file not found locally
    print(f"Benchmark dataset not found locally at {ds_path}. Downloading from GCS ({GCS_BENCHMARK_URI})...")
    import subprocess
    import shutil
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_src = GCS_BENCHMARK_URI
    if shutil.which("gcloud"):
      subprocess.run(["gcloud", "storage", "cp", gcs_src, str(ds_path)], check=True)
    else:
      raise FileNotFoundError(f"Benchmark dataset not found at {ds_path} and gcloud not installed.")

  df = pd.read_parquet(ds_path)
  print(f"Loaded benchmark dataset: {len(df)} candidate basins from {ds_path.name}")

  if continents:
    df = df[df["continent"].isin(continents)]
  if size_tiers:
    df = df[df["size_tier"].isin(size_tiers)]

  if samples and samples < len(df):
    # Balanced stratified subsample across continent and size tier
    sampled_dfs = []
    for _, grp in df.groupby(["continent", "size_tier"]):
      n = max(1, int(len(grp) * samples / len(df)))
      sampled_dfs.append(grp.sample(n=min(len(grp), n), random_state=42))
    df = pd.concat(sampled_dfs, ignore_index=True)
    if len(df) > samples:
      df = df.sample(n=samples, random_state=42)

  print(f"Benchmarking {len(df)} basins across {df['continent'].nunique()} continents using {workers} workers...")
  if tiles_dir:
    print(f"Using local tile directory: {tiles_dir}")
  cache_path = Path(cache_dir).expanduser() if cache_dir else get_default_cache_dir()
  if not tiles_dir:
    print(f"Using GCS tile bucket: {gcs_uri} (cached in {cache_path})")
    cache_path.mkdir(parents=True, exist_ok=True)
    needed_tile_keys = set()
    for row in df.itertuples():
      lat = float(row.latitude)
      lon = float(row.longitude)
      if is_coord_in_coverage(lat, lon):
        tk = latlon_to_tile_key(lat, lon)
        if is_tile_in_coverage(tk[0], tk[1]):
          needed_tile_keys.add(tk)

    missing_tiles = [
        tk
        for tk in sorted(needed_tile_keys)
        if not (cache_path / tile_key_to_filename(tk[0], tk[1])).exists()
    ]
    if missing_tiles:
      print(
          f"Pre-caching {len(missing_tiles)} required DEM tiles in main process before spawning workers..."
      )
      def _download_one(tk: Tuple[int, int]):
        try:
          download_tile_from_gcs(
              lat_top=tk[0],
              lon_left=tk[1],
              target_dir=cache_path,
              source_uri=gcs_uri,
          )
        except Exception as e:
          logger.warning(
              "Could not pre-cache tile %s: %s",
              tile_key_to_filename(tk[0], tk[1]),
              e,
          )

      with ThreadPoolExecutor(max_workers=min(16, len(missing_tiles))) as pool:
        list(pool.map(_download_one, missing_tiles))
      print(f"Pre-cached {len(missing_tiles)} DEM tiles successfully.")

  rows = df.to_dict(orient="records")
  results = []
  t_start = time.time()

  try:
    with ProcessPoolExecutor(max_workers=workers) as executor:
      futures = {
          executor.submit(
              _evaluate_single_basin,
              row,
              tiles_dir=tiles_dir,
              gcs_uri=gcs_uri,
              cache_dir=cache_dir,
              snap_window_cells=snap_window_cells,
          ): row["gauge_id"]
          for row in rows
      }

      done_count = 0
      total = len(futures)
      for fut in as_completed(futures):
        res = fut.result()
        results.append(res)
        done_count += 1
        if done_count % 50 == 0 or done_count == total:
          print(f"Progress: [{done_count}/{total}] basins evaluated ({(done_count/total)*100:.1f}%)")

    total_time = time.time() - t_start
    res_df = pd.DataFrame(results)

    successful = int((res_df['status'] == 'SUCCESS').sum())
    out_of_coverage = int((res_df['status'].str.startswith('OUT_OF_COVERAGE')).sum())
    unexpected_errors = len(res_df) - successful - out_of_coverage

    # Overall Report
    print("\n" + "=" * 80)
    print("GLOBAL CATCHMENT DELINEATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Basins Evaluated : {len(res_df)}")
    print(f"Total Wall-Clock Time   : {total_time:.1f}s ({total_time / max(1, len(res_df)):.3f}s / basin)")
    print(f"Successful Delineations : {successful} / {len(res_df)}")
    if out_of_coverage > 0:
      print(f"Out-of-Coverage Basins  : {out_of_coverage} / {len(res_df)} (detected & aborted cleanly)")
    if unexpected_errors > 0:
      print(f"Unexpected Errors       : {unexpected_errors} / {len(res_df)}")
    print(f"Overall Median IoU      : {res_df['iou'].median():.3f}")
    print(f"Overall Median Dice     : {res_df['dice'].median():.3f}")
    print(f"Basins with IoU >= 0.80 : {(res_df['iou'] >= 0.80).mean() * 100:.1f}%")
    print(f"Basins with IoU >= 0.90 : {(res_df['iou'] >= 0.90).mean() * 100:.1f}%")
    print(f"Median Absolute Area Err: {res_df['abs_area_err_pct'].median():.1f}%")
    print(f"Median Outlet Snap Dist : {res_df['snap_dist_m'].median():.1f} meters")

    # Breakdown Tables
    print_summary_table(res_df, "PERFORMANCE BY CONTINENT", "continent")
    print_summary_table(res_df, "PERFORMANCE BY HEMISPHERE QUADRANT", "hemisphere")
    print_summary_table(res_df, "PERFORMANCE BY BASIN SIZE TIER", "size_tier")
    print("=" * 80)

    if output_path:
      out_p = Path(output_path).expanduser().resolve()
      out_p.parent.mkdir(parents=True, exist_ok=True)
      if str(out_p).endswith(".parquet"):
        res_df.to_parquet(out_p, index=False)
      else:
        res_df.to_csv(out_p, index=False)
      print(f"Detailed per-basin metrics saved to: {out_p}")

    return res_df
  finally:
    if clean_cache and cache_path.exists():
      import shutil
      print(f"Cleaning up local DEM cache at {cache_path}...")
      shutil.rmtree(cache_path, ignore_errors=True)
      print("Local DEM cache cleaned.")


def main():
  parser = argparse.ArgumentParser(
      description="Run global catchment delineation benchmark across 1,000+ reference basins."
  )
  parser.add_argument(
      "--samples",
      type=int,
      default=1000,
      help="Number of basins to evaluate (default: 1000, max: 1200)",
  )
  parser.add_argument(
      "--continents",
      nargs="+",
      default=None,
      help="Filter by continents (e.g. Africa Asia Europe 'North America' 'South America' Oceania)",
  )
  parser.add_argument(
      "--size-tiers",
      nargs="+",
      default=None,
      help="Filter by size tiers (1_micro, 2_small, 3_medium, 4_large, 5_macro)",
  )
  parser.add_argument(
      "--workers",
      type=int,
      default=8,
      help="Number of parallel worker processes (default: 8)",
  )
  parser.add_argument(
      "--tiles-dir",
      type=str,
      default=None,
      help="Optional local DEM tiles directory. If omitted, downloads on demand from GCS.",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default=None,
      help="Path to custom benchmark dataset (.parquet or gs:// URI). Defaults to ~/ancillary-data/benchmarks/benchmark_basins_1000.parquet.",
  )
  parser.add_argument(
      "-o",
      "--output",
      type=str,
      default="benchmark_results.csv",
      help="Path to save detailed per-basin results (.csv or .parquet)",
  )
  parser.add_argument(
      "--snap-window",
      type=int,
      default=12,
      help="Outlet snap search window half-width in cells (default: 12 cells ~1.1 km)",
  )
  parser.add_argument(
      "--clean-cache",
      action="store_true",
      help="Automatically clean up downloaded local DEM tiles from cache after benchmark completes.",
  )

  args = parser.parse_args()

  run_benchmark(
      dataset_path=args.dataset,
      samples=args.samples,
      continents=args.continents,
      size_tiers=args.size_tiers,
      workers=args.workers,
      tiles_dir=args.tiles_dir,
      output_path=args.output,
      snap_window_cells=args.snap_window,
      clean_cache=args.clean_cache,
  )


if __name__ == "__main__":
  main()
