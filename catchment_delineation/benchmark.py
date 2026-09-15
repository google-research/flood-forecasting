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

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import shape
import shapely.wkt

from catchment_delineation.config import GCS_TILES_URI, get_default_cache_dir
from catchment_delineation.delineator import DemDelineator

logger = logging.getLogger("catchment_delineation.benchmark")

DEFAULT_BENCHMARK_PATH = (
    Path(__file__).parent / "data" / "benchmark_basins_1000.parquet"
)


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
        snap_window_cells=4,
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
) -> pd.DataFrame:
  """Executes the global catchment delineation benchmark across test basins."""
  ds_path = (
      Path(dataset_path) if dataset_path else DEFAULT_BENCHMARK_PATH
  ).resolve()

  if not ds_path.exists():
    # Attempt to download from GCS if dataset file not found locally
    print(f"Benchmark dataset not found locally at {ds_path}. Downloading from GCS...")
    import subprocess
    import shutil
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_src = f"{GCS_TILES_URI.replace('/tiles_5deg', '')}/benchmark_basins_1000.parquet"
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
  else:
    print(f"Using GCS tile bucket: {gcs_uri} (cached in {cache_dir or get_default_cache_dir()})")

  rows = df.to_dict(orient="records")
  results = []
  t_start = time.time()

  with ProcessPoolExecutor(max_workers=workers) as executor:
    futures = {
        executor.submit(
            _evaluate_single_basin,
            row,
            tiles_dir=tiles_dir,
            gcs_uri=gcs_uri,
            cache_dir=cache_dir,
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

  # Overall Report
  print("\n" + "=" * 80)
  print("GLOBAL CATCHMENT DELINEATION BENCHMARK RESULTS")
  print("=" * 80)
  print(f"Total Basins Evaluated : {len(res_df)}")
  print(f"Total Wall-Clock Time   : {total_time:.1f}s ({total_time / max(1, len(res_df)):.3f}s / basin)")
  print(f"Successful Delineations : {(res_df['status'] == 'SUCCESS').sum()} / {len(res_df)}")
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
      help="Path to custom benchmark dataset (.parquet). Defaults to bundled 1,200 basin dataset.",
  )
  parser.add_argument(
      "-o",
      "--output",
      type=str,
      default="benchmark_results.csv",
      help="Path to save detailed per-basin results (.csv or .parquet)",
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
  )


if __name__ == "__main__":
  main()
