#!/usr/bin/env python3
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

"""Builds a globally balanced 500-basin benchmark dataset for static attribute extraction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("build_benchmark_dataset")

import os
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "static_extractor"
    / "data"
    / "benchmark_basins_500.parquet"
)

CORE_DATASETS = [
    "camels",
    "camelsaus",
    "camelsbr",
    "camelscl",
    "camelsgb",
    "hysets",
    "lamah",
]

SIZE_TIERS = ["1_micro", "2_small", "3_medium", "4_large", "5_macro"]


def get_size_tier(area_km2: float) -> str:
  """Categorizes basin drainage area into one of 5 standardized size tiers."""
  if area_km2 < 100:
    return "1_micro"
  elif area_km2 < 500:
    return "2_small"
  elif area_km2 < 2500:
    return "3_medium"
  elif area_km2 < 10000:
    return "4_large"
  else:
    return "5_macro"


def build_benchmark_dataset(
    caravan_dir: Path,
    output_path: Path,
    target_per_dataset: int = 70,
    random_seed: int = 42,
) -> pd.DataFrame:
  """Samples balanced basins across all 7 Caravan datasets and 5 size tiers."""
  caravan_dir = Path(caravan_dir).resolve()
  output_path = Path(output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  all_basins: List[pd.DataFrame] = []

  for ds in CORE_DATASETS:
    logger.info("Processing dataset '%s'...", ds)
    attr_o_file = caravan_dir / "attributes" / ds / f"attributes_other_{ds}.csv"
    attr_h_file = (
        caravan_dir / "attributes" / ds / f"attributes_hydroatlas_{ds}.csv"
    )
    attr_c_file = (
        caravan_dir / "attributes" / ds / f"attributes_caravan_{ds}.csv"
    )
    shp_file = caravan_dir / "shapefiles" / ds / f"{ds}_basin_shapes.shp"

    if not all(
        f.exists() for f in [attr_o_file, attr_h_file, attr_c_file, shp_file]
    ):
      raise FileNotFoundError(
          f"Missing required Caravan files for dataset '{ds}' under"
          f" {caravan_dir}"
      )

    # 1. Load metadata & size tiers
    df_o = pd.read_csv(attr_o_file)
    df_o["dataset"] = ds
    df_o["size_tier"] = df_o["area"].apply(get_size_tier)

    # 2. Stratified sampling across size tiers
    allocated = {}
    remaining_budget = target_per_dataset
    ideal_per_tier = max(1, target_per_dataset // len(SIZE_TIERS))

    for t in SIZE_TIERS:
      avail = len(df_o[df_o["size_tier"] == t])
      n = min(avail, ideal_per_tier)
      allocated[t] = n
      remaining_budget -= n

    while remaining_budget > 0:
      added_any = False
      for t in SIZE_TIERS:
        avail = len(df_o[df_o["size_tier"] == t])
        if avail > allocated[t] and remaining_budget > 0:
          allocated[t] += 1
          remaining_budget -= 1
          added_any = True
      if not added_any:
        break

    ds_samples = []
    for t, n in allocated.items():
      if n > 0:
        sub = df_o[df_o["size_tier"] == t]
        ds_samples.append(sub.sample(n=n, random_state=random_seed))

    ds_sampled_o = pd.concat(ds_samples, ignore_index=True)
    sampled_gauge_ids = set(ds_sampled_o["gauge_id"])
    logger.info(
        "  Sampled %d basins for '%s' across tiers: %s",
        len(ds_sampled_o),
        ds,
        allocated,
    )

    # 3. Load reference HydroATLAS & Caravan ERA5 attributes
    df_h = pd.read_csv(attr_h_file)
    df_h = df_h[df_h["gauge_id"].isin(sampled_gauge_ids)].copy()
    h_renamed = {c: f"ref_{c}" for c in df_h.columns if c != "gauge_id"}
    df_h = df_h.rename(columns=h_renamed)

    df_c = pd.read_csv(attr_c_file)
    df_c = df_c[df_c["gauge_id"].isin(sampled_gauge_ids)].copy()
    c_renamed = {c: f"ref_{c}" for c in df_c.columns if c != "gauge_id"}
    df_c = df_c.rename(columns=c_renamed)

    # 4. Load geometries from shapefile
    logger.info("  Loading polygon geometries from %s...", shp_file.name)
    gdf = gpd.read_file(shp_file)
    gdf = gdf[gdf["gauge_id"].isin(sampled_gauge_ids)].copy()
    gdf["geometry_wkt"] = gdf.geometry.to_wkt()

    # 5. Merge all fields
    merged = ds_sampled_o.merge(
        gdf[["gauge_id", "geometry_wkt"]], on="gauge_id"
    )
    merged = merged.merge(df_h, on="gauge_id")
    merged = merged.merge(df_c, on="gauge_id")

    # Normalize coordinate and area column names
    merged = merged.rename(
        columns={
            "gauge_lat": "latitude",
            "gauge_lon": "longitude",
            "area": "ref_area_km2",
        }
    )

    all_basins.append(merged)

  final_df = pd.concat(all_basins, ignore_index=True)
  logger.info(
      "Compiled benchmark dataset with %d basins across %d datasets and %d"
      " total columns.",
      len(final_df),
      final_df["dataset"].nunique(),
      len(final_df.columns),
  )

  # Save to Parquet
  logger.info("Writing benchmark dataset to %s...", output_path)
  final_df.to_parquet(output_path, index=False, engine="pyarrow")
  file_size_mb = output_path.stat().st_size / (1024 * 1024)
  logger.info("Successfully created %s (%.2f MB).", output_path, file_size_mb)

  return final_df


def main():
  parser = argparse.ArgumentParser(
      description="Build balanced global benchmark dataset for static extractor."
  )
  parser.add_argument(
      "--caravan-dir",
      type=str,
      default=os.environ.get("CARAVAN_DIR"),
      required="CARAVAN_DIR" not in os.environ,
      help="Path to Caravan root directory containing attributes/ and shapefiles/.",
  )
  parser.add_argument(
      "--output",
      type=str,
      default=str(DEFAULT_OUTPUT_PATH),
      help="Target parquet output path.",
  )
  parser.add_argument(
      "--target-per-dataset",
      type=int,
      default=70,
      help="Number of basins to sample per dataset (default: 70 -> 490 basins).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for deterministic sampling.",
  )

  args = parser.parse_args()
  build_benchmark_dataset(
      caravan_dir=Path(args.caravan_dir),
      output_path=Path(args.output),
      target_per_dataset=args.target_per_dataset,
      random_seed=args.seed,
  )


if __name__ == "__main__":
  main()
