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

"""MultiMet Canary Test Runner.

Allows running local MultiMet forcing extractions over arbitrary catchment
geometries and date ranges.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure flood-forecasting-multimet repository root is in sys.path
_CANDIDATE_ROOTS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Projects/flood-forecasting-multimet")),
    os.path.expanduser("~/Projects/flood-forecasting-multimet"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../flood-forecasting-multimet")),
]
for _root in _CANDIDATE_ROOTS:
  if os.path.exists(os.path.join(_root, "multimet")):
    if _root not in sys.path:
      sys.path.insert(0, _root)
    break

import pandas as pd
import xarray as xr

from multimet.config import Product
from multimet.cpc import CPCExtractor
from multimet.era5_land import ERA5LandExtractor
from multimet.geometry import load_basin_geometries
from multimet.graphcast import GraphCastExtractor
from multimet.hres import HRESExtractor
from multimet.imerg import IMERGExtractor
from multimet.dynamical import AIFSExtractor, DynamicalIMERGExtractor
from multimet.zarr_writer import MultiMetZarrWriter
from multimet.zonal import ZonalWeightMatrix
from multimet.dask_runner import extract_product_dask

# Placeholders for upcoming modules
CHIRPSExtractor = None
CHIRPSGEFSExtractor = None

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_BASIN_PATHS = [
    os.path.join(_SCRIPT_DIR, "wabash_test_data/shapefiles/us/us_basin_shapes.geojson"),
    os.path.expanduser("~/multimet/canary/wabash_test_data/shapefiles/us/us_basin_shapes.geojson"),
    os.path.abspath(os.path.join(_SCRIPT_DIR, "../test/test_data/shapefiles/us/us_basin_shapes.geojson")),
    os.path.abspath(os.path.join(_SCRIPT_DIR, "../../multimet/test/test_data/shapefiles/us/us_basin_shapes.geojson")),
    os.path.expanduser("~/Projects/flood-forecasting-multimet/multimet/test/test_data/shapefiles/us/us_basin_shapes.geojson"),
    os.path.expanduser("~/Projects/flood-forecasting-multimet/test/test_data/shapefiles/us/us_basin_shapes.geojson"),
]
DEFAULT_TEST_BASINS = next(
    (p for p in _CANDIDATE_BASIN_PATHS if os.path.exists(p)),
    _CANDIDATE_BASIN_PATHS[0],
)
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")

PRODUCT_MAP = {
    "CPC": (Product.CPC, CPCExtractor),
    "IMERG": (Product.IMERG, IMERGExtractor),
    "ERA5_LAND": (Product.ERA5_LAND, ERA5LandExtractor),
    "HRES": (Product.HRES, HRESExtractor),
    "GRAPHCAST": (Product.GRAPHCAST, GraphCastExtractor),
    "AIFS": (Product.AIFS, AIFSExtractor),
    "DYNAMICAL_IMERG": (Product.DYNAMICAL_IMERG, DynamicalIMERGExtractor),
}
if CHIRPSExtractor is not None and hasattr(Product, "CHIRPS"):
  PRODUCT_MAP["CHIRPS"] = (Product.CHIRPS, CHIRPSExtractor)
if CHIRPSGEFSExtractor is not None and hasattr(Product, "CHIRPS_GEFS"):
  PRODUCT_MAP["CHIRPS_GEFS"] = (Product.CHIRPS_GEFS, CHIRPSGEFSExtractor)


def _str2bool(v: Union[str, bool]) -> bool:
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def parse_canary_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="MultiMet Canary Extraction Runner",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--basins_path",
      type=str,
      default=DEFAULT_TEST_BASINS,
      help="Path to GeoJSON catchment boundaries file.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=DEFAULT_OUTPUT_DIR,
      help="Directory to save extracted Zarr stores.",
  )
  parser.add_argument(
      "--products",
      type=str,
      default="CPC",
      help="Comma-separated products to run (e.g. CPC,IMERG,HRES,GRAPHCAST,ERA5_LAND).",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default="2020-01-01",
      help="Start date in YYYY-MM-DD format.",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default="2020-01-02",
      help="End date in YYYY-MM-DD format.",
  )
  parser.add_argument(
      "--id_column",
      type=str,
      default=None,
      help="Optional ID column name in the GeoJSON.",
  )
  parser.add_argument(
      "--source",
      type=str,
      default="public",
      help="Data source mode: 'public' (WeatherBench 2 / PSL / GES DISC / CHC) or 'local'.",
  )
  parser.add_argument(
      "--overwrite",
      nargs="?",
      const=True,
      default=True,
      type=_str2bool,
      help=(
          "If True, first deletes existing destination Zarr store for products"
          " in this canary run before extracting (default: True). Pass"
          " --no-overwrite to preserve/append existing data."
      ),
  )
  parser.add_argument(
      "--no-overwrite",
      "--no_overwrite",
      dest="overwrite",
      action="store_false",
      help="Do not delete existing Zarr stores before extracting.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=1,
      help="Number of parallel worker processes for temporal batch extraction.",
  )
  parser.add_argument(
      "--weights_cache",
      type=str,
      default=None,
      help="Path to precomputed/cached ZonalWeightMatrix (.npz) file.",
  )
  parser.add_argument(
      "--chunk_freq",
      type=str,
      default=None,
      help="Temporal chunk frequency for parallel extraction (e.g. 'YS', 'MS', 'auto').",
  )
  parser.add_argument(
      "--earthdata_username",
      type=str,
      default=None,
      help="Optional Earthdata Login username (defaults to ~/.netrc).",
  )
  parser.add_argument(
      "--earthdata_password",
      type=str,
      default=None,
      help="Optional Earthdata Login password (defaults to ~/.netrc).",
  )
  parser.add_argument(
      "--earthdata_token",
      type=str,
      default=None,
      help="Optional Earthdata Login Bearer token.",
  )
  parser.add_argument(
      "--netrc_path",
      type=str,
      default=None,
      help="Optional custom path to .netrc file.",
  )
  parser.add_argument(
      "--use_bounding_box",
      nargs="?",
      const=True,
      default=True,
      type=_str2bool,
      help=(
          "Whether to use spatial bounding box slicing to reduce"
          " memory/compute (default: True)."
      ),
  )
  parser.add_argument(
      "--no-bounding-box",
      "--no_bounding_box",
      dest="use_bounding_box",
      action="store_false",
      help="Disable spatial bounding box slicing and process full global grid.",
  )
  return parser.parse_args(argv)


def run_canary(args: argparse.Namespace) -> None:
  print("=" * 70)
  print("🦅 MULTIMET EXTRACTION CANARY TEST RUNNER")
  print("=" * 70)
  print(f"  • Catchment GeoJSON : {args.basins_path}")
  print(f"  • Output Directory  : {args.output_dir}")
  print(f"  • Target Products   : {args.products}")
  print(f"  • Source Mode       : {args.source}")
  print(f"  • Date Range        : {args.start_date} to {args.end_date}")
  print(f"  • Overwrite Mode    : {args.overwrite}")
  print(f"  • Bounding Box      : {args.use_bounding_box}")
  print("=" * 70)

  # 1. Load Geometries
  if not os.path.exists(args.basins_path):
    print(f"❌ Error: Basins file not found at: {args.basins_path}")
    sys.exit(1)

  t0 = time.time()
  gdf = load_basin_geometries(args.basins_path, id_column=args.id_column)
  basin_ids = list(gdf.index)
  print(f"\n✅ Loaded {len(gdf)} basin geometries in {time.time() - t0:.2f}s")
  print(f"   Basin IDs: {basin_ids[:10]}{' ...' if len(basin_ids) > 10 else ''}")

  # 2. Parse Requested Products
  raw_prods = list(
      dict.fromkeys(
          [p.strip().upper() for p in args.products.split(",") if p.strip()]
      )
  )
  writer = MultiMetZarrWriter(args.output_dir)

  # 3. Handle Overwrite: Delete existing Zarr stores for products in this run
  if args.overwrite:
    for prod_name in raw_prods:
      if prod_name in PRODUCT_MAP:
        prod_enum, _ = PRODUCT_MAP[prod_name]
        target_store = writer.get_store_path(prod_enum)
        if os.path.exists(target_store):
          print(
              f"🗑️  [Overwrite] Deleting existing Zarr store for {prod_name}:"
              f" {target_store}"
          )
          if os.path.isdir(target_store):
            shutil.rmtree(target_store, ignore_errors=True)
          else:
            os.remove(target_store)

  results_summary = []

  for prod_name in raw_prods:
    if prod_name not in PRODUCT_MAP:
      print(f"\n⚠️ Unknown product '{prod_name}'. Valid options: {list(PRODUCT_MAP.keys())}")
      continue

    prod_enum, extractor_cls = PRODUCT_MAP[prod_name]
    print(f"\n▶ Running extraction for: {prod_name}...")
    t_start = time.time()

    prod_start = args.start_date
    prod_end = args.end_date
    if prod_name == "AIFS":
      if pd.to_datetime(prod_start) < pd.to_datetime("2024-04-01"):
        print(
            f"  ℹ️ Note: AIFS operational forecasts begin on 2024-04-01."
            f" Adjusting canary test dates from {prod_start}..{prod_end} to 2024-05-01..2024-05-02."
        )
        prod_start = "2024-05-01"
        prod_end = "2024-05-02"

    if prod_name == "CPC":
      src = "psl" if args.source in ("public", "auto") else ("binary" if args.source == "local" else args.source)
      extractor_kwargs = {"source": src}
    elif prod_name == "IMERG":
      if args.source in ("dynamical", "icechunk", "catalog"):
        extractor_cls = DynamicalIMERGExtractor
        extractor_kwargs = {"source": args.source}
      else:
        src = "gesdisc" if args.source in ("public", "auto") else ("h5" if args.source == "local" else args.source)
        extractor_kwargs = {
            "source": src,
            "username": args.earthdata_username,
            "password": args.earthdata_password,
            "token": args.earthdata_token,
            "netrc_path": args.netrc_path,
        }
    elif prod_name in ("GRAPHCAST", "HRES", "ERA5_LAND"):
      src = "wb2" if args.source in ("public", "auto") else ("local" if args.source == "local" else args.source)
      extractor_kwargs = {"source": src}
    elif prod_name in ("AIFS", "DYNAMICAL_IMERG"):
      extractor_kwargs = {"source": args.source}
    else:
      extractor_kwargs = {}

    weights_path = None
    if args.weights_cache:
      if os.path.isdir(args.weights_cache):
        weights_path = os.path.join(
            args.weights_cache, f"weights_{prod_name.lower()}.npz"
        )
      elif len(raw_prods) > 1:
        root, ext = os.path.splitext(args.weights_cache)
        weights_path = f"{root}_{prod_name.lower()}{ext or '.npz'}"
      else:
        weights_path = args.weights_cache

    if args.num_workers > 1:
      print(
          f"  Spawning {args.num_workers} parallel workers via Dask..."
      )
      store_path = extract_product_dask(
          product=prod_enum,
          basins=gdf,
          output_dir=args.output_dir,
          start_date=prod_start,
          end_date=prod_end,
          num_workers=args.num_workers,
          weights_cache=weights_path,
          overwrite=args.overwrite,
          use_bounding_box=args.use_bounding_box,
          **extractor_kwargs,
      )
    else:
      extractor = extractor_cls(**extractor_kwargs)
      weights = None
      if weights_path:
        if os.path.exists(weights_path):
          print(f"  Loaded precomputed weights from: {weights_path}")
          weights = ZonalWeightMatrix.load(weights_path)
        elif extractor.lats is not None and extractor.lons is not None:
          print(f"  Precomputing weights matrix -> {weights_path}...")
          weights = ZonalWeightMatrix.from_geodataframe(
              gdf, extractor.lats, extractor.lons, num_workers=4
          )
          weights.save(weights_path)
          print(f"  Saved weights matrix: {weights_path}")

      ds = extractor.extract_for_basins(
          gdf,
          start_date=prod_start,
          end_date=prod_end,
          weights_matrix=weights,
          use_bounding_box=args.use_bounding_box,
      )
      # Save to Zarr
      store_path = writer.write_or_append(
          ds, prod_enum, overwrite_existing_basins=args.overwrite
      )

    elapsed = time.time() - t_start
    start_dt = pd.to_datetime(prod_start)
    end_dt = pd.to_datetime(prod_end)
    total_days = max(1, (end_dt - start_dt).days + 1)
    total_basin_days = len(basin_ids) * total_days
    throughput = total_basin_days / elapsed if elapsed > 0 else 0.0
    print(
        f"  Extraction finished in {elapsed:.2f}s ({throughput:.1f} basin-days/s)."
    )
    print(f"  Saved Zarr store: {store_path}")

    # Verify and inspect extracted dataset
    ds_verify = xr.open_zarr(store_path)
    vars_list = list(ds_verify.data_vars.keys())

    print("\n  🔍 Extracted Data Preview:")
    for var in vars_list:
      val_arr = ds_verify[var].values
      finite_count = int((~pd.isna(val_arr)).sum())
      total_count = int(val_arr.size)
      series = pd.Series(val_arr.flatten()).dropna()
      if len(series) > 0:
        min_val = f"{series.min():.4f}"
        max_val = f"{series.max():.4f}"
        mean_val = f"{series.mean():.4f}"
      else:
        min_val = max_val = mean_val = "nan"
      print(f"    - Variable: {var}")
      print(f"      Dimensions : {dict(ds_verify[var].sizes)}")
      print(f"      Valid data : {finite_count}/{total_count} points")
      print(f"      Range      : min={min_val}, max={max_val}, mean={mean_val}")

    results_summary.append({
        "product": prod_name,
        "status": "SUCCESS",
        "elapsed_s": round(elapsed, 2),
        "basin_days": total_basin_days,
        "throughput_b_days_per_s": round(throughput, 1),
        "variables": ", ".join(vars_list),
        "zarr_store": store_path,
    })

  # Summary Table
  print("\n" + "=" * 70)
  print("🏁 CANARY RUN SUMMARY")
  print("=" * 70)
  summary_df = pd.DataFrame(results_summary)
  print(summary_df.to_string(index=False))
  print("=" * 70)


def main(argv: Optional[Sequence[str]] = None) -> None:
  cli_args = None
  if argv is not None:
    if len(argv) > 0 and (
        argv[0] == sys.argv[0]
        or argv[0].endswith(".py")
        or "canary" in argv[0]
    ):
      cli_args = list(argv[1:])
    else:
      cli_args = list(argv)
  args = parse_canary_args(cli_args)
  run_canary(args)


if __name__ == "__main__":
  main(sys.argv[1:])
