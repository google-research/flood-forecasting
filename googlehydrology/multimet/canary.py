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

"""Canary test script for running local MultiMet forcing extractions.

Allows developers and researchers to quickly test forcing extraction for
arbitrary date ranges, basin GeoJSON files, and target products locally on
Cloudtop.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import Optional, Sequence

import pandas as pd
import xarray as xr

from googlehydrology.multimet.config import Product
from googlehydrology.multimet.cpc import CPCExtractor
from googlehydrology.multimet.era5_land import ERA5LandExtractor
from googlehydrology.multimet.geometry import load_basin_geometries
from googlehydrology.multimet.graphcast import GraphCastExtractor
from googlehydrology.multimet.hres import HRESExtractor
from googlehydrology.multimet.imerg import IMERGExtractor
from googlehydrology.multimet.zarr_writer import MultiMetZarrWriter
from googlehydrology.multimet.zonal import ZonalWeightMatrix

# Optional imports for upcoming PRs
try:
  from googlehydrology.multimet.chirps import CHIRPSExtractor
except ImportError:
  CHIRPSExtractor = None

try:
  from googlehydrology.multimet.chirps_gefs import CHIRPSGEFSExtractor
except ImportError:
  CHIRPSGEFSExtractor = None

try:
  from googlehydrology.multimet.parallel import extract_in_parallel
except ImportError:
  extract_in_parallel = None

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_BASIN_PATHS = [
    os.path.join(_BASE_DIR, "../../test/test_data/shapefiles/us/us_basin_shapes.geojson"),
    os.path.join(_BASE_DIR, "test/test_data/shapefiles/us/us_basin_shapes.geojson"),
    os.path.abspath("test/test_data/shapefiles/us/us_basin_shapes.geojson"),
]
DEFAULT_TEST_BASINS = next(
    (p for p in _CANDIDATE_BASIN_PATHS if os.path.exists(p)),
    _CANDIDATE_BASIN_PATHS[0],
)

PRODUCT_MAP = {
    "CPC": (Product.CPC, CPCExtractor),
    "IMERG": (Product.IMERG, IMERGExtractor),
    "ERA5_LAND": (Product.ERA5_LAND, ERA5LandExtractor),
    "HRES": (Product.HRES, HRESExtractor),
    "GRAPHCAST": (Product.GRAPHCAST, GraphCastExtractor),
}
if CHIRPSExtractor is not None and hasattr(Product, "CHIRPS"):
  PRODUCT_MAP["CHIRPS"] = (Product.CHIRPS, CHIRPSExtractor)
if CHIRPSGEFSExtractor is not None and hasattr(Product, "CHIRPS_GEFS"):
  PRODUCT_MAP["CHIRPS_GEFS"] = (Product.CHIRPS_GEFS, CHIRPSGEFSExtractor)


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
      default=os.path.join(tempfile.gettempdir(), "multimet_canary"),
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
      action="store_true",
      help="Overwrite existing basins in destination Zarr stores.",
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
  raw_prods = [p.strip().upper() for p in args.products.split(",") if p.strip()]
  writer = MultiMetZarrWriter(args.output_dir)

  results_summary = []

  for prod_name in raw_prods:
    if prod_name not in PRODUCT_MAP:
      print(f"\n⚠️ Unknown product '{prod_name}'. Valid options: {list(PRODUCT_MAP.keys())}")
      continue

    prod_enum, extractor_cls = PRODUCT_MAP[prod_name]
    print(f"\n▶ Running extraction for: {prod_name}...")
    t_start = time.time()

    try:
      if prod_name == "CPC":
        src = "psl" if args.source in ("public", "auto") else ("binary" if args.source == "local" else args.source)
        extractor_kwargs = {"source": src}
      elif prod_name == "IMERG":
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
        if extract_in_parallel is None:
          raise NotImplementedError(
              "Parallel extraction is not yet installed in this branch. Run with"
              " --num_workers=1."
          )
        print(
            f"  Spawning {args.num_workers} parallel workers (chunk_freq="
            f"{args.chunk_freq or 'auto'})..."
        )
        ds = extract_in_parallel(
            extractor_cls,
            gdf,
            start_date=args.start_date,
            end_date=args.end_date,
            num_workers=args.num_workers,
            chunk_freq=args.chunk_freq,
            weights_cache=weights_path,
            output_dir=args.output_dir,
            product=prod_enum,
            overwrite=args.overwrite,
            **extractor_kwargs,
        )
        store_path = writer.get_store_path(prod_enum)
      else:
        extractor = extractor_cls(**extractor_kwargs)
        weights = None
        if weights_path:
          if os.path.exists(weights_path):
            print(f"  Loaded precomputed weights from: {weights_path}")
            weights = ZonalWeightMatrix.load(weights_path)
          else:
            print(f"  Precomputing weights matrix -> {weights_path}...")
            weights = ZonalWeightMatrix.from_geodataframe(
                gdf, extractor.lats, extractor.lons, num_workers=4
            )
            weights.save(weights_path)
            print(f"  Saved weights matrix: {weights_path}")

        ds = extractor.extract_for_basins(
            gdf,
            start_date=args.start_date,
            end_date=args.end_date,
            weights_matrix=weights,
        )
        # Save to Zarr
        store_path = writer.write_or_append(
            ds, prod_enum, overwrite_existing_basins=args.overwrite
        )

      elapsed = time.time() - t_start
      start_dt = pd.to_datetime(args.start_date)
      end_dt = pd.to_datetime(args.end_date)
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

    except Exception as e:
      print(f"❌ Failed to extract {prod_name}: {e}")
      results_summary.append({
          "product": prod_name,
          "status": f"FAILED: {e}",
          "elapsed_s": round(time.time() - t_start, 2),
          "basin_days": 0,
          "throughput_b_days_per_s": 0.0,
          "variables": "N/A",
          "zarr_store": "N/A",
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
