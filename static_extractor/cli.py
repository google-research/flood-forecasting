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

"""Command-line interface for Caravan static attributes extraction."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from static_extractor.extractor import StaticAttributesExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("static_extractor")


def parse_args(args=None):
  parser = argparse.ArgumentParser(
      description="Extract Caravan HydroATLAS and ERA5 static attributes for watershed polygons.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--input",
      "-i",
      required=True,
      type=str,
      help="Path to input vector watershed polygon file (GeoJSON, Shapefile, GPKG).",
  )
  parser.add_argument(
      "--output",
      "-o",
      required=True,
      type=str,
      help="Path to output CSV file for extracted attributes.",
  )
  parser.add_argument(
      "--gdb-path",
      "-g",
      default=None,
      type=str,
      help="Path to BasinATLAS_v10.gdb directory or BasinATLAS_v10_lev12.shp. Defaults to cache or auto-discovery.",
  )
  parser.add_argument(
      "--era5-cache-dir",
      default=None,
      type=str,
      help="Directory where continental ERA5 climate index files are cached.",
  )
  parser.add_argument(
      "--id-column",
      default=None,
      type=str,
      help="Column name in vector file containing gauge or catchment ID.",
  )
  parser.add_argument(
      "--min-overlap-threshold",
      default=0.0,
      type=float,
      help="Minimum sub-basin intersection area threshold in km².",
  )
  parser.add_argument(
      "--auto-download",
      action="store_true",
      default=True,
      help="Automatically download data from Google Cloud Storage if not staged locally.",
  )
  parser.add_argument(
      "--era5-source",
      choices=["hybas", "gridded"],
      required=True,
      help="Source for ERA5 climate metrics (required): 'hybas' (fast area-weighted aggregation of precalculated Level 12 sub-basin statistics) or 'gridded' (recalculated on the fly from archived gridded ERA5 daily surface data on GCS).",
  )
  parser.add_argument(
      "--gridded-era5-uri",
      default=None,
      type=str,
      help="GCS URI or path to gridded daily ERA5 Zarr store. Defaults to gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr.",
  )
  parser.add_argument(
      "--cache-dir",
      default=None,
      type=str,
      help="Base directory for runtime cache (defaults to ~/.cache/googlehydrology).",
  )
  parser.add_argument(
      "--no-download",
      action="store_false",
      dest="auto_download",
      default=True,
      help="Disable automatic GCS downloads. Requires local files to be present.",
  )
  parser.add_argument(
      "--clean-cache",
      action="store_true",
      help="Automatically clean up the entire local cache directory (~/.cache/googlehydrology) after extraction finishes.",
  )
  parser.add_argument(
      "--workers",
      "-w",
      default=1,
      type=int,
      help="Number of parallel worker processes to use (default: 1).",
  )
  parser.add_argument(
      "--verbose",
      "-v",
      action="store_true",
      help="Show detailed debug/info log messages (disabled by default for clean progress bars).",
  )
  return parser.parse_args(args)


def main(args=None):
  parsed = parse_args(args)
  level = logging.DEBUG if parsed.verbose else logging.WARNING
  logging.basicConfig(
      level=level,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
      force=True,
  )
  if not parsed.verbose:
    for name in ["static_extractor", "urllib3", "google", "gcsfs", "fiona", "pyogrio"]:
      logging.getLogger(name).setLevel(logging.WARNING)
  logging.getLogger("asyncio").setLevel(logging.CRITICAL)

  input_path = Path(parsed.input)
  if not input_path.exists():
    logger.error("Input file '%s' does not exist.", input_path)
    sys.exit(1)

  cache_root = Path(parsed.cache_dir) if parsed.cache_dir else Path.home() / ".cache" / "googlehydrology"
  gdb_path = parsed.gdb_path or (cache_root / "hydroatlas" / "BasinATLAS_v10.gdb")
  era5_cache_dir = parsed.era5_cache_dir or (cache_root / "era5_climate")

  try:
    logger.debug("Initializing Caravan Static Attributes Extractor (ERA5 source: %s)...", parsed.era5_source)
    extractor = StaticAttributesExtractor(
        gdb_path=str(gdb_path),
        era5_cache_dir=str(era5_cache_dir),
        auto_download=parsed.auto_download,
        era5_source=parsed.era5_source,
        gridded_era5_uri=parsed.gridded_era5_uri,
    )

    logger.debug(
        "Extracting static attributes from '%s' (workers=%d)...",
        input_path,
        parsed.workers,
    )
    df = extractor.extract_attributes_from_file(
        input_path=input_path,
        output_csv_path=parsed.output,
        id_column=parsed.id_column,
        min_overlap_threshold=parsed.min_overlap_threshold,
        workers=parsed.workers,
        show_progress=True,
    )

    print(
        f"\n✓ Extracted {df.shape[1]} attributes for {df.shape[0]} catchments -> {parsed.output}"
    )
  finally:
    if parsed.clean_cache and cache_root.exists():
      import shutil
      logger.debug("Cleaning up cache root directory %s...", cache_root)
      shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
  main()
