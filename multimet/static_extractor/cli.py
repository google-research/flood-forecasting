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
from pathlib import Path
import shutil
import sys

from multimet.static_extractor.extractor import StaticAttributesExtractor

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
      help="Path to input vector watershed polygon file (GeoJSON, Shapefile, GPKG, Parquet).",
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
      help="Path to local BasinATLAS_v10.gdb directory, shapefile, or GeoParquet file (required unless --no-download is used with --gcs-gdb-uri).",
  )
  parser.add_argument(
      "--era5-source",
      choices=["hybas", "gridded"],
      required=True,
      help="Source for ERA5 climate metrics: 'hybas' (precalculated Level 12 sub-basin statistics) or 'gridded' (recalculated from daily ERA5 Zarr data).",
  )
  parser.add_argument(
      "--era5-cache-dir",
      default=None,
      type=str,
      help="Local directory containing continental ERA5 climate index files (required when --era5-source=hybas unless --no-download is used with --gcs-era5-climate-uri).",
  )
  parser.add_argument(
      "--gridded-era5-uri",
      default=None,
      type=str,
      help="GCS URI or local path to gridded daily ERA5 Zarr store (required when --era5-source=gridded; optional when --era5-source=hybas to compute *_ERA5_LAND columns).",
  )
  parser.add_argument(
      "--gcs-gdb-uri",
      default=None,
      type=str,
      help="Optional GCS URI for HydroATLAS data (downloaded into --gdb-path by default, or streamed in memory when --no-download is set).",
  )
  parser.add_argument(
      "--gcs-era5-climate-uri",
      default=None,
      type=str,
      help="Optional GCS URI for continental ERA5 climate tables (downloaded into --era5-cache-dir by default, or streamed in memory when --no-download is set).",
  )
  parser.add_argument(
      "--no-download",
      action="store_true",
      help="Stream HydroATLAS and ERA5 data directly from Google Cloud Storage in memory without downloading files to local disk.",
  )
  parser.add_argument(
      "--id-column",
      default="gauge_id",
      type=str,
      help="Column name in vector file containing the watershed or gauge ID.",
  )
  parser.add_argument(
      "--min-overlap-threshold",
      default=0.0,
      type=float,
      help="Minimum sub-basin intersection area threshold in km².",
  )
  parser.add_argument(
      "--clean-cache",
      action="store_true",
      help="Delete local --gdb-path and --era5-cache-dir directories after extraction finishes.",
  )
  parser.add_argument(
      "--workers",
      "-w",
      default=1,
      type=int,
      help="Number of parallel worker processes to use.",
  )
  parser.add_argument(
      "--verbose",
      "-v",
      action="store_true",
      help="Show detailed debug/info log messages.",
  )
  parsed = parser.parse_args(args)
  if parsed.no_download:
    if not parsed.gdb_path and not parsed.gcs_gdb_uri:
      parser.error(
          "Either --gdb-path or --gcs-gdb-uri is required when --no-download is set."
      )
    if (
        parsed.era5_source == "hybas"
        and not parsed.era5_cache_dir
        and not parsed.gcs_era5_climate_uri
    ):
      parser.error(
          "Either --era5-cache-dir or --gcs-era5-climate-uri is required when --era5-source is 'hybas' with --no-download."
      )
  else:
    if not parsed.gdb_path:
      parser.error(
          "--gdb-path is required unless --no-download is set with --gcs-gdb-uri."
      )
    if parsed.era5_source == "hybas" and not parsed.era5_cache_dir:
      parser.error("--era5-cache-dir is required when --era5-source is 'hybas'.")
  if parsed.era5_source == "gridded" and not parsed.gridded_era5_uri:
    parser.error("--gridded-era5-uri is required when --era5-source is 'gridded'.")
  return parsed


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

  logger.debug(
      "Initializing Caravan Static Attributes Extractor (ERA5 source: %s)...",
      parsed.era5_source,
  )
  extractor = StaticAttributesExtractor(
      gdb_path=parsed.gdb_path,
      era5_source=parsed.era5_source,
      era5_cache_dir=parsed.era5_cache_dir,
      gridded_era5_uri=parsed.gridded_era5_uri,
      gcs_gdb_uri=parsed.gcs_gdb_uri,
      gcs_era5_climate_uri=parsed.gcs_era5_climate_uri,
      no_download=parsed.no_download,
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

  if parsed.clean_cache and not parsed.no_download:
    if parsed.gdb_path and not str(parsed.gdb_path).startswith(("gs://", "gcs://")):
      gdb_p = Path(parsed.gdb_path)
      if gdb_p.exists():
        if gdb_p.is_dir():
          shutil.rmtree(gdb_p)
        else:
          gdb_p.unlink()
    if parsed.era5_cache_dir and not str(parsed.era5_cache_dir).startswith(("gs://", "gcs://")):
      era5_p = Path(parsed.era5_cache_dir)
      if era5_p.exists() and era5_p.is_dir():
        shutil.rmtree(era5_p)


if __name__ == "__main__":
  main()
