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

"""Multi-dataset batch runner for Caravan static attribute extraction."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth.*")

import pandas as pd
from tqdm.auto import tqdm

from multimet.static_extractor.extractor import StaticAttributesExtractor

logger = logging.getLogger("static_extractor.batch_runner")


def setup_logging(verbose: bool = False) -> None:
  """Configures logging levels, suppressing noisy info logs unless verbose."""
  level = logging.DEBUG if verbose else logging.WARNING
  logging.basicConfig(
      level=level,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
      force=True,
  )
  if not verbose:
    for name in [
        "static_extractor",
        "static_extractor.batch_runner",
        "static_extractor.extractor",
        "static_extractor.gcs",
        "static_extractor.climate",
        "urllib3",
        "google",
        "google.auth",
        "gcsfs",
        "fiona",
        "pyogrio",
    ]:
      logging.getLogger(name).setLevel(logging.WARNING)
  logging.getLogger("asyncio").setLevel(logging.CRITICAL)


SUPPORTED_EXTENSIONS = [".shp", ".geojson", ".gpkg", ".json", ".parquet", ".geoparquet"]

CARAVAN_CLIMATE_INDICES = {
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    "moisture_index",
    "seasonality",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "aridity_ERA5_LAND",
    "aridity_FAO_PM",
    "pet_mean_ERA5_LAND",
    "pet_mean_FAO_PM",
    "moisture_index_ERA5_LAND",
    "moisture_index_FAO_PM",
    "seasonality_ERA5_LAND",
    "seasonality_FAO_PM",
}


def export_subdataset_partitioned_files(
    df: pd.DataFrame,
    ds_name: str,
    output_sub_dir: Path,
) -> Dict[str, Path]:
  """Exports extracted attributes into partitioned HydroATLAS, Caravan, and Parquet files.

  Schema per subdataset directory:
    - attributes_hydroatlas_<ds>.csv (HydroATLAS attributes)
    - attributes_caravan_<ds>.csv (ERA5 Caravan climate indices)
    - attributes_<ds>.parquet (single unified table merging both tables on gauge_id)
  """
  output_sub_dir.mkdir(parents=True, exist_ok=True)
  df_work = df.copy()

  caravan_cols = [
      c for c in df_work.columns
      if c in CARAVAN_CLIMATE_INDICES
  ]
  hydro_cols = [c for c in df_work.columns if c not in caravan_cols]

  sorted_hydro = sorted(hydro_cols)
  if "basin_area" in sorted_hydro:
    sorted_hydro.remove("basin_area")
    sorted_hydro = ["basin_area"] + sorted_hydro

  sorted_caravan = sorted(caravan_cols)

  df_hydro = df_work[sorted_hydro].sort_index()
  df_caravan = df_work[sorted_caravan].sort_index()

  hydro_path = output_sub_dir / f"attributes_hydroatlas_{ds_name}.csv"
  caravan_path = output_sub_dir / f"attributes_caravan_{ds_name}.csv"
  df_hydro.to_csv(hydro_path)
  df_caravan.to_csv(caravan_path)

  df_unified = df_hydro.join(df_caravan, how="outer")
  parquet_path = output_sub_dir / f"attributes_{ds_name}.parquet"
  df_unified.to_parquet(parquet_path)

  return {
      "hydroatlas": hydro_path,
      "caravan": caravan_path,
      "parquet": parquet_path,
  }


from multimet.utils.gcs import (
    gcs_path_exists,
    sync_gcs_directory,
    upload_to_gcs,
)


def find_vector_file_in_dir(dataset_dir: Path) -> Optional[Path]:
  """Finds the primary watershed polygon file in a dataset directory."""
  dataset_name = dataset_dir.name.lower()

  preferred_names = [
      f"{dataset_name}_basin_shapes.shp",
      f"{dataset_name}_basins.shp",
      f"{dataset_name}.shp",
      f"{dataset_name}.geojson",
      f"{dataset_name}.geoparquet",
      f"{dataset_name}.parquet",
  ]
  for pref in preferred_names:
    p = dataset_dir / pref
    if p.exists():
      return p

  shps = sorted(dataset_dir.glob("*.shp"))
  if shps:
    basin_shps = [
        s
        for s in shps
        if "gauge" not in s.name.lower() and "point" not in s.name.lower()
    ]
    return basin_shps[0] if basin_shps else shps[0]

  for ext in [".geojson", ".gpkg", ".json", ".parquet", ".geoparquet"]:
    matches = sorted(dataset_dir.glob(f"*{ext}"))
    if matches:
      return matches[0]

  return None


def find_all_dataset_dirs(root_dir: Path) -> Dict[str, Path]:
  """Recursively finds all dataset directories containing vector files under root_dir."""
  datasets: Dict[str, Path] = {}
  direct_vfs = [
      f for f in root_dir.iterdir()
      if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
      and "gauge" not in f.name.lower() and "point" not in f.name.lower()
  ]
  if len(direct_vfs) > 1:
    for vf in sorted(direct_vfs):
      ds_key = f"{root_dir.name}_{vf.stem}" if root_dir.name not in ["staged_shapefiles", "data", "shapes"] else vf.stem
      datasets[ds_key] = vf
    return datasets

  root_vf = find_vector_file_in_dir(root_dir)
  if root_vf:
    datasets[root_dir.name] = root_vf
    return datasets

  for dirpath, dirnames, _ in os.walk(root_dir):
    d = Path(dirpath)
    if d == root_dir:
      continue
    d_vfs = [
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and "gauge" not in f.name.lower() and "point" not in f.name.lower()
    ]
    if len(d_vfs) > 1:
      for vf in sorted(d_vfs):
        ds_key = f"{d.name}_{vf.stem}" if d.name not in ["staged_shapefiles", "data", "shapes"] else vf.stem
        datasets[ds_key] = vf
      dirnames.clear()
      continue

    vf = find_vector_file_in_dir(d)
    if vf:
      datasets[d.name] = vf
      dirnames.clear()
  return datasets


def discover_datasets(
    parent_dirs: Optional[List[str]] = None,
    input_dirs: Optional[List[str]] = None,
    input_files: Optional[List[str]] = None,
    staging_cache_dir: Optional[Path] = None,
) -> Dict[str, Path]:
  """Discovers dataset names and their corresponding vector files.

  Returns a mapping of dataset_name -> vector_file_path.
  """
  dataset_map: Dict[str, Path] = {}

  # 1. Process specific input files
  if input_files:
    for f_str in input_files:
      p = Path(f_str)
      if not p.exists():
        raise FileNotFoundError(f"Input file does not exist: {p}")
      d_name = p.stem.replace("_basin_shapes", "").replace("_basins", "")
      dataset_map[d_name] = p

  # 2. Process specific input directories
  if input_dirs:
    for d_str in input_dirs:
      if d_str.startswith("gs://"):
        if staging_cache_dir is None:
          raise ValueError(
              "staging_cache_dir (--staging-dir) must be provided when input directories use gs:// URIs."
          )
        d_name = d_str.rstrip("/").split("/")[-1]
        local_dir = sync_gcs_directory(d_str, Path(staging_cache_dir) / d_name)
      else:
        local_dir = Path(d_str)

      if not local_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {local_dir}")

      found = find_all_dataset_dirs(local_dir)
      if not found:
        raise FileNotFoundError(f"No vector polygon file found in {local_dir}")
      dataset_map.update(found)

  # 3. Process parent directories (containing subdirectories for each dataset)
  if parent_dirs:
    for p_str in parent_dirs:
      if p_str.startswith("gs://"):
        if staging_cache_dir is None:
          raise ValueError(
              "staging_cache_dir (--staging-dir) must be provided when parent directories use gs:// URIs."
          )
        clean_p = p_str.rstrip("/")
        parts = [p for p in clean_p.split("/") if p and p != "gs:"]
        if len(parts) >= 2 and parts[-1] in ["shapefiles", "shapefiles-rederived", "data"]:
          parent_name = f"{parts[-2]}_{parts[-1]}"
        else:
          parent_name = parts[-1] if parts else "parent"
        local_parent = sync_gcs_directory(
            p_str, Path(staging_cache_dir) / parent_name
        )
      else:
        local_parent = Path(p_str)

      if not local_parent.is_dir():
        raise FileNotFoundError(
            f"Parent directory does not exist or is not a directory: {local_parent}"
        )

      found = find_all_dataset_dirs(local_parent)
      if not found:
        raise FileNotFoundError(
            f"No dataset vector files found under parent directory {local_parent}"
        )
      dataset_map.update(found)

  return dataset_map


def run_batch_extraction(
    dataset_map: Dict[str, Path],
    output_dir: Union[str, Path],
    gdb_path: Optional[Union[str, Path]] = None,
    era5_source: str = "",
    era5_cache_dir: Optional[Union[str, Path]] = None,
    gridded_era5_uri: Optional[str] = None,
    gcs_gdb_uri: Optional[str] = None,
    gcs_era5_climate_uri: Optional[str] = None,
    no_download: bool = False,
    staging_cache_dir: Optional[Union[str, Path]] = None,
    gcs_output_uri: Optional[str] = None,
    id_column: str = "gauge_id",
    min_overlap_threshold: float = 0.0,
    workers: int = 1,
    combine: bool = False,
    resume: bool = True,
    show_progress: bool = True,
    partition_outputs: bool = False,
) -> Dict[str, pd.DataFrame]:
  """Runs static attribute extraction across all discovered datasets."""
  if not era5_source or era5_source.lower() not in {"hybas", "gridded"}:
    raise ValueError(
        "era5_source must be explicitly specified as either 'hybas' or 'gridded'."
    )
  era5_source = era5_source.lower()
  is_gcs_output = str(output_dir).startswith("gs://")
  target_gcs_uri = str(output_dir) if is_gcs_output else gcs_output_uri

  if is_gcs_output:
    if staging_cache_dir is None:
      raise ValueError(
          "staging_cache_dir (--staging-dir) must be provided when output_dir is a gs:// URI."
      )
    out_dir = Path(staging_cache_dir) / "output_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)
  else:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

  if resume and target_gcs_uri and gcs_path_exists(target_gcs_uri):
    logger.debug(
        "Pre-syncing existing extracted files from GCS destination %s to allow resume...",
        target_gcs_uri,
    )
    sync_gcs_directory(target_gcs_uri, out_dir)

  logger.debug("Initializing StaticAttributesExtractor (era5_source=%s)...", era5_source)
  extractor = StaticAttributesExtractor(
      gdb_path=gdb_path,
      era5_source=era5_source,
      era5_cache_dir=era5_cache_dir,
      gridded_era5_uri=gridded_era5_uri,
      gcs_gdb_uri=gcs_gdb_uri,
      gcs_era5_climate_uri=gcs_era5_climate_uri,
      no_download=no_download,
  )

  extracted_dfs: Dict[str, pd.DataFrame] = {}
  total_datasets = len(dataset_map)
  start_all = time.time()

  logger.debug("Found %d datasets to process: %s", total_datasets, list(dataset_map.keys()))

  dataset_pbar = tqdm(
      total=total_datasets,
      desc="Datasets",
      unit="dataset",
      dynamic_ncols=True,
      disable=not show_progress,
  )

  for ds_name, vector_path in dataset_map.items():
    if partition_outputs:
      sub_dir = out_dir / ds_name
      sub_dir.mkdir(parents=True, exist_ok=True)
    else:
      sub_dir = out_dir

    if target_gcs_uri:
      base_gcs = target_gcs_uri.rstrip("/")
      target_ds_gcs = f"{base_gcs}/{ds_name}/" if partition_outputs else f"{base_gcs}/"
    else:
      target_ds_gcs = None

    if partition_outputs:
      parquet_file = sub_dir / f"attributes_{ds_name}.parquet"
      hydro_file = sub_dir / f"attributes_hydroatlas_{ds_name}.csv"
      caravan_file = sub_dir / f"attributes_caravan_{ds_name}.csv"
      if resume and (parquet_file.exists() or (hydro_file.exists() and caravan_file.exists())):
        if parquet_file.exists():
          df = pd.read_parquet(parquet_file)
        else:
          df = pd.read_csv(hydro_file, index_col=0).join(
              pd.read_csv(caravan_file, index_col=0), how="outer"
          )
        if not df.empty:
          extracted_dfs[ds_name] = df
          dataset_pbar.set_postfix_str(f"Skipped {ds_name} (already done)")
          dataset_pbar.update(1)
          continue
      out_file = None
    else:
      out_file = out_dir / f"attributes_caravan_{ds_name}.csv"
      if resume and out_file.exists():
        df = pd.read_csv(out_file, index_col=0)
        if not df.empty:
          extracted_dfs[ds_name] = df
          dataset_pbar.set_postfix_str(f"Skipped {ds_name} (already done)")
          dataset_pbar.update(1)
          continue

    dataset_pbar.set_postfix_str(f"Extracting {ds_name}...")
    t0 = time.time()
    df = extractor.extract_attributes_from_file(
        input_path=vector_path,
        output_csv_path=out_file,
        id_column=id_column,
        min_overlap_threshold=min_overlap_threshold,
        era5_source=era5_source,
        workers=workers,
        show_progress=show_progress,
        dataset_name=ds_name,
    )
    elapsed = time.time() - t0

    if partition_outputs:
      exported_files = export_subdataset_partitioned_files(
          df=df,
          ds_name=ds_name,
          output_sub_dir=sub_dir,
      )
      if target_ds_gcs:
        for f_path in exported_files.values():
          upload_to_gcs(f_path, target_ds_gcs)
      extracted_dfs[ds_name] = pd.read_parquet(exported_files["parquet"])
      logger.debug(
          "Successfully completed '%s' partitioned: %d basins in %.1f seconds (saved to %s)",
          ds_name,
          len(df),
          elapsed,
          sub_dir,
      )
    else:
      if target_gcs_uri and out_file is not None:
        upload_to_gcs(out_file, target_gcs_uri)
      extracted_dfs[ds_name] = df
      logger.debug(
          "Successfully completed '%s': %d basins extracted in %.1f seconds (saved to %s)",
          ds_name,
          len(df),
          elapsed,
          out_file,
      )

    dataset_pbar.set_postfix_str(f"Done {ds_name} ({len(df):,} basins, {elapsed:.1f}s)")
    dataset_pbar.update(1)

  dataset_pbar.close()

  total_elapsed = time.time() - start_all
  total_basins = sum(len(df) for df in extracted_dfs.values())
  logger.debug(
      "All datasets processed: %d total basins across %d datasets in %.1f seconds.",
      total_basins,
      len(extracted_dfs),
      total_elapsed,
  )

  if combine and extracted_dfs:
    combined_path = out_dir / "attributes_caravan_combined.csv"
    logger.debug("Combining all %d datasets into %s...", len(extracted_dfs), combined_path)
    combined_df = pd.concat(list(extracted_dfs.values()), axis=0)
    combined_df.to_csv(combined_path)
    if target_gcs_uri:
      upload_to_gcs(combined_path, target_gcs_uri)

    if partition_outputs:
      combined_parquet = out_dir / "attributes_combined.parquet"
      combined_df.to_parquet(combined_parquet)
      if target_gcs_uri:
        upload_to_gcs(combined_parquet, target_gcs_uri)
      logger.debug("Combined CSV & Parquet written: %d total rows.", len(combined_df))
    else:
      logger.debug("Combined CSV written: %d total rows.", len(combined_df))

  if show_progress:
    print(
        f"\n✓ Extracted {total_basins:,} basins across {len(extracted_dfs)} datasets "
        f"in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)."
    )
    if combine and extracted_dfs:
      if partition_outputs:
        print(f"✓ Combined files generated: {combined_path.name} & {combined_parquet.name} ({len(combined_df):,} total rows)")
      else:
        print(f"✓ Combined CSV generated: {combined_path.name} ({len(combined_df):,} total rows)")
    if target_gcs_uri:
      print(f"✓ Results stored in GCS destination: {target_gcs_uri}")
    else:
      print(f"✓ Results stored locally at: {out_dir}")

  return extracted_dfs


def parse_args(args=None):
  parser = argparse.ArgumentParser(
      description="Batch runner for Caravan static attribute extraction across multiple datasets.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--parent-dir",
      "-p",
      action="append",
      nargs="+",
      dest="parent_dirs",
      help="Parent directory containing dataset subdirectories (local path or gs://...). Can specify multiple times or provide multiple paths.",
  )
  parser.add_argument(
      "--input-dirs",
      "-d",
      nargs="+",
      default=None,
      help="List of dataset directories.",
  )
  parser.add_argument(
      "--input-files",
      "-f",
      nargs="+",
      default=None,
      help="List of specific vector polygon files (.shp, .geojson, .gpkg, .parquet).",
  )
  parser.add_argument(
      "--output-dir",
      "-o",
      required=True,
      type=str,
      help="Directory to save extracted attributes (local filesystem path or gs:// bucket URI).",
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
      help="Source for ERA5 climate metrics: 'hybas' (precalculated sub-basin statistics) or 'gridded' (daily Zarr recalculation).",
  )
  parser.add_argument(
      "--era5-cache-dir",
      default=None,
      type=str,
      help="Local directory containing continental ERA5 climate tables (required when --era5-source=hybas unless --no-download is used with --gcs-era5-climate-uri).",
  )
  parser.add_argument(
      "--gridded-era5-uri",
      default=None,
      type=str,
      help="GCS URI or local path to gridded daily ERA5 Zarr store (required when --era5-source=gridded; optional when --era5-source=hybas).",
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
      "--staging-dir",
      default=None,
      type=str,
      help="Local directory used for staging input shapefiles or output CSVs when reading from or writing to gs:// URIs.",
  )
  parser.add_argument(
      "--id-column",
      default="gauge_id",
      type=str,
      help="Column name in vector files containing the watershed or gauge ID.",
  )
  parser.add_argument(
      "--partition-outputs",
      "-P",
      action="store_true",
      default=False,
      help="Partition output attributes per subdataset directory into attributes_hydroatlas_<ds>.csv, attributes_caravan_<ds>.csv, and attributes_<ds>.parquet.",
  )
  parser.add_argument(
      "--gcs-output-uri",
      default=None,
      type=str,
      help="Optional GCS bucket URI to upload extracted files to when --output-dir is a local path.",
  )
  parser.add_argument(
      "--workers",
      "-w",
      type=int,
      default=1,
      help="Number of parallel worker processes to use.",
  )
  parser.add_argument(
      "--min-overlap-threshold",
      default=0.0,
      type=float,
      help="Minimum sub-basin intersection area threshold in km².",
  )
  parser.add_argument(
      "--clean-staging",
      action="store_true",
      help="Delete --staging-dir after batch extraction finishes.",
  )
  parser.add_argument(
      "--clean-cache",
      action="store_true",
      help="Delete --gdb-path, --era5-cache-dir, and --staging-dir after batch extraction finishes.",
  )
  parser.add_argument(
      "--combine",
      action="store_true",
      help="Also save a combined attributes_caravan_combined.csv containing all processed datasets.",
  )
  parser.add_argument(
      "--no-resume",
      action="store_false",
      dest="resume",
      default=True,
      help="Do not skip datasets that have already been extracted.",
  )
  parser.add_argument(
      "--verbose",
      "-v",
      action="store_true",
      help="Show detailed debug/info log messages.",
  )
  parser.add_argument(
      "--no-progress",
      action="store_false",
      dest="show_progress",
      default=True,
      help="Disable interactive progress bars.",
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
  setup_logging(parsed.verbose)

  staging_cache = Path(parsed.staging_dir) if parsed.staging_dir else None

  if parsed.parent_dirs:
    flat_parents = []
    for item in parsed.parent_dirs:
      if isinstance(item, list):
        flat_parents.extend(item)
      else:
        flat_parents.append(item)
    parsed.parent_dirs = flat_parents

  if not parsed.parent_dirs and not parsed.input_dirs and not parsed.input_files:
    logger.error("Must provide at least one of --parent-dir, --input-dirs, or --input-files.")
    sys.exit(1)

  if parsed.show_progress:
    print("Discovering dataset polygons...")

  dataset_map = discover_datasets(
      parent_dirs=parsed.parent_dirs,
      input_dirs=parsed.input_dirs,
      input_files=parsed.input_files,
      staging_cache_dir=staging_cache,
  )

  if not dataset_map:
    logger.error("No valid dataset vector files found matching provided paths.")
    sys.exit(1)

  if parsed.show_progress:
    print(f"Found {len(dataset_map)} datasets to process: {', '.join(sorted(dataset_map.keys()))}\n")

  run_batch_extraction(
      dataset_map=dataset_map,
      output_dir=parsed.output_dir,
      gdb_path=parsed.gdb_path,
      era5_source=parsed.era5_source,
      era5_cache_dir=parsed.era5_cache_dir,
      gridded_era5_uri=parsed.gridded_era5_uri,
      gcs_gdb_uri=parsed.gcs_gdb_uri,
      gcs_era5_climate_uri=parsed.gcs_era5_climate_uri,
      no_download=parsed.no_download,
      staging_cache_dir=staging_cache,
      gcs_output_uri=parsed.gcs_output_uri,
      id_column=parsed.id_column,
      min_overlap_threshold=parsed.min_overlap_threshold,
      workers=parsed.workers,
      combine=parsed.combine,
      resume=parsed.resume,
      show_progress=parsed.show_progress,
      partition_outputs=parsed.partition_outputs,
  )

  if parsed.clean_cache:
    if not parsed.no_download:
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
    if staging_cache and staging_cache.exists():
      shutil.rmtree(staging_cache)
  elif parsed.clean_staging and staging_cache and staging_cache.exists():
    shutil.rmtree(staging_cache)


if __name__ == "__main__":
  main()
