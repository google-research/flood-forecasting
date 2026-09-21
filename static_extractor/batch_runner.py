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
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth.*")

import pandas as pd
from tqdm.auto import tqdm

from static_extractor.config import CARAVAN_SUBDIR_MAPPING
from static_extractor.extractor import StaticAttributesExtractor

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
    vector_path: Optional[Path] = None,
) -> Dict[str, Path]:
  """Exports extracted attributes into partitioned HydroATLAS, Caravan, and Parquet files.

  Schema per subdataset directory:
    - attributes_hydroatlas_<ds>.csv (~198 HydroATLAS attributes)
    - attributes_caravan_<ds>.csv (long-term hydrologic & climate indices + gauge_lat/lon)
    - attributes_other_<ds>.csv (provider metadata, if present)
    - attributes_<ds>.parquet (single unified table merging all above tables on gauge_id)
  """
  output_sub_dir.mkdir(parents=True, exist_ok=True)
  df_work = df.copy()

  # 1. Attach gauge_lat and gauge_lon from coordinates.csv if available
  coords_file = None
  if vector_path is not None:
    candidate = vector_path.parent / "coordinates.csv"
    if candidate.exists():
      coords_file = candidate
    else:
      candidate2 = (
          vector_path.parent.parent.parent
          / "shapefiles"
          / ds_name
          / "coordinates.csv"
      )
      if candidate2.exists():
        coords_file = candidate2

  if coords_file is not None and coords_file.exists():
    coords_df = pd.read_csv(coords_file)
    if "gauge_id" in coords_df.columns:
      coords_df = coords_df.set_index("gauge_id")
    for col in ["gauge_lat", "gauge_lon"]:
      if col in coords_df.columns and col not in df_work.columns:
        df_work[col] = df_work.index.map(coords_df[col])

  # 2. Partition columns
  caravan_cols = [
      c for c in df_work.columns
      if c in CARAVAN_CLIMATE_INDICES or c in ["gauge_lat", "gauge_lon"]
  ]
  hydro_cols = [c for c in df_work.columns if c not in caravan_cols]

  # Sort HydroATLAS columns, ensuring basin_area is first
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

  # 3. Check for attributes_other_<ds>.csv
  other_path = output_sub_dir / f"attributes_other_{ds_name}.csv"
  df_other = None
  if not other_path.exists() and vector_path is not None:
    cand_other = vector_path.parent / f"attributes_other_{ds_name}.csv"
    if cand_other.exists():
      shutil.copy(cand_other, other_path)
  if other_path.exists():
    df_other = pd.read_csv(other_path, index_col=0)

  # 4. Construct unified table and save to Parquet
  df_unified = df_hydro.join(df_caravan, how="outer")
  if df_other is not None:
    df_unified = df_unified.join(df_other, how="left")

  parquet_path = output_sub_dir / f"attributes_{ds_name}.parquet"
  df_unified.to_parquet(parquet_path)

  result = {
      "hydroatlas": hydro_path,
      "caravan": caravan_path,
      "parquet": parquet_path,
  }
  if other_path.exists():
    result["other"] = other_path
  return result


def get_collection_and_subdataset(
    ds_name: str, vector_path: Optional[Path] = None
) -> Tuple[str, str]:
  """Resolves the Caravan collection and subdataset name for a given dataset."""
  key = ds_name.upper().replace("_BASIN_SHAPES", "").replace("_BASINS", "")
  if key in CARAVAN_SUBDIR_MAPPING:
    return CARAVAN_SUBDIR_MAPPING[key]

  if vector_path is not None:
    path_str = str(vector_path)
    for coll in ["caravan-original", "caravan-extensions", "google-internal"]:
      if coll in path_str:
        return coll, ds_name.lower()

  return "other", ds_name.lower()


def get_target_output_dir(
    base_output_dir: str,
    coll: str,
    ds_name: str,
    preserve_caravan_dirs: bool = False,
) -> str:
  """Builds the destination directory path for a subdataset."""
  base_clean = base_output_dir.rstrip("/")
  if not preserve_caravan_dirs:
    return f"{base_clean}/{ds_name}"

  # If base already ends with /attributes
  if base_clean.endswith("/attributes"):
    return f"{base_clean}/{ds_name}"

  # If base ends with a specific collection
  last_segment = base_clean.split("/")[-1]
  if last_segment in ["caravan-original", "caravan-extensions", "google-internal"]:
    return f"{base_clean}/attributes/{ds_name}"

  # Otherwise: <base_output>/<collection>/attributes/<subdataset>
  return f"{base_clean}/{coll}/attributes/{ds_name}"


def gcs_path_exists(gcs_uri: str) -> bool:
  """Checks if a GCS URI exists or contains any objects."""
  if shutil.which("gcloud"):
    try:
      res = subprocess.run(
          ["gcloud", "storage", "ls", gcs_uri],
          capture_output=True,
          text=True,
          check=False,
      )
      return res.returncode == 0
    except Exception:
      pass
  if shutil.which("gsutil"):
    try:
      res = subprocess.run(
          ["gsutil", "ls", gcs_uri],
          capture_output=True,
          text=True,
          check=False,
      )
      return res.returncode == 0
    except Exception:
      pass
  return False


def upload_to_gcs(local_path: Path, gcs_dest_uri: str) -> bool:
  """Uploads a local file or directory to a GCS destination."""
  gcs_dest_clean = gcs_dest_uri if gcs_dest_uri.endswith("/") else gcs_dest_uri + "/"
  target_uri = f"{gcs_dest_clean}{local_path.name}" if local_path.is_file() else gcs_dest_clean
  logger.debug("Uploading %s to %s...", local_path, target_uri)

  # 1. Try gcloud storage cp / rsync
  if shutil.which("gcloud"):
    try:
      cmd = (
          ["gcloud", "storage", "cp", str(local_path), target_uri]
          if local_path.is_file()
          else ["gcloud", "storage", "rsync", "-r", str(local_path), gcs_dest_clean]
      )
      res = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if res.returncode == 0:
        logger.debug("Successfully uploaded %s to %s via gcloud storage.", local_path.name, target_uri)
        return True
      logger.debug("gcloud storage upload failed: %s. Trying gsutil...", res.stderr)
    except Exception as e:
      logger.debug("gcloud storage upload error: %s. Trying gsutil...", e)

  # 2. Try gsutil cp / rsync
  if shutil.which("gsutil"):
    try:
      cmd = (
          ["gsutil", "cp", str(local_path), target_uri]
          if local_path.is_file()
          else ["gsutil", "-m", "rsync", "-r", str(local_path), gcs_dest_clean]
      )
      res = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if res.returncode == 0:
        logger.debug("Successfully uploaded %s to %s via gsutil.", local_path.name, target_uri)
        return True
      logger.debug("gsutil upload failed: %s", res.stderr)
    except Exception as e:
      logger.debug("gsutil error: %s", e)

  # 3. Fallback to gcsfs
  try:
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    clean_target = target_uri.replace("gs://", "")
    if local_path.is_file():
      fs.put(str(local_path), clean_target)
    else:
      fs.put(str(local_path), clean_target, recursive=True)
    logger.debug("Successfully uploaded %s to %s via gcsfs.", local_path.name, target_uri)
    return True
  except Exception as e:
    logger.error("Failed to upload %s to %s: %s", local_path, target_uri, e)
    return False


def sync_gcs_directory(gcs_uri: str, local_dest: Path) -> Path:
  """Syncs a GCS directory to a local directory using gcloud storage or gsutil."""
  local_dest.mkdir(parents=True, exist_ok=True)
  logger.debug("Syncing %s to local staging directory %s...", gcs_uri, local_dest)

  gcs_uri_clean = gcs_uri if gcs_uri.endswith("/") else gcs_uri + "/"

  # Try gcloud storage rsync first (preferred)
  try:
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "-r", gcs_uri_clean, str(local_dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
      return local_dest
    logger.debug("gcloud storage rsync failed: %s. Trying gsutil...", res.stderr)
  except Exception as e:
    logger.debug("gcloud storage rsync error: %s. Trying gsutil...", e)

  # Try gsutil rsync
  try:
    res = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_uri_clean, str(local_dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
      return local_dest
    logger.debug("gsutil rsync failed: %s. Trying gcloud storage cp...", res.stderr)
  except Exception as e:
    logger.debug("gsutil rsync error: %s", e)

  # Fallback to gcloud storage cp
  res = subprocess.run(
      ["gcloud", "storage", "cp", "-r", gcs_uri_clean, str(local_dest)],
      capture_output=True,
      text=True,
      check=False,
  )
  if res.returncode != 0:
    logger.warning("Could not sync GCS URI %s: %s", gcs_uri, res.stderr)
  return local_dest


def find_vector_file_in_dir(dataset_dir: Path) -> Optional[Path]:
  """Finds the primary watershed polygon file in a dataset directory."""
  dataset_name = dataset_dir.name.lower()

  # Check for typical Caravan naming first (e.g. camels_basin_shapes.shp)
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

  # Check any shapefiles, excluding auxiliary/gauge points if basin shapes exist
  shps = list(dataset_dir.glob("*.shp"))
  if shps:
    basin_shps = [
        s
        for s in shps
        if "gauge" not in s.name.lower() and "point" not in s.name.lower()
    ]
    return basin_shps[0] if basin_shps else shps[0]

  # Check GeoJSON, GPKG, or GeoParquet
  for ext in [".geojson", ".gpkg", ".json", ".parquet", ".geoparquet"]:
    matches = list(dataset_dir.glob(f"*{ext}"))
    if matches:
      return matches[0]

  return None


def find_all_dataset_dirs(root_dir: Path) -> Dict[str, Path]:
  """Recursively finds all dataset directories containing vector files under root_dir."""
  datasets: Dict[str, Path] = {}
  # Check if root_dir directly contains multiple vector files (e.g. continental geoparquet files)
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

  # First check if root_dir itself is a single dataset directory
  root_vf = find_vector_file_in_dir(root_dir)
  if root_vf:
    datasets[root_dir.name] = root_vf
    return datasets

  for dirpath, dirnames, _ in os.walk(root_dir):
    d = Path(dirpath)
    if d == root_dir:
      continue
    # Check if this subdirectory has multiple vector files directly
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
      # Stop recursing into subdirectories of a discovered dataset
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

  if staging_cache_dir is None:
    staging_cache_dir = (
        Path.home() / ".cache" / "googlehydrology" / "staged_shapefiles"
    )

  # 1. Process specific input files
  if input_files:
    for f_str in input_files:
      p = Path(f_str)
      if p.exists():
        d_name = p.stem.replace("_basin_shapes", "").replace("_basins", "")
        dataset_map[d_name] = p

  # 2. Process specific input directories
  if input_dirs:
    for d_str in input_dirs:
      if d_str.startswith("gs://"):
        d_name = d_str.rstrip("/").split("/")[-1]
        local_dir = sync_gcs_directory(d_str, staging_cache_dir / d_name)
      else:
        local_dir = Path(d_str)

      if local_dir.is_dir():
        found = find_all_dataset_dirs(local_dir)
        if found:
          dataset_map.update(found)
        else:
          logger.warning("No vector polygon file found in %s", local_dir)

  # 3. Process parent directories (containing subdirectories for each dataset)
  if parent_dirs:
    for p_str in parent_dirs:
      if p_str.startswith("gs://"):
        clean_p = p_str.rstrip("/")
        # If pointing to caravan-new root, automatically expand to the collections' shapefile directories
        if clean_p.endswith("caravan-new"):
          for coll in ["caravan-original", "caravan-extensions", "google-internal"]:
            coll_shapefiles = f"{clean_p}/{coll}/shapefiles"
            local_parent = sync_gcs_directory(
                coll_shapefiles, staging_cache_dir / f"{coll}_shapefiles"
            )
            found = find_all_dataset_dirs(local_parent)
            if found:
              dataset_map.update(found)
          continue

        parts = [p for p in clean_p.split("/") if p and p != "gs:"]
        if len(parts) >= 2 and parts[-1] in ["shapefiles", "shapefiles-rederived", "data"]:
          parent_name = f"{parts[-2]}_{parts[-1]}"
        else:
          parent_name = parts[-1] if parts else "parent"
        local_parent = sync_gcs_directory(
            p_str, staging_cache_dir / parent_name
        )
      else:
        clean_p = str(p_str).rstrip("/")
        if clean_p.endswith("caravan-new") and Path(clean_p).is_dir():
          for coll in ["caravan-original", "caravan-extensions", "google-internal"]:
            coll_shapefiles = Path(clean_p) / coll / "shapefiles"
            if coll_shapefiles.is_dir():
              found = find_all_dataset_dirs(coll_shapefiles)
              if found:
                dataset_map.update(found)
          continue
        local_parent = Path(p_str)

      if not local_parent.is_dir():
        logger.warning(
            "Parent directory %s does not exist or is not a directory.",
            local_parent,
        )
        continue

      found = find_all_dataset_dirs(local_parent)
      if found:
        dataset_map.update(found)
      else:
        logger.warning(
            "No dataset vector files found under parent directory %s",
            local_parent,
        )

  return dataset_map


def run_batch_extraction(
    dataset_map: Dict[str, Path],
    output_dir: Union[str, Path],
    workers: int = 4,
    era5_source: Optional[str] = None,
    gridded_era5_uri: Optional[str] = None,
    gdb_path: Optional[Union[str, Path]] = None,
    era5_cache_dir: Optional[Union[str, Path]] = None,
    staging_cache_dir: Optional[Path] = None,
    gcs_output_uri: Optional[str] = None,
    min_overlap_threshold: float = 0.0,
    combine: bool = False,
    resume: bool = True,
    show_progress: bool = True,
    partition_outputs: Optional[bool] = None,
    preserve_caravan_dirs: bool = False,
) -> Dict[str, pd.DataFrame]:
  """Runs static attribute extraction across all discovered datasets."""
  if not era5_source or era5_source.lower() not in {"hybas", "gridded"}:
    raise ValueError(
        "era5_source must be explicitly specified as either 'hybas' or 'gridded' "
        "(no default is assumed)."
    )
  era5_source = era5_source.lower()
  is_gcs_output = str(output_dir).startswith("gs://")
  target_gcs_uri = str(output_dir) if is_gcs_output else gcs_output_uri

  if preserve_caravan_dirs:
    partition_outputs = True
  elif partition_outputs is None:
    out_str = str(output_dir)
    gcs_str = str(gcs_output_uri) if gcs_output_uri else ""
    partition_outputs = ("caravan-new" in out_str) or ("caravan-new" in gcs_str)

  if is_gcs_output:
    if staging_cache_dir is not None:
      out_dir = staging_cache_dir.parent / "output_csvs"
    else:
      out_dir = Path.home() / ".cache" / "googlehydrology" / "output_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "Output configured for GCS: %s (staging locally in %s, partition_outputs=%s, preserve_caravan_dirs=%s)",
        target_gcs_uri,
        out_dir,
        partition_outputs,
        preserve_caravan_dirs,
    )
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
      era5_cache_dir=era5_cache_dir,
      era5_source=era5_source,
      gridded_era5_uri=gridded_era5_uri,
      auto_download=True,
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

  for i, (ds_name, vector_path) in enumerate(dataset_map.items(), start=1):
    coll, sub_name = get_collection_and_subdataset(ds_name, vector_path)
    if preserve_caravan_dirs:
      target_folder_local = get_target_output_dir(
          str(out_dir), coll, sub_name, preserve_caravan_dirs=True
      )
      sub_dir = Path(target_folder_local)
      sub_dir.mkdir(parents=True, exist_ok=True)
    elif partition_outputs:
      sub_dir = out_dir / ds_name
      sub_dir.mkdir(parents=True, exist_ok=True)
    else:
      sub_dir = out_dir

    if target_gcs_uri:
      target_ds_gcs = get_target_output_dir(
          target_gcs_uri, coll, sub_name, preserve_caravan_dirs=preserve_caravan_dirs
      )
      if not target_ds_gcs.endswith("/"):
        target_ds_gcs += "/"
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
    try:
      df = extractor.extract_attributes_from_file(
          input_path=vector_path,
          output_csv_path=out_file if not partition_outputs else None,
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
            vector_path=vector_path,
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
        if target_gcs_uri:
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
    finally:
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

  if target_gcs_uri:
    logger.debug(
        "All extracted datasets uploaded to GCS destination: %s",
        target_gcs_uri,
    )

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
      help="Parent directory containing dataset subdirectories (e.g. /path/to/caravan/ or gs://...). Can specify multiple times or provide multiple paths.",
  )
  parser.add_argument(
      "--input-dirs",
      "-d",
      nargs="+",
      default=None,
      help="List of dataset directories (e.g., path/to/camels path/to/hysets).",
  )
  parser.add_argument(
      "--input-files",
      "-f",
      nargs="+",
      default=None,
      help="List of specific vector polygon files (.shp, .geojson, .gpkg).",
  )
  parser.add_argument(
      "--output-dir",
      "-o",
      required=True,
      type=str,
      help="Directory to save extracted attributes. Can be a local filesystem path (e.g. /data/attributes/) or a GCS bucket URI (e.g. gs://my-bucket/attributes/).",
  )
  parser.add_argument(
      "--partition-outputs",
      "-P",
      action=argparse.BooleanOptionalAction,
      default=None,
      help="Partition output attributes per subdataset directory into attributes_hydroatlas_<ds>.csv, attributes_caravan_<ds>.csv, and attributes_<ds>.parquet.",
  )
  parser.add_argument(
      "--preserve-caravan-dirs",
      action="store_true",
      help="Partition static attributes by Caravan collection and subdataset into <collection>/attributes/<subdataset>/ per the storage layout.",
  )
  parser.add_argument(
      "--gcs-output-uri",
      default=None,
      type=str,
      help="Optional GCS bucket URI to upload extracted CSVs to when --output-dir is a local path.",
  )
  parser.add_argument(
      "--workers",
      "-w",
      type=int,
      default=1,
      help="Number of parallel worker processes to use.",
  )
  parser.add_argument(
      "--era5-source",
      choices=["hybas", "gridded"],
      required=True,
      help="Source for ERA5 climate metrics (required): 'hybas' (fast precalculated) or 'gridded' (on-the-fly Zarr recalculation).",
  )
  parser.add_argument(
      "--gridded-era5-uri",
      default=None,
      type=str,
      help="GCS URI or path to gridded daily ERA5 Zarr store.",
  )
  parser.add_argument(
      "--gdb-path",
      "-g",
      default=None,
      type=str,
      help="Optional path to local BasinATLAS_v10.gdb.",
  )
  parser.add_argument(
      "--era5-cache-dir",
      default=None,
      type=str,
      help="Optional directory for ERA5 climate tables.",
  )
  parser.add_argument(
      "--min-overlap-threshold",
      default=0.0,
      type=float,
      help="Minimum sub-basin intersection area threshold in km².",
  )
  parser.add_argument(
      "--cache-dir",
      default=None,
      type=str,
      help="Base directory for runtime cache (defaults to ~/.cache/googlehydrology).",
  )
  parser.add_argument(
      "--no-download",
      action="store_true",
      help="Disable automatic GCS downloads. Requires local files to be present.",
  )
  parser.add_argument(
      "--clean-staging",
      action="store_true",
      help="Automatically clean up staged shapefiles after batch extraction finishes.",
  )
  parser.add_argument(
      "--clean-cache",
      action="store_true",
      help="Automatically clean up the entire local cache directory (~/.cache/googlehydrology) after extraction finishes.",
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
      help="Show detailed debug/info log messages (disabled by default for clean progress bar output).",
  )
  parser.add_argument(
      "--no-progress",
      action="store_false",
      dest="show_progress",
      default=True,
      help="Disable interactive progress bars.",
  )
  return parser.parse_args(args)


def main(args=None):
  parsed = parse_args(args)
  setup_logging(parsed.verbose)
  cache_root = Path(parsed.cache_dir) if parsed.cache_dir else Path.home() / ".cache" / "googlehydrology"
  staging_cache = cache_root / "staged_shapefiles"
  gdb_path = parsed.gdb_path or (cache_root / "hydroatlas" / "BasinATLAS_v10.gdb")
  era5_cache_dir = parsed.era5_cache_dir or (cache_root / "era5_climate")

  # Flatten parent_dirs if multiple paths were passed or repeated
  if parsed.parent_dirs:
    flat_parents = []
    for item in parsed.parent_dirs:
      if isinstance(item, list):
        flat_parents.extend(item)
      else:
        flat_parents.append(item)
    parsed.parent_dirs = flat_parents

  try:
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
        workers=parsed.workers,
        era5_source=parsed.era5_source,
        gridded_era5_uri=parsed.gridded_era5_uri,
        gdb_path=str(gdb_path),
        era5_cache_dir=str(era5_cache_dir),
        staging_cache_dir=staging_cache,
        gcs_output_uri=parsed.gcs_output_uri,
        min_overlap_threshold=parsed.min_overlap_threshold,
        combine=parsed.combine,
        resume=parsed.resume,
        show_progress=parsed.show_progress,
        partition_outputs=parsed.partition_outputs,
        preserve_caravan_dirs=parsed.preserve_caravan_dirs,
    )
  finally:
    if parsed.clean_cache and cache_root.exists():
      logger.debug("Cleaning up cache root directory %s...", cache_root)
      shutil.rmtree(cache_root, ignore_errors=True)
    elif parsed.clean_staging and staging_cache.exists():
      logger.debug("Cleaning up staged shapefiles directory %s...", staging_cache)
      shutil.rmtree(staging_cache, ignore_errors=True)



if __name__ == "__main__":
  main()
