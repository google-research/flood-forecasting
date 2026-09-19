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

"""Automated global benchmarking suite for Caravan static attribute extraction against published reference data."""

from __future__ import annotations

import os

# Configure environment before NumPy / C-extensions initialize for fork and thread safety
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import shapely.wkt
from tqdm.auto import tqdm

from static_extractor.config import (
    ATTRIBUTE_DEFINITIONS,
    GCS_BENCHMARK_URI,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
    get_default_era5_cache_dir,
    get_default_gdb_path,
)
from static_extractor.extractor import (
    StaticAttributesExtractor,
    _get_worker_extractor,
)

logger = logging.getLogger("static_extractor.benchmark")

DEFAULT_BENCHMARK_PATH = (
    Path(__file__).parent / "data" / "benchmark_basins_500.parquet"
)

THEMATIC_DOMAINS = [
    "Topography",
    "Climate (HydroATLAS)",
    "Caravan ERA5 Climate",
    "Hydrology",
    "Soils & Geology",
    "Land Cover",
    "Anthropogenic",
]


def get_attribute_category(attr_name: str) -> str:
  """Categorizes an attribute name into one of 7 standardized thematic domains."""
  if attr_name in [
      "p_mean",
      "pet_mean",
      "pet_mean_ERA5_LAND",
      "pet_mean_FAO_PM",
      "aridity",
      "aridity_ERA5_LAND",
      "aridity_FAO_PM",
      "frac_snow",
      "moisture_index",
      "moisture_index_ERA5_LAND",
      "moisture_index_FAO_PM",
      "seasonality",
      "seasonality_ERA5_LAND",
      "seasonality_FAO_PM",
      "high_prec_freq",
      "high_prec_dur",
      "low_prec_freq",
      "low_prec_dur",
  ]:
    return "Caravan ERA5 Climate"

  if attr_name.startswith(("ele_", "slp_", "sgr_", "basin_area", "area")):
    return "Topography"
  if attr_name.startswith(
      ("tmp_", "pre_", "pet_", "aet_", "ari_", "cmi_", "snw_", "crf_")
  ):
    return "Climate (HydroATLAS)"
  if attr_name.startswith(
      ("run_", "lka_", "dis_", "inu_", "rev_", "ria_", "riv_", "dor_", "fmh_", "fec_", "wet_", "lkv_")
  ):
    return "Hydrology"
  if attr_name.startswith(
      ("cly_", "slt_", "snd_", "soc_", "swc_", "gwt_", "kar_", "ero_", "lit_")
  ):
    return "Soils & Geology"
  if attr_name.startswith(
      ("for_", "crp_", "pst_", "ire_", "urb_", "gla_", "prm_", "pac_", "glc_", "pnv_", "tbi_", "clz_", "cls_", "tec_")
  ):
    return "Land Cover"
  if attr_name.startswith(
      ("ppd_", "nli_", "rdd_", "hft_", "gdp_", "hdi_", "gad_", "pop_")
  ):
    return "Anthropogenic"

  return "Other"


def compute_continuous_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
  """Computes statistical validation metrics between reference and extracted continuous attributes."""
  mask = ~(np.isnan(y_true) | np.isnan(y_pred))
  y_t = y_true[mask]
  y_p = y_pred[mask]
  n = len(y_t)
  if n < 2:
    return {
        "n": n,
        "pearson_r": np.nan,
        "spearman_rho": np.nan,
        "r2": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "max_abs_error": np.nan,
        "med_rel_error_pct": np.nan,
        "max_rel_error_pct": np.nan,
    }

  std_t = float(np.std(y_t))
  std_p = float(np.std(y_p))
  if std_t < 1e-9 or std_p < 1e-9:
    pearson_r_val = np.nan
    spearman_rho_val = np.nan
  else:
    pearson_r_val, _ = pearsonr(y_t, y_p)
    spearman_rho_val, _ = spearmanr(y_t, y_p)

  diff = y_p - y_t
  abs_diff = np.abs(diff)
  mae = float(np.mean(abs_diff))
  rmse = float(np.sqrt(np.mean(diff**2)))
  max_abs_error = float(np.max(abs_diff))

  ss_res = float(np.sum(diff**2))
  ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
  if ss_tot > 1e-9:
    r2_val = 1.0 - (ss_res / ss_tot)
  else:
    r2_val = np.nan

  non_zero = np.abs(y_t) > 1e-5
  if np.any(non_zero):
    rel_errors = (abs_diff[non_zero] / np.abs(y_t[non_zero])) * 100.0
    med_rel_err = float(np.median(rel_errors))
    max_rel_err = float(np.max(rel_errors))
  else:
    med_rel_err = np.nan
    max_rel_err = np.nan

  return {
      "n": n,
      "pearson_r": round(float(pearson_r_val), 5),
      "spearman_rho": round(float(spearman_rho_val), 5),
      "r2": round(float(r2_val), 5),
      "mae": round(mae, 4),
      "rmse": round(rmse, 4),
      "max_abs_error": round(max_abs_error, 4),
      "med_rel_error_pct": round(med_rel_err, 3),
      "max_rel_error_pct": round(max_rel_err, 3),
  }


def compute_categorical_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
  """Computes classification accuracy for discrete majority attributes."""
  mask = ~(np.isnan(y_true) | np.isnan(y_pred))
  y_t = y_true[mask].astype(int)
  y_p = y_pred[mask].astype(int)
  n = len(y_t)
  if n == 0:
    return {"n": 0, "accuracy_pct": np.nan, "classes_count": 0}
  acc = float(np.mean(y_t == y_p) * 100.0)
  num_classes = len(np.unique(np.concatenate([y_t, y_p])))
  return {
      "n": n,
      "accuracy_pct": round(acc, 2),
      "classes_count": num_classes,
  }


def _worker_evaluate_basin(args: tuple) -> Dict[str, Any]:
  """ProcessPool worker evaluating a single basin geometry."""
  row_dict, gdb_path, era5_cache_dir, era5_source = args
  gauge_id = row_dict["gauge_id"]
  geom_wkt = row_dict["geometry_wkt"]
  dataset = row_dict.get("dataset", "unknown")
  size_tier = row_dict.get("size_tier", "unknown")
  country = row_dict.get("country", "unknown")
  ref_area_km2 = float(row_dict["ref_area_km2"])

  t0 = time.time()
  try:
    ext = _get_worker_extractor(
        gdb_path=gdb_path,
        era5_cache_dir=era5_cache_dir,
        gridded_era5_uri=None,
    )
    geom = shapely.wkt.loads(geom_wkt)
    res = ext.extract_attributes_for_polygon(
        geom,
        catchment_id=gauge_id,
        era5_source=era5_source,
    )
    elapsed = time.time() - t0

    extracted_attrs = res.get("caravan_attributes", {})
    calc_area = float(
        extracted_attrs.get("basin_area", res.get("total_area_km2", np.nan))
    )
    subbasins_count = int(res.get("intersected_subbasins_count", 0))

    area_bias_pct = (
        float((calc_area - ref_area_km2) / ref_area_km2 * 100.0)
        if ref_area_km2 > 0
        else np.nan
    )

    return {
        "gauge_id": gauge_id,
        "dataset": dataset,
        "size_tier": size_tier,
        "country": country,
        "ref_area_km2": ref_area_km2,
        "calc_area_km2": round(calc_area, 2),
        "area_bias_pct": round(area_bias_pct, 2),
        "abs_area_err_pct": round(abs(area_bias_pct), 2),
        "subbasins_count": subbasins_count,
        "elapsed_sec": round(elapsed, 3),
        "extracted_attrs": extracted_attrs,
        "status": "SUCCESS",
    }
  except Exception as e:
    elapsed = time.time() - t0
    logger.warning("Error evaluating basin %s: %s", gauge_id, e)
    return {
        "gauge_id": gauge_id,
        "dataset": dataset,
        "size_tier": size_tier,
        "country": country,
        "ref_area_km2": ref_area_km2,
        "calc_area_km2": np.nan,
        "area_bias_pct": np.nan,
        "abs_area_err_pct": np.nan,
        "subbasins_count": 0,
        "elapsed_sec": round(elapsed, 3),
        "extracted_attrs": {},
        "status": f"ERROR: {e}",
    }


def print_table(headers: List[str], rows: List[List[Any]], title: str):
  """Prints a formatted ASCII summary table."""
  print(f"\n--- {title} ---")
  col_widths = [len(h) for h in headers]
  for row in rows:
    for i, val in enumerate(row):
      col_widths[i] = max(col_widths[i], len(str(val)))

  fmt_header = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
  separator = "-" * len(fmt_header)
  print(fmt_header)
  print(separator)

  for row in rows:
    row_strs = []
    for i, val in enumerate(row):
      # Right-align numeric columns
      if isinstance(val, (int, float)) or (isinstance(val, str) and (val.endswith("%") or val.endswith("s"))):
        row_strs.append(f"{str(val):>{col_widths[i]}}")
      else:
        row_strs.append(f"{str(val):<{col_widths[i]}}")
    print(" | ".join(row_strs))


def run_benchmark(
    dataset_path: Optional[Path] = None,
    samples: Optional[int] = None,
    regions: Optional[List[str]] = None,
    size_tiers: Optional[List[str]] = None,
    workers: int = 8,
    gdb_path: Optional[str] = None,
    era5_cache_dir: Optional[str] = None,
    era5_source: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Executes the comprehensive Caravan static attributes extraction benchmark."""
  if not era5_source or era5_source.lower() not in {"hybas", "gridded"}:
    raise ValueError(
        "era5_source must be explicitly specified as either 'hybas' or 'gridded' "
        "(no default is assumed)."
    )
  era5_source = era5_source.lower()
  ds_path = (
      Path(dataset_path) if dataset_path else DEFAULT_BENCHMARK_PATH
  ).resolve()

  if not ds_path.exists():
    print(f"Benchmark dataset not found locally at {ds_path}. Downloading from GCS...")
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_src = GCS_BENCHMARK_URI
    try:
      import shutil
      import subprocess
      if shutil.which("gcloud"):
        subprocess.run(["gcloud", "storage", "cp", gcs_src, str(ds_path)], check=True)
      else:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket("open-multimet")
        blob_path = gcs_src.replace("gs://open-multimet/", "")
        blob = bucket.blob(blob_path)
        blob.download_to_filename(str(ds_path))
      print(f"Successfully downloaded benchmark dataset to {ds_path}")
    except Exception as e:
      raise FileNotFoundError(
          f"Benchmark dataset not found at {ds_path} and failed to download from GCS: {e}. "
          "Please generate it using 'python scripts/build_static_benchmark_dataset.py'."
      )

  df = pd.read_parquet(ds_path)
  print(f"Loaded benchmark dataset: {len(df)} reference basins from {ds_path.name}")

  # Apply dataset and size tier filters
  if regions:
    df = df[df["dataset"].isin(regions)]
  if size_tiers:
    df = df[df["size_tier"].isin(size_tiers)]

  if samples and samples < len(df):
    # Balanced stratified subsampling across dataset and size tier
    sampled_dfs = []
    for _, grp in df.groupby(["dataset", "size_tier"]):
      n = max(1, int(len(grp) * samples / len(df)))
      sampled_dfs.append(grp.sample(n=min(len(grp), n), random_state=42))
    df = pd.concat(sampled_dfs, ignore_index=True)
    if len(df) > samples:
      df = df.sample(n=samples, random_state=42)

  print(
      f"Benchmarking {len(df)} basins across {df['dataset'].nunique()} datasets "
      f"using {workers} workers (era5_source='{era5_source}')..."
  )

  # Pre-verify BasinATLAS GDB and pre-cache continental ERA5 tables
  effective_gdb = Path(gdb_path) if gdb_path else get_default_gdb_path()
  effective_cache_dir = (
      Path(era5_cache_dir) if era5_cache_dir else get_default_era5_cache_dir()
  )

  print(f"Using BasinATLAS GDB: {effective_gdb}")
  print(f"Using ERA5 climate cache: {effective_cache_dir}")

  # Pre-initialize master extractor to ensure GDB and continent tables are present
  print("Pre-initializing master extractor and checking caches...")
  master_ext = StaticAttributesExtractor(
      gdb_path=effective_gdb,
      era5_cache_dir=effective_cache_dir,
      era5_source=era5_source,
      auto_download=True,
  )

  # Prepare task payloads
  basin_records = df.to_dict(orient="records")
  worker_args = [
      (
          rec,
          str(effective_gdb),
          str(effective_cache_dir),
          era5_source,
      )
      for rec in basin_records
  ]

  t_start = time.time()
  results: List[Dict[str, Any]] = []

  ctx = mp.get_context("spawn")
  with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
    futures = {
        executor.submit(_worker_evaluate_basin, arg): arg[0]["gauge_id"]
        for arg in worker_args
    }

    for fut in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Evaluating basins",
        unit="basin",
        dynamic_ncols=True,
    ):
      res = fut.result()
      results.append(res)

  total_wall_time = time.time() - t_start
  basin_metrics_df = pd.DataFrame(
      [
          {k: v for k, v in r.items() if k != "extracted_attrs"}
          for r in results
      ]
  )

  # Construct extraction results map: {gauge_id: {attr: val}}
  extracted_map = {r["gauge_id"]: r.get("extracted_attrs", {}) for r in results}

  # Identify reference attribute columns
  ref_cols = [c for c in df.columns if c.startswith("ref_") and c != "ref_area_km2"]
  attr_names = [c.replace("ref_", "") for c in ref_cols]

  # Compute per-attribute statistical validation metrics
  attr_rows = []
  for attr in attr_names:
    ref_col = f"ref_{attr}"
    y_true_vals = []
    y_pred_vals = []

    for _, row in df.iterrows():
      gid = row["gauge_id"]
      if gid in extracted_map and row[ref_col] is not None:
        y_true_vals.append(row[ref_col])
        pred_val = extracted_map[gid].get(attr, np.nan)
        y_pred_vals.append(pred_val)

    y_t = np.array(y_true_vals, dtype=float)
    y_p = np.array(y_pred_vals, dtype=float)
    cat = get_attribute_category(attr)
    is_majority = attr in MAJORITY_PROPERTIES

    if is_majority:
      cat_res = compute_categorical_metrics(y_t, y_p)
      attr_rows.append({
          "attribute": attr,
          "category": cat,
          "type": "categorical",
          "n": cat_res["n"],
          "pearson_r": np.nan,
          "spearman_rho": np.nan,
          "r2": np.nan,
          "mae": np.nan,
          "rmse": np.nan,
          "max_abs_error": np.nan,
          "med_rel_error_pct": np.nan,
          "max_rel_error_pct": np.nan,
          "accuracy_pct": cat_res["accuracy_pct"],
          "classes_count": cat_res["classes_count"],
      })
    else:
      cont_res = compute_continuous_metrics(y_t, y_p)
      attr_rows.append({
          "attribute": attr,
          "category": cat,
          "type": "continuous",
          "n": cont_res["n"],
          "pearson_r": cont_res["pearson_r"],
          "spearman_rho": cont_res["spearman_rho"],
          "r2": cont_res["r2"],
          "mae": cont_res["mae"],
          "rmse": cont_res["rmse"],
          "max_abs_error": cont_res["max_abs_error"],
          "med_rel_error_pct": cont_res["med_rel_error_pct"],
          "max_rel_error_pct": cont_res["max_rel_error_pct"],
          "accuracy_pct": np.nan,
          "classes_count": np.nan,
      })

  attr_metrics_df = pd.DataFrame(attr_rows)

  # Calculate per-basin mean relative error and categorical accuracy
  basin_mean_errs = []
  basin_max_errs = []
  basin_worst_attrs = []
  basin_cat_accs = []
  cont_attr_names = [a for a in attr_names if a not in MAJORITY_PROPERTIES]
  cat_attr_names = [a for a in attr_names if a in MAJORITY_PROPERTIES]

  for r in results:
    gid = r["gauge_id"]
    ext_dict = r.get("extracted_attrs", {})
    row_ref = df[df["gauge_id"] == gid].iloc[0]

    # Continuous relative errors
    rel_errs = []
    max_err_val = -1.0
    max_err_attr = None
    for a in cont_attr_names:
      ref_v = float(row_ref[f"ref_{a}"])
      pred_v = float(ext_dict.get(a, np.nan))
      if not np.isnan(ref_v) and not np.isnan(pred_v) and abs(ref_v) > 1e-5:
        err = abs(pred_v - ref_v) / abs(ref_v) * 100.0
        rel_errs.append(err)
        if err > max_err_val:
          max_err_val = err
          max_err_attr = a

    basin_mean_errs.append(
        round(float(np.median(rel_errs)), 2) if rel_errs else np.nan
    )
    basin_max_errs.append(
        round(float(max_err_val), 2) if max_err_val >= 0 else np.nan
    )
    basin_worst_attrs.append(max_err_attr or "none")

    # Categorical matches
    matches = 0
    total_cat = 0
    for a in cat_attr_names:
      ref_v = row_ref[f"ref_{a}"]
      pred_v = ext_dict.get(a, np.nan)
      if not pd.isna(ref_v) and not pd.isna(pred_v):
        total_cat += 1
        if int(ref_v) == int(pred_v):
          matches += 1
    basin_cat_accs.append(
        round(float(matches / total_cat * 100.0), 1) if total_cat > 0 else np.nan
    )

  basin_metrics_df["median_attr_rel_err_pct"] = basin_mean_errs
  basin_metrics_df["max_attr_rel_err_pct"] = basin_max_errs
  basin_metrics_df["worst_attribute"] = basin_worst_attrs
  basin_metrics_df["categorical_acc_pct"] = basin_cat_accs

  # Summarize Overall Benchmark Performance
  successful = int((basin_metrics_df["status"] == "SUCCESS").sum())
  total_basins = len(basin_metrics_df)
  time_per_basin = total_wall_time / max(1, total_basins)

  cont_metrics = attr_metrics_df[attr_metrics_df["type"] == "continuous"]
  cat_metrics = attr_metrics_df[attr_metrics_df["type"] == "categorical"]

  mean_r = float(cont_metrics["pearson_r"].mean())
  median_r = float(cont_metrics["pearson_r"].median())
  pct_r_99 = float((cont_metrics["pearson_r"] >= 0.99).mean() * 100.0)
  pct_r_95 = float((cont_metrics["pearson_r"] >= 0.95).mean() * 100.0)
  pct_r_90 = float((cont_metrics["pearson_r"] >= 0.90).mean() * 100.0)
  mean_cat_acc = float(cat_metrics["accuracy_pct"].mean())

  med_area_err = float(basin_metrics_df["abs_area_err_pct"].median())
  max_area_err = float(basin_metrics_df["abs_area_err_pct"].max())
  worst_area_basin = str(
      basin_metrics_df.loc[basin_metrics_df["abs_area_err_pct"].idxmax()][
          "gauge_id"
      ]
  )

  med_attr_err = float(cont_metrics["med_rel_error_pct"].median())
  max_attr_err = float(cont_metrics["max_rel_error_pct"].max())
  worst_attr_name = str(
      cont_metrics.loc[cont_metrics["max_rel_error_pct"].idxmax()]["attribute"]
  )

  max_abs_err_val = float(cont_metrics["max_abs_error"].max())
  worst_abs_attr_name = str(
      cont_metrics.loc[cont_metrics["max_abs_error"].idxmax()]["attribute"]
  )

  print("\n" + "=" * 80)
  print("GLOBAL CARAVAN STATIC ATTRIBUTE EXTRACTION BENCHMARK RESULTS")
  print("=" * 80)
  print(f"Total Basins Evaluated      : {total_basins}")
  print(f"Total Wall-Clock Time        : {total_wall_time:.2f}s ({time_per_basin:.3f}s / basin)")
  success_pct = (100.0 * successful / total_basins) if total_basins else 0.0
  print(f"Successful Extractions       : {successful} / {total_basins} ({success_pct:.1f}%)")
  print(f"Total Attributes Checked     : {len(attr_metrics_df)} (196 HydroATLAS + 14 Caravan ERA5)")
  print(f"Continuous Attributes Mean r : {mean_r:.4f}")
  print(f"Continuous Attributes Med r  : {median_r:.5f}")
  print(f"Attributes with r >= 0.99    : {pct_r_99:.1f}% ({int((cont_metrics['pearson_r'] >= 0.99).sum())} / {len(cont_metrics)})")
  print(f"Attributes with r >= 0.95    : {pct_r_95:.1f}%")
  print(f"Attributes with r >= 0.90    : {pct_r_90:.1f}%")
  print(f"Categorical Majority Acc     : {mean_cat_acc:.2f}%")
  print(f"Median Basin Area Error      : {med_area_err:.2f}%")
  print(f"Maximum Basin Area Error     : {max_area_err:.2f}% (Basin: {worst_area_basin})")
  print(f"Median Attribute Rel Error   : {med_attr_err:.2f}%")
  print(f"Maximum Attribute Rel Error  : {max_attr_err:.2f}% (Attribute: {worst_attr_name})")
  print(f"Maximum Absolute Error       : {max_abs_err_val:.4f} (Attribute: {worst_abs_attr_name})")

  # 1. Performance by Thematic Domain Table
  cat_table_headers = [
      "Thematic Domain", "Attributes", "Mean r", "Median r", "r >= 0.99 %", "Med Rel Err %", "Max Rel Err %", "Max Abs Err"
  ]
  cat_table_rows = []
  for dom in THEMATIC_DOMAINS:
    sub = cont_metrics[cont_metrics["category"] == dom]
    if len(sub) > 0:
      dom_mean_r = f"{sub['pearson_r'].mean():.4f}"
      dom_med_r = f"{sub['pearson_r'].median():.5f}"
      dom_pct_99 = f"{(sub['pearson_r'] >= 0.99).mean() * 100:.1f}%"
      dom_med_err = f"{sub['med_rel_error_pct'].median():.2f}%"
      dom_max_err = f"{sub['max_rel_error_pct'].max():.2f}%"
      dom_max_abs = f"{sub['max_abs_error'].max():.2f}"
    else:
      dom_mean_r = "N/A"
      dom_med_r = "N/A"
      dom_pct_99 = "N/A"
      dom_med_err = "N/A"
      dom_max_err = "N/A"
      dom_max_abs = "N/A"
    cat_table_rows.append([
        dom,
        len(attr_metrics_df[attr_metrics_df["category"] == dom]),
        dom_mean_r,
        dom_med_r,
        dom_pct_99,
        dom_med_err,
        dom_max_err,
        dom_max_abs,
    ])
  print_table(cat_table_headers, cat_table_rows, "PERFORMANCE BY THEMATIC DOMAIN")

  # 2. Performance by Dataset / Region Table
  ds_table_headers = [
      "Dataset", "Basins", "Med Area Err %", "Max Area Err %", "Med Attr Err %", "Max Attr Err %", "Cat Acc %", "Mean Time"
  ]
  ds_table_rows = []
  for ds, grp in basin_metrics_df.groupby("dataset"):
    n = len(grp)
    med_area = f"{grp['abs_area_err_pct'].median():.2f}%"
    max_area = f"{grp['abs_area_err_pct'].max():.2f}%"
    med_attr_err_str = f"{grp['median_attr_rel_err_pct'].median():.2f}%"
    max_attr_err_str = f"{grp['max_attr_rel_err_pct'].max():.2f}%"
    cat_acc = f"{grp['categorical_acc_pct'].mean():.1f}%"
    mean_t = f"{grp['elapsed_sec'].mean():.3f}s"
    ds_table_rows.append([str(ds), n, med_area, max_area, med_attr_err_str, max_attr_err_str, cat_acc, mean_t])
  print_table(ds_table_headers, ds_table_rows, "PERFORMANCE BY DATASET / REGION")

  # 3. Performance by Size Tier Table
  tier_table_headers = [
      "Size Tier", "Basins", "Med Area Err %", "Max Area Err %", "Med Attr Err %", "Max Attr Err %", "Cat Acc %", "Mean Time"
  ]
  tier_table_rows = []
  for tier, grp in basin_metrics_df.groupby("size_tier"):
    n = len(grp)
    med_area = f"{grp['abs_area_err_pct'].median():.2f}%"
    max_area = f"{grp['abs_area_err_pct'].max():.2f}%"
    med_attr_err_str = f"{grp['median_attr_rel_err_pct'].median():.2f}%"
    max_attr_err_str = f"{grp['max_attr_rel_err_pct'].max():.2f}%"
    cat_acc = f"{grp['categorical_acc_pct'].mean():.1f}%"
    mean_t = f"{grp['elapsed_sec'].mean():.3f}s"
    tier_table_rows.append([str(tier), n, med_area, max_area, med_attr_err_str, max_attr_err_str, cat_acc, mean_t])
  print_table(tier_table_headers, tier_table_rows, "PERFORMANCE BY BASIN SIZE TIER")

  # 4. Discrete Majority Classification Accuracy Table
  maj_table_headers = ["Majority Attribute", "Domain", "Classes", "Exact Match Accuracy %"]
  maj_table_rows = []
  for _, row in cat_metrics.iterrows():
    maj_table_rows.append([
        row["attribute"],
        row["category"],
        int(row["classes_count"]),
        f"{row['accuracy_pct']:.2f}%",
    ])
  print_table(maj_table_headers, maj_table_rows, "DISCRETE CATEGORICAL CLASSIFICATION ACCURACY")

  # 5. Top 10 Attributes by Maximum Relative Error
  max_err_headers = [
      "Attribute", "Thematic Domain", "Pearson r", "Med Rel Err %", "Max Rel Err %", "Max Abs Err", "MAE"
  ]
  top_max_err = cont_metrics.sort_values(by="max_rel_error_pct", ascending=False).head(10)
  max_err_rows = []
  for _, r in top_max_err.iterrows():
    max_err_rows.append([
        r["attribute"],
        r["category"],
        f"{r['pearson_r']:.4f}",
        f"{r['med_rel_error_pct']:.2f}%",
        f"{r['max_rel_error_pct']:.2f}%",
        f"{r['max_abs_error']:.4f}",
        f"{r['mae']:.4f}",
    ])
  print_table(max_err_headers, max_err_rows, "TOP 10 ATTRIBUTES BY MAXIMUM RELATIVE ERROR")
  print("=" * 80)

  # Output Reports and CSVs
  out_dir = Path(output_dir).resolve() if output_dir else Path.cwd()
  out_dir.mkdir(parents=True, exist_ok=True)

  attr_csv_path = out_dir / "benchmark_attribute_metrics.csv"
  attr_metrics_df.to_csv(attr_csv_path, index=False)
  print(f"\nSaved attribute-level metrics to: {attr_csv_path}")

  basin_csv_path = out_dir / "benchmark_basin_metrics.csv"
  basin_metrics_df.to_csv(basin_csv_path, index=False)
  print(f"Saved basin-level metrics to: {basin_csv_path}")

  # Generate markdown report
  report_md_path = out_dir / "benchmark_report.md"
  _generate_markdown_report(
      report_path=report_md_path,
      total_basins=total_basins,
      successful=successful,
      total_wall_time=total_wall_time,
      time_per_basin=time_per_basin,
      mean_r=mean_r,
      median_r=median_r,
      pct_r_99=pct_r_99,
      pct_r_95=pct_r_95,
      pct_r_90=pct_r_90,
      mean_cat_acc=mean_cat_acc,
      med_area_err=med_area_err,
      max_area_err=max_area_err,
      worst_area_basin=worst_area_basin,
      med_attr_err=med_attr_err,
      max_attr_err=max_attr_err,
      worst_attr_name=worst_attr_name,
      max_abs_err_val=max_abs_err_val,
      worst_abs_attr_name=worst_abs_attr_name,
      cat_table_headers=cat_table_headers,
      cat_table_rows=cat_table_rows,
      ds_table_headers=ds_table_headers,
      ds_table_rows=ds_table_rows,
      tier_table_headers=tier_table_headers,
      tier_table_rows=tier_table_rows,
      maj_table_headers=maj_table_headers,
      maj_table_rows=maj_table_rows,
      max_err_headers=max_err_headers,
      max_err_rows=max_err_rows,
      cont_metrics=cont_metrics,
  )
  print(f"Generated comprehensive Markdown report: {report_md_path}")

  return attr_metrics_df, basin_metrics_df


def _generate_markdown_report(
    report_path: Path,
    total_basins: int,
    successful: int,
    total_wall_time: float,
    time_per_basin: float,
    mean_r: float,
    median_r: float,
    pct_r_99: float,
    pct_r_95: float,
    pct_r_90: float,
    mean_cat_acc: float,
    med_area_err: float,
    max_area_err: float,
    worst_area_basin: str,
    med_attr_err: float,
    max_attr_err: float,
    worst_attr_name: str,
    max_abs_err_val: float,
    worst_abs_attr_name: str,
    cat_table_headers: List[str],
    cat_table_rows: List[List[Any]],
    ds_table_headers: List[str],
    ds_table_rows: List[List[Any]],
    tier_table_headers: List[str],
    tier_table_rows: List[List[Any]],
    maj_table_headers: List[str],
    maj_table_rows: List[List[Any]],
    max_err_headers: List[str],
    max_err_rows: List[List[Any]],
    cont_metrics: pd.DataFrame,
):
  """Writes formatted GitHub markdown report."""
  top_10 = cont_metrics.sort_values(by="pearson_r", ascending=False).head(10)
  bottom_10 = cont_metrics.sort_values(by="pearson_r", ascending=True).head(10)

  lines = [
      "# Caravan Static Attributes Extractor: Global Benchmark Report",
      "",
      "> **Validation Benchmark** against published Caravan datasets",
      "> covering 196 HydroATLAS v1.0 Level 12 physiographic/hydro-environmental attributes and 14 ERA5 climate metrics.",
      "",
      "## 1. Executive Summary",
      "",
      f"- **Total Basins Evaluated**: {total_basins}",
      f"- **Extraction Success Rate**: {successful} / {total_basins} ({(100.0 * successful / total_basins if total_basins else 0.0):.1f}%)",
      f"- **Total Benchmark Runtime**: {total_wall_time:.2f}s ({time_per_basin:.3f}s / basin)",
      f"- **Continuous Attributes Mean Pearson r**: **{mean_r:.4f}**",
      f"- **Continuous Attributes Median Pearson r**: **{median_r:.5f}**",
      f"- **Attributes with r >= 0.99**: **{pct_r_99:.1f}%** ({int((cont_metrics['pearson_r'] >= 0.99).sum())} / {len(cont_metrics)})",
      f"- **Attributes with r >= 0.95**: **{pct_r_95:.1f}%**",
      f"- **Attributes with r >= 0.90**: **{pct_r_90:.1f}%**",
      f"- **Categorical Majority Accuracy**: **{mean_cat_acc:.2f}%**",
      f"- **Median Basin Drainage Area Discrepancy**: **{med_area_err:.2f}%**",
      f"- **Maximum Basin Drainage Area Discrepancy**: **{max_area_err:.2f}%** (Basin: `{worst_area_basin}`)",
      f"- **Median Attribute Relative Error**: **{med_attr_err:.2f}%**",
      f"- **Maximum Attribute Relative Error**: **{max_attr_err:.2f}%** (Attribute: `{worst_attr_name}`)",
      f"- **Maximum Absolute Error**: **{max_abs_err_val:.4f}** (Attribute: `{worst_abs_attr_name}`)",
      "",
      "## 2. Performance by Thematic Domain",
      "",
      _to_markdown_table(cat_table_headers, cat_table_rows),
      "",
      "## 3. Performance by Geographic Region / Dataset",
      "",
      _to_markdown_table(ds_table_headers, ds_table_rows),
      "",
      "## 4. Performance by Basin Size Tier",
      "",
      _to_markdown_table(tier_table_headers, tier_table_rows),
      "",
      "## 5. Discrete Categorical Majority Attributes Accuracy",
      "",
      _to_markdown_table(maj_table_headers, maj_table_rows),
      "",
      "## 6. Top 10 Most Accurately Extracted Continuous Attributes",
      "",
      _to_markdown_table(
          ["Attribute", "Thematic Domain", "Pearson r", "R^2", "Med Rel Err %", "Max Rel Err %", "Max Abs Err"],
          [
              [r["attribute"], r["category"], f"{r['pearson_r']:.5f}", f"{r['r2']:.5f}", f"{r['med_rel_error_pct']:.2f}%", f"{r['max_rel_error_pct']:.2f}%", f"{r['max_abs_error']:.4f}"]
              for _, r in top_10.iterrows()
          ],
      ),
      "",
      "## 7. Bottom 10 Continuous Attributes by Correlation",
      "",
      _to_markdown_table(
          ["Attribute", "Thematic Domain", "Pearson r", "R^2", "Med Rel Err %", "Max Rel Err %", "Max Abs Err"],
          [
              [r["attribute"], r["category"], f"{r['pearson_r']:.5f}", f"{r['r2']:.5f}", f"{r['med_rel_error_pct']:.2f}%", f"{r['max_rel_error_pct']:.2f}%", f"{r['max_abs_error']:.4f}"]
              for _, r in bottom_10.iterrows()
          ],
      ),
      "",
      "## 8. Top 10 Attributes by Maximum Relative Error",
      "",
      _to_markdown_table(max_err_headers, max_err_rows),
      "",
  ]

  report_path.write_text("\n".join(lines))


def _to_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
  header_line = "| " + " | ".join(headers) + " |"
  sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
  data_lines = ["| " + " | ".join(str(val) for val in row) + " |" for row in rows]
  return "\n".join([header_line, sep_line] + data_lines)


def main():
  parser = argparse.ArgumentParser(
      description="Run global Caravan static attributes extraction benchmark."
  )
  parser.add_argument(
      "--samples",
      type=int,
      default=None,
      help="Number of basins to benchmark (default: all basins in dataset).",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default=None,
      help="Path to custom benchmark dataset (.parquet).",
  )
  parser.add_argument(
      "--regions",
      "--datasets",
      nargs="+",
      default=None,
      dest="regions",
      help="Filter by datasets / regions (camels, camelsaus, camelsbr, camelscl, camelsgb, hysets, lamah).",
  )
  parser.add_argument(
      "--size-tiers",
      nargs="+",
      default=None,
      help="Filter by size tiers (1_micro, 2_small, 3_medium, 4_large, 5_macro).",
  )
  parser.add_argument(
      "--workers",
      type=int,
      default=8,
      help="Number of parallel worker processes (default: 8).",
  )
  parser.add_argument(
      "--gdb-path",
      type=str,
      default=None,
      help="BasinATLAS GDB or shapefile path. Defaults to runtime cache.",
  )
  parser.add_argument(
      "--era5-cache-dir",
      type=str,
      default=None,
      help="Directory containing precomputed continental ERA5 tables.",
  )
  parser.add_argument(
      "--era5-source",
      type=str,
      required=True,
      choices=["hybas", "gridded"],
      help="ERA5 data source (required): 'hybas' (precomputed subbasins) or 'gridded' (Zarr).",
  )
  parser.add_argument(
      "-o",
      "--output-dir",
      type=str,
      default="./benchmark_results",
      help="Directory to save benchmark reports and CSV files.",
  )

  args = parser.parse_args()

  run_benchmark(
      dataset_path=args.dataset,
      samples=args.samples,
      regions=args.regions,
      size_tiers=args.size_tiers,
      workers=args.workers,
      gdb_path=args.gdb_path,
      era5_cache_dir=args.era5_cache_dir,
      era5_source=args.era5_source,
      output_dir=args.output_dir,
  )


if __name__ == "__main__":
  main()
