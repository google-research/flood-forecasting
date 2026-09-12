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

"""Massively parallel Dask runner for MultiMet meteorological forcing extraction.

Enables distributed extraction across time (days or temporal batches) for all core
MultiMet products (ERA5-Land, CPC, IMERG, HRES, GraphCast, AIFS) on Dask clusters
(Google Cloud, Kubernetes, or local multi-core machines).

Uses lock-free direct chunk writing to pre-allocated Zarr stores, eliminating
race conditions and scheduler memory bottlenecks.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import distributed
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import zarr

from multimet.base import BaseExtractor
from multimet.config import (
    FORECAST_LEAD_DAYS,
    PRODUCT_BANDS,
    PRODUCT_TYPES,
    Product,
    ProductType,
)
from multimet.cpc import CPCExtractor
from multimet.dynamical import AIFSExtractor, DynamicalIMERGExtractor
from multimet.era5_land import ERA5LandExtractor
from multimet.geometry import load_basin_geometries
from multimet.graphcast import GraphCastExtractor
from multimet.hres import HRESExtractor
from multimet.imerg import IMERGExtractor
from multimet.spatial import slice_coordinates_by_bounds
from multimet.zarr_writer import MultiMetZarrWriter
from multimet.zonal import ZonalWeightMatrix

logger = logging.getLogger(__name__)

PRODUCT_MAP: Dict[str, Tuple[Product, type[BaseExtractor]]] = {
    "CPC": (Product.CPC, CPCExtractor),
    "ERA5_LAND": (Product.ERA5_LAND, ERA5LandExtractor),
    "IMERG": (Product.IMERG, IMERGExtractor),
    "HRES": (Product.HRES, HRESExtractor),
    "GRAPHCAST": (Product.GRAPHCAST, GraphCastExtractor),
    "AIFS": (Product.AIFS, AIFSExtractor),
    "DYNAMICAL_IMERG": (Product.DYNAMICAL_IMERG, DynamicalIMERGExtractor),
}


def init_dask_client(
    scheduler_address: Optional[str] = None,
    num_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    memory_limit: str = "auto",
    dashboard_address: Optional[str] = None,
) -> distributed.Client:
  """Initializes or connects to a Dask distributed Client.

  Args:
    scheduler_address: Optional address of remote Dask scheduler (e.g. tcp://10.0.0.1:8786).
    num_workers: Number of workers for LocalCluster if scheduler_address is None.
    threads_per_worker: Threads per worker (default 1 for GIL-bound python tasks).
    memory_limit: Per-worker RAM limit (e.g. '4GB' or 'auto').
    dashboard_address: Optional dashboard port/address (e.g. ':8787').

  Returns:
    Connected distributed.Client instance.
  """
  if scheduler_address:
    logger.info("Connecting to remote Dask scheduler at %s", scheduler_address)
    return distributed.Client(scheduler_address)

  try:
    existing_client = distributed.get_client()
    logger.info("Reusing existing active Dask client: %s", existing_client)
    return existing_client
  except ValueError:
    pass

  n_workers = num_workers or max(1, (os.cpu_count() or 2) - 1)
  logger.info(
      "Spawning local Dask cluster with %d workers (threads_per_worker=%d)...",
      n_workers,
      threads_per_worker,
  )
  cluster = distributed.LocalCluster(
      n_workers=n_workers,
      threads_per_worker=threads_per_worker,
      memory_limit=memory_limit,
      dashboard_address=dashboard_address,
      processes=True,
  )
  return distributed.Client(cluster)


def _extract_and_write_chunk_task(
    store_path: str,
    product_name: str,
    extractor_cls: type[BaseExtractor],
    extractor_kwargs: Dict[str, Any],
    basins_gdf: gpd.GeoDataFrame,
    weights_matrix: Optional[ZonalWeightMatrix],
    start_date_str: str,
    end_date_str: str,
    start_idx: int,
    end_idx: int,
    use_bounding_box: bool = True,
) -> Dict[str, Any]:
  """Worker task: extracts data for a date batch and writes directly to Zarr.

  Args:
    store_path: Target Zarr store URI.
    product_name: MultiMet product enum name string.
    extractor_cls: Extractor class to instantiate.
    extractor_kwargs: Keyword arguments for extractor instantiation.
    basins_gdf: Catchment GeoDataFrame.
    weights_matrix: Precomputed and cropped ZonalWeightMatrix.
    start_date_str: Start date string (YYYY-MM-DD).
    end_date_str: End date string (YYYY-MM-DD).
    start_idx: Start integer index along the date dimension (inclusive).
    end_idx: End integer index along the date dimension (exclusive).
    use_bounding_box: Whether to spatially slice gridded inputs to bounds.

  Returns:
    Status dictionary summarizing extracted chunk metadata.
  """
  prod_enum = Product[product_name]
  extractor = extractor_cls(**extractor_kwargs)

  ds = extractor.extract_for_basins(
      basins_gdf,
      start_date=start_date_str,
      end_date=end_date_str,
      weights_matrix=weights_matrix,
      use_bounding_box=use_bounding_box,
  )

  num_days = end_idx - start_idx
  z_root = zarr.open_group(store_path, mode="r+")

  for band in ds.data_vars:
    if band not in z_root:
      continue
    vals = ds[band].values.astype(np.float32)
    if num_days == 1:
      if vals.ndim == 2:
        z_root[band][:, start_idx] = vals[:, 0]
      elif vals.ndim == 3:
        z_root[band][:, start_idx, :] = vals[:, 0, :]
    else:
      if vals.ndim == 2:
        z_root[band][:, start_idx:end_idx] = vals
      elif vals.ndim == 3:
        z_root[band][:, start_idx:end_idx, :] = vals

  return {
      "status": "ok",
      "product": product_name,
      "start_date": start_date_str,
      "end_date": end_date_str,
      "start_idx": start_idx,
      "end_idx": end_idx,
      "num_days": num_days,
  }


def extract_product_dask(
    product: Union[str, Product],
    basins: Union[str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any], Sequence[Any]],
    output_dir: Union[str, os.PathLike],
    start_date: Union[str, pd.Timestamp] = "2020-01-01",
    end_date: Union[str, pd.Timestamp] = "2020-01-02",
    client: Optional[distributed.Client] = None,
    num_workers: Optional[int] = None,
    dask_scheduler: Optional[str] = None,
    batch_days: int = 1,
    source: str = "public",
    id_column: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = True,
    weights_cache: Optional[str] = None,
    use_bounding_box: bool = True,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
    show_progress: bool = True,
    **extractor_extra_kwargs: Any,
) -> str:
  """Extracts a single meteorological product in parallel across days using Dask.

  Args:
    product: Target Product enum or name string.
    basins: Catchment geometries (file path or GeoDataFrame).
    output_dir: Destination directory for consolidated Zarr stores.
    start_date: Start date (YYYY-MM-DD or Timestamp).
    end_date: End date (YYYY-MM-DD or Timestamp).
    client: Optional existing Dask Client. If None, one will be created or retrieved.
    num_workers: Number of workers if creating a LocalCluster.
    dask_scheduler: Address of remote Dask scheduler if applicable.
    batch_days: Number of consecutive days per worker task (default 1).
    source: Data source mode ('public', 'local', etc.).
    id_column: Optional basin identifier column name in geometry source.
    overwrite: If True, deletes existing destination store before extraction.
    resume: If True and store exists, inspects chunks and only extracts missing days.
    weights_cache: Optional path to cached weights .npz file.
    use_bounding_box: Whether to geographically slice grids to basin bounds.
    earthdata_username: Optional NASA Earthdata username.
    earthdata_password: Optional NASA Earthdata password.
    earthdata_token: Optional NASA Earthdata Bearer token.
    netrc_path: Optional custom .netrc path.
    show_progress: Whether to display a tqdm progress bar.
    **extractor_extra_kwargs: Additional arguments passed to extractor constructor.

  Returns:
    Target Zarr store path written to.
  """
  prod_name = product.value if isinstance(product, Product) else str(product).upper()
  if prod_name not in PRODUCT_MAP:
    raise ValueError(
        f"Unsupported product '{prod_name}'. Supported: {list(PRODUCT_MAP.keys())}"
    )

  prod_enum, extractor_cls = PRODUCT_MAP[prod_name]
  basins_gdf = load_basin_geometries(basins, id_column=id_column)
  basin_ids = list(basins_gdf.index)

  start_dt = pd.to_datetime(start_date)
  end_dt = pd.to_datetime(end_date)
  if end_dt < start_dt:
    raise ValueError(f"end_date ({end_dt}) cannot be before start_date ({start_dt})")

  all_dates = pd.date_range(start_dt, end_dt, freq="D")
  total_days = len(all_dates)

  writer = MultiMetZarrWriter(output_dir)
  store_path = writer.get_store_path(prod_enum)

  # Configure extractor kwargs
  extractor_kwargs = dict(extractor_extra_kwargs)
  if prod_name == "CPC":
    src = "psl" if source in ("public", "auto") else ("binary" if source == "local" else source)
    extractor_kwargs["source"] = src
  elif prod_name == "IMERG":
    if source in ("dynamical", "icechunk", "catalog"):
      extractor_cls = DynamicalIMERGExtractor
      extractor_kwargs["source"] = source
    else:
      src = "gesdisc" if source in ("public", "auto") else ("h5" if source == "local" else source)
      extractor_kwargs.update({
          "source": src,
          "username": earthdata_username,
          "password": earthdata_password,
          "token": earthdata_token,
          "netrc_path": netrc_path,
      })
  elif prod_name in ("ERA5_LAND", "HRES", "GRAPHCAST"):
    src = "wb2" if source in ("public", "auto") else ("local" if source == "local" else source)
    extractor_kwargs["source"] = src
  elif prod_name in ("AIFS", "DYNAMICAL_IMERG"):
    extractor_kwargs["source"] = source

  # Handle store initialization & overwrite
  def _store_exists(p: str) -> bool:
    return (
        os.path.exists(os.path.join(p, "zarr.json"))
        or os.path.exists(os.path.join(p, ".zgroup"))
        or os.path.exists(os.path.join(p, ".zmetadata"))
    )

  store_already_exists = _store_exists(store_path)

  if overwrite and store_already_exists:
    logger.info("Overwrite requested: removing existing store %s", store_path)
    if os.path.isdir(store_path):
      shutil.rmtree(store_path, ignore_errors=True)
    elif os.path.exists(store_path):
      os.remove(store_path)
    store_already_exists = False

  if not store_already_exists:
    logger.info("Initializing skeleton Zarr store for %s at %s...", prod_name, store_path)
    writer.initialize_zarr_store(prod_enum, basin_ids, all_dates)
    missing_indices = list(range(total_days))
  else:
    # Store exists: check date coordinates
    try:
      with xr.open_zarr(store_path) as existing_ds:
        existing_dates = pd.to_datetime(existing_ds["date"].values)
    except Exception:
      existing_dates = None

    if existing_dates is not None and not all_dates.equals(existing_dates) and not resume:
      logger.info("Dates mismatch: reinitializing store %s for requested date range...", store_path)
      writer.initialize_zarr_store(prod_enum, basin_ids, all_dates)
      existing_z = zarr.open_group(store_path, mode="r")
      missing_indices = list(range(total_days))
    else:
      existing_z = zarr.open_group(store_path, mode="r")
      if resume:
        missing_indices = [
            i
            for i in range(total_days)
            if not writer.is_date_chunk_written(prod_enum, i, root_group=existing_z)
        ]
        logger.info(
            "Resume mode: %d of %d days already written in %s",
            total_days - len(missing_indices),
            total_days,
            prod_name,
        )
      else:
        missing_indices = list(range(total_days))

  if not missing_indices:
    logger.info("Product %s is already 100%% complete. Consolidating metadata...", prod_name)
    writer.consolidate_metadata(prod_enum)
    return store_path

  # Precompute and crop ZonalWeightMatrix once on the driver
  weights_matrix: Optional[ZonalWeightMatrix] = None
  if weights_cache and os.path.exists(weights_cache):
    weights_matrix = ZonalWeightMatrix.load(weights_cache)
    logger.info("Loaded precomputed weights matrix from %s", weights_cache)

  sample_extractor = extractor_cls(**extractor_kwargs)
  if weights_matrix is None and hasattr(sample_extractor, "lats") and sample_extractor.lats is not None:
    if use_bounding_box:
      sub_lats, sub_lons, _, _ = slice_coordinates_by_bounds(
          sample_extractor.lats, sample_extractor.lons, bounds=basins_gdf, buffer_degrees=0.5
      )
      res_lat = float(abs(sample_extractor.lats[1] - sample_extractor.lats[0]))
      res_lon = float(abs(sample_extractor.lons[1] - sample_extractor.lons[0]))
      weights_matrix = ZonalWeightMatrix.from_geodataframe(
          basins_gdf, sub_lats, sub_lons, cell_res_lat=res_lat, cell_res_lon=res_lon
      )
    else:
      res_lat = float(abs(sample_extractor.lats[1] - sample_extractor.lats[0]))
      res_lon = float(abs(sample_extractor.lons[1] - sample_extractor.lons[0]))
      weights_matrix = ZonalWeightMatrix.from_geodataframe(
          basins_gdf, sample_extractor.lats, sample_extractor.lons, cell_res_lat=res_lat, cell_res_lon=res_lon
      )

  # Partition missing dates into contiguous tasks
  batches: List[Tuple[int, int]] = []
  i = 0
  while i < len(missing_indices):
    start_pos = i
    while (
        i + 1 < len(missing_indices)
        and missing_indices[i + 1] == missing_indices[i] + 1
        and (i + 1 - start_pos) < batch_days
    ):
      i += 1
    batches.append((missing_indices[start_pos], missing_indices[i] + 1))
    i += 1

  # Initialize Dask Client
  dask_client = client or init_dask_client(
      scheduler_address=dask_scheduler,
      num_workers=num_workers,
  )

  # Scatter large immutable objects to cluster workers
  gdf_future = dask_client.scatter(basins_gdf, broadcast=True)
  matrix_future = (
      dask_client.scatter(weights_matrix, broadcast=True)
      if weights_matrix is not None
      else None
  )

  logger.info(
      "Dispatching %d Dask tasks (%d days) for %s across cluster...",
      len(batches),
      len(missing_indices),
      prod_name,
  )

  task_futures = []
  for b_start, b_end in batches:
    b_start_dt_str = all_dates[b_start].strftime("%Y-%m-%d")
    b_end_dt_str = all_dates[b_end - 1].strftime("%Y-%m-%d")
    future = dask_client.submit(
        _extract_and_write_chunk_task,
        store_path=store_path,
        product_name=prod_enum.name,
        extractor_cls=extractor_cls,
        extractor_kwargs=extractor_kwargs,
        basins_gdf=gdf_future,
        weights_matrix=matrix_future,
        start_date_str=b_start_dt_str,
        end_date_str=b_end_dt_str,
        start_idx=b_start,
        end_idx=b_end,
        use_bounding_box=use_bounding_box,
        retries=3,
    )
    task_futures.append(future)

  t0 = time.time()
  pbar = tqdm.tqdm(
      total=len(missing_indices),
      desc=f"Dask {prod_name} [{start_dt.strftime('%Y-%m-%d')}..{end_dt.strftime('%Y-%m-%d')}]",
      unit="day",
      disable=not show_progress,
  )

  completed_days = 0
  for future in distributed.as_completed(task_futures):
    res = future.result()
    n_done = res.get("num_days", 1)
    completed_days += n_done
    pbar.update(n_done)

  pbar.close()
  elapsed = time.time() - t0
  basin_days = len(basin_ids) * completed_days
  throughput = basin_days / elapsed if elapsed > 0 else 0.0
  logger.info(
      "Completed %s extraction of %d days in %.2fs (%.1f basin-days/s)",
      prod_name,
      completed_days,
      elapsed,
      throughput,
  )

  # Consolidate metadata post-flight
  writer.consolidate_metadata(prod_enum)
  logger.info("Successfully consolidated metadata for %s at %s", prod_name, store_path)
  return store_path


def extract_multimet_dask(
    basins: Union[str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any], Sequence[Any]],
    output_dir: Union[str, os.PathLike],
    products: Optional[Sequence[Union[str, Product]]] = None,
    start_date: Union[str, pd.Timestamp] = "2020-01-01",
    end_date: Union[str, pd.Timestamp] = "2020-01-02",
    dask_scheduler: Optional[str] = None,
    num_workers: Optional[int] = None,
    batch_days: int = 1,
    source: str = "public",
    id_column: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = True,
    weights_cache: Optional[str] = None,
    use_bounding_box: bool = True,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
) -> Dict[str, str]:
  """Runs massively parallel Dask extraction across requested products.

  Args:
    basins: Catchment geometries source (path or GeoDataFrame).
    output_dir: Target directory for consolidated Zarr stores.
    products: List of products to extract. Defaults to all 5 core products.
    start_date: Start date string (YYYY-MM-DD) or Timestamp.
    end_date: End date string (YYYY-MM-DD) or Timestamp.
    dask_scheduler: Address of remote Dask scheduler if applicable.
    num_workers: Number of workers for LocalCluster if local.
    batch_days: Number of days per Dask worker task (default 1).
    source: Source mode ('public' or 'local').
    id_column: Optional column name for gauge ID in geometry file.
    overwrite: Whether to overwrite existing stores.
    resume: Whether to resume and only process missing days.
    weights_cache: Optional path to cached weights .npz file or directory.
    use_bounding_box: Whether to use spatial bounding box reduction.
    earthdata_username: Optional NASA Earthdata username.
    earthdata_password: Optional NASA Earthdata password.
    earthdata_token: Optional NASA Earthdata Bearer token.
    netrc_path: Optional path to custom .netrc file.

  Returns:
    Dictionary mapping product name to output Zarr store path.
  """
  client = init_dask_client(
      scheduler_address=dask_scheduler,
      num_workers=num_workers,
  )

  target_prods = (
      [p.value if isinstance(p, Product) else str(p).upper() for p in products]
      if products is not None
      else ["CPC", "ERA5_LAND", "IMERG", "HRES", "GRAPHCAST"]
  )

  output_stores: Dict[str, str] = {}
  for prod_name in target_prods:
    w_path = weights_cache
    if weights_cache and os.path.isdir(weights_cache):
      w_path = os.path.join(weights_cache, f"weights_{prod_name.lower()}.npz")

    store_path = extract_product_dask(
        product=prod_name,
        basins=basins,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        client=client,
        batch_days=batch_days,
        source=source,
        id_column=id_column,
        overwrite=overwrite,
        resume=resume,
        weights_cache=w_path,
        use_bounding_box=use_bounding_box,
        earthdata_username=earthdata_username,
        earthdata_password=earthdata_password,
        earthdata_token=earthdata_token,
        netrc_path=netrc_path,
    )
    output_stores[prod_name] = store_path

  return output_stores


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="MultiMet Massively Parallel Dask Meteorological Extractor",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--basins_path",
      nargs="+",
      required=True,
      help=(
          "Path(s) to GeoJSON or Shapefile catchment boundaries. "
          "Accepts one or more files, glob patterns, directories, or comma-separated strings."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Directory or GCS URI to save extracted consolidated Zarr stores.",
  )
  parser.add_argument(
      "--products",
      type=str,
      default="CPC,ERA5_LAND,IMERG,HRES,GRAPHCAST",
      help="Comma-separated product list to extract.",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default="2020-01-01",
      help="Start date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default="2020-01-02",
      help="End date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--dask_scheduler",
      type=str,
      default=None,
      help="Optional remote Dask scheduler address (tcp://host:port).",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=None,
      help="Number of local Dask workers (defaults to CPU count - 1).",
  )
  parser.add_argument(
      "--batch_days",
      type=int,
      default=1,
      help="Number of consecutive days per worker task.",
  )
  parser.add_argument(
      "--source",
      type=str,
      default="public",
      help="Source mode: 'public' or 'local'.",
  )
  parser.add_argument(
      "--id_column",
      type=str,
      default=None,
      help="Column name for basin ID in geometry file.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite existing Zarr stores.",
  )
  parser.add_argument(
      "--no-resume",
      dest="resume",
      action="store_false",
      default=True,
      help="Disable resumption checking and re-extract all dates.",
  )
  parser.add_argument(
      "--weights_cache",
      type=str,
      default=None,
      help="Path to precomputed/cached .npz weights archive or directory.",
  )
  parser.add_argument(
      "--no-bounding-box",
      dest="use_bounding_box",
      action="store_false",
      default=True,
      help="Disable spatial bounding box slicing.",
  )
  parser.add_argument(
      "--earthdata_username",
      type=str,
      default=None,
      help="NASA Earthdata Login username.",
  )
  parser.add_argument(
      "--earthdata_password",
      type=str,
      default=None,
      help="NASA Earthdata Login password.",
  )
  parser.add_argument(
      "--earthdata_token",
      type=str,
      default=None,
      help="NASA Earthdata Bearer token.",
  )
  parser.add_argument(
      "--netrc_path",
      type=str,
      default=None,
      help="Custom path to .netrc file for Earthdata credentials.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )
  parser = _build_parser()
  args = parser.parse_args(argv)

  prods = [p.strip() for p in args.products.split(",") if p.strip()]
  t0 = time.time()
  print(f"▶ Starting MultiMet Dask parallel extraction for: {prods}")
  stores = extract_multimet_dask(
      basins=args.basins_path,
      output_dir=args.output_dir,
      products=prods,
      start_date=args.start_date,
      end_date=args.end_date,
      dask_scheduler=args.dask_scheduler,
      num_workers=args.num_workers,
      batch_days=args.batch_days,
      source=args.source,
      id_column=args.id_column,
      overwrite=args.overwrite,
      resume=args.resume,
      weights_cache=args.weights_cache,
      use_bounding_box=args.use_bounding_box,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
  )
  print(f"\n✓ Completed extraction of {len(stores)} products in {time.time() - t0:.2f}s:")
  for prod, store_path in stores.items():
    print(f"  • {prod:12s} -> {store_path}")


if __name__ == "__main__":
  main()
