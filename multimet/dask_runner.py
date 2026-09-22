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
import gc
import logging
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask
import distributed
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
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
from multimet.gcp import configure_gcp_project
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
  mem_limit = memory_limit
  if str(mem_limit).strip().lower() in ("0", "none", "false", "unlimited"):
    mem_limit = 0
    mem_str = "unlimited (no nanny limit)"
  elif mem_limit == "auto":
    try:
      total_ram = psutil.virtual_memory().total
      # Proportional allocation: divide 90% of host RAM evenly across workers
      mem_limit = int((total_ram * 0.9) / n_workers)
      mem_str = f"{mem_limit / (1024**3):.2f} GiB"
    except Exception:
      mem_limit = "auto"
      mem_str = "auto"
  else:
    mem_str = str(mem_limit)

  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
  existing_py_path = os.environ.get("PYTHONPATH", "")
  worker_py_path = (
      f"{repo_root}{os.pathsep}{existing_py_path}"
      if existing_py_path
      else repo_root
  )

  logger.info(
      "Spawning local Dask cluster with %d workers (threads_per_worker=%d, memory_limit=%s)...",
      n_workers,
      threads_per_worker,
      mem_str,
  )
  cluster = distributed.LocalCluster(
      n_workers=n_workers,
      threads_per_worker=threads_per_worker,
      memory_limit=mem_limit,
      dashboard_address=dashboard_address,
      processes=True,
      env={
          "PYTHONPATH": worker_py_path,
          "PYTHONWARNINGS": "ignore::FutureWarning,ignore::UserWarning",
          "MULTIMET_DASK_WORKER": "1",
      },
  )
  client = distributed.Client(cluster)

  def _setup_worker(r: str) -> None:
    import sys
    import warnings
    if r not in sys.path:
      sys.path.insert(0, r)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"zarr.*")
    try:
      from multimet.gcp import _patch_gcsfs_shutdown
      _patch_gcsfs_shutdown()
    except Exception:
      pass

  client.run(_setup_worker, repo_root)
  return client


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
    gcp_project: Optional[str] = None,
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
    gcp_project: Optional Google Cloud project ID for GCS quota/billing.

  Returns:
    Status dictionary summarizing extracted chunk metadata.
  """
  if gcp_project or store_path.startswith(("gs://", "gcs://")):
    configure_gcp_project(gcp_project)

  prod_enum = Product[product_name]
  extractor = extractor_cls(**extractor_kwargs)

  with dask.config.set(scheduler="threads"):
    ds = extractor.extract_for_basins(
        basins_gdf,
        start_date=start_date_str,
        end_date=end_date_str,
        weights_matrix=weights_matrix,
        use_bounding_box=use_bounding_box,
    )

  num_days = end_idx - start_idx
  max_write_retries = 5
  for attempt in range(max_write_retries):
    try:
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
      break
    except Exception as e:
      if attempt == max_write_retries - 1:
        logger.error(
            "Failed to write chunk [%d:%d] to %s after %d attempts: %s",
            start_idx,
            end_idx,
            store_path,
            max_write_retries,
            e,
        )
        raise
      backoff = (2 ** attempt) + random.uniform(0.5, 2.0)
      logger.warning(
          "Transient error writing chunk [%d:%d] to %s (attempt %d/%d): %s. Retrying in %.2fs...",
          start_idx,
          end_idx,
          store_path,
          attempt + 1,
          max_write_retries,
          e,
          backoff,
      )
      time.sleep(backoff)

  del ds
  gc.collect()

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
    memory_limit: Union[str, int, float, None] = "auto",
    source: str = "public",
    id_column: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = True,
    append: bool = False,
    weights_cache: Optional[str] = None,
    use_bounding_box: bool = True,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
    gcp_project: Optional[str] = None,
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
    append: If True and store exists, appends either new dates or new basins.
      Cannot append both simultaneously.
    weights_cache: Optional path to cached weights .npz file.
    use_bounding_box: Whether to geographically slice grids to basin bounds.
    earthdata_username: Optional NASA Earthdata username.
    earthdata_password: Optional NASA Earthdata password.
    earthdata_token: Optional NASA Earthdata Bearer token.
    netrc_path: Optional custom .netrc path.
    gcp_project: Optional Google Cloud project ID for GCS quota/billing.
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

  if start_date is None or end_date is None:
    raise ValueError(
        "extract_product_dask requires both start_date and end_date to be "
        "explicitly provided; default placeholder dates are not permitted."
    )
  start_dt = pd.to_datetime(start_date)
  end_dt = pd.to_datetime(end_date)
  if end_dt < start_dt:
    raise ValueError(
        f"end_date ({end_dt}) cannot be before start_date ({start_dt})"
    )

  all_dates = pd.date_range(start_dt, end_dt, freq="D")
  total_days = len(all_dates)

  writer = MultiMetZarrWriter(output_dir)
  store_path = writer.get_store_path(prod_enum)

  if gcp_project or store_path.startswith(("gs://", "gcs://")):
    gcp_project = configure_gcp_project(gcp_project)

  # Configure extractor kwargs
  extractor_kwargs = dict(extractor_extra_kwargs)
  if "archive_store" in extractor_kwargs:
    arch_val = extractor_kwargs.pop("archive_store")
    if arch_val and not extractor_kwargs.get("data_dir"):
      extractor_kwargs["data_dir"] = arch_val
  source_lower = source.lower().strip()
  if prod_name == "ERA5_LAND":
    extractor_kwargs["source"] = "archive"
    if not extractor_kwargs.get("data_dir"):
      raise ValueError(
          "ERA5_LAND requires an explicit gridded archive URI via data_dir "
          "or archive_stores['ERA5_LAND']."
      )
  elif source_lower in ("archive", "gridded_archive", "zarr_archive"):
    extractor_kwargs["source"] = "archive"
    if not extractor_kwargs.get("data_dir"):
      raise ValueError(
          f"Product {prod_name} in archive mode requires an explicit store "
          f"URI via data_dir or archive_stores[{prod_name!r}]."
      )
  elif prod_name == "CPC":
    src = (
        "psl"
        if source_lower in ("public", "auto", "upstream")
        else ("binary" if source_lower == "local" else source_lower)
    )
    extractor_kwargs["source"] = src
  elif prod_name == "IMERG":
    if source_lower in ("dynamical", "icechunk", "catalog"):
      extractor_cls = DynamicalIMERGExtractor
      extractor_kwargs["source"] = source_lower
    else:
      src = (
          "gesdisc"
          if source_lower in ("public", "auto", "upstream")
          else ("h5" if source_lower == "local" else source_lower)
      )
      extractor_kwargs.update({
          "source": src,
          "username": earthdata_username,
          "password": earthdata_password,
          "token": earthdata_token,
          "netrc_path": netrc_path,
      })
  elif prod_name in ("HRES", "GRAPHCAST"):
    src = (
        "wb2"
        if source_lower in ("public", "auto", "upstream")
        else ("local" if source_lower == "local" else source_lower)
    )
    extractor_kwargs["source"] = src
  elif prod_name in ("AIFS", "DYNAMICAL_IMERG"):
    extractor_kwargs["source"] = source_lower

  from multimet.zarr_writer import check_zarr_store_exists

  def _store_exists(p: str) -> bool:
    return check_zarr_store_exists(p)

  def _remove_store(p: str) -> None:
    try:
      fs, fs_path = fsspec.core.url_to_fs(p)
      if fs.exists(fs_path):
        fs.rm(fs_path, recursive=True)
        time.sleep(1.0)
    except Exception as e:
      logger.warning("Error removing store %s with fsspec: %s", p, e)
      if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
      elif os.path.exists(p):
        os.remove(p)
    if _store_exists(p):
      raise RuntimeError(
          f"Failed to overwrite/remove existing store at {p}. "
          "Please verify storage permissions and quota project."
      )

  store_already_exists = _store_exists(store_path)

  if overwrite and store_already_exists:
    logger.info("Overwrite requested: removing existing store %s", store_path)
    _remove_store(store_path)
    store_already_exists = False

  provenance_attrs: Dict[str, Any] = {
      "Code_Repository": "https://github.com/google-research/flood-forecasting",
      "Code_Package": (
          "https://github.com/google-research/flood-forecasting/tree/main/multimet"
      ),
      "Generated_By": "multimet.dask_runner",
      "Extracted_Date_Range": (
          f"{all_dates[0].strftime('%Y-%m-%d')} to"
          f" {all_dates[-1].strftime('%Y-%m-%d')}"
      ),
  }
  if source == "archive" and "data_dir" in extractor_extra_kwargs:
    provenance_attrs["Extracted_From_Gridded_Archive"] = str(
        extractor_extra_kwargs["data_dir"]
    )
  if isinstance(basins, (str, os.PathLike)):
    provenance_attrs["Extracted_From_Catchment_Shapefiles"] = str(basins)
  elif isinstance(basins, (list, tuple, set)):
    provenance_attrs["Extracted_From_Catchment_Shapefiles"] = [
        str(b) for b in basins
    ]

  if not store_already_exists:
    logger.info("Initializing skeleton Zarr store for %s at %s...", prod_name, store_path)
    writer.initialize_zarr_store(
        prod_enum, basin_ids, all_dates, extra_attrs=provenance_attrs
    )
    missing_indices = list(range(total_days))
  else:
    # Store already exists and overwrite is False: update/append mode
    store_info = writer.get_store_info(prod_enum)
    if store_info is not None:
      store_basins = set(store_info["basins"])
      store_dates = pd.DatetimeIndex(store_info["dates"])
      new_basins = [b for b in basin_ids if b not in store_basins]
      new_dates = [d for d in all_dates if d not in set(store_dates)]

      # STRICT CHECK: Disallow simultaneous expansion along both dimensions
      if new_basins and new_dates:
        raise ValueError(
            f"Cannot append both new basins ({len(new_basins)}) and new dates ({len(new_dates)}) "
            f"simultaneously to existing Zarr store at {store_path}. "
            "Zarr arrays require a dense rectangular coordinate grid; expanding two dimensions at once "
            "leaves unpopulated cross-quadrants. Please run two sequential steps: "
            "first update dates for existing basins, then append new basins for the full date range (or vice versa)."
        )

      if new_basins:
        # Appending new basins across existing store date range
        logger.info(
            "Appending %d new basins to %s across %d existing dates [%s..%s]...",
            len(new_basins),
            prod_name,
            len(store_dates),
            store_dates[0].strftime("%Y-%m-%d"),
            store_dates[-1].strftime("%Y-%m-%d"),
        )
        tmp_output_dir = os.path.join(
            output_dir, f".tmp_append_{prod_enum.value}_{int(time.time() * 1000)}"
        )
        try:
          extract_product_dask(
              product=prod_enum,
              basins=basins_gdf.loc[new_basins],
              output_dir=tmp_output_dir,
              start_date=store_dates[0],
              end_date=store_dates[-1],
              client=client,
              num_workers=num_workers,
              dask_scheduler=dask_scheduler,
              batch_days=batch_days,
              memory_limit=memory_limit,
              source=source,
              id_column=id_column,
              overwrite=True,
              resume=False,
              weights_cache=weights_cache,
              use_bounding_box=use_bounding_box,
              earthdata_username=earthdata_username,
              earthdata_password=earthdata_password,
              earthdata_token=earthdata_token,
              netrc_path=netrc_path,
              gcp_project=gcp_project,
              show_progress=show_progress,
              **extractor_extra_kwargs,
          )
          tmp_writer = MultiMetZarrWriter(tmp_output_dir)
          tmp_store = tmp_writer.get_store_path(prod_enum)
          with xr.open_zarr(tmp_store) as tmp_ds:
            writer.append_basins(prod_enum, tmp_ds)
          logger.info(
              "Successfully appended %d new basins to %s store at %s",
              len(new_basins),
              prod_name,
              store_path,
          )
        finally:
          try:
            fs, fs_path = fsspec.core.url_to_fs(tmp_output_dir)
            if fs.exists(fs_path):
              fs.rm(fs_path, recursive=True)
          except Exception:
            if os.path.exists(tmp_output_dir):
              shutil.rmtree(tmp_output_dir, ignore_errors=True)
        return store_path

      else:
        # Existing basins: expand date range if needed (prepending, postpending, overlap, or rewrite)
        all_store_dates, requested_indices = writer.expand_date_range(prod_enum, all_dates)
        all_dates = all_store_dates
        total_days = len(all_dates)
        existing_z = zarr.open_group(store_path, mode="r")
        if resume:
          missing_indices = [
              i
              for i in requested_indices
              if not writer.is_date_chunk_written(prod_enum, i, root_group=existing_z)
          ]
          logger.info(
              "Resume mode: %d of %d requested days already populated in %s",
              len(requested_indices) - len(missing_indices),
              len(requested_indices),
              prod_name,
          )
        else:
          # Rewrite mode: re-extract all requested days
          missing_indices = list(requested_indices)
          logger.info(
              "Rewrite mode: extracting all %d requested days for %s",
              len(missing_indices),
              prod_name,
          )
    else:
      writer.initialize_zarr_store(
          prod_enum, basin_ids, all_dates, extra_attrs=provenance_attrs
      )
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

  # Pre-check archive in the parent process so any missing-variable warning
  # is emitted strictly ONCE before Dask workers start.
  if extractor_kwargs.get("source") == "archive" and extractor_kwargs.get("data_dir"):
    from multimet.gridded_archive import (
        _resolve_band_sources,
        get_archive_spec,
        open_gridded_archive,
    )
    _arch_uri = str(extractor_kwargs["data_dir"])
    _ds_pre = open_gridded_archive(_arch_uri)
    _resolve_band_sources(_ds_pre, get_archive_spec(prod_enum), _arch_uri)

  # Initialize Dask Client
  created_client = client is None
  dask_client = client or init_dask_client(
      scheduler_address=dask_scheduler,
      num_workers=num_workers,
      memory_limit=memory_limit,
  )

  try:
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
          gcp_project=gcp_project,
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
  finally:
    if created_client:
      try:
        cluster_obj = getattr(dask_client, "cluster", None)
        dask_client.close(timeout=5)
        if cluster_obj is not None:
          cluster_obj.close(timeout=5)
      except Exception:
        pass

  # Consolidate metadata post-flight
  writer.consolidate_metadata(prod_enum)
  logger.info("Successfully consolidated metadata for %s at %s", prod_name, store_path)
  return store_path


def extract_multimet_dask(
    basins: Union[
        str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any], Sequence[Any]
    ],
    output_dir: Union[str, os.PathLike],
    products: Optional[Sequence[Union[str, Product]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    dask_scheduler: Optional[str] = None,
    num_workers: Optional[int] = None,
    batch_days: int = 1,
    memory_limit: Union[str, int, float, None] = "auto",
    source: str = "public",
    archive_stores: Optional[Mapping[str, str]] = None,
    data_dirs: Optional[Mapping[str, str]] = None,
    id_column: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = True,
    append: bool = False,
    weights_cache: Optional[str] = None,
    use_bounding_box: bool = True,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
    gcp_project: Optional[str] = None,
) -> Dict[str, str]:
  """Runs massively parallel Dask extraction across requested products."""
  if start_date is None or end_date is None:
    raise ValueError(
        "extract_multimet_dask requires both start_date and end_date to be "
        "explicitly provided; default placeholder dates are not permitted."
    )

  norm_archive_stores: Dict[str, str] = {
      (k.value if isinstance(k, Product) else str(k).upper()): str(v)
      for k, v in (archive_stores or {}).items()
  }
  norm_data_dirs: Dict[str, str] = {
      (k.value if isinstance(k, Product) else str(k).upper()): str(v)
      for k, v in (data_dirs or {}).items()
  }

  all_paths = [str(output_dir)]
  if isinstance(basins, (str, os.PathLike)):
    all_paths.append(str(basins))
  elif isinstance(basins, (list, tuple, set)):
    all_paths.extend(str(x) for x in basins)
  if weights_cache:
    all_paths.append(str(weights_cache))
  all_paths.extend(norm_archive_stores.values())
  all_paths.extend(norm_data_dirs.values())

  if any(p.startswith(("gs://", "gcs://")) for p in all_paths) or gcp_project:
    gcp_project = configure_gcp_project(gcp_project)
    if gcp_project:
      logger.info(
          "Configured Google Cloud project for GCS operations: %s", gcp_project
      )

  client = init_dask_client(
      scheduler_address=dask_scheduler,
      num_workers=num_workers,
      memory_limit=memory_limit,
  )

  target_prods = (
      [p.value if isinstance(p, Product) else str(p).upper() for p in products]
      if products is not None
      else ["CPC", "ERA5_LAND", "IMERG", "HRES"]
  )

  output_stores: Dict[str, str] = {}
  for prod_name in target_prods:
    w_path = weights_cache
    if weights_cache and os.path.isdir(weights_cache):
      w_path = os.path.join(weights_cache, f"weights_{prod_name.lower()}.npz")

    prod_data_dir = norm_archive_stores.get(
        prod_name, norm_data_dirs.get(prod_name)
    )
    prod_source = (
        "archive" if prod_name in norm_archive_stores else source
    )
    extra_kw = {"data_dir": prod_data_dir} if prod_data_dir else {}

    store_path = extract_product_dask(
        product=prod_name,
        basins=basins,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        client=client,
        batch_days=batch_days,
        memory_limit=memory_limit,
        source=prod_source,
        id_column=id_column,
        overwrite=overwrite,
        resume=resume,
        append=append,
        weights_cache=w_path,
        use_bounding_box=use_bounding_box,
        earthdata_username=earthdata_username,
        earthdata_password=earthdata_password,
        earthdata_token=earthdata_token,
        netrc_path=netrc_path,
        gcp_project=gcp_project,
        **extra_kw,
    )
    output_stores[prod_name] = store_path

  return output_stores


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="MultiMet Massively Parallel Dask Meteorological Extractor",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--basins-path",
      "--basins_path",
      dest="basins_path",
      nargs="+",
      required=True,
      help=(
          "Path(s) to GeoJSON or Shapefile catchment boundaries. "
          "Accepts one or more files, glob patterns, directories, or comma-separated strings."
      ),
  )
  parser.add_argument(
      "--output-dir",
      "--output_dir",
      dest="output_dir",
      type=str,
      required=True,
      help="Directory or GCS URI to save extracted consolidated Zarr stores.",
  )
  parser.add_argument(
      "--products",
      type=str,
      default="CPC,ERA5_LAND,IMERG,HRES",
      help="Comma-separated product list to extract.",
  )
  parser.add_argument(
      "--start-date",
      "--start_date",
      dest="start_date",
      type=str,
      required=True,
      help="Required start date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--end-date",
      "--end_date",
      dest="end_date",
      type=str,
      required=True,
      help="Required end date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--dask-scheduler",
      "--dask_scheduler",
      dest="dask_scheduler",
      type=str,
      default=None,
      help="Optional remote Dask scheduler address (tcp://host:port).",
  )
  parser.add_argument(
      "--num-workers",
      "--num_workers",
      dest="num_workers",
      type=int,
      default=None,
      help="Number of local Dask workers (defaults to CPU count - 1).",
  )
  parser.add_argument(
      "--batch-days",
      "--batch_days",
      dest="batch_days",
      type=int,
      default=1,
      help="Number of consecutive days per worker task.",
  )
  parser.add_argument(
      "--memory-limit",
      "--memory_limit",
      dest="memory_limit",
      type=str,
      default="auto",
      help="Per-worker memory limit (e.g. '8GB', '0' to disable nanny kills, or 'auto').",
  )
  parser.add_argument(
      "--source",
      type=str,
      default="public",
      help=(
          "Source mode: 'archive' (gridded Zarr archives via --archive-store), "
          "'public'/'upstream', or 'local'."
      ),
  )
  parser.add_argument(
      "--archive-store",
      "--archive_store",
      dest="archive_stores",
      action="append",
      default=None,
      metavar="PRODUCT=URI",
      help=(
          "Explicit gridded archive Zarr store URI for a product (repeatable), "
          "e.g. --archive-store CPC=gs://.../CPC/daily_surface.zarr."
      ),
  )
  parser.add_argument(
      "--id-column",
      "--id_column",
      dest="id_column",
      type=str,
      default=None,
      help="Column name for basin ID in geometry file.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      default=False,
      help="Delete existing Zarr store before starting extraction.",
  )
  parser.add_argument(
      "--append",
      action="store_true",
      default=False,
      help="Append to existing Zarr store (either new dates or new basins).",
  )
  parser.add_argument(
      "--no-resume",
      dest="resume",
      action="store_false",
      default=True,
      help="Disable resumption checking and re-extract all dates.",
  )
  parser.add_argument(
      "--weights-cache",
      "--weights_cache",
      dest="weights_cache",
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
      "--earthdata-username",
      "--earthdata_username",
      dest="earthdata_username",
      type=str,
      default=None,
      help="NASA Earthdata Login username.",
  )
  parser.add_argument(
      "--earthdata-password",
      "--earthdata_password",
      dest="earthdata_password",
      type=str,
      default=None,
      help="NASA Earthdata Login password.",
  )
  parser.add_argument(
      "--earthdata-token",
      "--earthdata_token",
      dest="earthdata_token",
      type=str,
      default=None,
      help="NASA Earthdata Bearer token.",
  )
  parser.add_argument(
      "--netrc-path",
      "--netrc_path",
      dest="netrc_path",
      type=str,
      default=None,
      help="Custom path to .netrc file for Earthdata credentials.",
  )
  parser.add_argument(
      "--gcp-project",
      "--gcp_project",
      dest="gcp_project",
      type=str,
      default=None,
      help="Optional Google Cloud project ID for GCS quota/billing. Auto-detected if omitted.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
  from multimet.runner import _parse_product_uri_pairs

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )
  parser = _build_parser()
  args = parser.parse_args(argv)

  prods = [p.strip() for p in args.products.split(",") if p.strip()]
  archive_stores = _parse_product_uri_pairs(args.archive_stores)
  t0 = time.time()
  print(f"▶ Starting MultiMet Dask parallel extraction for: {prods}")
  try:
    stores = extract_multimet_dask(
        basins=args.basins_path,
        output_dir=args.output_dir,
        products=prods,
        start_date=args.start_date,
        end_date=args.end_date,
        dask_scheduler=args.dask_scheduler,
        num_workers=args.num_workers,
        batch_days=args.batch_days,
        memory_limit=args.memory_limit,
        source=args.source,
        archive_stores=archive_stores,
        id_column=args.id_column,
        overwrite=args.overwrite,
        resume=args.resume,
        append=args.append,
        weights_cache=args.weights_cache,
        use_bounding_box=args.use_bounding_box,
        earthdata_username=args.earthdata_username,
        earthdata_password=args.earthdata_password,
        earthdata_token=args.earthdata_token,
        netrc_path=args.netrc_path,
        gcp_project=args.gcp_project,
    )
    print(
        f"\n✓ Completed extraction of {len(stores)} products in"
        f" {time.time() - t0:.2f}s:"
    )
    for prod, store_path in stores.items():
      print(f"  • {prod:12s} -> {store_path}")
  finally:
    try:
      client = distributed.get_client()
      if client.cluster:
        client.cluster.close()
      client.close()
    except Exception:
      pass


if __name__ == "__main__":
  main()
