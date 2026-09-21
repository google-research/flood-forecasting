# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Storage location, batch write, and resume primitives for archive builders.

Provides target classification (requiring explicit URI schemes for remote cloud
stores), atomic batch append/in-place Zarr writers with retry backoff, and
self-healing resume planning that distinguishes genuine upstream missing dates
from operational failures while preventing trailing NaN initialization.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr

try:
  import gcsfs  # type: ignore[import-untyped]
except ImportError:
  gcsfs = None

if TYPE_CHECKING:
  pass

_logger = logging.getLogger(__name__)

# Matches an explicit URI scheme prefix such as "gs://", "s3://", or "file://".
# A Windows drive letter ("C:\\data" or "C:/data") has no "//" after the colon.
_URI_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://")

_FILE_SCHEME_PREFIX = "file://"

# Errors that will fail identically on every attempt and must not be retried.
NON_RETRYABLE_ERRORS = (ImportError, TypeError, ValueError, PermissionError)


class UpstreamDataMissingError(FileNotFoundError):
  """Raised when an upstream archive genuinely has no data published for a date."""


def resolve_zarr_target(target: str) -> tuple[str, bool]:
  """Classifies a Zarr target as either remote (cloud URI) or local path.

  Rules:
  1. ``file://<path>`` is unwrapped to a local filesystem path ``(path, False)``.
  2. Any other explicit URI scheme (``gs://``, ``s3://``, ``az://``, ...) is
     treated as a remote object-store URL ``(target, True)``.
  3. Any path without a URI scheme (POSIX paths, relative paths, ``~``, Windows
     drive paths, UNC paths) is treated as a local filesystem path
     ``(target, False)``. Bare relative paths are never implicitly rewritten to
     cloud buckets.

  Args:
    target: Target Zarr store path or URI.

  Returns:
    A ``(resolved_location, is_remote)`` tuple.

  Raises:
    ValueError: If ``target`` is empty or whitespace.
  """
  if not target or not target.strip():
    raise ValueError("Zarr target must be a non-empty string.")

  scheme_match = _URI_SCHEME_RE.match(target)
  if scheme_match:
    if scheme_match.group("scheme").lower() == "file":
      return target[len(_FILE_SCHEME_PREFIX) :], False
    return target, True

  return target, False


def is_remote_target(target: str) -> bool:
  """Returns whether ``target`` refers to a remote cloud store."""
  return resolve_zarr_target(target)[1]


def get_zarr_mapper(
    target_zarr_url: str, project: str | None = None
) -> tuple[str, bool, Any]:
  """Resolves a Zarr target into ``(full_url, is_remote, mapper)``."""
  full_url, is_remote = resolve_zarr_target(target_zarr_url)
  if is_remote:
    if full_url.startswith("gs://"):
      if gcsfs is None:
        raise ImportError(
            "gcsfs is required to access gs:// Zarr targets. Install gcsfs or "
            "provide a local filesystem path."
        )
      clean_path = full_url.removeprefix("gs://")
      fs_kwargs: dict[str, Any] = {}
      if project:
        fs_kwargs["project"] = project
      fs = gcsfs.GCSFileSystem(**fs_kwargs)
      return full_url, True, fs.get_mapper(clean_path)
    return full_url, True, fsspec.get_mapper(full_url)
  return full_url, False, full_url


def inspect_zarr_store(
    target_zarr_url: str,
    project: str | None = None,
    overwrite: bool = False,
) -> tuple[str, bool, bool, bool, Any]:
  """Inspects target Zarr store existence and handles ``overwrite=True``.

  Does not swallow storage or authentication exceptions, ensuring an existing
  store is never accidentally treated as non-existent and overwritten.

  Args:
    target_zarr_url: Destination Zarr store URI or local path.
    project: Optional cloud project for GCS billing/authentication.
    overwrite: If ``True``, deletes the existing store before building.

  Returns:
    Tuple of ``(full_url, is_remote, store_exists, has_consolidated, mapper)``.
  """
  full_url, is_remote, mapper = get_zarr_mapper(target_zarr_url, project)
  if is_remote:
    if full_url.startswith("gs://"):
      clean_path = full_url.removeprefix("gs://")
      fs_kwargs: dict[str, Any] = {}
      if project:
        fs_kwargs["project"] = project
      fs = gcsfs.GCSFileSystem(**fs_kwargs)
      has_consolidated = bool(fs.exists(f"{clean_path}/.zmetadata"))
      store_exists = has_consolidated or bool(
          fs.exists(f"{clean_path}/zarr.json")
      )
      if store_exists and overwrite:
        _logger.info("Overwriting existing remote store at %s...", full_url)
        fs.rm(clean_path, recursive=True)
        store_exists = False
        has_consolidated = False
    else:
      has_consolidated = ".zmetadata" in mapper
      store_exists = has_consolidated or "zarr.json" in mapper
      if store_exists and overwrite:
        _logger.info("Overwriting existing remote store at %s...", full_url)
        mapper.clear()
        store_exists = False
        has_consolidated = False
  else:
    has_consolidated = os.path.exists(os.path.join(full_url, ".zmetadata"))
    store_exists = has_consolidated or os.path.exists(
        os.path.join(full_url, "zarr.json")
    )
    if store_exists and overwrite:
      _logger.info("Overwriting existing local store at %s...", full_url)
      if os.path.exists(full_url):
        shutil.rmtree(full_url)
      store_exists = False
      has_consolidated = False

  return full_url, is_remote, store_exists, has_consolidated, mapper


def decode_zarr_time_index(mapper: Any) -> pd.DatetimeIndex:  # noqa: ANN401
  """Reads and decodes the ``time`` coordinate directly from a Zarr group."""
  root = zarr.open_group(mapper, mode="r")
  raw_time = root["time"][:]
  attrs = dict(root["time"].attrs)
  units = attrs.get("units", "")
  if isinstance(units, str) and "days since" in units:
    origin = units.split("days since")[-1].strip().split()[0]
    return pd.to_datetime(raw_time, unit="D", origin=origin)
  return pd.to_datetime(raw_time)


def _is_slice_all_nan(ds: xr.Dataset, time_idx: int) -> bool:
  """Checks whether all data variables at ``time_idx`` are entirely NaN."""
  sub = ds.isel(time=time_idx)
  for var in ds.data_vars:
    vals = np.asarray(sub[var].values)
    if np.isfinite(vals).any():
      return False
  return True


def plan_archive_resume(
    mapper: Any,  # noqa: ANN401
    requested_dates: pd.DatetimeIndex,
    has_consolidated: bool = False,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, dict[str, int]]:
  """Plans in-place retries and new date appends when resuming an existing store.

  Inspects the existing store for:
  1. Dates recorded in ``attrs["failed_dates"]`` or ``attrs["missing_dates"]``.
  2. Any trailing all-NaN slices at the end of the existing store (for example,
     if a store was previously initialized with dates beyond the latest
     available upstream publication).

  Any such incomplete dates that fall within ``requested_dates`` are scheduled
  for in-place backfill (`in_place_dates`), and any requested dates after the
  store's maximum timestamp are scheduled for append (`append_dates`).

  Args:
    mapper: Zarr store path or fsspec mapper.
    requested_dates: Full sequence of dates requested by the caller.
    has_consolidated: Whether consolidated metadata is present.

  Returns:
    Tuple of ``(in_place_dates, append_dates, date_to_idx)``.
  """
  root = zarr.open_group(mapper, mode="r")
  recorded_missing = set(root.attrs.get("missing_dates", []))
  recorded_failed = set(root.attrs.get("failed_dates", []))
  needs_retry = recorded_missing | recorded_failed

  with xr.open_zarr(mapper, consolidated=has_consolidated) as existing_ds:
    existing_times = pd.to_datetime(existing_ds["time"].values)
    date_to_idx = {
        t.strftime("%Y-%m-%d"): idx for idx, t in enumerate(existing_times)
    }

    # Walk backwards from the end of the store to detect any trailing all-NaN
    # slices so that an over-extended time axis can self-heal on update runs.
    for idx in range(len(existing_times) - 1, -1, -1):
      d_str = existing_times[idx].strftime("%Y-%m-%d")
      if d_str in needs_retry or _is_slice_all_nan(existing_ds, idx):
        needs_retry.add(d_str)
      else:
        break

  max_existing_time = pd.Timestamp(existing_times.max())
  in_place_list: list[pd.Timestamp] = []
  append_list: list[pd.Timestamp] = []

  for dt in requested_dates:
    d_str = dt.strftime("%Y-%m-%d")
    if d_str in date_to_idx:
      if d_str in needs_retry:
        in_place_list.append(dt)
    elif dt > max_existing_time:
      append_list.append(dt)

  return (
      pd.DatetimeIndex(in_place_list),
      pd.DatetimeIndex(append_list),
      date_to_idx,
  )


def update_store_tracking_attrs(
    mapper: Any,  # noqa: ANN401
    *,
    newly_valid_dates: Sequence[str] = (),
    newly_missing_dates: Sequence[str] = (),
    newly_failed_dates: Sequence[str] = (),
) -> None:
  """Updates ``missing_dates`` and ``failed_dates`` in the Zarr root attrs."""
  root = zarr.open_group(mapper, mode="r+")
  missing = set(root.attrs.get("missing_dates", []))
  failed = set(root.attrs.get("failed_dates", []))

  valid_set = set(newly_valid_dates)
  missing = (missing - valid_set) | set(newly_missing_dates)
  failed = (failed - valid_set) | set(newly_failed_dates)

  root.attrs["missing_dates"] = sorted(missing)
  root.attrs["failed_dates"] = sorted(failed)


def write_failure_log(
    failure_log_path: str | None,
    *,
    failed_dates: Sequence[str],
    missing_dates: Sequence[str],
) -> None:
  """Writes a structured JSON log of failed and upstream-missing dates."""
  if not failure_log_path:
    return
  parent = os.path.dirname(os.path.abspath(failure_log_path))
  if parent:
    os.makedirs(parent, exist_ok=True)
  payload = {
      "failed_dates": sorted(set(failed_dates)),
      "missing_dates": sorted(set(missing_dates)),
  }
  with open(failure_log_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)


def write_dataset_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    *,
    project: str | None = None,
    is_initial_write: bool = False,
    consolidated: bool = True,
    time_chunk_size: int = 1,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch ``xr.Dataset`` to a Zarr store with backoff."""
  for attempt in range(max_retries):
    try:
      full_url, _, mapper = get_zarr_mapper(target_zarr_url, project)
      if is_initial_write:
        _logger.info("Writing initial Zarr schema to %s...", full_url)
        encoding = {
            var: {
                "chunks": (time_chunk_size,)
                + tuple(
                    len(ds_batch[dim])
                    for dim in ds_batch[var].dims
                    if dim != "time"
                )
            }
            for var in ds_batch.data_vars
        }
        ds_batch.to_zarr(
            mapper, mode="w", consolidated=consolidated, encoding=encoding
        )
      else:
        _logger.info(
            "Appending %d dates along time dimension...", len(ds_batch["time"])
        )
        ds_batch.to_zarr(
            mapper, mode="a", append_dim="time", consolidated=consolidated
        )
      return
    except NON_RETRYABLE_ERRORS:
      raise
    except Exception as err:  # noqa: BLE001
      wait_secs = 5 * (2**attempt)
      _logger.warning(
          "Error writing batch to Zarr (attempt %d/%d): %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          err,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      time.sleep(wait_secs)


def write_dataset_batch_in_place(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    *,
    project: str | None = None,
    date_to_idx: dict[str, int] | None = None,
    max_retries: int = 5,
) -> None:
  """Writes a batch of dates directly in-place into existing Zarr slices."""
  for attempt in range(max_retries):
    try:
      _, _, mapper = get_zarr_mapper(target_zarr_url, project)
      if date_to_idx is None:
        time_pd = decode_zarr_time_index(mapper)
        date_to_idx = {
            t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)
        }

      batch_times = pd.to_datetime(ds_batch["time"].values)
      missing_dates = [
          t.strftime("%Y-%m-%d")
          for t in batch_times
          if t.strftime("%Y-%m-%d") not in date_to_idx
      ]
      if missing_dates:
        raise ValueError(
            "Dates not found in target store for in-place write: "
            f"{missing_dates}"
        )

      indices = [date_to_idx[t.strftime("%Y-%m-%d")] for t in batch_times]
      root = zarr.open_group(mapper, mode="r+")
      is_contiguous = indices == list(
          range(indices[0], indices[0] + len(indices))
      )

      for var in ds_batch.data_vars:
        vals = ds_batch[var].values
        if is_contiguous:
          root[var][indices[0] : indices[-1] + 1] = vals
        else:
          for i, target_idx in enumerate(indices):
            root[var][target_idx : target_idx + 1] = vals[i : i + 1]
      return
    except NON_RETRYABLE_ERRORS:
      raise
    except Exception as err:  # noqa: BLE001
      wait_secs = 5 * (2**attempt)
      _logger.warning(
          "Error writing batch in-place (attempt %d/%d): %s. Retrying in "
          "%ds...",
          attempt + 1,
          max_retries,
          err,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      time.sleep(wait_secs)
