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

"""Builds or updates an HRES daily forecast archive from ECMWF Open Data.

Ingests the public ECMWF Open Data stream (IFS HRES, 00 UTC run, 0.25 degree,
``ifs/0p25/oper``) and writes daily aggregated surface variables for lead days
1..10. The grid, step schedule, variables, units, and daily aggregation rules
are defined in :mod:`multimet.hres_schema`.

The earliest supported run date is ``2024-03-06``, the first date on which
ECMWF Open Data provides all required surface variables at 0.25 degree. Earlier
years (2016–2024) are built separately using the same schema and published as a
pre-built archive.

Strict missing-data policy: an unpublished or partially published run date is
never filled from another source, run hour, or resolution. Unpublished dates at
the end of the requested range are omitted so the archive never ends with
trailing ``NaN`` slices; unpublished dates inside the range are written as
``NaN`` and recorded in ``missing_dates`` so subsequent runs can backfill them.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing as mp
import os
import time
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import zarr

try:
  import eccodes  # type: ignore[import-untyped]
except ImportError:
  eccodes = None

try:
  import gcsfs  # type: ignore[import-untyped]
except ImportError:
  gcsfs = None

from multimet import hres_schema
from multimet import storage

ECMWF_OPEN_DATA_BUCKET = "ecmwf-open-data"
OPEN_DATA_SOURCE_NAME = "ECMWF Open Data, IFS HRES ifs/0p25/oper, 00 UTC run"

# First run date on which ifs/0p25/oper includes all required surface variables.
MIN_START_DATE = pd.Timestamp("2024-03-06")

NUM_LEAD_DAYS = hres_schema.NUM_LEAD_DAYS
HRES_VARIABLES = hres_schema.VARIABLES
HRES_ATTRS = hres_schema.GLOBAL_ATTRS

# Grid written to the archive. Tests override these with a smaller grid.
HRES_LATS = hres_schema.LATITUDES
HRES_LONS = hres_schema.LONGITUDES

DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 4)

_GEOMETRY_KEYS = {
    "ni": "Ni",
    "nj": "Nj",
    "lat_first": "latitudeOfFirstGridPointInDegrees",
    "lat_last": "latitudeOfLastGridPointInDegrees",
    "lon_first": "longitudeOfFirstGridPointInDegrees",
    "lon_last": "longitudeOfLastGridPointInDegrees",
    "i_scans_negatively": "iScansNegatively",
    "j_scans_positively": "jScansPositively",
    "j_points_are_consecutive": "jPointsAreConsecutive",
}


def decode_grib_message(
    message: bytes, *, param: str, step: int, date: pd.Timestamp
) -> np.ndarray:
  """Decodes a single GRIB2 message and reorients it onto the schema grid.

  Verifies that the message headers match ``param``, ``step``, and ``date`` (00
  UTC run) and extracts the grid geometry directly from the message headers.

  Args:
    message: Raw bytes of a single GRIB2 message.
    param: Expected ECMWF ``shortName`` (e.g., ``"2t"``).
    step: Expected forecast step in hours.
    date: Expected forecast initialization date.

  Returns:
    2-D array of shape ``(721, 1440)`` in raw GRIB units.
  """
  if eccodes is None:
    raise ImportError(
        "eccodes is required to decode ECMWF Open Data. Install the"
        " 'eccodes' Python package."
    )
  gid = eccodes.codes_new_from_message(message)
  try:
    short_name = eccodes.codes_get(gid, "shortName")
    end_step = int(eccodes.codes_get(gid, "endStep"))
    data_date = str(eccodes.codes_get(gid, "dataDate"))
    data_time = int(eccodes.codes_get(gid, "dataTime"))
    bitmap = int(eccodes.codes_get(gid, "bitmapPresent"))
    geometry = hres_schema.GribGeometry(
        **{
            field: eccodes.codes_get(gid, key)
            for field, key in _GEOMETRY_KEYS.items()
        }
    )
    values = eccodes.codes_get_values(gid)
  finally:
    eccodes.codes_release(gid)

  expected = (param, step, date.strftime("%Y%m%d"), 0)
  found = (short_name, end_step, data_date, data_time)
  if found != expected:
    raise ValueError(
        "GRIB message is not the one requested: expected"
        f" (param, step, date, time) = {expected}, found {found}."
    )
  if bitmap:
    raise ValueError(
        f"GRIB message {param} +{step}h on {data_date} has missing points"
        " (bitmap). This is not expected for a global field."
    )
  return hres_schema.reorient_to_schema_grid(values, geometry)


class ECMWFOpenDataSource:
  """Reads one run date from the public ECMWF Open Data bucket on GCS."""

  def __init__(self, bucket: str = ECMWF_OPEN_DATA_BUCKET):
    if gcsfs is None:
      raise ImportError("gcsfs is required to read ECMWF Open Data from GCS.")
    self.bucket = bucket.removeprefix("gs://").strip("/")
    self.fs = gcsfs.GCSFileSystem(token="anon")

  def _run_prefix(self, date: pd.Timestamp) -> str:
    stamp = date.strftime("%Y%m%d")
    return f"{self.bucket}/{stamp}/00z/ifs/0p25/oper/{stamp}000000"

  def _read_index(self, path: str) -> dict[str, tuple[int, int]]:
    """Returns ``{param: (offset, length)}`` for the surface params we need."""
    offsets: dict[str, tuple[int, int]] = {}
    with self.fs.open(path, "r") as f:
      for line in f:
        entry = json.loads(line)
        param = entry.get("param")
        if (
            entry.get("levtype") != "sfc"
            or param not in hres_schema.GRIB_PARAMS
        ):
          continue
        if param in offsets:
          raise ValueError(f"{path} lists {param} more than once.")
        offsets[param] = (int(entry["_offset"]), int(entry["_length"]))
    return offsets

  def extract_date(self, date: pd.Timestamp) -> dict[str, np.ndarray]:
    """Returns the daily archive variables for one run date.

    Raises:
      storage.UpstreamDataMissingError: If any forecast step of this run is not
        published (yet).
      ValueError: If a published file is missing a variable, holds an
        unexpected message, or has an unexpected grid.
    """
    prefix = self._run_prefix(date)
    folder = prefix.rsplit("/", 1)[0]
    try:
      published = set(self.fs.ls(folder, detail=False, refresh=True))
    except FileNotFoundError as err:
      raise storage.UpstreamDataMissingError(
          f"No ECMWF Open Data run published at gs://{folder}"
      ) from err
    missing_steps = [
        step
        for step in hres_schema.FORECAST_STEPS
        if f"{prefix}-{step}h-oper-fc.index" not in published
    ]
    if missing_steps:
      raise storage.UpstreamDataMissingError(
          f"Run {date:%Y-%m-%d} is not complete in gs://{folder}: steps"
          f" {missing_steps} are not published."
      )

    needed: dict[int, list[str]] = {}
    for param, step in hres_schema.required_fields():
      needed.setdefault(step, []).append(param)

    aggregator = hres_schema.DailyAggregator(
        (len(hres_schema.LATITUDES), len(hres_schema.LONGITUDES))
    )
    for step in sorted(needed):
      params = sorted(needed[step])
      index_path = f"{prefix}-{step}h-oper-fc.index"
      offsets = self._read_index(index_path)
      absent = [p for p in params if p not in offsets]
      if absent:
        raise ValueError(f"{index_path} does not list {absent}.")
      grib_path = f"{prefix}-{step}h-oper-fc.grib2"
      messages = self.fs.cat_ranges(
          [grib_path] * len(params),
          [offsets[p][0] for p in params],
          [offsets[p][0] + offsets[p][1] for p in params],
      )
      for param, message in zip(params, messages, strict=True):
        if isinstance(message, Exception):
          raise message
        field = decode_grib_message(message, param=param, step=step, date=date)
        aggregator.add(param, step, field)
    return aggregator.finalize()


_worker_source: ECMWFOpenDataSource | None = None


def _init_worker(ecmwf_open_data_bucket: str = ECMWF_OPEN_DATA_BUCKET) -> None:
  """Creates the Open Data client in a worker process."""
  global _worker_source  # noqa: PLW0603
  _worker_source = ECMWFOpenDataSource(ecmwf_open_data_bucket)


def _make_nan_date_payload() -> dict[str, np.ndarray]:
  """Creates an all-NaN payload for a date with no published data."""
  shape = (NUM_LEAD_DAYS, len(HRES_LATS), len(HRES_LONS))
  return {
      var: np.full(shape, np.nan, dtype=np.float32) for var in HRES_VARIABLES
  }


def _extract_single_date(
    dt: pd.Timestamp,
    max_retries: int = 3,
) -> tuple[pd.Timestamp, dict[str, np.ndarray] | None]:
  """Extracts one run date, retrying network errors.

  Returns ``(dt, None)`` when the run is not (fully) published. Any other
  error that is still there after ``max_retries`` attempts is raised, so bad
  data is never written as NaN. Errors that cannot change on retry (bad grid,
  missing variable) are raised at once.
  """
  if _worker_source is None:
    raise RuntimeError("ECMWFOpenDataSource was not initialized.")
  last_err: Exception | None = None
  for attempt in range(max_retries):
    try:
      return dt, _worker_source.extract_date(dt)
    except storage.UpstreamDataMissingError as err:
      logging.warning("%s", err)
      return dt, None
    except storage.NON_RETRYABLE_ERRORS:
      raise
    except Exception as err:  # noqa: BLE001
      last_err = err
      logging.warning(
          "Error reading %s (attempt %d/%d): %s",
          dt.strftime("%Y-%m-%d"),
          attempt + 1,
          max_retries,
          err,
      )
      if attempt < max_retries - 1:
        time.sleep(2 * (2**attempt))

  raise RuntimeError(
      f"Failed to read HRES data for {dt.strftime('%Y-%m-%d')} after"
      f" {max_retries} attempts: {last_err}"
  ) from last_err


def build_batch_dataset(
    batch_dates: Sequence[pd.Timestamp],
    batch_data: dict[str, Sequence[np.ndarray]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> xr.Dataset:
  """Assembles one write batch in the archive schema."""
  return xr.Dataset(
      data_vars={
          var: (
              ["time", "lead_time", "latitude", "longitude"],
              np.stack(batch_data[var], axis=0),
              hres_schema.variable_attrs(var),
          )
          for var in HRES_VARIABLES
      },
      coords={
          "time": list(batch_dates),
          "lead_time": hres_schema.LEAD_TIMES,
          "latitude": latitudes,
          "longitude": longitudes,
      },
      attrs=dict(HRES_ATTRS),
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    is_initial_write: bool = False,
    consolidated: bool = False,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store."""
  storage.write_dataset_batch_to_zarr(
      ds_batch,
      target_zarr_url,
      project=project,
      is_initial_write=is_initial_write,
      consolidated=consolidated,
      time_chunk_size=1,
      max_retries=max_retries,
  )


def write_batch_in_place(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    date_to_idx: dict[str, int] | None = None,
    max_retries: int = 5,
) -> None:
  """Overwrites existing dates in the target Zarr store."""
  storage.write_dataset_batch_in_place(
      ds_batch,
      target_zarr_url,
      project=project,
      date_to_idx=date_to_idx,
      max_retries=max_retries,
  )


def check_target_schema(mapper: object, has_consolidated: bool) -> None:
  """Verifies that an existing Zarr archive matches the HRES archive schema."""
  with xr.open_zarr(mapper, consolidated=has_consolidated) as existing:
    problems = hres_schema.find_schema_problems(existing, HRES_LATS, HRES_LONS)
  if problems:
    raise ValueError(
        "The target archive does not match the HRES archive schema"
        f" (version {hres_schema.SCHEMA_VERSION}), so new dates cannot be"
        " added to it:\n  - "
        + "\n  - ".join(problems)
        + "\nUse a matching archive, or a new --target_zarr path."
    )


def default_start_date(mapper: object, has_consolidated: bool) -> pd.Timestamp:
  """Determines the default resume start date for an existing archive.

  Returns the day after the last date in the archive, or the earliest date
  ``>= MIN_START_DATE`` recorded in ``missing_dates`` / ``failed_dates`` if
  earlier.
  """
  with xr.open_zarr(mapper, consolidated=has_consolidated) as existing:
    start = pd.Timestamp(pd.to_datetime(existing["time"].values).max())
    start += pd.Timedelta(days=1)
    recorded = list(existing.attrs.get("missing_dates", []))
    recorded += list(existing.attrs.get("failed_dates", []))
  fillable = [
      pd.Timestamp(d) for d in recorded if pd.Timestamp(d) >= MIN_START_DATE
  ]
  return min([start, *fillable])


def resolve_date_range(
    start_date: str | None,
    end_date: str | None,
    *,
    default_start: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
  """Resolves and validates the ``(start, end)`` forecast date range.

  If ``start_date`` is omitted, ``default_start`` is used (see
  :func:`default_start_date`). Creating a new archive requires an explicit
  ``start_date``.
  """
  if start_date is None:
    if default_start is None:
      raise ValueError(
          "--start_date is required when the target archive does not exist."
      )
    start = default_start
  else:
    start = pd.Timestamp(start_date)
  end = pd.Timestamp(end_date or datetime.date.today().isoformat())
  if start < MIN_START_DATE:
    raise ValueError(
        f"ECMWF Open Data has all HRES variables at 0.25 degree only from"
        f" {MIN_START_DATE:%Y-%m-%d}. Got --start_date {start:%Y-%m-%d}. For"
        " older dates, use the published Open-MultiMet HRES archive."
    )
  return start, end


def _update_source_records(mapper: object, dates: Sequence[str]) -> None:
  if not dates:
    return
  root = zarr.open_group(mapper, mode="r+")
  root.attrs[hres_schema.SOURCE_ATTR] = hres_schema.update_source_records(
      root.attrs.get(hres_schema.SOURCE_ATTR, []),
      OPEN_DATA_SOURCE_NAME,
      dates,
  )


def build_hres_archive(
    target_zarr: str,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    project: str | None = None,
    ecmwf_open_data_bucket: str = ECMWF_OPEN_DATA_BUCKET,
    batch_size: int = 10,
    overwrite: bool = False,
    in_place: bool = False,
    num_workers: int | None = None,
    failure_log: str | None = None,
) -> None:
  """Builds or updates an HRES daily forecast archive from ECMWF Open Data.

  Args:
    target_zarr: Destination Zarr path (local directory or ``gs://`` URI).
    start_date: First forecast date (``YYYY-MM-DD``). If omitted when updating
      an existing archive, resumes from the day after the last date in the
      archive (or the earliest fillable missing date).
    end_date: Last forecast date (``YYYY-MM-DD``). Defaults to today.
    project: Optional GCP project ID for writing to a ``gs://`` target.
    ecmwf_open_data_bucket: GCS bucket hosting ECMWF Open Data.
    batch_size: Number of forecast dates accumulated per Zarr write.
    overwrite: If True, deletes the existing target archive and rebuilds from
      ``start_date``.
    in_place: If True, overwrites ``start_date .. end_date`` in-place inside an
      existing archive.
    num_workers: Number of parallel download and decode worker processes.
    failure_log: Optional file path for a JSON report of unpublished dates.
  """
  full_target_url, _, store_exists, has_consolidated, mapper = (
      storage.inspect_zarr_store(
          target_zarr, project=project, overwrite=overwrite
      )
  )
  if store_exists:
    check_target_schema(mapper, has_consolidated)

  start, end = resolve_date_range(
      start_date,
      end_date,
      default_start=(
          default_start_date(mapper, has_consolidated) if store_exists else None
      ),
  )
  if start > end:
    logging.info("Archive is already up to date (last date %s).", end.date())
    return

  if num_workers is None or num_workers <= 0:
    num_workers = DEFAULT_NUM_WORKERS

  requested_dates = pd.date_range(start, end, freq="1D")
  date_to_idx: dict[str, int] = {}
  in_place_dates = pd.DatetimeIndex([])
  append_dates = requested_dates

  if in_place:
    if not store_exists:
      raise ValueError(
          f"Cannot run --in_place: target store {full_target_url} does not"
          " exist."
      )
    with xr.open_zarr(mapper, consolidated=has_consolidated) as existing_ds:
      time_pd = pd.to_datetime(existing_ds["time"].values)
      date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)}
    in_place_dates = requested_dates
    append_dates = pd.DatetimeIndex([])
  elif store_exists:
    in_place_dates, append_dates, date_to_idx = storage.plan_archive_resume(
        mapper, requested_dates, has_consolidated=has_consolidated
    )
    if len(in_place_dates) == 0 and len(append_dates) == 0:
      logging.info("Store already holds every date up to %s.", end.date())
      return

  is_first_write = not store_exists
  newly_valid_dates: list[str] = []
  newly_missing_dates: list[str] = []

  def _run_extraction(
      dates_to_run: pd.DatetimeIndex,
  ) -> Iterator[tuple[pd.Timestamp, dict[str, np.ndarray] | None]]:
    """Yields extracted dates in order for streaming batch writes."""
    if len(dates_to_run) == 0:
      return
    if num_workers > 1 and len(dates_to_run) > 1:
      mp_ctx = mp.get_context("spawn")
      with mp_ctx.Pool(
          processes=min(num_workers, len(dates_to_run)),
          initializer=_init_worker,
          initargs=(ecmwf_open_data_bucket,),
      ) as pool:
        yield from tqdm.tqdm(
            pool.imap(_extract_single_date, dates_to_run, chunksize=1),
            total=len(dates_to_run),
            desc=f"Processing HRES ({num_workers} workers)",
        )
      return
    _init_worker(ecmwf_open_data_bucket)
    for dt in tqdm.tqdm(dates_to_run, desc="Processing HRES (sequential)"):
      yield _extract_single_date(dt)

  try:
    # 1. Overwrite existing dates (--in_place or backfilling previously missing
    # interior dates).
    if len(in_place_dates) > 0:
      ip_dates: list[pd.Timestamp] = []
      ip_data: dict[str, list[np.ndarray]] = {var: [] for var in HRES_VARIABLES}

      for dt, date_data in _run_extraction(in_place_dates):
        d_str = dt.strftime("%Y-%m-%d")
        if date_data is None:
          newly_missing_dates.append(d_str)
          if not in_place:
            continue
          date_data = _make_nan_date_payload()
        else:
          newly_valid_dates.append(d_str)

        ip_dates.append(dt)
        for k, k_list in ip_data.items():
          k_list.append(date_data[k])
        if len(ip_dates) >= batch_size:
          write_batch_in_place(
              build_batch_dataset(ip_dates, ip_data, HRES_LATS, HRES_LONS),
              full_target_url,
              project=project,
              date_to_idx=date_to_idx,
          )
          ip_dates = []
          ip_data = {var: [] for var in HRES_VARIABLES}

      if ip_dates:
        write_batch_in_place(
            build_batch_dataset(ip_dates, ip_data, HRES_LATS, HRES_LONS),
            full_target_url,
            project=project,
            date_to_idx=date_to_idx,
        )

    # 2. Append new dates after the end of the existing store. Unpublished
    # dates are buffered and written as NaN only if a subsequent date has valid
    # data, ensuring the time axis never ends with trailing NaN slices.
    if len(append_dates) > 0:
      batch_dates: list[pd.Timestamp] = []
      batch_data: dict[str, list[np.ndarray]] = {
          var: [] for var in HRES_VARIABLES
      }
      pending_missing: list[pd.Timestamp] = []

      def _flush_append_batch() -> None:
        nonlocal batch_dates, batch_data, is_first_write
        if not batch_dates:
          return
        write_batch_to_zarr(
            build_batch_dataset(batch_dates, batch_data, HRES_LATS, HRES_LONS),
            full_target_url,
            project=project,
            is_initial_write=is_first_write,
            consolidated=has_consolidated,
        )
        is_first_write = False
        batch_dates = []
        batch_data = {var: [] for var in HRES_VARIABLES}

      def _add_to_batch(dt: pd.Timestamp, data: dict[str, np.ndarray]) -> None:
        batch_dates.append(dt)
        for k, k_list in batch_data.items():
          k_list.append(data[k])
        if len(batch_dates) >= batch_size:
          _flush_append_batch()

      for dt, date_data in _run_extraction(append_dates):
        if date_data is None:
          pending_missing.append(dt)
          continue
        for m_dt in pending_missing:
          newly_missing_dates.append(m_dt.strftime("%Y-%m-%d"))
          _add_to_batch(m_dt, _make_nan_date_payload())
        pending_missing.clear()
        newly_valid_dates.append(dt.strftime("%Y-%m-%d"))
        _add_to_batch(dt, date_data)

      if pending_missing:
        logging.warning(
            "Skipping %d unpublished trailing dates (%s to %s); re-run later"
            " once published.",
            len(pending_missing),
            pending_missing[0].strftime("%Y-%m-%d"),
            pending_missing[-1].strftime("%Y-%m-%d"),
        )

      _flush_append_batch()

  finally:
    # Persist tracking and provenance attributes for any batches already
    # committed, even if an exception interrupted a later batch.
    if not is_first_write:
      storage.update_store_tracking_attrs(
          mapper,
          newly_valid_dates=newly_valid_dates,
          newly_missing_dates=newly_missing_dates,
      )
      _update_source_records(mapper, newly_valid_dates)
    storage.write_failure_log(
        failure_log,
        failed_dates=[],
        missing_dates=newly_missing_dates,
    )

  logging.info(
      "HRES archive update complete for %s to %s.", start.date(), end.date()
  )


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line argument parser for ``build-hres-archive``."""
  parser = argparse.ArgumentParser(
      prog="build-hres-archive",
      description=(
          "Build or update an ECMWF IFS HRES daily surface forecast archive"
          f" from public ECMWF Open Data (run dates from"
          f" {MIN_START_DATE:%Y-%m-%d})."
      ),
  )
  parser.add_argument(
      "--target_zarr",
      type=str,
      required=True,
      help="Destination Zarr archive path (local directory or gs:// URI).",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=None,
      help=(
          "First forecast date (YYYY-MM-DD), on or after"
          f" {MIN_START_DATE:%Y-%m-%d}. Defaults to the day after the last"
          " date in the archive when updating. Required for a new archive."
      ),
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=None,
      help="Last forecast date (YYYY-MM-DD). Defaults to today.",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="Optional GCP project ID used when writing to a gs:// target.",
  )
  parser.add_argument(
      "--ecmwf_open_data_bucket",
      type=str,
      default=ECMWF_OPEN_DATA_BUCKET,
      help="GCS bucket hosting ECMWF Open Data (default: ecmwf-open-data).",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=10,
      help="Number of forecast dates accumulated per Zarr write.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Delete the existing target archive and rebuild from --start_date.",
  )
  parser.add_argument(
      "--in_place",
      action="store_true",
      help=(
          "Overwrite --start_date to --end_date in-place in an existing"
          " archive."
      ),
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=DEFAULT_NUM_WORKERS,
      help="Number of forecast dates downloaded and decoded in parallel.",
  )
  parser.add_argument(
      "--failure_log",
      type=str,
      default=None,
      help="Optional path to write a JSON report of unpublished dates.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-hres-archive``."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  build_hres_archive(
      target_zarr=args.target_zarr,
      start_date=args.start_date,
      end_date=args.end_date,
      project=args.project,
      ecmwf_open_data_bucket=args.ecmwf_open_data_bucket,
      batch_size=args.batch_size,
      overwrite=args.overwrite,
      in_place=args.in_place,
      num_workers=args.num_workers,
      failure_log=args.failure_log,
  )


if __name__ == "__main__":
  main()
