# Copyright 2025 Google LLC
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

"""Thin NetCDF directory loader for the GoogleHydrology model pipeline.

Supports two explicit, non-recursive NetCDF directory layouts:

1. **Multi-basin files:** One or more ``.nc`` / ``.nc4`` files directly in the
   given directory that each contain a ``basin`` dimension (e.g., a single
   combined file, non-overlapping time-split files, or basin-batch files).
   Files are lazily opened and combined across time and/or disjoint basin
   groups.
2. **Per-basin files (``<basin_id>.nc`` / ``<basin_id>.nc4``):** One file per
   basin directly in the given directory, without a ``basin`` dimension inside
   each file (an optional scalar ``basin`` coordinate is ignored). Only files
   matching the requested basins are opened.

Numeric and boolean data variables (e.g., ``float64``, ``int64``, ``bool``) are
lazily cast to ``float32`` on read to satisfy ``Multimet``'s ``float32``
invariant and avoid integer overflow or dtype promotion in ``Scaler``.
"""

import itertools
import logging
import math
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from xarray.coders import CFTimedeltaCoder

from googlehydrology.utils.configutils import normalize_product_key
from googlehydrology.utils.tqdm import AutoRefreshTqdm

__all__ = [
    'NETCDF_SUFFIXES',
    'find_product_netcdf_dir',
    'has_netcdf_files',
    'list_netcdf_files',
    'load_netcdf_attributes',
    'load_netcdf_timeseries',
]

LOGGER = logging.getLogger(__name__)

NETCDF_SUFFIXES = ('.nc', '.nc4')
_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'
_ONE_DAY = pd.Timedelta(days=1)


def _is_cloud_path(path: Path | str) -> bool:
    normalized = str(path).replace('\\', '/')
    return normalized.startswith(('gs://', 'gs:/'))


def _is_netcdf_filename(name: str) -> bool:
    if name.startswith('.'):
        return False
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in NETCDF_SUFFIXES)


def _netcdf_engine_for_path(path: Path) -> str:
    """Pick an xarray engine from the file magic header.

    Uses ``h5netcdf`` for HDF5 / NetCDF4 files (avoiding ``libnetcdf``
    reopen/threading crashes under Dask) and ``netcdf4`` for NetCDF3 classic
    and CDF-5 files (lazy reads without ``scipy``'s eager ``mmap=False`` RAM
    overhead or ``mmap=True`` cache-eviction warnings).
    """
    with path.open('rb') as handle:
        magic = handle.read(8)
    if magic.startswith(_HDF5_MAGIC):
        return 'h5netcdf'
    return 'netcdf4'


def _open_dataset_raw(
    path: Path, chunks: str | dict[str, str | int]
) -> xr.Dataset:
    """Open one NetCDF file lazily without global timedelta auto-decoding.

    Using ``decode_timedelta=False`` prevents non-``lead_time`` variables that
    happen to carry CF time units (e.g. ``units='days'``) from turning into
    ``timedelta64`` arrays. Only ``lead_time`` is decoded explicitly afterward.
    """
    engine = _netcdf_engine_for_path(path)
    try:
        return xr.open_dataset(
            path,
            engine=engine,
            chunks=chunks,
            decode_timedelta=False,
        )
    except (OSError, ValueError, KeyError) as primary_err:
        if engine != 'h5netcdf':
            raise
        LOGGER.debug(
            'h5netcdf failed to open %s (%s); falling back to netcdf4 engine.',
            path,
            primary_err,
        )
        try:
            return xr.open_dataset(
                path,
                engine='netcdf4',
                chunks=chunks,
                decode_timedelta=False,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as fallback_err:
            raise OSError(
                f'Failed to open {path} with h5netcdf ({primary_err}) and '
                f'netcdf4 ({fallback_err}).'
            ) from primary_err


def _decode_lead_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """Decode only the ``lead_time`` coordinate using CF duration units."""
    if 'lead_time' not in ds.coords:
        return ds
    coord = ds.coords['lead_time']
    if np.issubdtype(coord.dtype, np.timedelta64):
        return ds
    decoded = CFTimedeltaCoder().decode(
        ds.variables['lead_time'], name='lead_time'
    )
    if np.issubdtype(decoded.dtype, np.timedelta64):
        ds = ds.assign_coords(
            lead_time=(decoded.dims, np.asarray(decoded.data))
        )
    return ds


def _make_closer(handles: Sequence[xr.Dataset]) -> Callable[[], None]:
    """Return a close callback that closes every underlying raw handle."""
    raw_handles = list(handles)

    def _close() -> None:
        for handle in raw_handles:
            handle.close()

    return _close


def _close_all(handles: Sequence[xr.Dataset]) -> None:
    for handle in handles:
        handle.close()


def list_netcdf_files(directory: Path | str) -> list[Path]:
    """Return sorted visible ``.nc`` / ``.nc4`` files directly in dir."""
    if _is_cloud_path(directory):
        return []
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(
        Path(entry.path)
        for entry in os.scandir(p)
        if entry.is_file() and _is_netcdf_filename(entry.name)
    )


def has_netcdf_files(directory: Path | str) -> bool:
    """Return True if ``directory`` directly holds visible NetCDF files."""
    if _is_cloud_path(directory):
        return False
    p = Path(directory)
    if not p.is_dir():
        return False
    with os.scandir(p) as entries:
        return any(
            entry.is_file() and _is_netcdf_filename(entry.name)
            for entry in entries
        )


def find_product_netcdf_dir(
    dynamics_path: Path | str, product: str
) -> Path | None:
    """Find a product subfolder under ``dynamics_path`` with NetCDF files."""
    if _is_cloud_path(dynamics_path):
        return None
    p = Path(dynamics_path)
    if not p.is_dir():
        return None

    direct = p / product
    if has_netcdf_files(direct):
        return direct

    product_norm = normalize_product_key(product)
    for sub in sorted(p.iterdir()):
        if (
            sub.is_dir()
            and normalize_product_key(sub.name) == product_norm
            and has_netcdf_files(sub)
        ):
            return sub
    return None


def _cast_numeric_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Lazily cast numeric/bool data_vars and floating coords to float32."""
    cast_vars = {
        name: da.astype(np.float32)
        for name, da in ds.data_vars.items()
        if (
            np.issubdtype(da.dtype, np.number)
            or np.issubdtype(da.dtype, np.bool_)
        )
        and da.dtype != np.float32
    }
    if cast_vars:
        ds = ds.assign(cast_vars)

    cast_coords = {
        name: coord.astype(np.float32)
        for name, coord in ds.coords.items()
        if np.issubdtype(coord.dtype, np.floating) and coord.dtype != np.float32
    }
    if cast_coords:
        ds = ds.assign_coords(cast_coords)
    return ds


def _normalize_basin_coord(ds: xr.Dataset, source: Path | str) -> xr.Dataset:
    """Decode byte-string ``basin`` coordinates and reject numeric IDs."""
    if 'basin' not in ds.coords:
        return ds
    coord = ds.coords['basin']
    if np.issubdtype(coord.dtype, np.number) or np.issubdtype(
        coord.dtype, np.bool_
    ):
        raise ValueError(
            f"'basin' coordinate in {source} must contain string basin IDs, "
            f'got numeric dtype {coord.dtype}.'
        )
    vals = coord.to_numpy()
    if np.issubdtype(coord.dtype, np.bytes_) or (
        coord.dtype == object
        and any(isinstance(b, (bytes, np.bytes_)) for b in vals)
    ):
        decoded = [
            b.decode('utf-8') if isinstance(b, (bytes, np.bytes_)) else str(b)
            for b in vals
        ]
        ds = ds.assign_coords(basin=decoded)
    return ds


def _build_per_basin_path_map(
    nc_files: Sequence[Path],
) -> dict[str, list[Path]]:
    stem_map: dict[str, list[Path]] = {}
    for path in nc_files:
        stem_map.setdefault(path.stem, []).append(path)
    return stem_map


def _detect_layout(
    nc_files: Sequence[Path],
    basins: Sequence[str] | None = None,
) -> Literal['multi_basin', 'per_basin']:
    """Inspect a representative file to classify multi-basin vs per-basin.

    When ``basins`` is supplied, prefers probing a file matching a requested
    basin stem so unrelated/corrupt files sorting earlier are never opened in
    per-basin directories.
    """
    probe = nc_files[0]
    if basins:
        stem_map = _build_per_basin_path_map(nc_files)
        for basin in basins:
            matches = stem_map.get(basin)
            if matches:
                probe = matches[0]
                break

    raw = _open_dataset_raw(probe, chunks={})
    try:
        if 'basin' in raw.dims:
            return 'multi_basin'
        if 'basin' in raw.coords and raw.coords['basin'].ndim > 0:
            raise ValueError(
                f"Non-dimension 'basin' coordinate with dims "
                f'{raw.coords["basin"].dims} in {probe}: multi-basin files '
                "must use 'basin' as a dimension, and per-basin files may "
                "only contain a scalar (0D) or absent 'basin' coordinate."
            )
        return 'per_basin'
    finally:
        raw.close()


def _validate_required_features(
    ds: xr.Dataset,
    features: Sequence[str],
    source: Path | str,
    *,
    include_coords: bool = False,
) -> None:
    available = set(ds.data_vars)
    if include_coords:
        available |= set(ds.coords)
    missing = sorted(set(features) - available)
    if missing:
        available_list = sorted(available)
        label = 'variables/coordinates' if include_coords else 'variables'
        raise ValueError(
            f'Requested features {missing} not found in {source}. '
            f'Available {label}: {available_list}'
        )


def _validate_basin_coord(
    ds: xr.Dataset, basins: Sequence[str] | None, source: Path | str
) -> None:
    if 'basin' not in ds.dims or 'basin' not in ds.coords:
        raise ValueError(
            f"Required 'basin' dimension/coordinate not found in {source}. "
            f'Found dims: {list(ds.dims)}.'
        )
    basin_index = pd.Index(ds.coords['basin'].to_numpy())
    if not basin_index.is_unique:
        dup = basin_index[basin_index.duplicated()][0]
        raise ValueError(
            f"Duplicate 'basin' coordinate value {dup!r} in {source}."
        )
    if basins is not None:
        if len(basins) == 0:
            raise ValueError('Basin list must be non-empty.')
        present = set(basin_index.tolist())
        missing = sorted(set(basins) - present)
        if missing:
            raise ValueError(
                f'{len(missing)} basins are missing in {source}: '
                f'{", ".join(missing)}'
            )


def _validate_date_coord(ds: xr.Dataset, source: Path | str) -> None:
    if 'date' not in ds.dims or 'date' not in ds.coords:
        raise ValueError(
            f"Required 'date' dimension/coordinate not found in {source}. "
            f'Found dims: {list(ds.dims)}.'
        )
    if not np.issubdtype(ds['date'].dtype, np.datetime64):
        raise ValueError(
            f"'date' coordinate in {source} must decode as datetime64, "
            f'got dtype {ds["date"].dtype}.'
        )
    if ds.sizes['date'] == 0:
        raise ValueError(f"'date' coordinate in {source} is empty.")

    idx = pd.DatetimeIndex(ds['date'].to_numpy())
    if idx.isna().any():
        raise ValueError(
            f"'date' coordinate in {source} contains missing (NaT) values."
        )
    if not idx.is_unique:
        dup = idx[idx.duplicated()][0]
        raise ValueError(
            f"'date' coordinate in {source} contains duplicate timestamp "
            f'{dup.isoformat()}.'
        )
    non_midnight = idx != idx.normalize()
    if non_midnight.any():
        bad_ts = idx[non_midnight][0]
        raise ValueError(
            f"'date' coordinate in {source} must contain daily midnight "
            f'(00:00:00) timestamps; found {bad_ts.isoformat()}.'
        )


@dataclass(frozen=True)
class _MultiBasinReadOpts:
    features: Sequence[str] | None = None
    include_coords: bool = False
    only_1d_basin_vars: bool = False
    validate_dates: bool = False


def _validate_lead_time_values(
    lt: pd.TimedeltaIndex, source: Path | str
) -> None:
    if lt.isna().any():
        raise ValueError(
            f"'lead_time' coordinate in {source} contains missing (NaT) values."
        )
    if not lt.is_unique:
        dup = lt[lt.duplicated()][0]
        raise ValueError(
            f"'lead_time' coordinate in {source} contains duplicate value "
            f'{dup}.'
        )
    if ((lt % _ONE_DAY) != pd.Timedelta(0)).any():
        bad_lt = lt[(lt % _ONE_DAY) != pd.Timedelta(0)][0]
        raise ValueError(
            f"'lead_time' coordinate in {source} must use whole-day multiples; "
            f'found {bad_lt}.'
        )
    if lt.min() < _ONE_DAY:
        raise ValueError(
            f"'lead_time' coordinate in {source} must start at >= 1 day; "
            f'found minimum {lt.min()}.'
        )
    if not lt.is_monotonic_increasing:
        raise ValueError(
            f"'lead_time' coordinate in {source} must be strictly increasing; "
            f'got {[int(td / _ONE_DAY) for td in lt]} days.'
        )
    if len(lt) > 1 and not (np.diff(lt) == _ONE_DAY).all():
        raise ValueError(
            f"'lead_time' coordinate in {source} must be contiguous daily "
            f'steps without gaps; got {[int(td / _ONE_DAY) for td in lt]} days.'
        )


def _validate_lead_time_coord(
    ds: xr.Dataset,
    source: Path | str,
    *,
    require_on_all_vars: bool = False,
) -> None:
    if 'lead_time' not in ds.dims or 'lead_time' not in ds.coords:
        raise ValueError(
            f"Lead times ('lead_time') do not exist in dataset at {source}."
        )
    if not np.issubdtype(ds['lead_time'].dtype, np.timedelta64):
        raise ValueError(
            f"'lead_time' coordinate in {source} must decode as timedelta64 "
            f"(e.g. CF units='days'), got dtype {ds['lead_time'].dtype}."
        )
    if ds.sizes['lead_time'] == 0:
        raise ValueError(f"'lead_time' coordinate in {source} is empty.")

    _validate_lead_time_values(
        pd.TimedeltaIndex(ds['lead_time'].to_numpy()), source
    )

    if require_on_all_vars:
        missing_lt_vars = sorted(
            name
            for name, da in ds.data_vars.items()
            if 'lead_time' not in da.dims
        )
        if missing_lt_vars:
            raise ValueError(
                f'Forecast variables {missing_lt_vars} in {source} are missing '
                "the 'lead_time' dimension."
            )


def _check_var_date_overlaps(
    group: Sequence[tuple[Path, xr.Dataset]], var_name: str, directory: Path
) -> None:
    """Vectorized interval check rejecting overlapping dates within a group."""
    intervals: list[
        tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex, Path]
    ] = []
    for path, ds in group:
        if 'date' not in ds.dims:
            continue
        idx = pd.DatetimeIndex(ds.coords['date'].to_numpy())
        intervals.append((idx.min(), idx.max(), idx, path))
    intervals.sort(key=lambda item: (item[0], item[1]))

    active: list[tuple[pd.Timestamp, pd.DatetimeIndex, Path]] = []
    for curr_min, curr_max, curr_idx, curr_path in intervals:
        active = [item for item in active if item[0] >= curr_min]
        for _, prev_idx, prev_path in active:
            overlap = prev_idx.intersection(curr_idx)
            if len(overlap) > 0:
                raise ValueError(
                    f"Overlapping 'date' {overlap[0].isoformat()} for "
                    f"variable '{var_name}' in {directory}: "
                    f'{prev_path.name} and {curr_path.name}.'
                )
        active.append((curr_max, curr_idx, curr_path))


def _check_var_partial_basin_overlaps(
    var_parts: Sequence[tuple[Path, xr.Dataset]],
    var_name: str,
    directory: Path,
) -> None:
    """Reject pairs of files that overlap on both basin and date (if 2D/3D)."""
    meta: list[
        tuple[
            Path,
            set[str],
            pd.DatetimeIndex | None,
            pd.Timestamp | None,
            pd.Timestamp | None,
        ]
    ] = []
    for path, ds in var_parts:
        basins_set = set(ds.coords['basin'].to_numpy().tolist())
        if 'date' in ds.dims:
            idx = pd.DatetimeIndex(ds.coords['date'].to_numpy())
            meta.append((path, basins_set, idx, idx.min(), idx.max()))
        else:
            meta.append((path, basins_set, None, None, None))

    for i, (p1, b1, d1, d1_min, d1_max) in enumerate(meta):
        for p2, b2, d2, d2_min, d2_max in meta[i + 1 :]:
            shared_basins = b1 & b2
            if not shared_basins:
                continue
            shared_b = min(shared_basins)
            if d1 is None or d2 is None:
                raise ValueError(
                    f"Overlapping 'basin' {shared_b!r} for variable "
                    f"'{var_name}' in {directory}: {p1.name} and {p2.name}."
                )
            if d1_max >= d2_min and d2_max >= d1_min:
                overlap = d1.intersection(d2)
                if len(overlap) > 0:
                    raise ValueError(
                        f"Overlapping 'basin' {shared_b!r} and 'date' "
                        f"{overlap[0].isoformat()} for variable '{var_name}' "
                        f'in {directory}: {p1.name} and {p2.name}.'
                    )


def _validate_combined_indexes(ds: xr.Dataset, directory: Path) -> None:
    if 'date' in ds.dims:
        date_idx = pd.DatetimeIndex(ds.coords['date'].to_numpy())
        if not date_idx.is_unique:
            dup = date_idx[date_idx.duplicated()][0]
            raise ValueError(
                f"Duplicate 'date' {dup.isoformat()} after combining files in "
                f'{directory}.'
            )
    basin_idx = pd.Index(ds.coords['basin'].to_numpy())
    if not basin_idx.is_unique:
        dup_b = basin_idx[basin_idx.duplicated()][0]
        raise ValueError(
            f"Duplicate 'basin' {dup_b!r} after combining files in {directory}."
        )


def _combine_var_same_basin_group(
    group: Sequence[tuple[Path, xr.Dataset]], var_name: str, directory: Path
) -> xr.Dataset:
    """Combine one variable's datasets sharing an identical basin set."""
    if len(group) == 1:
        return group[0][1]
    ref_basins = group[0][1].coords['basin'].to_numpy()
    aligned = [
        ds
        if np.array_equal(ds.coords['basin'].to_numpy(), ref_basins)
        else ds.sel(basin=ref_basins)
        for _, ds in group
    ]
    if all('date' not in ds.dims for ds in aligned):
        # Static attribute repeated identically across time-split files.
        return xr.merge(aligned, join='outer', compat='no_conflicts')
    _check_var_date_overlaps(group, var_name, directory)
    return xr.combine_by_coords(
        aligned,
        data_vars='minimal',
        coords='minimal',
        join='outer',
        compat='no_conflicts',
    )


def _combine_single_variable(
    var_name: str,
    var_parts: Sequence[tuple[Path, xr.Dataset]],
    directory: Path,
) -> xr.Dataset:
    """Combine all file fragments for one variable across time/basin splits."""
    if len(var_parts) == 1:
        return var_parts[0][1]

    groups: dict[frozenset[str], list[tuple[Path, xr.Dataset]]] = {}
    for item in var_parts:
        key = frozenset(item[1].coords['basin'].to_numpy().tolist())
        groups.setdefault(key, []).append(item)

    if len(groups) == 1:
        return _combine_var_same_basin_group(
            next(iter(groups.values())), var_name, directory
        )

    total_basins = sum(len(key) for key in groups)
    union_basins: set[str] = set().union(*groups)
    if total_basins == len(union_basins):
        combined_groups = [
            _combine_var_same_basin_group(group, var_name, directory)
            for group in groups.values()
        ]
        return xr.concat(
            combined_groups,
            dim='basin',
            data_vars='minimal',
            coords='minimal',
            join='outer',
            compat='no_conflicts',
        )

    _check_var_partial_basin_overlaps(var_parts, var_name, directory)
    return xr.combine_by_coords(
        [ds.sortby('basin') for _, ds in var_parts],
        data_vars='minimal',
        coords='minimal',
        join='outer',
        compat='no_conflicts',
    )


def _combine_multi_basin_datasets(
    opened: Sequence[tuple[Path, xr.Dataset]], directory: Path
) -> xr.Dataset:
    """Combine multi-basin datasets per variable across time, basin, or 2D.

    Combining each variable independently before merging allows distinct
    variables to live in files with different basin partitions (e.g., separate
    attribute tables or mixed basin-batch and global variable files) while
    still rejecting true per-variable basin/date overlaps.
    """
    by_var: dict[str, list[tuple[Path, xr.Dataset]]] = {}
    for path, ds in opened:
        for var_name in ds.data_vars:
            by_var.setdefault(str(var_name), []).append((path, ds[[var_name]]))

    per_var_dss = [
        _combine_single_variable(var_name, var_parts, directory)
        for var_name, var_parts in by_var.items()
    ]
    combined = (
        per_var_dss[0]
        if len(per_var_dss) == 1
        else xr.merge(per_var_dss, join='outer', compat='no_conflicts')
    )
    _validate_combined_indexes(combined, directory)
    return combined


def _subset_multi_basin_vars(
    ds: xr.Dataset, opts: _MultiBasinReadOpts
) -> xr.Dataset | None:
    """Select requested (or 1D static) variables and drop unused coords."""
    if opts.features is not None:
        present_vars = [f for f in opts.features if f in ds.data_vars]
        present_coords = (
            [
                f
                for f in opts.features
                if f not in ds.data_vars and f in ds.coords and f not in ds.dims
            ]
            if opts.include_coords
            else []
        )
        if not present_vars and not present_coords:
            return None
        if present_coords:
            ds = ds.reset_coords(present_coords)
        ds = ds[present_vars + present_coords]
    elif opts.only_1d_basin_vars:
        one_d_vars = [
            name for name, da in ds.data_vars.items() if da.dims == ('basin',)
        ]
        if not one_d_vars:
            return None
        ds = ds[one_d_vars]

    non_index_coords = [c for c in ds.coords if c not in ds.dims]
    if non_index_coords:
        ds = ds.drop_vars(non_index_coords)
    return ds


def _preprocess_multi_basin_part(
    ds: xr.Dataset,
    path: Path,
    directory: Path,
    *,
    is_multi_file: bool,
    opts: _MultiBasinReadOpts,
) -> xr.Dataset | None:
    """Validate and subset a single multi-basin file prior to combining."""
    if 'basin' not in ds.dims:
        if is_multi_file:
            raise ValueError(
                f'Cannot mix multi-basin and per-basin NetCDF files in '
                f"{directory}: {path.name} is missing a 'basin' dimension."
            )
        raise ValueError(
            f"Required 'basin' dimension not found in {path}. "
            f'Found dims: {list(ds.dims)}.'
        )
    if 'basin' not in ds.coords:
        raise ValueError(
            f"Required 'basin' coordinate variable not found in {path}."
        )

    ds = _decode_lead_time_coord(ds)
    ds = _normalize_basin_coord(ds, path)
    basin_idx = pd.Index(ds.coords['basin'].to_numpy())
    if not basin_idx.is_unique:
        dup = basin_idx[basin_idx.duplicated()][0]
        raise ValueError(
            f"Duplicate 'basin' coordinate value {dup!r} in {path}."
        )

    subset = _subset_multi_basin_vars(ds, opts)
    if subset is None:
        return None
    if opts.validate_dates:
        _validate_date_coord(subset, path)
    if 'lead_time' in subset.dims:
        _validate_lead_time_coord(subset, path)
    return subset


def _raise_no_matching_vars(
    directory: Path, features: Sequence[str] | None, seen_vars: set[str]
) -> None:
    if features:
        raise ValueError(
            f'Requested features {sorted(features)} not found in '
            f'{directory}. Available variables: {sorted(seen_vars)}'
        )
    raise ValueError(
        f'No matching variables found in NetCDF files at {directory}.'
    )


def _open_multi_basin_netcdf(
    directory: Path,
    nc_files: Sequence[Path],
    opts: _MultiBasinReadOpts,
) -> tuple[xr.Dataset, list[xr.Dataset]]:
    """Open one or more multi-basin NetCDF files lazily as one Dataset."""
    is_multi_file = len(nc_files) > 1
    raw_handles: list[xr.Dataset] = []
    opened: list[tuple[Path, xr.Dataset]] = []
    seen_vars: set[str] = set()
    try:
        for path in nc_files:
            raw = _open_dataset_raw(path, chunks='auto')
            raw_handles.append(raw)
            seen_vars.update(str(v) for v in raw.data_vars)
            if opts.include_coords:
                seen_vars.update(str(c) for c in raw.coords)
            ds_part = _preprocess_multi_basin_part(
                raw,
                path,
                directory,
                is_multi_file=is_multi_file,
                opts=opts,
            )
            if ds_part is None:
                raw.close()
                raw_handles.pop()
                continue
            opened.append((path, ds_part))

        if not opened:
            _raise_no_matching_vars(directory, opts.features, seen_vars)

        combined = (
            opened[0][1]
            if len(opened) == 1
            else _combine_multi_basin_datasets(opened, directory)
        )
    except Exception:
        _close_all(raw_handles)
        raise
    else:
        return combined, raw_handles


def _resolve_basin_netcdf_path(
    directory: Path, basin: str, stem_map: dict[str, list[Path]]
) -> Path:
    matches = stem_map.get(basin, [])
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous per-basin NetCDF files for basin '{basin}' in "
            f'{directory}: {[m.name for m in matches]}'
        )
    if not matches:
        raise FileNotFoundError(
            f"No basin NetCDF file found for '{basin}' in {directory} "
            f'(expected {basin}.nc or {basin}.nc4).'
        )
    return matches[0]


def _validate_no_basin_dim(
    ds: xr.Dataset, directory: Path, ds_path: Path
) -> None:
    if 'basin' in ds.dims:
        raise ValueError(
            f'Cannot mix per-basin and multi-basin NetCDF files in '
            f"{directory}: {ds_path.name} unexpectedly has a 'basin' "
            'dimension.'
        )


def _open_single_basin_netcdf(
    directory: Path,
    ds_path: Path,
    features: Sequence[str] | None,
    *,
    require_lead_time: bool,
) -> tuple[xr.Dataset, xr.Dataset, np.datetime64, np.datetime64]:
    raw = _open_dataset_raw(ds_path, chunks={'date': 'auto'})
    try:
        ds = _decode_lead_time_coord(raw)
        _validate_no_basin_dim(ds, directory, ds_path)
        if features:
            _validate_required_features(ds, features, ds_path)
            ds = ds[list(features)]
        # Drop scalar/non-index coords (e.g. 0D 'basin' or per-basin 'lat') so
        # xr.concat does not misclassify or broadcast basin #1's scalars.
        non_index_coords = [c for c in ds.coords if c not in ds.dims]
        if non_index_coords:
            ds = ds.drop_vars(non_index_coords)
        _validate_date_coord(ds, ds_path)
        if require_lead_time or 'lead_time' in ds.dims:
            _validate_lead_time_coord(
                ds, ds_path, require_on_all_vars=require_lead_time
            )
        ds = _cast_numeric_to_float32(ds)
        date_values = ds['date'].to_numpy()
        first_date = date_values.min()
        last_date = date_values.max()
    except Exception:
        raw.close()
        raise
    return raw, ds, first_date, last_date


def _open_per_basin_netcdf(
    directory: Path,
    basins: Sequence[str],
    features: Sequence[str] | None,
    *,
    require_lead_time: bool = False,
    batch_size: int = 500,
) -> xr.Dataset:
    """Open per-basin ``<basin_id>.nc`` files in batches and stack by basin."""
    bar_off = logging.getLogger().level > logging.DEBUG
    basin_list = list(basins)
    if not basin_list:
        raise ValueError('Basin list must be non-empty for per-basin NetCDF.')

    stem_map = _build_per_basin_path_map(list_netcdf_files(directory))
    paths = tuple(
        _resolve_basin_netcdf_path(directory, basin, stem_map)
        for basin in basin_list
    )

    raw_handles: list[xr.Dataset] = []

    def open_batch(
        batch_paths: tuple[Path, ...],
    ) -> Iterator[tuple[xr.Dataset, np.datetime64, np.datetime64]]:
        for ds_path in AutoRefreshTqdm(
            batch_paths,
            desc='Read',
            unit='file',
            leave=False,
            disable=bar_off,
            total=len(batch_paths),
        ):
            raw, ds, first_date, last_date = _open_single_basin_netcdf(
                directory,
                ds_path,
                features,
                require_lead_time=require_lead_time,
            )
            raw_handles.append(raw)
            yield ds, first_date, last_date

    def process_batch(
        batch_items: tuple[tuple[str, Path], ...],
    ) -> xr.Dataset:
        batch_basins, batch_paths = zip(*batch_items, strict=True)
        datasets, starts, ends = zip(*open_batch(batch_paths), strict=True)
        start, end = min(starts), max(ends)
        date = pd.date_range(start=start, end=end, freq='D', name='date')
        reindexed = [
            _cast_numeric_to_float32(ds.reindex(date=date)) for ds in datasets
        ]
        return xr.concat(
            reindexed,
            dim=pd.Index(batch_basins, name='basin'),
            data_vars='all',
            coords='minimal',
            compat='override',
            combine_attrs='override',
            join='outer',
        )

    def batchify() -> Iterator[xr.Dataset]:
        paired = list(zip(basin_list, paths, strict=True))
        batches: Iterable[tuple[tuple[str, Path], ...]] = itertools.batched(
            paired, batch_size
        )
        total = math.ceil(len(paired) / batch_size)
        yield from AutoRefreshTqdm(
            map(process_batch, batches),
            desc='Gather',
            unit='batch',
            total=total,
            disable=bar_off,
        )

    try:
        batch_datasets = tuple(batchify())
        if len(batch_datasets) == 1:
            result = batch_datasets[0]
        else:
            result = xr.concat(
                batch_datasets,
                dim='basin',
                data_vars='all',
                coords='minimal',
                compat='override',
                combine_attrs='override',
                join='outer',
            )
        result = _cast_numeric_to_float32(result)
        result.set_close(_make_closer(raw_handles))
    except Exception:
        _close_all(raw_handles)
        raise
    else:
        return result


def load_netcdf_attributes(
    data_dir: Path | str,
    basins: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
) -> xr.Dataset:
    """Load static catchment attributes from multi-basin NetCDF file(s).

    Parameters
    ----------
    data_dir : Path | str
        Directory directly containing one or more multi-basin ``.nc`` / ``.nc4``
        files with a ``basin`` dimension.
    basins : Sequence[str], optional
        If provided, validates and selects only these basins.
    features : Sequence[str], optional
        If provided, validates and selects only these static attributes.

    Returns
    -------
    xr.Dataset
        Dataset indexed by ``basin`` with numeric/boolean data variables cast
        to ``float32``.
    """
    if basins is not None and len(basins) == 0:
        raise ValueError('Basin list must be non-empty.')

    directory = Path(data_dir)
    if directory.is_file() and _is_netcdf_filename(directory.name):
        raise ValueError(
            f'Expected a directory containing NetCDF files, got file: '
            f'{directory}. Pass its parent directory instead.'
        )

    nc_files = list_netcdf_files(directory)
    if not nc_files:
        raise FileNotFoundError(
            f'No NetCDF (.nc/.nc4) files found in {directory}.'
        )

    if _detect_layout(nc_files) != 'multi_basin':
        raise ValueError(
            f'Static attribute NetCDF files in {directory} must use the '
            "multi-basin layout (containing a 'basin' dimension)."
        )

    LOGGER.debug(
        'Loading static attributes from NetCDF directory: %s', directory
    )
    opts = _MultiBasinReadOpts(
        features=features,
        include_coords=True,
        only_1d_basin_vars=(features is None),
        validate_dates=False,
    )
    ds, raw_handles = _open_multi_basin_netcdf(directory, nc_files, opts)
    try:
        if features:
            _validate_required_features(
                ds, features, directory, include_coords=True
            )
            ds = ds[list(features)]
        _validate_basin_coord(ds, basins, directory)
        if basins is not None:
            ds = ds.sel(basin=list(basins))
        ds = _cast_numeric_to_float32(ds)
        ds.set_close(_make_closer(raw_handles))
    except Exception:
        _close_all(raw_handles)
        raise
    else:
        return ds


def load_netcdf_timeseries(
    data_dir: Path | str,
    basins: Sequence[str],
    features: Sequence[str] | None = None,
    *,
    require_lead_time: bool = False,
    batch_size: int = 500,
) -> xr.Dataset:
    """Load target, hindcast, or forecast timeseries from a NetCDF directory.

    Supports both multi-basin NetCDF files (with a ``basin`` dimension) and
    per-basin ``<basin_id>.nc`` / ``<basin_id>.nc4`` files directly inside
    ``data_dir``.

    Parameters
    ----------
    data_dir : Path | str
        Directory directly containing ``.nc`` / ``.nc4`` files.
    basins : Sequence[str]
        Basin IDs to load.
    features : Sequence[str], optional
        Data variable names to select and validate.
    require_lead_time : bool, optional
        If True, validates that ``lead_time`` exists as a whole-day
        ``timedelta64`` dimension/coordinate (used for forecast datasets).
    batch_size : int, optional
        Number of basin files combined per batch in per-basin mode.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(basin, date)`` or
        ``(basin, date, lead_time)`` and numeric/boolean data variables cast to
        ``float32``.
    """
    if len(basins) == 0:
        raise ValueError('Basin list must be non-empty.')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}.')

    directory = Path(data_dir)
    if directory.is_file() and _is_netcdf_filename(directory.name):
        raise ValueError(
            f'Expected a directory containing NetCDF files, got file: '
            f'{directory}. Pass its parent directory instead.'
        )

    nc_files = list_netcdf_files(directory)
    if not nc_files:
        raise FileNotFoundError(
            f'No NetCDF (.nc/.nc4) files found in {directory}.'
        )

    layout = _detect_layout(nc_files, basins=basins)
    LOGGER.debug(
        'Loading timeseries from NetCDF directory (%s layout): %s',
        layout,
        directory,
    )
    if layout == 'per_basin':
        return _open_per_basin_netcdf(
            directory=directory,
            basins=basins,
            features=features,
            require_lead_time=require_lead_time,
            batch_size=batch_size,
        )

    opts = _MultiBasinReadOpts(
        features=features,
        include_coords=False,
        only_1d_basin_vars=False,
        validate_dates=True,
    )
    ds, raw_handles = _open_multi_basin_netcdf(directory, nc_files, opts)
    try:
        if features:
            _validate_required_features(ds, features, directory)
            ds = ds[list(features)]
        _validate_basin_coord(ds, basins, directory)
        _validate_date_coord(ds, directory)
        if require_lead_time or 'lead_time' in ds.dims:
            _validate_lead_time_coord(
                ds, directory, require_on_all_vars=require_lead_time
            )
        ds = ds.sel(basin=list(basins))
        ds = _cast_numeric_to_float32(ds)
        ds.set_close(_make_closer(raw_handles))
    except Exception:
        _close_all(raw_handles)
        raise
    else:
        return ds
