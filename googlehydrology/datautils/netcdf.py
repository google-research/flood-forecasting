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

1. **Multi-basin files:** One or more ``.nc`` / ``.nc4`` files in the given
   directory that each contain a ``basin`` dimension (e.g., a single combined
   file, time-split files, or basin-batch files). Combined by coordinates via
   ``xarray.open_mfdataset``.
2. **Per-basin files (``<basin_id>.nc`` / ``<basin_id>.nc4``):** One file per
   basin directly in the given directory, without a ``basin`` dimension inside
   each file. Only files for the requested basins are opened.

Floating-point variables and coordinates (e.g. ``float64``) are lazily cast to
``float32`` on read to satisfy ``Multimet``'s ``float32`` invariant.
"""

import itertools
import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from googlehydrology.utils.tqdm import AutoRefreshTqdm

LOGGER = logging.getLogger(__name__)

NETCDF_SUFFIXES = ('.nc', '.nc4')
_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'
_NETCDF3_CLASSIC_MAGIC = (b'CDF\x01', b'CDF\x02')


def _is_cloud_path(path: Path | str) -> bool:
    path_str = str(path)
    return path_str.startswith(('gs://', 'gs:/', 'gs/'))


def _netcdf_engine_for_path(path: Path) -> str:
    """Pick an xarray engine from the file magic header.

    Uses ``h5netcdf`` for HDF5 / NetCDF4 files (avoiding ``libnetcdf``
    reopen/threading crashes under Dask), ``scipy`` for NetCDF3 classic files,
    and falls back to ``netcdf4`` for CDF-5 or other variants.
    """
    with path.open('rb') as handle:
        magic = handle.read(8)
    if magic.startswith(_HDF5_MAGIC):
        return 'h5netcdf'
    if magic[:4] in _NETCDF3_CLASSIC_MAGIC:
        return 'scipy'
    return 'netcdf4'


def list_netcdf_files(directory: Path | str) -> list[Path]:
    """Return sorted ``.nc`` / ``.nc4`` files directly inside ``directory``."""
    if _is_cloud_path(directory):
        return []
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(
        entry
        for entry in p.iterdir()
        if entry.is_file() and entry.suffix.lower() in NETCDF_SUFFIXES
    )


def has_netcdf_files(directory: Path | str) -> bool:
    """Return True if ``directory`` directly holds ``.nc`` / ``.nc4`` files."""
    return bool(list_netcdf_files(directory))


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

    product_norm = product.lower().replace('_', '')
    for sub in sorted(p.iterdir()):
        if (
            sub.is_dir()
            and sub.name.lower().replace('_', '') == product_norm
            and has_netcdf_files(sub)
        ):
            return sub
    return None


def _cast_floats_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Lazily cast non-float32 floating-point vars and coords to float32."""
    cast_vars = {
        name: da.astype(np.float32)
        for name, da in ds.data_vars.items()
        if np.issubdtype(da.dtype, np.floating) and da.dtype != np.float32
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


def _normalize_basin_coord(ds: xr.Dataset) -> xr.Dataset:
    """Decode byte-string ``basin`` coordinates into Python strings."""
    if 'basin' in ds.coords and np.issubdtype(
        ds.coords['basin'].dtype, np.bytes_
    ):
        decoded = [
            b.decode('utf-8') if isinstance(b, (bytes, np.bytes_)) else str(b)
            for b in ds.coords['basin'].to_numpy()
        ]
        ds = ds.assign_coords(basin=decoded)
    return ds


def _detect_layout(
    nc_files: Sequence[Path],
) -> Literal['multi_basin', 'per_basin']:
    """Inspect the first file to distinguish multi-basin vs per-basin layout."""
    first = nc_files[0]
    engine = _netcdf_engine_for_path(first)
    with xr.open_dataset(
        first, engine=engine, chunks={}, decode_timedelta=True
    ) as ds0:
        if 'basin' in ds0.dims or 'basin' in ds0.coords:
            return 'multi_basin'
    return 'per_basin'


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
        raise ValueError(
            f'Requested features {missing} not found in {source}. '
            f'Available variables: {sorted(ds.data_vars)}'
        )


def _validate_basin_coord(
    ds: xr.Dataset, basins: Sequence[str] | None, source: Path | str
) -> None:
    if 'basin' not in ds.dims or 'basin' not in ds.coords:
        raise ValueError(
            f"Required 'basin' dimension/coordinate not found in {source}. "
            f'Found dims: {list(ds.dims)}.'
        )
    if basins:
        present = set(ds.coords['basin'].to_numpy().tolist())
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
            f"got dtype {ds['date'].dtype}."
        )
    if ds.sizes['date'] == 0:
        raise ValueError(f"'date' coordinate in {source} is empty.")


def _validate_lead_time_coord(
    ds: xr.Dataset,
    source: Path | str,
    *,
    require_on_all_vars: bool = False,
) -> None:
    if 'lead_time' not in ds.dims or 'lead_time' not in ds.coords:
        raise ValueError(
            f"Lead times ('lead_time') do not exist in forecast dataset at "
            f'{source}.'
        )
    if not np.issubdtype(ds['lead_time'].dtype, np.timedelta64):
        raise ValueError(
            f"'lead_time' coordinate in {source} must decode as timedelta64 "
            f"(e.g. CF units='days'), got dtype {ds['lead_time'].dtype}."
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


def _combine_same_basin_group(group: Sequence[xr.Dataset]) -> xr.Dataset:
    """Combine datasets sharing the same basin set along remaining coords."""
    if len(group) == 1:
        return group[0]
    ref_basins = group[0].coords['basin'].to_numpy()
    aligned = [
        ds
        if np.array_equal(ds.coords['basin'].to_numpy(), ref_basins)
        else ds.sel(basin=ref_basins)
        for ds in group
    ]
    return xr.combine_by_coords(
        aligned,
        data_vars='minimal',
        coords='minimal',
        join='outer',
        compat='no_conflicts',
    )


def _combine_multi_basin_datasets(opened: Sequence[xr.Dataset]) -> xr.Dataset:
    """Combine multi-basin datasets across time, basin, or 2D splits.

    Grouping by each file's basin set and concatenating mutually disjoint basin
    groups via ``xr.concat`` allows arbitrary (unsorted or interleaved) basin ID
    partitions that ``xr.combine_by_coords`` otherwise rejects as non-monotonic.
    """
    groups: dict[frozenset[str], list[xr.Dataset]] = {}
    for ds in opened:
        key = frozenset(ds.coords['basin'].to_numpy().tolist())
        groups.setdefault(key, []).append(ds)

    if len(groups) == 1:
        return _combine_same_basin_group(next(iter(groups.values())))

    total_basins = sum(len(key) for key in groups)
    union_basins: set[str] = set().union(*groups)
    if total_basins == len(union_basins):
        combined_groups = [
            _combine_same_basin_group(group) for group in groups.values()
        ]
        return xr.concat(
            combined_groups,
            dim='basin',
            data_vars='minimal',
            coords='minimal',
            join='outer',
            compat='no_conflicts',
        )

    return xr.combine_by_coords(
        [ds.sortby('basin') for ds in opened],
        data_vars='minimal',
        coords='minimal',
        join='outer',
        compat='no_conflicts',
    )


def _open_multi_basin_netcdf(
    directory: Path, nc_files: Sequence[Path]
) -> xr.Dataset:
    """Open one or more multi-basin NetCDF files lazily as one Dataset."""

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        source = ds.encoding.get('source', str(directory))
        if 'basin' not in ds.dims or 'basin' not in ds.coords:
            raise ValueError(
                f'Cannot mix multi-basin and per-basin NetCDF files in '
                f"{directory}: {source} is missing a 'basin' dimension."
            )
        return _normalize_basin_coord(ds)

    if len(nc_files) == 1:
        engine = _netcdf_engine_for_path(nc_files[0])
        ds = xr.open_dataset(
            nc_files[0],
            engine=engine,
            chunks='auto',
            decode_timedelta=True,
        )
        try:
            return _preprocess(ds)
        except Exception:
            ds.close()
            raise

    opened: list[xr.Dataset] = []
    try:
        for path in nc_files:
            ds_part = xr.open_dataset(
                path,
                engine=_netcdf_engine_for_path(path),
                chunks='auto',
                decode_timedelta=True,
            )
            opened.append(_preprocess(ds_part))
        return _combine_multi_basin_datasets(opened)
    except Exception:
        for ds_part in opened:
            ds_part.close()
        raise


def _resolve_basin_netcdf_path(directory: Path, basin: str) -> Path:
    matches = [
        directory / f'{basin}{ext}'
        for ext in NETCDF_SUFFIXES
        if (directory / f'{basin}{ext}').is_file()
    ]
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
) -> tuple[xr.Dataset, np.datetime64, np.datetime64]:
    engine = _netcdf_engine_for_path(ds_path)
    ds = xr.open_dataset(
        ds_path,
        engine=engine,
        chunks={'date': 'auto'},
        decode_timedelta=True,
    )
    try:
        _validate_no_basin_dim(ds, directory, ds_path)
        if features:
            _validate_required_features(ds, features, ds_path)
            ds = ds[list(features)]
        _validate_date_coord(ds, ds_path)
        if require_lead_time or 'lead_time' in ds.dims:
            _validate_lead_time_coord(
                ds, ds_path, require_on_all_vars=require_lead_time
            )
        ds = _cast_floats_to_float32(ds)
        date_values = ds['date'].to_numpy()
        first_date = date_values.min()
        last_date = date_values.max()
    except Exception:
        ds.close()
        raise
    return ds, first_date, last_date


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

    paths = tuple(
        _resolve_basin_netcdf_path(directory, basin) for basin in basin_list
    )

    def open_batch(
        batch_paths: tuple[Path, ...],
    ) -> Iterator[tuple[xr.Dataset, np.datetime64, np.datetime64]]:
        yield from AutoRefreshTqdm(
            (
                _open_single_basin_netcdf(
                    directory,
                    ds_path,
                    features,
                    require_lead_time=require_lead_time,
                )
                for ds_path in batch_paths
            ),
            desc='Read',
            unit='file',
            leave=False,
            disable=bar_off,
            total=len(batch_paths),
        )

    def process_batch(
        batch_items: tuple[tuple[str, Path], ...],
    ) -> xr.Dataset:
        batch_basins, batch_paths = zip(*batch_items, strict=True)
        datasets, starts, ends = zip(*open_batch(batch_paths), strict=True)
        start, end = min(starts), max(ends)
        date = pd.date_range(start=start, end=end, freq='D', name='date')
        reindexed = [ds.reindex(date=date) for ds in datasets]
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

    batch_datasets = tuple(batchify())
    if len(batch_datasets) == 1:
        return batch_datasets[0]
    return xr.concat(
        batch_datasets,
        dim='basin',
        data_vars='all',
        coords='minimal',
        compat='override',
        combine_attrs='override',
        join='outer',
    )


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
        Dataset indexed by ``basin`` with floating-point arrays cast to
        ``float32``.
    """
    directory = Path(data_dir)
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
    ds = _open_multi_basin_netcdf(directory, nc_files)
    try:
        if features:
            _validate_required_features(
                ds, features, directory, include_coords=True
            )
            ds = ds[list(features)]
        _validate_basin_coord(ds, basins, directory)
        if basins:
            ds = ds.sel(basin=list(basins))
        return _cast_floats_to_float32(ds)
    except Exception:
        ds.close()
        raise


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
        If True, validates that ``lead_time`` exists as a ``timedelta64``
        dimension/coordinate (used for forecast datasets).
    batch_size : int, optional
        Number of basin files combined per batch in per-basin mode.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(basin, date)`` or
        ``(basin, date, lead_time)`` and floating-point arrays cast to
        ``float32``.
    """
    directory = Path(data_dir)
    nc_files = list_netcdf_files(directory)
    if not nc_files:
        raise FileNotFoundError(
            f'No NetCDF (.nc/.nc4) files found in {directory}.'
        )

    layout = _detect_layout(nc_files)
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

    ds = _open_multi_basin_netcdf(directory, nc_files)
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
        if basins:
            ds = ds.sel(basin=list(basins))
        return _cast_floats_to_float32(ds)
    except Exception:
        ds.close()
        raise
