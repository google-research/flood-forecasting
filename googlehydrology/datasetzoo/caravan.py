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

import functools
import itertools
import logging
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import dask
import dask.dataframe as dd
import dask.delayed
import numpy as np
import pandas as pd
import xarray

from googlehydrology.utils.tqdm import AutoRefreshTqdm as tqdm

LOGGER = logging.getLogger(__name__)


def _find_zarr_store(path: Path | str, preferred_names: list[str]) -> Path | None:
    """Finds a Zarr store given a directory or file path."""
    path_str = str(path)
    if path_str.startswith('gs://') or path_str.startswith('gs:/'):
        # For GCS paths, assume valid store if ends with zarr or subpath
        if path_str.endswith('.zarr'):
            return Path(path_str)
        for name in preferred_names:
            return Path(f"{path_str.rstrip('/')}/{name}")
        return Path(path_str)

    p = Path(path)
    if p.suffix == '.zarr' or (p / '.zgroup').exists() or (p / 'zarr.json').exists() or (p / '.zmetadata').exists():
        return p
    for name in preferred_names:
        candidate = p / name
        if candidate.exists() and (
            candidate.suffix == '.zarr'
            or (candidate / '.zgroup').exists()
            or (candidate / 'zarr.json').exists()
            or (candidate / '.zmetadata').exists()
        ):
            return candidate
    return None


def load_caravan_attributes(
    data_dir: Path | str,
    basins: list[str] | None = None,
    subdataset: str | None = None,
    features: list[str] | None = None,
) -> xarray.Dataset:
    """Load the attributes of the Caravan dataset.

    Supports Zarr stores (preferred) and legacy Caravan CSV directories.

    Parameters
    ----------
    data_dir : Path | str
        Path to attributes Zarr store or root directory of Caravan attributes.
    basins : list[str], optional
        If passed, returns only attributes for the basins specified in this list. Otherwise, the attributes of all
        basins are returned.
    subdataset : str, optional
        If passed (legacy CSV mode), returns only the attributes of one sub-dataset.
    features: list[str], optional
        If passed, will only return the specified features (columns) in the statics datasets.

    Returns
    -------
    xarray.Dataset
        A basin indexed Dataset with all attributes as coordinates.
    """
    LOGGER.debug('load caravan attributes')
    zarr_store = _find_zarr_store(data_dir, ['attributes.zarr', 'attributes', 'statics.zarr', 'statics'])
    if zarr_store is not None:
        LOGGER.debug(f'Loading attributes from Zarr store: {zarr_store}')
        store_path = zarr_store.as_posix().replace('gs:/', 'gs://')
        ds = xarray.open_zarr(store_path, chunks='auto')
        if features:
            available_features = [f for f in features if f in ds.data_vars or f in ds.coords]
            ds = ds[available_features]
        if basins:
            if 'basin' in ds.coords:
                missing = set(basins).difference(ds.coords['basin'].data)
                if missing:
                    raise ValueError(
                        f'{len(missing)} basins are missing static attributes: {", ".join(missing)}'
                    )
                ds = ds.sel(basin=basins)
        return ds

    # Legacy CSV loader fallback
    data_dir = Path(data_dir)
    if subdataset:
        subdataset_dir = data_dir / 'attributes' / subdataset
        if not subdataset_dir.is_dir():
            raise FileNotFoundError(
                f'No subdataset {subdataset} found at {subdataset_dir}.'
            )
        subdataset_dirs = [subdataset_dir]
    else:
        attr_dir = data_dir / 'attributes' if (data_dir / 'attributes').is_dir() else data_dir
        subdataset_dirs = [
            d for d in attr_dir.glob('*') if d.is_dir()
        ]

    if basins:
        subdataset_names = list(set(x.split('_')[0] for x in basins))
        if subdataset:
            if len(subdataset_names) > 1 or subdataset_names[0] != subdataset:
                raise ValueError(
                    'At least one of the passed basins is not part of the passed subdataset.'
                )
        else:
            attr_dir = data_dir / 'attributes' if (data_dir / 'attributes').is_dir() else data_dir
            missing_subdatasets = [
                s
                for s in subdataset_names
                if not (attr_dir / s).is_dir()
            ]
            if missing_subdatasets:
                raise FileNotFoundError(
                    f'Could not find subdataset directories for {missing_subdatasets}.'
                )
        subdataset_dirs = [
            s for s in subdataset_dirs if s.name in subdataset_names
        ]

    LOGGER.debug('load legacy attribute files')
    ds = _load_attribute_files_of_subdatasets(subdataset_dirs, features or [])

    if basins:
        missing = set(basins).difference(ds.coords['basin'].data)
        if missing:
            raise ValueError(
                f'{len(missing)} basins are missing static attributes: {", ".join(missing)}'
            )
        ds = ds.sel(basin=basins)

    return ds


def load_caravan_timeseries(
    data_dir: Path | str,
    basins: list[str],
    target_features: list[str],
    *,
    csv: bool = False,
    batch_size: int = 500,
) -> xarray.Dataset:
    """Load the timeseries data of basins from the Caravan dataset.

    Supports Zarr stores (preferred) and legacy multi-file NetCDF/CSV datasets.

    Parameters
    ----------
    data_dir : Path | str
        Path to timeseries Zarr store or root directory of Caravan timeseries.
    basins : list[str]
        List of basin ID strings.
    target_features : list[str]
        The target variables to select.
    csv: bool, optional
        Whether to load CSV files instead of NC files (legacy mode).
    batch_size : int, optional
        Batch size for legacy multi-file loader.

    Returns
    -------
    xarray.Dataset
        A combined Dataset with 'basin' and 'date' coordinates.
    """
    LOGGER.debug('load caravan timeseries')
    zarr_store = _find_zarr_store(
        data_dir, ['streamflow.zarr', 'targets.zarr', 'timeseries.zarr', 'timeseries']
    )
    if zarr_store is not None:
        LOGGER.debug(f'Loading timeseries from Zarr store: {zarr_store}')
        store_path = zarr_store.as_posix().replace('gs:/', 'gs://')
        ds = xarray.open_zarr(store_path, chunks='auto')
        if target_features:
            available = [f for f in target_features if f in ds.data_vars]
            ds = ds[available]
        if basins:
            ds = ds.sel(basin=basins)
        return ds

    return load_caravan_timeseries_together(
        data_dir=Path(data_dir),
        basins=basins,
        target_features=target_features,
        csv=csv,
        batch_size=batch_size,
    )


def load_csvs_as_ds(basin_to_path: dict[str, Path]) -> xarray.Dataset:
    """Load timeseries data from CSV files into a single xarray Dataset."""
    datas = (
        dd.read_csv(path, parse_dates=['date'], dtype=np.float32)
        for path in basin_to_path.values()
    )
    datas = [df.assign(basin=basin) for basin, df in zip(basin_to_path, datas)]
    return dd.concat(datas).compute().set_index(['basin', 'date']).to_xarray()


def load_caravan_timeseries_together(
    data_dir: Path,
    basins: list[str],
    target_features: list[str],
    *,
    csv: bool = False,
    batch_size: int = 500,
) -> xarray.Dataset:
    """Legacy multi-file Caravan timeseries loader."""
    bar_off = logging.getLogger().level > logging.DEBUG

    def basin_to_path(basin: str) -> Path:
        subdataset = basin.partition('_')[0]
        kind = 'csv' if csv else 'netcdf'
        ext = 'csv' if csv else 'nc'
        path = data_dir / 'timeseries' / kind / subdataset / f'{basin}.{ext}'
        if path.is_file():
            return path
        raise FileNotFoundError(f'No basin file found at {path}.')

    def select(ds: xarray.Dataset) -> xarray.Dataset:
        return ds[target_features]

    paths = tuple(map(basin_to_path, basins))

    if csv:
        return select(load_csvs_as_ds(dict(zip(basins, paths))))

    combine = functools.partial(
        xarray.combine_nested,
        concat_dim='basin',
        coords='minimal',
        compat='override',
        combine_attrs='override',
    )

    open_dataset_args = {'chunks': {'date': 'auto'}, 'engine': 'netcdf4'}

    def open_dataset(ds_path: Path) -> tuple[xarray.Dataset, float, float]:
        ds = select(xarray.open_dataset(ds_path, **open_dataset_args))
        first_date, last_date = ds['date'].isel(date=[0, -1]).data
        return ds, first_date, last_date

    def open_datasets(
        batch_paths: tuple[Path],
    ) -> Iterator[tuple[xarray.Dataset, float, float]]:
        dss, n = map(open_dataset, batch_paths), len(batch_paths)
        yield from tqdm(
            dss, desc='Read', unit='file', leave=False, disable=bar_off, total=n
        )

    def process_batch(batch_paths: Iterable[Path]) -> xarray.Dataset:
        datasets, starts, ends = zip(*open_datasets(batch_paths))
        start, end = min(starts), max(ends)
        date = pd.date_range(start=start, end=end, freq='D', name='date')
        datasets = [ds.reindex(date=date) for ds in datasets]
        return combine(datasets, join='override')

    def batchify() -> Iterator[xarray.Dataset]:
        batches = map(process_batch, itertools.batched(paths, batch_size))
        total = math.ceil(len(paths) / batch_size)
        yield from tqdm(
            batches, desc='Gather', unit='batch', total=total, disable=bar_off
        )

    return combine(tuple(batchify()), join='outer').assign_coords(basin=basins)


def _load_attribute_files_of_subdatasets(
    datasets: list[Path], features: list[str]
) -> xarray.Dataset:
    """Loads all attribute CSV files, indexing gauge_id to basin."""
    @dask.delayed
    def process(csv_file: Path) -> xarray.Dataset:
        df64 = pd.read_csv(csv_file, index_col='gauge_id')
        df = df64.astype(
            {
                col: np.float32
                for col in df64.select_dtypes(include=[np.number]).columns
            }
        )
        df.rename_axis('basin', inplace=True)
        if features:
            df.drop(
                columns=(e for e in df.columns if e not in features), inplace=True
            )
        return df.to_xarray().chunk(
            {'basin': -1}
        )  # Uses underlying numpy arrays in df

    dss = map(
        process,
        itertools.chain.from_iterable(e.glob('*.csv') for e in datasets),
    )
    dss = dask.compute(*dss)

    return xarray.merge(dss, join='outer', compat='no_conflicts')
