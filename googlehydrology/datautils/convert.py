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

"""Utilities to convert legacy Caravan NetCDF/CSV into unified Zarr format."""

import itertools
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)


def convert_caravan_attributes(
    attributes_dir: Path | str,
    output_zarr_path: Path | str,
    subdatasets: Sequence[str] | None = None,
) -> xr.Dataset:
    """Converts Caravan attribute CSV files into a consolidated Zarr store.

    Parameters
    ----------
    attributes_dir : Path | str
        Directory containing Caravan attribute subdirectories or CSV files.
    output_zarr_path : Path | str
        Destination path for the attributes Zarr store.
    subdatasets : Sequence[str], optional
        Optional list of subdataset directory names to convert.

    Returns
    -------
    xr.Dataset
        The combined static attributes Dataset with dimension ('basin').
    """
    attributes_dir = Path(attributes_dir)
    output_zarr_path = Path(output_zarr_path)

    # Find all attribute CSV files
    if subdatasets:
        csv_files = []
        for sub in subdatasets:
            sub_dir = (
                attributes_dir / sub
                if (attributes_dir / sub).is_dir()
                else attributes_dir
            )
            csv_files.extend(list(sub_dir.glob('*.csv')))
    else:
        csv_files = list(attributes_dir.glob('**/*.csv'))

    if not csv_files:
        raise FileNotFoundError(
            f'No attribute CSV files found in {attributes_dir}'
        )

    # Process each CSV
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'gauge_id' in df.columns:
                df = df.set_index('gauge_id')
            elif df.index.name != 'gauge_id' and 'basin' in df.columns:
                df = df.set_index('basin')
            df.index.name = 'basin'

            # Cast float columns to float32
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].astype(np.float32)
            dfs.append(df)
        except Exception as e:
            LOGGER.warning('Error reading %s: %s', csv_file, e)

    if not dfs:
        raise ValueError(
            f'Could not extract valid attribute data from {attributes_dir}'
        )

    # Merge dataframes
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.combine_first(df)

    ds = combined_df.to_xarray()

    # Save as Zarr
    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(output_zarr_path, mode='w', consolidated=True)
    LOGGER.info('Successfully wrote attributes Zarr to %s', output_zarr_path)
    return ds


def convert_caravan_timeseries(
    timeseries_dir: Path | str,
    output_zarr_path: Path | str,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    """Converts Caravan timeseries NetCDF/CSV files into a single Zarr store.

    Parameters
    ----------
    timeseries_dir : Path | str
        Directory containing timeseries files (e.g. timeseries/netcdf/...).
    output_zarr_path : Path | str
        Destination path for the unified timeseries Zarr store.
    variables : Sequence[str], optional
        Variables to extract (e.g. ['streamflow']). If None, extracts all.

    Returns
    -------
    xr.Dataset
        The unified timeseries Dataset with dimensions ('basin', 'date').
    """
    timeseries_dir = Path(timeseries_dir)
    output_zarr_path = Path(output_zarr_path)

    # Find all NC files first, then fallback to CSV files
    nc_files = sorted(list(timeseries_dir.glob('**/*.nc')))
    csv_files = (
        sorted(list(timeseries_dir.glob('**/*.csv'))) if not nc_files else []
    )
    files = nc_files if nc_files else csv_files

    if not files:
        raise FileNotFoundError(
            f'No timeseries files (.nc or .csv) found in {timeseries_dir}'
        )

    datasets = []
    basins = []

    for file_path in files:
        basin_id = file_path.stem
        if nc_files:
            ds = xr.open_dataset(file_path)
            if variables:
                available_vars = [v for v in variables if v in ds.data_vars]
                ds = ds[available_vars]
        else:
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            if variables:
                available_vars = [v for v in variables if v in df.columns]
                df = df[available_vars]
            df = df.astype(np.float32)
            ds = df.to_xarray()

        # Cast floats to float32
        for v in ds.data_vars:
            if np.issubdtype(ds[v].dtype, np.floating):
                ds[v] = ds[v].astype(np.float32)

        datasets.append(ds)
        basins.append(basin_id)

    # Align dates across all basins
    combined_ds = xr.concat(
        datasets, dim=pd.Index(basins, name='basin'), join='outer'
    )

    # Save to Zarr
    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    combined_ds = combined_ds.chunk('auto')
    combined_ds.to_zarr(output_zarr_path, mode='w', consolidated=True)
    LOGGER.info('Successfully wrote timeseries Zarr to %s', output_zarr_path)
    return combined_ds


def convert_caravan_to_zarr(
    caravan_dir: Path | str,
    output_dir: Path | str,
    variables: Sequence[str] | None = ('streamflow',),
) -> tuple[xr.Dataset, xr.Dataset]:
    """Converts a full Caravan directory (attributes and timeseries) to Zarr stores.

    Parameters
    ----------
    caravan_dir : Path | str
        Root directory of the Caravan dataset.
    output_dir : Path | str
        Destination directory for the converted Zarr stores.
    variables : Sequence[str], optional
        Variables to extract for timeseries. Default is ('streamflow',).

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        (attributes_ds, timeseries_ds)
    """
    caravan_dir = Path(caravan_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Attributes
    attr_dir = (
        caravan_dir / 'attributes'
        if (caravan_dir / 'attributes').exists()
        else caravan_dir
    )
    attr_ds = convert_caravan_attributes(
        attr_dir, output_dir / 'attributes.zarr'
    )

    # Timeseries / Targets
    ts_dir = (
        caravan_dir / 'timeseries' / 'netcdf'
        if (caravan_dir / 'timeseries' / 'netcdf').exists()
        else caravan_dir / 'timeseries'
        if (caravan_dir / 'timeseries').exists()
        else caravan_dir
    )
    ts_ds = convert_caravan_timeseries(
        ts_dir, output_dir / 'streamflow.zarr', variables=variables
    )

    return attr_ds, ts_ds
