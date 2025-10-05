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

import itertools
import logging
from pathlib import Path
from typing import List, Optional

import dask
import dask.delayed
import numpy as np
import pandas as pd
import xarray

LOGGER = logging.getLogger(__name__)


def load_caravan_attributes(data_dir: Path,
                            basins: Optional[List[str]] = None,
                            subdataset: Optional[str] = None,
                            features: Optional[List[str]] = None) -> xarray.Dataset:
    """Load the attributes of the Caravan dataset.

    Parameters
    ----------
    data_dir : Path
        Path to the root directory of Caravan that has to include a sub-directory called 'attributes' which contain the 
        attributes of all sub-datasets in separate folders.
    basins : List[str], optional
        If passed, returns only attributes for the basins specified in this list. Otherwise, the attributes of all 
        basins are returned.
    subdataset : str, optional
        If passed, returns only the attributes of one sub-dataset. Otherwise, the attributes of all sub-datasets are 
        loaded.

    Raises
    ------
    FileNotFoundError
        If the requested sub-dataset does not exist or any sub-dataset for the requested basins is missing.
    ValueError
        If any of the requested basins does not exist in the attribute files or if both, basins and sub-dataset are 
        passed but at least one of the basins is not part of the corresponding sub-dataset.

    Returns
    -------
    xarray.Dataset
        A basin indexed Dataset with all attributes as coordinates.
    """
    LOGGER.debug('')
    if subdataset:
        subdataset_dir = data_dir / "attributes" / subdataset
        if not subdataset_dir.is_dir():
            raise FileNotFoundError(f"No subdataset {subdataset} found at {subdataset_dir}.")
        subdataset_dirs = [subdataset_dir]

    else:
        subdataset_dirs = [d for d in (data_dir / "attributes").glob('*') if d.is_dir()]

    if basins:
        # Get list of unique sub datasets from the basin strings.
        subdataset_names = list(set(x.split('_')[0] for x in basins))

        # Make sure the subdatasets exist and are not in conflict with the subdataset argument, if passed.
        if subdataset:
            # subdataset_names is only allowed to be size 1 in this case.
            if len(subdataset_names) > 1 or subdataset_names[0] != subdataset:
                raise ValueError("At least one of the passed basins is not part of the passed subdataset.")
        else:
            # Check if all subdatasets exist.
            missing_subdatasets = [s for s in subdataset_names if not (data_dir / "attributes" / s).is_dir()]

            if missing_subdatasets:
                raise FileNotFoundError(f"Could not find subdataset directories for {missing_subdatasets}.")

        # Subset subdataset_dirs to only the required subsets.
        subdataset_dirs = [s for s in subdataset_dirs if s.name in subdataset_names]

    # Load all required attribute files.
    LOGGER.debug('load attribute files')
    ds = _load_attribute_files_of_subdatasets(subdataset_dirs, features or [])

    # If a specific list of basins is requested, subset the Dataset.
    if basins:
        LOGGER.debug('missing')
        # Check for any requested basins that are missing from the loaded data.
        missing = set(basins).difference(ds.coords["basin"].data)
        if missing:
            raise ValueError(f'{len(missing)} basins are missing static attributes: {", ".join(missing)}')

        # Subset to only the requested basins.
        ds = ds.sel(basin=basins)

    return ds


def load_caravan_timeseries(data_dir: Path, basin: str, filetype: str = "netcdf") -> pd.DataFrame|xarray.Dataset:
    """Loads the timeseries data of one basin from the Caravan dataset.
    
    Parameters
    ----------
    data_dir : Path
        Path to the root directory of Caravan that has to include a sub-directory called 'timeseries'. This 
        sub-directory has to contain another sub-directory called either 'csv' or 'netcdf', depending on the choice 
        of the filetype argument. By default, netCDF files are loaded from the 'netcdf' subdirectory.
    basin : str
        The Caravan gauge id string in the form of {subdataset_name}_{gauge_id}.
    filetype : str, optional
        Can be either 'csv' or 'netcdf'. Depending on this value, this function will load the timeseries data from the
        netcdf files (default) (xarray.Dataset) or csv files (pd.DataFrame).

    Raises
    ------
    ValueError
        If filetype is not in ['csv', 'netcdf'].
    FileNotFoundError
        If no timeseries file exists for the basin.
    """
    # Get the subdataset name from the basin string.
    subdataset_name = basin.split('_')[0]

    if filetype == "netcdf":
        filepath = data_dir / "timeseries" / "netcdf" / subdataset_name / f"{basin}.nc"
    elif filetype == "csv":
        filepath = data_dir / "timeseries" / "csv" / subdataset_name / f"{basin}.csv"
    else:
        raise ValueError("filetype has to be either 'csv' or 'netcdf'.")

    if not filepath.is_file():
        raise FileNotFoundError(f"No basin file found at {filepath}.")

    # Load timeseries data.
    if filetype == "netcdf":
        return xarray.open_dataset(filepath)
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date')
    return df


def load_caravan_timeseries_together(
    data_dir: Path, basins: list[str], target_features: list[str]
) -> xarray.Dataset:
    """Loads the timeseries data of basins from the Caravan dataset.

    Parameters
    ----------
    data_dir : Path
        Path to the root directory of Caravan that has to include a sub-directory called 'timeseries'. This
        sub-directory has to contain another sub-directory called either 'csv' or 'netcdf', depending on the choice
        of the filetype argument. By default, netCDF files are loaded from the 'netcdf' subdirectory.
    basins : list[str]
        The Caravan gauge id strings in the form of {subdataset_name}_{gauge_id}.
    target_features : list[str]
        The target variables to select.

    Raises
    ------
    FileNotFoundError
        If no timeseries file exists for the basin.
    """

    def basin_to_file_path(basin: str) -> Path:
        subdataset_name = basin.split("_")[0]
        filepath = data_dir / "timeseries" / "netcdf" / subdataset_name / f"{basin}.nc"
        if not filepath.is_file():
            raise FileNotFoundError(f"No basin file found at {filepath}.")
        return filepath

    def preprocess(ds: xarray.Dataset):
        return ds[target_features]

    ds = xarray.open_mfdataset(
        [basin_to_file_path(e) for e in basins],
        preprocess=preprocess,
        combine="nested",
        concat_dim="basin",
        parallel=False,  # open_mfdataset has a bug (seg fault) when True
        chunks={"date": "auto"},
    )
    return ds.assign_coords(basin=basins)


def _load_attribute_files_of_subdatasets(datasets: list[Path], features: List[str]) -> xarray.Dataset:
    """Loads all attribute CSV files, indexing gauge_id to basin.

    Converts float64 to float32.
    """

    @dask.delayed
    def process(csv_file: Path) -> xarray.Dataset:
        df64 = pd.read_csv(csv_file, index_col="gauge_id")
        df = df64.astype(
            {col: np.float32 for col in df64.select_dtypes(include=["float64"]).columns}
        )
        df.rename_axis("basin", inplace=True)
        df.drop(columns=(e for e in df.columns if e not in features), inplace=True)
        return df.to_xarray().chunk('auto')  # Uses underlying numpy arrays in df

    dss = map(process, itertools.chain.from_iterable(e.glob("*.csv") for e in datasets))
    dss = dask.compute(*dss)

    return xarray.merge(dss)
