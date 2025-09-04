import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class Caravan(BaseDataset):
    """Data set class for the Caravan data set by [#]_.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, the scaler must be calculated (`compute_scaler` must be True),
        and similarly the `id_to_int` input is required if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    compute_scaler : bool
        Forces the dataset to calculate a new scaler instead of loading a precalculated scaler. Used during training, but
        not finetuning.

    References
    ----------
    .. [#] Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., Gudmundsson, L., Hassidim, A., 
        Klotz, D., Nevo, S., Shalev, G., & Matias, Y.. (2022). Caravan - A global community dataset for large-sample 
        hydrology. Accepted for publication in Nature Scientific Data. Preprint: 
        https://eartharxiv.org/repository/view/3345/, 2023
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 compute_scaler: bool = True):
        # initialize parent class
        super(Caravan, self).__init__(cfg=cfg,
                                      is_train=is_train,
                                      period=period,
                                      basin=basin,
                                      additional_features=additional_features,
                                      id_to_int=id_to_int,
                                      compute_scaler=compute_scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load timeseries data from netcdf files."""
        # TODO: avoid converting to pd.DataFrame
        return load_caravan_timeseries(data_dir=self.cfg.data_dir, basin=basin).to_dataframe()

    def _load_attributes(self) -> pd.DataFrame:
        """Load input and output data from text files."""
        # TODO: avoid converting to pd.DataFrame
        return load_caravan_attributes(data_dir=self.cfg.data_dir, basins=self.basins).to_dataframe()


def load_caravan_attributes(data_dir: Path,
                            basins: Optional[List[str]] = None,
                            subdataset: Optional[str] = None) -> xarray.Dataset:
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
    dss = []
    for subdataset_dir in subdataset_dirs:
        dss.extend(_load_attribute_files_of_subdataset(subdataset_dir))

    # Merge all Datasets along the basin index.
    LOGGER.debug('merge')
    ds = xarray.merge(dss)

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


def _load_attribute_files_of_subdataset(subdataset_dir: Path) -> list[xarray.Dataset]:
    """Loads all attribute CSV files for one subdataset.

    NOTE: we change the index column from gauge_id to basin.
    NOTE: we convert float64 to float32 on data decoding.

    One may merge them, which is equivalent to a side-by-side concat (aligned along
    the shared, now called, "basin" coordinate).
    """
    return [
        xarray.Dataset.from_dataframe(
            read_csv_float32(csv_file, index_col="gauge_id").rename_axis("basin")
        )
        for csv_file in subdataset_dir.glob("*.csv")
    ]


def read_csv_float32(csv_file: Path, index_col: str) -> pd.DataFrame:
    df_peek = pd.read_csv(csv_file, index_col=index_col, nrows=100)
    dtype_map = {
        col: np.float32 for col, dtype in df_peek.dtypes.items() if dtype == "float64"
    }
    return pd.read_csv(csv_file, index_col=index_col, dtype=dtype_map)
