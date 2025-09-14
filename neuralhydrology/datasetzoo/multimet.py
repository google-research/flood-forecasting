from typing import Dict, List, Optional, Union

import logging
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import xarray as xr

from neuralhydrology.datasetzoo.caravan import load_caravan_attributes, load_caravan_timeseries_together
from neuralhydrology.datasetzoo.forecast_basedataset import ForecastDataset
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

MULTIMET_MINIMUM_LEAD_TIME = 1


def _open_zarr(path: Path) -> xr.Dataset:
    path = str(path).replace('gs:/', 'gs://')
    return xr.open_zarr(store=path, chunks='auto', decode_timedelta=True)


def _get_products_and_bands_from_feature_strings(
    features: List[str]
) -> Dict[str, List[str]]:
    """
    Processes feature strings to create a dictionary of product to band(s).
    
    Parameters
    ----------
    features : List[str]
        A list features in the format `<product>_<band>. This is the format for feature
        names in the Multimet dataset.
    
    Returns
    -------
    Dict[str, List[str]] 
        Keys are product names and values are a list of features for that product. Features
        remain in the format <product>_<band>.
    """
    product_bands = {}
    for feature in features:
        product = feature.split('_')[0].upper()
        if product == 'ERA5LAND':
            product = 'ERA5_LAND'
        if product not in product_bands:
            product_bands[product] = []
        product_bands[product].append(feature)
    return product_bands


class Multimet(ForecastDataset):
    """Data loader for the Caravan MultiMet dataset.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
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

    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 compute_scaler: bool = True):

        # Initialize parent class.
        super(Multimet, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       compute_scaler=compute_scaler)

    def _load_data(self) -> xr.Dataset:
        """Main loading function for Caravan-Multimet.
        
        Returns an xr dataset of features with the following dimensions: (basin, date, lead_time).
        This loading function aggregates hindcast, forecast, statics, and target data.
        
        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with various dimensions.
        """
        datasets = []
        if self._static_features is not None:
            LOGGER.debug('load attributes')
            datasets.append(self._load_static_features())
        if self._hindcast_features is not None:
            LOGGER.debug('load hindcast features')
            datasets.extend(self._load_hindcast_features())
        if self._forecast_features is not None:
            LOGGER.debug('load forecast features')
            datasets.extend(self._load_forecast_features())
        if self._target_features is not None:
            LOGGER.debug('load target features')
            datasets.append(self._load_target_features())
        if not datasets:
            raise ValueError('At least one type of data must be loaded.')
        LOGGER.debug('merge')
        return xr.merge(datasets)

    def _load_hindcast_features(self) -> list[xr.Dataset]:
        """Load Caravan-Multimet data for hindcast features.

        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (date, basin).
        """
        # Prepare hindcast features to load, including the masks of union_mapping
        features = []
        features.extend(self._hindcast_features)
        if self._union_mapping:
            features.extend(self._union_mapping.values())
        features = list(set(features))

        # Separate products and bands for each product from feature names.
        product_bands = _get_products_and_bands_from_feature_strings(features=features)

        # Initialize storage for product/band dataframes that will eventually be concatenated.
        product_dss = []

        # Load data for the selected products, bands, and basins.
        for product, bands in product_bands.items():
            product_path = self._hindcasts_data_path / product / 'timeseries.zarr'
            product_ds = _open_zarr(product_path)

            # If this is a forecast product, extract only shortest leadtime for hindcasts.
            if 'lead_time' in product_ds:
                lead_time_delta = np.timedelta64(MULTIMET_MINIMUM_LEAD_TIME, 'D')
                product_ds = product_ds.sel(lead_time=lead_time_delta).squeeze(
                    'lead_time').shift(date=MULTIMET_MINIMUM_LEAD_TIME)

            product_dss.append(product_ds.sel(basin=self._basins)[bands])

        return product_dss

    def _load_forecast_features(self) -> list[xr.Dataset]:
        """Load Caravan-Multimet data for forecast features.

        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (date, lead_time, basin).
        """
        # Separate products and bands for each product from feature names.
        product_bands = _get_products_and_bands_from_feature_strings(features=self._forecast_features)

        # Initialize storage for product/band dataframes that will eventually be concatenated.
        product_dss = []

        # Lead time array.
        lead_times = [pd.Timedelta(days=i) for i in range(MULTIMET_MINIMUM_LEAD_TIME, self.lead_time+1)]

        # Load data for the selected products, bands, and basins.
        for product, bands in product_bands.items():
            product_path = self._forecasts_data_path / product / 'timeseries.zarr'
            product_ds = _open_zarr(product_path)

            # If this is a forecast product, extract only leadtime 0 for hindcasts.
            if 'lead_time' not in product_ds:
                raise ValueError(f'Lead times do not exist for forecast product ({product}).')

            product_ds = product_ds.sel(basin=self._basins, lead_time=lead_times)[bands]
            product_dss.append(product_ds)

        return product_dss

    def _load_target_features(self) -> xr.Dataset:
        """Load Caravan streamflow data.
        
        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (date, basin).
        """
        return load_caravan_timeseries_together(self._targets_data_path, self._basins, self._target_features)

    def _load_static_features(self) -> xr.Dataset:
        """Load Caravan static attributes.

        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (basin).
        """
        return load_caravan_attributes(
            data_dir=self._statics_data_path,
            basins=self._basins
        )[self._static_features]
