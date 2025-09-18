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

from typing import Dict, List, Optional, Hashable

import logging
import itertools
import functools
import datetime
from pathlib import Path

import dask
import dask.array
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from ruamel.yaml import YAML
from collections import defaultdict

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.scaler import Scaler
from neuralhydrology.datautils.union_features import union_features
from neuralhydrology.datautils.utils import load_basin_file
from neuralhydrology.datautils.validate_samples import validate_samples, extract_feature_groups
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.configutils import flatten_feature_list
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError

LOGGER = logging.getLogger(__name__)

# Data types for all keys in the sample dictionary.
NUMPY_VARS = ['date']
TENSOR_VARS = ['x_s', 'x_d', 'x_d_hindcast', 'x_d_forecast', 'y', 'per_basin_target_stds', 'basin_index']


class ForecastDataset(BaseDataset):
    """Base data set class for forecast models.
    
    Use subclasses of this class for training/evaluating a model with forecast capabilities.
    Currently, the only supported forecast dataset is Caravan-Multimet.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, the scaler must be calculated (`compute_scaler` must be True). 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise, the basin(s) is(are) read from the
        appropriate basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        Not supported for forecast models and datasets.
    id_to_int : Dict[str, int], optional
        Not supported for forecast models and datasets.
    compute_scaler : bool
        Forces the dataset to calculate a new scaler instead of loading a precalculated scaler. Used during training, but
        not finetuning. 
    """

    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basin: str = None,
        # TODO (future) :: Why is this passed in separately?
        # Can we remove this functionality altogether or load in BaseDataset?
        additional_features: List[Dict[str, pd.DataFrame]] = [],
        id_to_int: Dict[str, int] = {},
        compute_scaler: bool = True
    ):
        # Sequence length parameters.
        # TODO (future) :: Remove all old forecast functionality from basedataset.
        self.lead_time = cfg.lead_time
        self._seq_length = cfg.seq_length
        self._predict_last_n = cfg.predict_last_n
        self._forecast_overlap = cfg.forecast_overlap

        # Feature lists by type.
        self._static_features = cfg.static_attributes
        self._target_features = cfg.target_variables
        self._forecast_features = []
        if cfg.forecast_inputs:
            self._forecast_features = flatten_feature_list(cfg.forecast_inputs)
        if cfg.hindcast_inputs:
            self._hindcast_features = flatten_feature_list(cfg.hindcast_inputs)
        elif cfg.dynamic_inputs:
            self._hindcast_features = flatten_feature_list(cfg.dynamic_inputs)
        else:
            raise ValueError('Either `hindcast_inputs` or `dynamic_inputs` must be supplied.')
        self._union_mapping = cfg.union_mapping

        # Feature data paths by type. This allows the option to load some data from cloud and some locally.
        self._statics_data_path = cfg.statics_data_dir
        self._dynamics_data_path = cfg.dynamics_data_dir
        self._targets_data_path = cfg.targets_data_dir

        # NaN-handling options are required to apply the correct sample validation algorithms.
        self._nan_handling_method = cfg.nan_handling_method
        self._feature_groups = [self._hindcast_features, self._forecast_features]
        if (isinstance(self._hindcast_features[0], str) or isinstance(self._forecast_features[0], str)) \
            and self._nan_handling_method in ['masked_mean', 'attention', 'unioning']:
            raise ValueError(f'Feature groups are required for {self._nan_handling_method} NaN-handling.')

        # Validating samples depends on whether we are training or testing.
        self.is_train = is_train
        # TODO (future) :: Necessary for tester. Remove dependency if possible.
        self.frequencies = ['1D']

        self._period = period
        if period not in ['train', 'validation', 'test']:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test' ")

        if period in ["validation", "test"] or cfg.is_finetuning:
            if compute_scaler:
                raise ValueError("Scaler must be loaded (not computed) for validation, test, and finetuning.")

        # TODO (future) :: Fix this broken functionality.
        if cfg.use_basin_id_encoding:
            raise ValueError('Forecast datasets do not currently support one-hot-encoding.')

        # TODO (future) :: Consolidate the basin list loading somewhere instead of in two different places.
        self._basins = [basin]
        if basin is None:
            self._basins = load_basin_file(getattr(cfg, f'{period}_basin_file'))

        # Load & preprocess the data.
        LOGGER.debug('load data')
        self._dataset = self._load_data()
        LOGGER.debug("validate all floats are float32")
        _assert_floats_are_float32(self._dataset)

        # Extract date ranges.
        # TODO (future) :: Make this work for non-continuous date ranges.
        # TODO (future) :: This only works for daily data.
        self._min_lead_time = 0
        self._lead_times = []
        if self._forecast_features:
            self._min_lead_time = int((self._dataset.lead_time.min() / np.timedelta64(1, 'D')).item())
            self._lead_times = list(range(self._min_lead_time, self.lead_time+1))
        if self._period == 'train':
            self._sample_dates = pd.date_range(cfg.train_start_date, cfg.train_end_date)
        elif self._period == 'test':
            self._sample_dates = pd.date_range(cfg.test_start_date, cfg.test_end_date)
        elif self._period == 'validation':
            self._sample_dates = pd.date_range(cfg.validation_start_date, cfg.validation_end_date)
        # The convention in NH is that the period dates define the SAMPLE dates.
        # All hindcast (and forecast) seqences are extra. Therefore, when cropping
        # the dataset for sampling, we keep all the hindcast and forecast sequence
        # data on both sides of the period dates. This would be more memory efficient
        # in `_load_data()` but that approach adds complexity to the child classes.
        data_start_date = self._sample_dates[0] - pd.Timedelta(days=self._seq_length)
        data_end_date = self._sample_dates[-1] + pd.Timedelta(days=self._seq_length)
        data_dates = pd.date_range(data_start_date, data_end_date)
        LOGGER.debug('reindex data')
        self._dataset = self._dataset.reindex(date=data_dates).sel(date=data_dates)

        # Timestep counters indicate the lead time of each forecast timestep.
        self._hindcast_counter = None
        self._forecast_counter = None
        if cfg.timestep_counter:
            self._hindcast_counter = np.full((self._seq_length,), 0)
            self._forecast_counter = self._lead_times           
            if self._forecast_overlap:
                overlap_counter = np.full((self._forecast_overlap,), self._min_lead_time)
                self._forecast_counter = np.concatenate([overlap_counter, self._forecast_counter], 0)

        # Union features to extend certain data records.
        # Martin suggests doing this step prior to training models and then saving the unioned dataset locally.
        # If you do that, then remove this line.
        if self._union_mapping:
            LOGGER.debug('union features')
            self._dataset = union_features(self._dataset, self._union_mapping)

        # Scale the dataset AFTER cropping dates so that we do not calcualte scalers using test or eval data.
        LOGGER.debug('init scaler')
        self.scaler = Scaler(
            scaler_dir=(cfg.base_run_dir if cfg.is_finetuning else cfg.run_dir),
            calculate_scaler=compute_scaler,
            custom_normalization=cfg.custom_normalization,
            dataset=(self._dataset if compute_scaler else None)
        )
        LOGGER.debug('scale data')
        self._dataset = self.scaler.scale(self._dataset)       
        if compute_scaler:
            LOGGER.debug('save scaler')
            self.scaler.save()

        # Create sample index lookup table for `__getitem__`.
        LOGGER.debug('create sample index')
        self._create_sample_index()

        # TODO: Optionally, optimize the data loader and trainer modules to work with chunked lazy data.
        LOGGER.debug("materialize data (compute)")
        self._dataset = self._dataset.compute()

        # Compute stats for NSE-based loss functions.
        # TODO (future) :: Find a better way to decide whether to calculate these. At least keep a list of
        # losses that require them somewhere like `training.__init__.py`. Perhaps simply always calculate.
        self._per_basin_target_stds = None
        if cfg.loss.lower() in ['nse', 'weightednse']:
            LOGGER.debug('create per_basin_target_stds')
            self._per_basin_target_stds = self._dataset[
                self._target_features].std(
                    dim=[d for d in self._dataset[self._target_features].dims if d != 'basin'],
                skipna=True
            )

        self._data_cache: dict[str, xr.DataArray] = {}

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self,
        item: int
    ) -> Dict[str, torch.Tensor | np.ndarray | Dict[str, torch.Tensor]]:
        """Retrieves a sample by integer index."""

        # Stop iteration.
        if item >= self._num_samples:
            raise IndexError(f'Requested index {item} > the total number of samples {self._num_samples}.')

        # Negative and non-integer indexes raise an error instead of stop iterating.
        if item < 0:
            raise ValueError(f'Requested index {item} < 0.')
        if item % 1 != 0:
            raise ValueError(f'Requested index {item} is not an integer.')

        # TODO (future) :: Suggest remove outer keys and use only feature names. Major change required.
        sample = {
            'date': self._extract_dates(item),
            'x_s': self._extract_statics(item),
            'x_d_hindcast': self._extract_hindcasts(item),
            'x_d_forecast': self._extract_forecasts(item),
            'y': self._extract_targets(item),
        }
        if self._per_basin_target_stds is not None:
            sample['per_basin_target_stds'] = self._extract_per_basin_stds(item)
        if self._hindcast_counter is not None:
            sample['x_d_hindcast']['hindcast_counter'] = np.expand_dims(self._hindcast_counter, -1)
        if self._forecast_counter is not None:
            sample['x_d_forecast']['forecast_counter'] = np.expand_dims(self._forecast_counter, -1)

        # Rename the hindcast data key if we are not doing forecasting.
        if not self._forecast_features:
            sample['x_d'] = sample.pop('x_d_hindcast')
            _ = sample.pop('x_d_forecast')

        sample['basin_index'] = np.array(self._sample_index[item]['basin'], dtype=np.int16)

        # Return sample with various required formats.
        return {key: _convert_to_tensor(key, value) for key, value in sample.items()}

    def _calc_date_range(self, item: int, *, lead: bool = False) -> range:
        date = self._sample_index[item]["date"]
        duration = self._seq_length - 1
        if not lead and not self._lead_times:
            return range(date - duration, date + 1)
        end = date + self.lead_time
        return range(end - duration, end + 1)

    def _extract_dates(self, item: int) -> np.ndarray:
        date = self._calc_date_range(item)
        features = self._extract_dataset(self._dataset, ['date'], {'date': date})
        return features['date']

    def _extract_statics(self, item: int) -> np.ndarray:
        basin = self._sample_index[item]['basin']
        features = self._extract_dataset(self._dataset, self._static_features, {'basin': basin})
        return np.stack([features[e] for e in self._static_features], axis=-1)

    def _extract_hindcasts(self, item: int) -> dict[str, np.ndarray]:
        dim_indexes = self._sample_index[item].copy()
        dim_indexes['date'] = range(dim_indexes['date']-self._seq_length+1, dim_indexes['date']+1)
        features = self._extract_dataset(self._dataset, self._hindcast_features, dim_indexes)
        return {name: np.expand_dims(feature, -1) for name, feature in features.items()}
        # TODO (future) :: This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works, and everything else expects
        # the trailing dim. Remove this dependency in the future.

    def _extract_forecasts(self, item: int) -> dict[str, np.ndarray]:
        features = self._extract_dataset(self._dataset, self._forecast_features, self._sample_index[item])
        if self._forecast_overlap is not None and self._forecast_overlap > 0:
            dim_indexes = self._sample_index[item].copy()
            dim_indexes['date'] = range(
                dim_indexes['date']+1-self._min_lead_time-self._forecast_overlap,
                dim_indexes['date']+1-self._min_lead_time
            )
            dim_indexes['lead_time'] = 0
            overlaps = self._extract_dataset(self._dataset, self._forecast_features, dim_indexes)
            features = {name: np.concatenate([overlaps[name], feature]) for name, feature in features.items()}
        return {name: np.expand_dims(feature, -1) for name, feature in features.items()}
        # TODO (future) :: This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works, and everything else expects
        # the trailing dim. Remove this dependency in the future.

    def _extract_targets(self, item: int) -> np.ndarray:
        dim_indexes = self._sample_index[item].copy()
        dim_indexes["date"] = self._calc_date_range(item, lead=True)
        features = self._extract_dataset(self._dataset, self._target_features, dim_indexes)
        return np.stack([features[e] for e in self._target_features], axis=-1)

    def _extract_per_basin_stds(self, item: int) -> np.ndarray:
        assert self._per_basin_target_stds is not None
        features = self._extract_dataset(
            self._per_basin_target_stds, 
            self._target_features,
            {'basin': self._sample_index[item]['basin']},
        )
        return np.expand_dims(np.stack([features[e] for e in self._target_features], axis=-1), axis=0)
        # TODO (future) :: This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works, and everything else expects
        # the trailing dim. Remove this dependency in the future.

    # This is run by the base class.
    def _create_sample_index(self):
        """Creates a map from integer sample indexes to the integer positions into the xr.Dataset.
        
        This allows index-based sample retrieval, which is faster than coordinate-based sample retrieval.
        """
        # Create a boolean mask for the original dataset noting valid (True) vs. invalid (False) samples.
        valid_sample_mask = validate_samples(
            is_train=self.is_train,
            dataset=self._dataset,
            nan_handling_method=self._nan_handling_method,
            sample_dates=self._sample_dates,
            lead_time=self.lead_time,
            seq_length=self._seq_length,
            predict_last_n=self._predict_last_n,
            forecast_overlap=self._forecast_overlap,
            min_lead_time=self._min_lead_time,
            static_features=self._static_features,
            forecast_features=self._forecast_features,
            hindcast_features=self._hindcast_features,
            target_features=self._target_features,
            feature_groups=self._feature_groups,
        )[0]

        LOGGER.debug('valid_sample_mask compute')
        # Convert boolean valid sample mask into indexes of all samples. This retains
        # only the portion of the valid sample mask with True values.
        # Each element is a list of valid integer positions (indexers) for which
        # values are True for a dimension.
        indices = dask.compute(*dask.array.nonzero(valid_sample_mask.data))

        # Count the number of valid samples.
        num_samples = len(indices[0]) if indices else 0
        if num_samples == 0:
            if self._period == 'train':
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError       

        # Maps dim name to its respective list of int indices (index arrays) i.e. columns
        # of all basins, all dates, etc.
        vectorized_indices = {
            dim: indices[i] for i, dim in enumerate(valid_sample_mask.dims) if dim != "sample"
        }

        LOGGER.debug("sample_index")
        # Reorg columns to rows, mapping sample index i [0, num_samples) to a dict that
        # maps an int position for that sample in each dim. E.g. {1: {'basin': 2, 'date': 3}}
        #
        # This allows integer indexing into each coordinate dimension of the original dataset,
        # while ONLY selecting valid samples. The full original dataset is retained (including
        # not-valid samples) for sequence construction.
        self._sample_index = {
            i: {dim: indexes[i] for dim, indexes in vectorized_indices.items()}
            for i in range(num_samples)
        }

        self._num_samples = num_samples

    def _load_data(self) -> xr.Dataset:
        """Returns an xr dataset of features with the following dimensions: (basin, date, lead_time).
        
        Must be implemented in the child class for a specific dataset.
        """
        raise NotImplementedError

    def _extract_dataset(self, data: xr.Dataset, features: list[str], indexers: dict[Hashable, int|range]) -> dict[str, np.ndarray | np.float32]:
        def extract(feature_name):
            key = f'{id(data)}{feature_name}'
            feature = self._data_cache.get(key)
            if feature is None:
                feature = self._data_cache[key] = data[feature_name]
            return _extract_dataarray(feature, indexers)

        return {feature_name: extract(feature_name) for feature_name in features}


def _extract_dataarray(data: xr.DataArray, indexers: dict[Hashable, int|range]) -> np.ndarray | np.float32:
    """Returns the values in array according to dims given by indexers.
    
    This function replaces uses of `isel` with data and indexers.
    """
    locators = (indexers[dim] if dim in indexers else slice(None) for dim in data.dims)
    return data.data[tuple(locators)]


def _assert_floats_are_float32(dataset: xr.Dataset):
    items = itertools.chain(dataset.data_vars.items(), dataset.coords.items())
    for name, data_array_or_coord in items:
        if np.issubdtype(data_array_or_coord.dtype, np.floating):
            assert data_array_or_coord.dtype == np.float32, (
                f"Data variable or coord '{name}' is a float but not float32. "
                f"Actual dtype: {data_array_or_coord.dtype}"
            )

def _convert_to_tensor(key: str, value: np.ndarray) -> torch.Tensor | np.ndarray:
    if key in NUMPY_VARS:
        return value
    elif key in TENSOR_VARS:
        if isinstance(value, dict):
            return {k: torch.from_numpy(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        else:
            raise ValueError(f'Unrecognized data type: {type(value)}')
    else:
        raise ValueError(f'Unrecognized data key: {key}')