from typing import Dict, List, Optional

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from ruamel.yaml import YAML
from collections import defaultdict

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.validate_samples import validate_samples, extract_feature_groups
from neuralhydrology.datautils.utils import load_basin_file
from neuralhydrology.datautils.scaler import Scaler
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError

# Data types for all keys in the sample dictionary.
NUMPY_VARS = ['date']
TENSOR_VARS = ['x_s', 'x_d', 'x_d_hindcast', 'x_d_forecast', 'y', 'per_basin_target_stds']


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
            self._forecast_features = cfg.forecast_inputs
        if cfg.hindcast_inputs:
            self._hindcast_features = cfg.hindcast_inputs
        elif cfg.dynamic_inputs:
            self._hindcast_features = cfg.dynamic_inputs
        else:
            raise ValueError('Either `hindcast_inputs` or `dynamic_inputs` must be supplied.')
        
        # Feature data paths by type. This allows the option to load some data from cloud and some locally.
        self._statics_data_path = cfg.statics_data_dir
        self._hindcasts_data_path = cfg.hindcasts_data_dir
        self._forecasts_data_path = cfg.forecasts_data_dir
        self._targets_data_path = cfg.targets_data_dir
        
        # NaN-handling options are required to apply the correct sample validation algorithms.
        self._nan_handling_method = cfg.nan_handling_method
        self._feature_groups = [self._hindcast_features, self._forecast_features]
        if (isinstance(self._hindcast_features[0], str) or isinstance(self._forecast_features[0], str)) \
            and self._nan_handling_method in ['masked_mean', 'attention']:
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
        self._dataset = self._load_data().astype('float32')

        # Extract date ranges.
        # TODO (future) :: Make this work for non-continuous date ranges.
        # TODO (future) :: This only works for daily data.
        self._min_lead_time = 0
        self.lead_times = []
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

        # Scale the dataset AFTER cropping dates so that we do not calcualte scalers using test or eval data.
        self.scaler = Scaler(
            scaler_dir=(cfg.base_run_dir if cfg.is_finetuning else cfg.run_dir),
            calculate_scaler=compute_scaler,
            custom_normalization=cfg.custom_normalization,
            dataset=(self._dataset if compute_scaler else None)
        )
        self._dataset = self.scaler.scale(self._dataset)       
        if compute_scaler:
            self.scaler.save()
        
        # Create sample index lookup table for `__getitem__`.
        self._create_sample_index()

        # Compute stats for NSE-based loss functions.
        # TODO (future) :: Find a better way to decide whether to calculate these. At least keep a list of
        # losses that require them somewhere like `training.__init__.py`. Perhaps simply always calculate.
        self._per_basin_target_stds = None
        if cfg.loss.lower() in ['nse', 'weightednse']:
            self._per_basin_target_stds = self._dataset[
                self._target_features].std(
                    dim=[d for d in self._dataset[self._target_features].dims if d != 'basin'],
                skipna=True
            )
        
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
        
        # Extract sample from `self._dataset`.
        def _extract_dates(item: int) -> np.ndarray:
            hindcast_date_indexes = range(
                self._sample_index[item]['date']+1-self._seq_length,
                self._sample_index[item]['date']+1
            )
            dates = self._dataset['date'].isel(date=hindcast_date_indexes).values
            if self._lead_times:
                forecast_dates = [
                    dates[-1] + np.timedelta64(i, 'D')
                    for i in self._lead_times
                ]
                dates = np.concatenate([dates, forecast_dates])[-self._seq_length:]
            return dates

        def _extract_statics(feature: str, item: int) -> np.ndarray:
            return self._dataset[feature].isel(basin=self._sample_index[item]['basin']).values
            
        def _extract_hindcasts(feature: str, item: int) -> np.ndarray:
            dim_indexes = {dim: val for dim, val in self._sample_index[item].items()}
            dim_indexes['date'] = range(dim_indexes['date']-self._seq_length+1, dim_indexes['date']+1)
            return self._dataset[feature].isel(dim_indexes).values
        
        def _extract_forecasts(feature: str, item: int) -> np.ndarray:
            dim_indexes = {dim: val for dim, val in self._sample_index[item].items()}
            forecast_array = self._dataset[feature].isel(dim_indexes).values
            if self._forecast_overlap is not None and self._forecast_overlap > 0:
                dim_indexes['date'] = range(
                    dim_indexes['date']+1-self._min_lead_time-self._forecast_overlap,
                    dim_indexes['date']+1-self._min_lead_time
                )
                dim_indexes['lead_time'] = 0
                overlap_array = self._dataset[feature].isel(dim_indexes).values
                forecast_array = np.concatenate([overlap_array, forecast_array])
            return forecast_array
        
        def _extract_targets(feature: str, item: int) -> np.ndarray:
            dim_indexes = {dim: val for dim, val in self._sample_index[item].items()}
            dim_indexes['date'] = list(range(dim_indexes['date']-self._seq_length+1, dim_indexes['date']+1))
            dim_indexes['date'] += [dim_indexes['date'][-1] + i for i in self._lead_times]
            return self._dataset[feature].isel(dim_indexes).values[-self._seq_length:]
    
        sample = {'date': _extract_dates(item)}
        # TODO (future) :: Suggest remove outer keys and use only feature names. Major change required.
        sample['x_s'] = np.stack([_extract_statics(feature, item) for feature in self._static_features], -1)
        sample['x_d_hindcast'] = {feature: _extract_hindcasts(feature, item) for feature in self._hindcast_features}
        sample['x_d_forecast'] = {feature: _extract_forecasts(feature, item) for feature in self._forecast_features}
        sample['y'] = np.stack([_extract_targets(feature, item) for feature in self._target_features], -1)
        if self._per_basin_target_stds is not None:
            sample['per_basin_target_stds'] = np.stack(
                [
                    self._per_basin_target_stds[feature].isel(basin=self._sample_index[item]['basin']).values
                    for feature in self._target_features
                ], -1
            )
        if self._hindcast_counter is not None:
            sample['x_d_hindcast']['hindcast_counter'] = self._hindcast_counter
            sample['x_d_forecast']['forecast_counter'] = self._forecast_counter            
        
        # TODO (future) :: This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works, and everything else expects
        # the trailing dim. Remove this dependency in the future.
        sample['x_d_hindcast'] = {feature: np.expand_dims(value, -1) for feature, value in sample['x_d_hindcast'].items()}
        sample['x_d_forecast'] = {feature: np.expand_dims(value, -1) for feature, value in sample['x_d_forecast'].items()}
        if self._per_basin_target_stds is not None:
            sample['per_basin_target_stds'] = np.expand_dims(sample['per_basin_target_stds'], 0)
        
        # Rename the hindcast data key if we are not doing forecasting.
        if not self._forecast_features:
            sample['x_d'] = sample.pop('x_d_hindcast')
            _ = sample.pop('x_d_forecast')
            
        # Return sample with various required formats.
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
             
        return {key: _convert_to_tensor(key, value) for key, value in sample.items()}

    def _create_sample_index(self):
        """Creates a map from integer sample indexes to the integer positions into the xr.Dataset.
        
        This allows index-based sample retrieval, which is faster than coordinate-based sample retrieval.
        """
        # Create a boolean mask for the original dataset noting valid (True) vs. invalid (False) samples.
        valid_sample_mask, masks = validate_samples(
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
        )
        
        # Convert boolean valid sample mask into indexes of all samples. This retains
        # only the portion of the valid sample mask with True values.
        stacked_mask = valid_sample_mask.stack(sample=valid_sample_mask.dims)
        valid_sample_da = stacked_mask[stacked_mask]
    
        # Count the number of valid samples.
        num_samples = valid_sample_da.size
        if num_samples == 0:
            if self._period == 'train':
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError       

        # This is a place where you can add checks on the types, formats, etc. of dataset coordinates.
        # TODO (future) :: Is there a way to enforce this that isn't manual?
        def _safe_coord(coord):
            """Can add any coordinate assurances here that might be needed."""
            if isinstance(coord, np.datetime64) or isinstance(coord, int):
                coord = pd.Timestamp(coord)
            # elif ...
            return coord
        
        # Map of integer indexes into each coordinate dimension of the original dataset.
        dimension_index = {
            dim: {
                _safe_coord(coord): i for i, coord in enumerate(self._dataset[dim].values)
            } for dim in valid_sample_da.coords if dim != 'sample'
        }
        
        # Map of integer sample indexes into maps from dimension name to integer indexes into
        # the coordinates of that dimension. Sample indexes come from the valid sample mask.
        # This allows integer indexing into each coordinate dimension of the original dataset,
        # while ONLY selecting valid samples. The full original dataset is retained (including
        # not-valid samples) for sequence construction.
        self._sample_index = {
            i: {
                dim: dimension_index[dim][_safe_coord(valid_sample_da[dim].isel(sample=i).item())]
                for dim in valid_sample_da.coords if dim != 'sample'
            } for i in range(num_samples)
        }
        self._num_samples = len(self._sample_index)

    def _load_data(self) -> xr.Dataset:
        """Returns an xr dataset of features with the following dimensions: (basin, date, lead_time).
        
        Must be implemented in the child class for a specific dataset.
        """
        raise NotImplementedError
