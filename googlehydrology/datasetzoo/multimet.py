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

from typing import Hashable, Iterable, Union
import time
import copy
import concurrent.futures
import logging
import itertools
import functools
from pathlib import Path

import dask
import dask.array
import dask.distributed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from googlehydrology.datasetzoo.caravan import load_caravan_attributes, load_caravan_timeseries_together
from googlehydrology.datautils.scaler import Scaler
from googlehydrology.datautils.union_features import union_features
from googlehydrology.datautils.utils import load_basin_file
from googlehydrology.datautils.validate_samples import validate_samples
from googlehydrology.utils.config import Config
from googlehydrology.utils.configutils import flatten_feature_list
from googlehydrology.utils.errors import NoTrainDataError, NoEvaluationDataError

LOGGER = logging.getLogger(__name__)

# Data types for all keys in the sample dictionary.
NUMPY_VARS = ["date"]
TENSOR_VARS = [
    "x_s",
    "x_d",
    "x_d_hindcast",
    "x_d_forecast",
    "y",
    "per_basin_target_stds",
    "basin_index",
]
MULTIMET_MINIMUM_LEAD_TIME = 1

class ThreadedDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self, *args, _lazy_memory: bool, max_pending: int = 4, **kwargs
    ):
        self._n_threads = kwargs.get('num_workers', 0)
        kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)

        self._lazy_memory = _lazy_memory
        self._max_pending = max_pending

        # TODO: use py instead of dask dist, unneeded overhead

        self._cluster = dask.distributed.LocalCluster(
            processes=False, n_workers=1, threads_per_worker=1
        )
        self._client = dask.distributed.Client(
            self._cluster, direct_to_workers=True, set_as_default=False
        )

        self._futures = None
        self._batches = None

    def _prepare(self):
        n = self._max_pending - self._futures.count()
        if n <= 0:
            return
        print(f'  prepare {n}')
        batches = itertools.islice(self._batches, n)
        self._futures.update(map(self._submit_task, batches))

    def _submit_task(self, indices):
        return self._client.submit(
            ThreadedDataLoader._compute_batch,
            self.dataset,
            self.collate_fn,
            indices,
        )

    @staticmethod
    def _compute_batch(dataset, collate_fn, batch_indices):
        tt = time.time()
        batch = [dataset[i] for i in batch_indices]

        t = time.time()
        (batch,) = dask.compute(batch, scheduler='single-threaded')
        t = time.time() - t
        print(f'worker {t=}')

        batch = [
            {k: _convert_to_tensor(k, v) for k, v in sample.items()}
            for sample in batch
        ]
        batch = collate_fn(batch)
        tt = time.time() - tt
        print(f'worker tara {tt-t=}')
        return batch

    def __iter__(self):
        self._batches = iter(self.batch_sampler)
        self._futures = dask.distributed.as_completed(loop=self._client.loop)
        self._prepare()
        for future in self._futures:
            result = future.result()
            self._prepare()
            print('yield')
            yield result
            print('yield post')


class Multimet(Dataset):
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
        compute_scaler: bool = True,
    ):
        self._lazy = cfg.lazy_memory  # Instantiate the cache if enabled
        self._comp = dask.array if self._lazy.enabled else np

        # Sequence length parameters.
        # TODO (future) :: Remove all old forecast functionality from basedataset.
        self.lead_time = cfg.lead_time
        self._seq_length = cfg.seq_length
        self._predict_last_n = cfg.predict_last_n
        self._forecast_overlap = cfg.forecast_overlap
        self._allzero_samples_are_invalid = cfg.allzero_samples_are_invalid

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
            raise ValueError(
                "Either `hindcast_inputs` or `dynamic_inputs` must be supplied."
            )
        self._union_mapping = cfg.union_mapping

        # Feature data paths by type. This allows the option to load some data from cloud and some locally.
        self._statics_data_path = cfg.statics_data_dir
        self._dynamics_data_path = cfg.dynamics_data_dir
        self._targets_data_path = cfg.targets_data_dir

        # NaN-handling options are required to apply the correct sample validation algorithms.
        self._nan_handling_method = cfg.nan_handling_method
        self._feature_groups = [self._hindcast_features, self._forecast_features]
        if (
            isinstance(self._hindcast_features[0], str)
            or isinstance(self._forecast_features[0], str)
        ) and self._nan_handling_method in ["masked_mean", "attention", "unioning"]:
            raise ValueError(
                f"Feature groups are required for {self._nan_handling_method} NaN-handling."
            )

        # Validating samples depends on whether we are training or testing.
        self.is_train = is_train
        # TODO (future) :: Necessary for tester. Remove dependency if possible.
        self.frequencies = ["1D"]

        self._period = period
        if period not in ["train", "validation", "test"]:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test' ")

        if period in ["validation", "test"] or cfg.is_finetuning:
            if compute_scaler:
                raise ValueError(
                    "Scaler must be loaded (not computed) for validation, test, and finetuning."
                )

        # TODO (future) :: Fix this broken functionality.
        if cfg.use_basin_id_encoding:
            raise ValueError(
                "Forecast datasets do not currently support one-hot-encoding."
            )

        # TODO (future) :: Consolidate the basin list loading somewhere instead of in two different places.
        self._basins = [basin]
        if basin is None:
            self._basins = load_basin_file(getattr(cfg, f"{period}_basin_file"))

        # Load & preprocess the data.
        LOGGER.debug("load data")
        self._dataset = self._load_data()
        LOGGER.debug("validate all floats are float32")
        _assert_floats_are_float32(self._dataset)

        # Extract date ranges.
        # TODO (future) :: Make this work for non-continuous date ranges.
        # TODO (future) :: This only works for daily data.
        self._min_lead_time = 0
        self._lead_times = []
        if self._forecast_features:
            self._min_lead_time = int(
                (self._dataset.lead_time.min() / np.timedelta64(1, "D")).item()
            )
            self._lead_times = list(range(self._min_lead_time, self.lead_time + 1))

        # Split hindcast features to groups with/without lead_time in the dataset.
        # These lists will be used for efficient data selection during sampling.
        self._hindcast_features_with_lead_time = [
            feature
            for feature in self._hindcast_features
            if "lead_time" in self._dataset[feature].dims
        ]
        self._hindcast_features_without_lead_time = [
            feature
            for feature in self._hindcast_features
            if feature not in self._hindcast_features_with_lead_time
        ]

        start_dates, end_dates = self._get_period_dates(cfg)
        self._sample_dates = self._union_ranges(start_dates, end_dates)
        # The convention in NH is that the period dates define the SAMPLE dates.
        # All hindcast (and forecast) seqences are extra. Therefore, when cropping
        # the dataset for sampling, we keep all the hindcast and forecast sequence
        # data on both sides of the period dates. This would be more memory efficient
        # in `_load_data()` but that approach adds complexity to the child classes.
        extended_start_dates = [
            start_date - pd.Timedelta(days=self._seq_length)
            for start_date in start_dates
        ]
        extended_end_dates = [
            end_date + pd.Timedelta(days=self.lead_time) for end_date in end_dates
        ]
        extended_dates = self._union_ranges(extended_start_dates, extended_end_dates)
        LOGGER.debug("reindex data")
        self._dataset = self._dataset.reindex(date=extended_dates).sel(
            date=extended_dates
        )

        # Timestep counters indicate the lead time of each forecast timestep.
        self._hindcast_counter = None
        self._forecast_counter = None
        if cfg.timestep_counter:
            self._hindcast_counter = np.full((self._seq_length,), 0)
            self._forecast_counter = self._lead_times
            if self._forecast_overlap:
                overlap_counter = np.full(
                    (self._forecast_overlap,), self._min_lead_time
                )
                self._forecast_counter = np.concatenate(
                    [overlap_counter, self._forecast_counter], 0
                )

        # Union features to extend certain data records.
        # Martin suggests doing this step prior to training models and then saving the unioned dataset locally.
        # If you do that, then remove this line.
        if self._union_mapping:
            LOGGER.debug("union features")
            self._dataset = union_features(self._dataset, self._union_mapping)

        # Scale the dataset AFTER cropping dates so that we do not calcualte scalers using test or eval data.
        LOGGER.debug("init scaler")
        self.scaler = Scaler(
            scaler_dir=(cfg.base_run_dir if cfg.is_finetuning else cfg.run_dir),
            calculate_scaler=compute_scaler,
            custom_normalization=cfg.custom_normalization,
            dataset=(self._dataset if compute_scaler else None),
        )
        LOGGER.debug("scale data")
        self._dataset = self.scaler.scale(self._dataset)

        # TODO: Optionally, optimize the data loader and trainer modules to work with chunked lazy data.
        # We explicitly keep the self.scaler.scaler computation since trainer uses it directly
        if self._lazy.enabled:
            LOGGER.debug("compute scaler")
            (self.scaler.scaler,) = dask.compute(self.scaler.scaler)
        else:
            LOGGER.debug("compute scaler and dataset")
            (self._dataset, self.scaler.scaler) = dask.compute(self._dataset, self.scaler.scaler)

        # Create sample index lookup table for `__getitem__`.
        # TODO: create sample index still must compute the dataset on its own
        LOGGER.debug("create sample index")
        self._create_sample_index()

        # Compute stats for NSE-based loss functions.
        # TODO (future) :: Find a better way to decide whether to calculate these. At least keep a list of
        # losses that require them somewhere like `training.__init__.py`. Perhaps simply always calculate.
        self._per_basin_target_stds = None
        if cfg.loss.lower() in ["nse", "weightednse"]:
            LOGGER.debug("create per_basin_target_stds")
            self._per_basin_target_stds = self._dataset[self._target_features].std(
                dim=[
                    d for d in self._dataset[self._target_features].dims if d != "basin"
                ],
                skipna=True,
            )

        self._data_cache: dict[str, xr.DataArray] = {}

        LOGGER.debug("forecast dataset init complete")

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self, item: int
    ) -> dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]:
        """Retrieves a sample by integer index."""
        # Stop iteration.
        if item >= self._num_samples:
            raise IndexError(
                f"Requested index {item} > the total number of samples {self._num_samples}."
            )

        # Negative and non-integer indexes raise an error instead of stop iterating.
        if item < 0:
            raise ValueError(f"Requested index {item} < 0.")
        if item % 1 != 0:
            raise ValueError(f"Requested index {item} is not an integer.")

        # TODO (future) :: Suggest remove outer keys and use only feature names. Major change required.
        sample = {
            "date": self._extract_dates(item),
            "x_s": self._extract_statics(item),
            "x_d_hindcast": self._extract_hindcasts(item),
            "x_d_forecast": self._extract_forecasts(item),
            "y": self._extract_targets(item),
        }
        if self._per_basin_target_stds is not None:
            sample["per_basin_target_stds"] = self._extract_per_basin_stds(item)
        if self._hindcast_counter is not None:
            sample["x_d_hindcast"]["hindcast_counter"] = np.expand_dims(
                self._hindcast_counter, -1
            )
        if self._forecast_counter is not None:
            sample["x_d_forecast"]["forecast_counter"] = np.expand_dims(
                self._forecast_counter, -1
            )

        # Rename the hindcast data key if we are not doing forecasting.
        if not self._forecast_features:
            sample["x_d"] = sample.pop("x_d_hindcast")
            _ = sample.pop("x_d_forecast")

        # Can't use strings. Torch does not support it in tensors.
        sample["basin_index"] = np.array(
            self._sample_index[item]["basin"], dtype=np.int16
        )

        # if self._lazy.enabled:
        #     (sample,) = dask.compute(sample, scheduler='single-threaded')
        #     # (sample,) = dask.compute(sample)

        # return {key: _convert_to_tensor(key, value) for key, value in sample.items()}

        return sample

    def _calc_date_range(self, item: int, *, lead: bool = False) -> range:
        date = self._sample_index[item]["date"]
        duration = self._seq_length - 1
        if not lead and not self._lead_times:
            return range(date - duration, date + 1)
        end = date + self.lead_time
        return range(end - duration, end + 1)

    def _extract_dates(self, item: int) -> np.ndarray:
        date = self._calc_date_range(item)
        features = self._extract_dataset(self._dataset, ["date"], {"date": date})
        return features["date"]

    def _extract_statics(self, item: int) -> np.ndarray:
        basin = self._sample_index[item]["basin"]
        features = self._extract_dataset(
            self._dataset, self._static_features, {"basin": basin}
        )
        return np.stack([features[e] for e in self._static_features], axis=-1)

    def _extract_hindcasts(self, item: int) -> dict[str, np.ndarray]:
        # Extract hindcast features without lead_time.
        dim_indexes_without_lead_time = self._sample_index[item].copy()
        dim_indexes_without_lead_time["date"] = range(
            dim_indexes_without_lead_time["date"] - self._seq_length + 1,
            dim_indexes_without_lead_time["date"] + 1,
        )
        features = self._extract_dataset(
            self._dataset,
            self._hindcast_features_without_lead_time,
            dim_indexes_without_lead_time,
        )

        # Forecast features with lead_time may be used as hindcast features. In that case, we select
        # only the first lead_time value, and move selection period one day backwards.
        dim_indexes_with_lead_time = self._sample_index[item].copy()
        dim_indexes_with_lead_time["lead_time"] = 0
        dim_indexes_with_lead_time["date"] = range(
            dim_indexes_with_lead_time["date"] - self._seq_length,
            dim_indexes_with_lead_time["date"],
        )
        features |= self._extract_dataset(
            self._dataset,
            self._hindcast_features_with_lead_time,
            dim_indexes_with_lead_time,
        )

        return {name: self._comp.expand_dims(feature, -1) for name, feature in features.items()}
        # TODO (future) :: This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works, and everything else expects
        # the trailing dim. Remove this dependency in the future.

    def _extract_forecasts(self, item: int) -> dict[str, np.ndarray]:
        forecast_indexer = self._sample_index[item].copy()
        forecast_indexer["lead_time"] = slice(MULTIMET_MINIMUM_LEAD_TIME, None)
        features = self._extract_dataset(
            self._dataset, self._forecast_features, forecast_indexer
        )

        if self._forecast_overlap is not None and self._forecast_overlap > 0:
            dim_indexes = self._sample_index[item].copy()
            dim_indexes["date"] = range(
                dim_indexes["date"] + 1 - self._min_lead_time - self._forecast_overlap,
                dim_indexes["date"] + 1 - self._min_lead_time,
            )
            dim_indexes["lead_time"] = 0
            overlaps = self._extract_dataset(
                self._dataset, self._forecast_features, dim_indexes
            )
            features = {
                name: self._comp.concatenate([overlaps[name], feature], axis=0)
                for name, feature in features.items()
            }
        return {name: self._comp.expand_dims(feature, axis=-1) for name, feature in features.items()}
        # TODO (future) :: This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works, and everything else expects
        # the trailing dim. Remove this dependency in the future.

    def _extract_targets(self, item: int) -> np.ndarray:
        dim_indexes = self._sample_index[item].copy()
        dim_indexes["date"] = self._calc_date_range(item, lead=True)
        features = self._extract_dataset(
            self._dataset, self._target_features, dim_indexes
        )
        return self._comp.stack([features[e] for e in self._target_features], axis=-1)

    def _extract_per_basin_stds(self, item: int) -> np.ndarray:
        assert self._per_basin_target_stds is not None
        features = self._extract_dataset(
            self._per_basin_target_stds,
            self._target_features,
            {"basin": self._sample_index[item]["basin"]},
        )
        return self._comp.expand_dims(
            self._comp.stack([features[e] for e in self._target_features], axis=-1), axis=0
        )
        # TODO (future) :: This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works, and everything else expects
        # the trailing dim. Remove this dependency in the future.

    def _get_period_dates(
        self, cfg: Config
    ) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        if self._period == "train":
            start_dates, end_dates = cfg.train_start_date, cfg.train_end_date
        elif self._period == "test":
            start_dates, end_dates = cfg.test_start_date, cfg.test_end_date
        elif self._period == "validation":
            start_dates, end_dates = cfg.validation_start_date, cfg.validation_end_date
        else:
            raise ValueError(f"Unknown period {self._period}")
        if len(start_dates) != len(end_dates):
            raise ValueError(
                f"Start and end date lists for period {self._period} must have the same length."
            )
        if any(start >= end for start, end in zip(start_dates, end_dates)):
            raise ValueError(
                f"Start dates {start_dates} are before matched end dates {end_dates}."
            )
        return start_dates, end_dates

    def _union_ranges(
        self, start_dates: list[pd.Timestamp], end_dates: list[pd.Timestamp]
    ) -> pd.DatetimeIndex:
        ranges = [
            pd.date_range(start, end) for start, end in zip(start_dates, end_dates)
        ]
        return functools.reduce(pd.Index.union, ranges)

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
            allzero_samples_are_invalid=self._allzero_samples_are_invalid,
        )[0]

        LOGGER.debug("valid_sample_mask compute")
        # Convert boolean valid sample mask into indexes of all samples. This retains
        # only the portion of the valid sample mask with True values.
        # Each element is a list of valid integer positions (indexers) for which
        # values are True for a dimension.
        indices = dask.compute(*dask.array.nonzero(valid_sample_mask.data))

        # Count the number of valid samples.
        num_samples = len(indices[0]) if indices else 0
        if num_samples == 0:
            if self._period == "train":
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError

        # Maps dim name to its respective list of int indices (index arrays) i.e. columns
        # of all basins, all dates, etc.
        vectorized_indices = {
            dim: indices[i]
            for i, dim in enumerate(valid_sample_mask.dims)
            if dim != "sample"
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

    def _extract_dataset(
        self,
        data: xr.Dataset,
        features: list[str],
        indexers: dict[Hashable, int | range | slice],
    ) -> dict[str, np.ndarray | np.float32]:
        def extract(feature_name: str):
            key = f"{id(data)}{feature_name}"
            feature = self._data_cache.get(key)
            if feature is None:
                feature = self._data_cache[key] = data[feature_name]
            return _extract_dataarray(feature, indexers)

        return {feature_name: extract(feature_name) for feature_name in features}

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
            LOGGER.debug("load attributes")
            datasets.append(self._load_static_features())
        if self._hindcast_features is not None:
            LOGGER.debug("load hindcast features")
            datasets.extend(self._load_hindcast_features())
        if self._forecast_features is not None:
            LOGGER.debug("load forecast features")
            datasets.extend(self._load_forecast_features())
        if self._target_features is not None:
            LOGGER.debug("load target features")
            datasets.append(self._load_target_features())
        if not datasets:
            raise ValueError("At least one type of data must be loaded.")
        LOGGER.debug("merge")
        return xr.merge(datasets, join='outer')

    def _load_hindcast_features(self) -> list[xr.Dataset]:
        """Load Caravan-Multimet data for hindcast features.

        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (date, basin).
        """
        # Prepare hindcast features to load, including the masks of union_mapping
        features = set(self._hindcast_features) | set(
            (self._union_mapping or {}).values()
        )

        # Separate products and bands for each product from feature names.
        product_bands = _get_products_and_bands_from_feature_strings(features=features)

        # Initialize storage for product/band dataframes that will eventually be concatenated.
        product_dss = []

        # Load data for the selected products, bands, and basins.
        for product, bands in product_bands.items():
            product_path = self._dynamics_data_path / product / "timeseries.zarr"
            product_ds = _open_zarr(product_path)

            if "lead_time" in product_ds:
                # The same product may be used both for forecast and hindcast features. For hindcast, we load it with the
                # full lead_time similar to forecast, and filter the minimal lead_time values during sampling.
                start = pd.Timedelta(days=MULTIMET_MINIMUM_LEAD_TIME)
                stop = pd.Timedelta(days=self.lead_time + 1)
                product_ds = product_ds.sel(
                    basin=self._basins,
                    lead_time=slice(start, stop),
                )
            else:
                product_ds = product_ds.sel(basin=self._basins)

            product_ds = product_ds[bands]

            product_dss.append(product_ds)

        return product_dss

    def _load_forecast_features(self) -> list[xr.Dataset]:
        """Load Caravan-Multimet data for forecast features.

        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (date, lead_time, basin).
        """
        # Separate products and bands for each product from feature names.
        product_bands = _get_products_and_bands_from_feature_strings(
            features=self._forecast_features
        )

        # Initialize storage for product/band dataframes that will eventually be concatenated.
        product_dss = []

        # Load data for the selected products, bands, and basins.
        for product, bands in product_bands.items():
            product_path = self._dynamics_data_path / product / "timeseries.zarr"
            product_ds = _open_zarr(product_path)

            # If this is a forecast product, extract only leadtime 0 for hindcasts.
            if "lead_time" not in product_ds:
                raise ValueError(
                    f"Lead times do not exist for forecast product ({product})."
                )

            start = pd.Timedelta(days=MULTIMET_MINIMUM_LEAD_TIME)
            stop = pd.Timedelta(days=self.lead_time + 1)
            product_ds = product_ds.sel(
                basin=self._basins,
                lead_time=slice(start, stop),
            )[bands]

            product_dss.append(product_ds)

        return product_dss

    def _load_target_features(self) -> xr.Dataset:
        """Load Caravan streamflow data.

        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (date, basin).
        """
        return load_caravan_timeseries_together(
            self._targets_data_path, self._basins, self._target_features
        )

    def _load_static_features(self) -> xr.Dataset:
        """Load Caravan static attributes.

        Returns
        -------
        xr.Dataset
            Dataset containing the loaded features with dimensions (basin).
        """
        return load_caravan_attributes(
            data_dir=self._statics_data_path,
            basins=self._basins,
            features=self._static_features,
        )

    @staticmethod
    def collate_fn(
        samples: list[
            dict[str, Union[torch.Tensor, np.ndarray, dict[str, torch.Tensor]]]
        ],
    ) -> dict[str, Union[torch.Tensor, np.ndarray, dict[str, torch.Tensor]]]:
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith("date"):
                # Dates are stored as a numpy array of datetime64, which we maintain as numpy array.
                batch[feature] = np.stack(
                    [sample[feature] for sample in samples], axis=0
                )
            elif feature.startswith("x_d"):
                # Dynamics are stored as dictionaries with feature names as keys.
                batch[feature] = {
                    k: torch.stack([sample[feature][k] for sample in samples], dim=0)
                    for k in samples[0][feature]
                }
            else:
                # Everything else is a torch.Tensor.
                batch[feature] = torch.stack(
                    [sample[feature] for sample in samples], dim=0
                )
        return batch


def _extract_dataarray(
    data: xr.DataArray, indexers: dict[Hashable, int | range | slice]
) -> np.ndarray | np.float32:
    """Returns the values in array according to dims given by indexers.

    This function replaces uses of `isel` with data and indexers.
    """
    locs = (indexers[dim] if dim in indexers else slice(None) for dim in data.dims)
    locs = (list(loc) if isinstance(loc, range) else loc for loc in locs)
    return data.data[tuple(locs)]


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
    if key not in TENSOR_VARS:
        raise ValueError(f"Unrecognized data key: {key}")
    if isinstance(value, dict):
        return {k: torch.from_numpy(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise ValueError(f"Unrecognized data type: {type(value)}")


def _open_zarr(path: Path) -> xr.Dataset:
    path = str(path).replace("gs:/", "gs://")
    return xr.open_zarr(store=path, chunks="auto", decode_timedelta=True)


def _get_products_and_bands_from_feature_strings(
    features: Iterable[str],
) -> dict[str, list[str]]:
    """
    Processes feature strings to create a dictionary of product to band(s).

    Parameters
    ----------
    features : list[str]
        A list features in the format `<product>_<band>. This is the format for feature
        names in the Multimet dataset.

    Returns
    -------
    dict[str, list[str]]
        Keys are product names and values are a list of features for that product. Features
        remain in the format <product>_<band>.
    """
    product_bands = {}
    for feature in features:
        product = feature.split("_")[0].upper()
        if product == "ERA5LAND":
            product = "ERA5_LAND"
        product_bands.setdefault(product, []).append(feature)
    return product_bands
