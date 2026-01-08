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
import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from torch.utils.data import BatchSampler, SequentialSampler

from googlehydrology.datasetzoo.multimet import SampleIndexer


def metrics_to_dataframe(
    results: dict, metrics: Iterable[str], targets: Iterable[str]
) -> pd.DataFrame:
    """Extract all metric values from result dictionary and convert to pandas.DataFrame

    Parameters
    ----------
    results : dict
        Dictionary, containing the results of the model evaluation as returned by the `Tester.evaluate()`.
    metrics : Iterable[str]
        Iterable of metric names (without frequency suffix).
    targets : Iterable[str]
        Iterable of target variable names.

    Returns
    -------
    A basin indexed DataFrame with one column per metric. In case of multi-frequency runs, the metric names contain
    the corresponding frequency as a suffix.
    """
    metrics_dict = defaultdict(dict)
    for basin, basin_data in results.items():
        for freq, freq_results in basin_data.items():
            for target, metric in itertools.product(targets, metrics):
                metric_key = metric
                if len(targets) > 1:
                    metric_key = f'{target}_{metric}'
                if len(basin_data) > 1:
                    # For multi-frequency runs, metrics include a frequency suffix.
                    metric_key = f'{metric_key}_{freq}'
                if metric_key in freq_results.keys():
                    metrics_dict[basin][metric_key] = freq_results[metric_key]
                else:
                    # in case the current period has no valid samples, the result dict has no metric-key
                    metrics_dict[basin][metric_key] = np.nan

    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.index.name = 'basin'

    return df


class BasinBatchSampler(BatchSampler):
    """Groups samples by basin.

    Maps every basin to samples for it, and on iterations chunks them by batch size.
    """

    def __init__(
        self,
        sample_index: SampleIndexer,
        batch_size: int,
        basins_indexes: np.typing.NDArray[np.integer],
    ):
        super().__init__(
            SequentialSampler(range(len(sample_index))),
            batch_size,
            drop_last=False,
        )
        self.batch_size = batch_size

        col = sample_index.get_column('basin')

        if len(basins_indexes):  # Binary search the already sorted indexes
            starts = np.searchsorted(col, basins_indexes, side='left')
            ends = np.searchsorted(col, basins_indexes, side='right')
            self._starts, self._counts = starts, ends - starts
        else:  # Find boundary changes eg for all [0,0,0,1,2,2,2,2,3,3]
            changes = np.flatnonzero(col[:-1] != col[1:]) + 1
            bounds = np.concatenate(([0], changes, [len(col)]))
            # Starts are bounds' left edges eg [0,3,4,8](,10)
            # Counts are ends - starts ie the distance (diff) eg [3,1,4,2]
            self._starts, self._counts = bounds[:-1], np.diff(bounds)
        # Array lengths are in terms of num basins eg len changes, len starts,
        # etc. Or use bool masks that are 1 byte instead of 8 bytes (64 bit) so
        # the col's compare size is 1/8 of col.

        fulls, remainders = np.divmod(self._counts, batch_size)
        self._num_batches = int(fulls.sum() + np.count_nonzero(remainders))


    def __iter__(self):
        for start, count in zip(self._starts, self._counts, strict=True):
            end = start + count
            yield from itertools.batched(range(start, end), self.batch_size)

    def __len__(self):
        return self._num_batches


def get_samples_indexes(values: list[str], *, samples: list[str]):
    """Returns indexes of elements from samples that are in values."""
    return np.flatnonzero(np.isin(values, samples))
