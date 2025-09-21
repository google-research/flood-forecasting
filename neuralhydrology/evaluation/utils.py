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
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from torch.utils.data import SequentialSampler, BatchSampler


def load_basin_id_encoding(run_dir: Path) -> Dict[str, int]:
    id_to_int_file = run_dir / "train_data" / "id_to_int.yml"
    if id_to_int_file.is_file():
        with id_to_int_file.open("r") as fp:
            yaml = YAML(typ="safe")
            id_to_int = yaml.load(fp)
        return id_to_int

    else:
        id_to_int_file = run_dir / "train_data" / "id_to_int.p"
        if id_to_int_file.is_file():
            with id_to_int_file.open("rb") as fp:
                id_to_int = pickle.load(fp)
            return id_to_int
        else:
            raise FileNotFoundError(f"No id-to-int file found in {id_to_int_file.parent}. "
                                    "Looked for (new) yaml file or (old) pickle file")


def metrics_to_dataframe(results: dict, metrics: Iterable[str], targets: Iterable[str]) -> pd.DataFrame:
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
                    metric_key = f"{target}_{metric}"
                if len(basin_data) > 1:
                    # For multi-frequency runs, metrics include a frequency suffix.
                    metric_key = f"{metric_key}_{freq}"
                if metric_key in freq_results.keys():
                    metrics_dict[basin][metric_key] = freq_results[metric_key]
                else:
                    # in case the current period has no valid samples, the result dict has no metric-key
                    metrics_dict[basin][metric_key] = np.nan

    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.index.name = "basin"

    return df


class BasinBatchSampler(BatchSampler):
    """Groups samples by basin.
    
    Maps every basin to samples for it, and on iterations chunks them by batch size.
    """

    def __init__(
        self,
        sample_index: dict[int, dict[str, int]],
        batch_size: int,
        basins_indexes: set[int] = set(),
    ):
        super().__init__(SequentialSampler(range(len(sample_index))), batch_size, drop_last=False)

        self._batch_size = batch_size

        self._basin_indices: dict[int, list[int]] = {}
        for sample, data in sample_index.items():
            basin_index = data['basin']
            if (not basins_indexes) or basin_index in basins_indexes:
                self._basin_indices.setdefault(basin_index, []).append(sample)

        self._num_batches = sum(
            math.ceil(len(indices) / batch_size)
            for indices in self._basin_indices.values()
        )

    def __iter__(self):
        for indices in self._basin_indices.values():  # for every basin
            for i in range(0, len(indices), self._batch_size):  # chunk into batches
                yield indices[i:i + self._batch_size]  # batch

    def __len__(self):
        return self._num_batches
