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

from typing import Type

from torch.utils.data import Dataset

from neuralhydrology.datasetzoo.multimet import Multimet
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo.datasetregistry import DatasetRegistry


def get_dataset(cfg: Config,
                is_train: bool,
                period: str,
                basin: str = None,
                additional_features: list = [],
                id_to_int: dict = {},
                compute_scaler: bool = False) -> Dataset:
    """Get data set instance, depending on the run configuration.

    Currently implemented datasets are 'multimet',
    as well as the 'generic' dataset class that can be used for any kind of dataset as long as it is
    in the correct format.

    New dataset classes can be added at the beginning of runtime using the function register_dataset().

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
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) is(are) read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    compute_scaler : bool
        Forces the dataset to calculate a new scaler instead of loading a precalculated scaler. Used during training, but
        not finetuning.

    Returns
    -------
    Dataset
        A new data set instance, depending on the run configuration.

    Raises
    ------
    NotImplementedError
        If no data set class is implemented for the 'dataset' argument in the config.
    """
    global _datasetZooRegistry

    return _datasetZooRegistry.instantiate_dataset(cfg, is_train, period, basin, additional_features, id_to_int, compute_scaler)


def register_dataset(key: str, new_class: Type):
    """Adds a dataset class to the dataset registry.
    
    This class must derive from Dataset. New dataset class has to be added at the beginning of runtime.

    Parameters
    ----------
    key : str
        The key of the dataset that is set in the configuration file.

    new_class : Type
        The new Dataset class to register.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the new class is not derived from Dataset.
    """
    global _datasetZooRegistry
    _datasetZooRegistry.register_dataset_class(key, new_class)


_datasetZooRegistry: DatasetRegistry = DatasetRegistry()

_datasetZooRegistry.register_dataset_class("multimet", Multimet)
