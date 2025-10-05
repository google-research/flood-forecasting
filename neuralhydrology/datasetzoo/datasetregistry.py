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

from neuralhydrology.utils.config import Config


class DatasetRegistry:
    """Class that registers dataset classes that can be used with neuralhydrology."""

    def __init__(self):
        self.__dataset_class = {}

    def register_dataset_class(self, key: str, new_class: Type):
        """Adds a new dataset class to the dataset registry.

        Parameters
        ----------
        key : str
            The unique identifier for the dataset class. This key will be used in configuration files
            to specify which dataset to use.
        new_class : Type
            The dataset class to register. Must be a subclass of Dataset.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the provided class is not a subclass of Dataset.

        Examples
        --------
        >>> registry = DatasetZooRegistry()
        >>> registry.register_dataset_class("my_dataset", MyCustomDataset)
        """
        if not issubclass(new_class, Dataset):
            raise TypeError(f"Class {type(new_class)} is not a subclass of Dataset.")
        self.__dataset_class[key] = new_class

    def instantiate_dataset(self,
                            cfg: Config,
                            is_train: bool,
                            period: str,
                            basin: str = None,
                            compute_scaler: bool = True) -> Dataset:
        """Creates and returns an instance of a dataset class based on the configuration.

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
            If passed, the data for only this basin will be loaded. Otherwise the basin(s) is(are) read from the appropriate
            basin file, corresponding to the `period`.
        compute_scaler : bool
            Forces the dataset to calculate a new scaler instead of loading a precalculated scaler. Used during training, but
            not finetuning.

        Returns
        -------
        Dataset
            An instance of the appropriate dataset class based on the configuration.

        Raises
        ------
        NotImplementedError
            If no dataset class is implemented for the dataset specified in the configuration.
        """
        dataset_key = cfg.dataset.lower()
        Dataset = self.__dataset_class.get(dataset_key, None)
        if Dataset is None:
            raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")

        return Dataset(cfg=cfg,
                       is_train=is_train,
                       period=period,
                       basin=basin,
                       compute_scaler=compute_scaler)
