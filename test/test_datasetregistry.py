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

"""Unit tests for googlehydrology.datasetzoo.datasetregistry."""

from unittest.mock import MagicMock

import pytest
from torch.utils.data import Dataset

from googlehydrology.datasetzoo import get_dataset, register_dataset
from googlehydrology.datasetzoo.datasetregistry import DatasetRegistry


class DummyValidDataset(Dataset):
    def __init__(self, cfg, is_train, period, basins=None, compute_scaler=True):
        self.cfg = cfg
        self.is_train = is_train
        self.period = period
        self.basins = basins

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


class DummyInvalidClass:
    pass


@pytest.mark.unit
def test_dataset_registry_registration():
    registry = DatasetRegistry()
    registry.register_dataset_class('dummy', DummyValidDataset)

    # Instantiate
    cfg = MagicMock(dataset='dummy')
    instance = registry.instantiate_dataset(
        cfg=cfg,
        is_train=True,
        period='train',
        basins=['basin1'],
        compute_scaler=True,
    )
    assert isinstance(instance, DummyValidDataset)
    assert instance.is_train is True
    assert instance.period == 'train'
    assert instance.basins == ['basin1']


@pytest.mark.unit
def test_dataset_registry_invalid_type():
    registry = DatasetRegistry()
    with pytest.raises(TypeError, match='is not a subclass of Dataset'):
        registry.register_dataset_class('invalid', DummyInvalidClass)


@pytest.mark.unit
def test_dataset_registry_unimplemented_dataset():
    registry = DatasetRegistry()
    cfg = MagicMock(dataset='unregistered_dataset')
    with pytest.raises(
        NotImplementedError, match='No dataset class implemented'
    ):
        registry.instantiate_dataset(
            cfg=cfg,
            is_train=True,
            period='train',
        )


@pytest.mark.unit
def test_module_level_register_and_get_dataset():
    register_dataset('dummy_module_dataset', DummyValidDataset)
    cfg = MagicMock(dataset='dummy_module_dataset')
    instance = get_dataset(cfg=cfg, is_train=False, period='test')
    assert isinstance(instance, DummyValidDataset)
