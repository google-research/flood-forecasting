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

"""Tests for the per-call overrides of `BaseLoss.forward`."""

from unittest.mock import MagicMock

import pytest
import torch

from googlehydrology.training.loss import (
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
)
from googlehydrology.training.regularization import BaseRegularization


def _cfg(predict_last_n):
    cfg = MagicMock()
    cfg.predict_last_n = predict_last_n
    cfg.no_loss_frequencies = []
    cfg.target_variables = ['streamflow']
    cfg.target_loss_weights = None
    return cfg


class _RecordingRegularization(BaseRegularization):
    def __init__(self):
        super().__init__(cfg=None, name='recording', weight=1.0)
        self.seen = None

    def forward(self, prediction, ground_truth, other_model_data):
        self.seen = other_model_data
        return torch.tensor(0.0)


def _data(seq_len=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    y_hat = torch.randn(2, seq_len, 1, generator=gen)
    y = torch.randn(2, seq_len, 1, generator=gen)
    stds = torch.rand(2, 1, 1, generator=gen) + 0.5
    return {'y_hat': y_hat}, {'y': y, 'per_basin_target_stds': stds}


@pytest.mark.unit
@pytest.mark.parametrize(
    'loss_cls', [MaskedMSELoss, MaskedRMSELoss, MaskedNSELoss]
)
@pytest.mark.parametrize('override', [3, {'1D': 3}])
def test_predict_last_n_override_matches_config(loss_cls, override):
    prediction, data = _data()
    expected, _ = loss_cls(_cfg({'1D': 3}))(prediction, data)
    actual, _ = loss_cls(_cfg({'1D': 6}))(
        prediction, data, predict_last_n=override
    )
    assert torch.equal(actual, expected)


@pytest.mark.unit
def test_predict_last_n_override_multi_frequency():
    cfg = _cfg({'1D': 2, '1h': 4})
    cfg_expected = _cfg({'1D': 1, '1h': 4})
    prediction = {
        'y_hat_1D': torch.randn(1, 3, 1),
        'y_hat_1h': torch.randn(1, 5, 1),
    }
    data = {'y_1D': torch.randn(1, 3, 1), 'y_1h': torch.randn(1, 5, 1)}
    expected, _ = MaskedMSELoss(cfg_expected)(prediction, data)
    actual, _ = MaskedMSELoss(cfg)(prediction, data, predict_last_n={'1D': 1})
    assert torch.equal(actual, expected)

    with pytest.raises(ValueError, match='unknown frequency'):
        MaskedMSELoss(cfg)(prediction, data, predict_last_n={'3h': 1})


@pytest.mark.unit
def test_default_path_unchanged_vs_explicit_none():
    prediction, data = _data()
    loss_fn = MaskedNSELoss(_cfg(4))
    default, default_all = loss_fn(prediction, data)
    explicit, explicit_all = loss_fn(
        prediction, data, predict_last_n=None, other_model_data=None
    )
    assert torch.equal(default, explicit)
    assert dict(default_all).keys() == dict(explicit_all).keys()


@pytest.mark.unit
def test_other_model_data_reaches_regularization():
    prediction, data = _data()
    prediction['extra'] = torch.ones(1)
    loss_fn = MaskedMSELoss(_cfg(4))
    reg = _RecordingRegularization()
    loss_fn.set_regularization_terms([reg])

    loss_fn(prediction, data)
    assert set(reg.seen) == {'extra'}

    marker = torch.zeros(3)
    loss_fn(prediction, data, other_model_data={'marker': marker})
    assert set(reg.seen) == {'extra', 'marker'}
    assert reg.seen['marker'] is marker
