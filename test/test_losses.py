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

"""Unit tests for googlehydrology.training.loss and regularization."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from googlehydrology.training.loss import (
    BaseLoss,
    MaskedCMALLoss,
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
    _get_predict_last_n,
)
from googlehydrology.training.regularization import BaseRegularization


class DummyRegularization(BaseRegularization):
    def __init__(self, cfg=None, weight: float = 0.5):
        super().__init__(cfg=cfg, name='dummy_reg', weight=weight)

    def forward(self, prediction, ground_truth, model_parameters):
        return torch.tensor(2.0)


@pytest.fixture
def dummy_config():
    cfg = MagicMock()
    cfg.predict_last_n = 10
    cfg.no_loss_frequencies = []
    cfg.target_variables = ['streamflow']
    cfg.target_loss_weights = None
    cfg.n_distributions = 3
    return cfg


@pytest.mark.unit
def test_get_predict_last_n():
    cfg_int = MagicMock(predict_last_n=5)
    assert _get_predict_last_n(cfg_int) == {'': 5}

    cfg_single_dict = MagicMock(predict_last_n={'1D': 7})
    assert _get_predict_last_n(cfg_single_dict) == {'': 7}

    cfg_multi_dict = MagicMock(predict_last_n={'1D': 7, '1h': 24})
    assert _get_predict_last_n(cfg_multi_dict) == {'1D': 7, '1h': 24}


@pytest.mark.unit
def test_base_loss_target_weights(dummy_config):
    # Single target default equal weights
    loss = MaskedMSELoss(dummy_config)
    assert torch.allclose(loss._target_weights, torch.tensor([1.0]))

    # Multi-target default equal weights
    dummy_config.target_variables = ['var1', 'var2']
    dummy_config.target_loss_weights = None
    loss = MaskedMSELoss(dummy_config)
    assert torch.allclose(loss._target_weights, torch.tensor([0.5, 0.5]))

    # Multi-target custom weights
    dummy_config.target_loss_weights = [0.8, 0.2]
    loss = MaskedMSELoss(dummy_config)
    assert torch.allclose(loss._target_weights, torch.tensor([0.8, 0.2]))

    # Weight length mismatch error
    dummy_config.target_loss_weights = [0.5]
    with pytest.raises(
        ValueError,
        match='Number of weights must be equal to the number of target',
    ):
        MaskedMSELoss(dummy_config)


@pytest.mark.unit
def test_masked_mse_loss(dummy_config):
    loss_fn = MaskedMSELoss(dummy_config)

    # Batch of 2, sequence of 10, 1 target
    y_hat = torch.tensor([[[2.0], [4.0]], [[6.0], [8.0]]])  # shape [2, 2, 1]
    y = torch.tensor([[[1.0], [np.nan]], [[4.0], [10.0]]])   # shape [2, 2, 1]

    dummy_config.predict_last_n = 2
    loss_fn = MaskedMSELoss(dummy_config)

    prediction = {'y_hat': y_hat}
    data = {'y': y}

    total_loss, all_losses = loss_fn(prediction, data)
    # Valid differences:
    # (2-1)=1 -> 1^2=1; (6-4)=2 -> 2^2=4; (8-10)=-2 -> (-2)^2=4
    # Mean of squared errors = (1 + 4 + 4) / 3 = 3.0
    # MaskedMSE multiplies by 0.5 -> 1.5
    assert np.isclose(total_loss.item(), 1.5)
    assert 'loss' in all_losses
    assert 'total_loss' in all_losses


@pytest.mark.unit
def test_masked_rmse_loss(dummy_config):
    dummy_config.predict_last_n = 2
    loss_fn = MaskedRMSELoss(dummy_config)

    y_hat = torch.tensor([[[2.0], [4.0]]])  # shape [1, 2, 1]
    y = torch.tensor([[[1.0], [np.nan]]])    # shape [1, 2, 1]

    prediction = {'y_hat': y_hat}
    data = {'y': y}

    total_loss, _ = loss_fn(prediction, data)
    # (2-1)^2 = 1.0 * 0.5 = 0.5 -> sqrt(0.5)
    expected = np.sqrt(0.5)
    assert np.isclose(total_loss.item(), expected)


@pytest.mark.unit
def test_masked_nse_loss(dummy_config):
    dummy_config.predict_last_n = 2
    loss_fn = MaskedNSELoss(dummy_config, eps=0.1)

    y_hat = torch.tensor([[[2.0], [4.0]]])  # shape [1, 2, 1]
    y = torch.tensor([[[1.0], [3.0]]])      # shape [1, 2, 1]
    per_basin_target_stds = torch.tensor([[[1.0]]])  # shape [1, 1, 1]

    prediction = {'y_hat': y_hat}
    data = {'y': y, 'per_basin_target_stds': per_basin_target_stds}

    total_loss, _ = loss_fn(prediction, data)
    # Squared errors: (2-1)^2 = 1.0, (4-3)^2 = 1.0 -> Mean = 1.0
    # Weights = 1 / (1.0 + 0.1)^2 = 1 / 1.21 = 0.826446
    # Scaled loss = 1.0 * 0.826446 = 0.826446
    expected = 1.0 / (1.1 ** 2)
    assert np.isclose(total_loss.item(), expected, atol=1e-5)


@pytest.mark.unit
def test_masked_cmal_loss(dummy_config):
    dummy_config.predict_last_n = 2
    dummy_config.n_distributions = 3
    loss_fn = MaskedCMALLoss(dummy_config)

    batch_size = 2
    seq_len = 2
    n_dist = 3

    # Create synthetic CMAL predictions
    mu = torch.zeros(batch_size, seq_len, n_dist)
    b = torch.ones(batch_size, seq_len, n_dist)
    tau = torch.full((batch_size, seq_len, n_dist), 0.5)
    pi = torch.full((batch_size, seq_len, n_dist), 1.0 / n_dist)

    y = torch.zeros(batch_size, seq_len, 1)

    prediction = {'mu': mu, 'b': b, 'tau': tau, 'pi': pi}
    data = {'y': y}

    total_loss, all_losses = loss_fn(prediction, data)
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0


@pytest.mark.unit
def test_loss_with_regularization(dummy_config):
    dummy_config.predict_last_n = 2
    loss_fn = MaskedMSELoss(dummy_config)

    reg = DummyRegularization(weight=0.5)
    loss_fn.set_regularization_terms([reg])

    y_hat = torch.tensor([[[2.0], [2.0]]])
    y = torch.tensor([[[2.0], [2.0]]])

    prediction = {'y_hat': y_hat}
    data = {'y': y}

    total_loss, all_losses = loss_fn(prediction, data)
    # MSE loss = 0.0, reg = 0.5 * 2.0 = 1.0
    assert np.isclose(total_loss.item(), 1.0)
    assert 'dummy_reg' in all_losses
    assert all_losses['dummy_reg'].item() == 2.0


@pytest.mark.unit
def test_multi_frequency_loss():
    cfg = MagicMock()
    cfg.predict_last_n = {'1D': 2, '1h': 4}
    cfg.no_loss_frequencies = ['1h']  # Exclude 1h from loss
    cfg.target_variables = ['streamflow']
    cfg.target_loss_weights = None

    loss_fn = MaskedMSELoss(cfg)

    prediction = {
        'y_hat_1D': torch.tensor([[[2.0], [3.0]]]),
        'y_hat_1h': torch.tensor([[[10.0], [10.0], [10.0], [10.0]]]),
    }
    data = {
        'y_1D': torch.tensor([[[1.0], [3.0]]]),
        'y_1h': torch.tensor([[[0.0], [0.0], [0.0], [0.0]]]),
    }

    total_loss, _ = loss_fn(prediction, data)
    # Only 1D considered: (2-1)^2 = 1, (3-3)^2 = 0 -> Mean = 0.5
    # Scaled loss: 0.5 * 0.5 = 0.25
    assert np.isclose(total_loss.item(), 0.25)
