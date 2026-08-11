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

"""Unit tests for googlehydrology.training helpers and BaseTrainer."""

from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn

from googlehydrology.training import (
    get_loss_obj,
    get_optimizer,
    get_regularization_obj,
)
from googlehydrology.training.loss import (
    MaskedCMALLoss,
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
)


@pytest.mark.unit
def test_get_optimizer_all_types():
    model = nn.Linear(5, 2)
    optimizers = ['adam', 'adamw', 'sgd', 'asgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax']
    for opt_name in optimizers:
        cfg = MagicMock()
        cfg.optimizer = opt_name
        cfg.initial_learning_rate = 0.001
        opt = get_optimizer(model, cfg)
        assert isinstance(opt, torch.optim.Optimizer)

    # Unsupported optimizer
    cfg_invalid = MagicMock(optimizer='invalid_optimizer', initial_learning_rate=0.001)
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_optimizer(model, cfg_invalid)


@pytest.mark.unit
def test_get_loss_obj():
    loss_types = {
        'mse': MaskedMSELoss,
        'rmse': MaskedRMSELoss,
        'nse': MaskedNSELoss,
        'cmalloss': MaskedCMALLoss,
    }
    for loss_name, expected_class in loss_types.items():
        cfg = MagicMock()
        cfg.loss = loss_name
        cfg.predict_last_n = 1
        cfg.no_loss_frequencies = []
        cfg.target_variables = ['flow']
        cfg.target_loss_weights = None
        cfg.n_distributions = 3

        loss_obj = get_loss_obj(cfg)
        assert isinstance(loss_obj, expected_class)

    # Unsupported loss
    cfg_invalid = MagicMock(loss='invalid_loss')
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_loss_obj(cfg_invalid)


@pytest.mark.unit
def test_get_regularization_obj():
    cfg = MagicMock()
    cfg.regularization = ['forecast_overlap', ('forecast_overlap', 0.5)]

    reg_objs = get_regularization_obj(cfg)
    assert len(reg_objs) == 2
    assert reg_objs[0].name == 'forecast_overlap'
    assert reg_objs[0].weight == 1.0
    assert reg_objs[1].name == 'forecast_overlap'
    assert reg_objs[1].weight == 0.5

    # Unsupported regularization
    cfg_invalid = MagicMock(regularization=['invalid_reg'])
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_regularization_obj(cfg_invalid)
