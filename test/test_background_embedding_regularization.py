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

"""Tests for `BackgroundEmbeddingRegularization` and the optimizer factory."""

from unittest.mock import MagicMock

import pytest
import torch

from googlehydrology.training import (
    get_optimizer,
    get_regularization_obj,
)
from googlehydrology.training.loss import MaskedMSELoss
from googlehydrology.training.regularization import (
    BackgroundEmbeddingRegularization,
)


def _components():
    gen = torch.Generator().manual_seed(0)
    opt = {
        'static': torch.randn(4, 8, generator=gen, requires_grad=True),
        'dynamic': torch.randn(4, 5, 8, generator=gen, requires_grad=True),
        'no_baseline': torch.randn(4, 3, generator=gen, requires_grad=True),
    }
    base = {
        'static': torch.randn(4, 8, generator=gen),
        'dynamic': torch.randn(4, 5, 8, generator=gen),
    }
    return opt, base


@pytest.mark.unit
def test_matches_manual_formula_and_skips_missing_baselines():
    opt, base = _components()
    weights = {'static': 1e-3, 'dynamic': 0.5, 'no_baseline': 100.0}
    reg = BackgroundEmbeddingRegularization(cfg=None)
    out = reg(
        {},
        {},
        {
            'optimized_components': opt,
            'baseline_components': base,
            'component_weights': weights,
        },
    )
    expected = 1e-3 * torch.mean((opt['static'] - base['static']) ** 2)
    expected = expected + 0.5 * torch.mean(
        (opt['dynamic'] - base['dynamic']) ** 2
    )
    torch.testing.assert_close(out, expected)
    assert reg.name == 'bg_embedding'

    out.backward()
    assert opt['static'].grad is not None
    assert opt['no_baseline'].grad is None


@pytest.mark.unit
def test_missing_weight_defaults_to_one():
    opt, base = _components()
    reg = BackgroundEmbeddingRegularization(cfg=None)
    out = reg(
        {},
        {},
        {
            'optimized_components': {'static': opt['static']},
            'baseline_components': base,
        },
    )
    torch.testing.assert_close(
        out, torch.mean((opt['static'] - base['static']) ** 2)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    'other',
    [
        {},
        {'optimized_components': {}, 'baseline_components': {}},
        {
            'optimized_components': {'a': torch.ones(2)},
            'baseline_components': {},
        },
    ],
)
def test_zero_when_nothing_to_regularize(other):
    out = BackgroundEmbeddingRegularization(cfg=None)({}, {}, other)
    assert out.dtype == torch.float32
    assert out.item() == 0.0


@pytest.mark.unit
def test_linked_in_factory_and_reached_via_loss():
    cfg = MagicMock()
    cfg.regularization = [('bg_embedding', 0.25)]
    cfg.predict_last_n = 2
    cfg.no_loss_frequencies = []
    cfg.target_variables = ['streamflow']
    cfg.target_loss_weights = None

    (reg,) = get_regularization_obj(cfg)
    assert isinstance(reg, BackgroundEmbeddingRegularization)
    assert reg.weight == 0.25

    cfg.regularization = ['bg_embedding']
    (reg,) = get_regularization_obj(cfg)
    assert reg.weight == 1.0

    loss_fn = MaskedMSELoss(cfg)
    loss_fn.set_regularization_terms([reg])
    opt, base = _components()
    prediction = {'y_hat': torch.zeros(1, 2, 1)}
    data = {'y': torch.zeros(1, 2, 1)}
    total, all_losses = loss_fn(
        prediction,
        data,
        other_model_data={
            'optimized_components': opt,
            'baseline_components': base,
            'component_weights': {'static': 1.0, 'dynamic': 0.0},
        },
    )
    expected = torch.mean((opt['static'] - base['static']) ** 2)
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(all_losses['bg_embedding'], expected)


def _optimizer_cfg(name='adam', lr=0.01):
    cfg = MagicMock()
    cfg.optimizer = name
    cfg.initial_learning_rate = lr
    return cfg


@pytest.mark.unit
def test_get_optimizer_accepts_tensors():
    tensors = [
        torch.zeros(3, requires_grad=True),
        torch.ones(2, requires_grad=True),
    ]
    optimizer = get_optimizer(tensors, _optimizer_cfg())
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.01
    assert optimizer.param_groups[0]['params'][0] is tensors[0]


@pytest.mark.unit
def test_get_optimizer_accepts_param_groups():
    a = torch.zeros(3, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    optimizer = get_optimizer(
        [{'params': [a], 'lr': 0.5}, {'params': [b]}],
        _optimizer_cfg('sgd', lr=0.01),
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert [g['lr'] for g in optimizer.param_groups] == [0.5, 0.01]


@pytest.mark.unit
def test_get_optimizer_accepts_module():
    model = torch.nn.Linear(2, 1)
    optimizer = get_optimizer(model, _optimizer_cfg('adamw'))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['params'][0] is model.weight
