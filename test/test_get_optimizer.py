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

"""Tests for `get_optimizer` with modules and raw parameter groups."""

from collections.abc import Callable

import pytest
import torch

from googlehydrology.training import get_optimizer
from googlehydrology.utils.config import Config


@pytest.fixture
def cfg(get_config: Callable[[str], Config]) -> Config:
    """Real run configuration."""
    return get_config('forecast')


_OPTIMIZERS = [
    'adam',
    'adamw',
    'sgd',
    'asgd',
    'rmsprop',
    'adagrad',
    'adadelta',
    'adamax',
]


@pytest.mark.unit
@pytest.mark.parametrize('name', _OPTIMIZERS)
def test_get_optimizer_param_groups_step(cfg: Config, name: str) -> None:
    """Every optimizer accepts param groups with their own lr and steps."""
    cfg.update_config({'optimizer': name, 'initial_learning_rate': 0.01})
    a = torch.zeros(3, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    grad_a = torch.tensor([1.0, -2.0, 3.0])
    grad_b = torch.tensor([0.5, -0.5])
    optimizer = get_optimizer(
        [{'params': [a], 'lr': 0.5}, {'params': [b]}], cfg
    )

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert [g['lr'] for g in optimizer.param_groups] == [0.5, 0.01]
    assert optimizer.param_groups[0]['params'][0] is a

    loss = (a * grad_a).sum() + (b * grad_b).sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(a, torch.zeros(3))
    assert not torch.equal(b, torch.zeros(2))
    if name == 'sgd':
        torch.testing.assert_close(a, -0.5 * grad_a)
        torch.testing.assert_close(b, -0.01 * grad_b)
    if name == 'adam':
        # First Adam step with bias correction: m_hat = g, v_hat = g^2, so
        # the update is -lr * g / (|g| + eps) ~ -lr * sign(g).
        eps = optimizer.defaults['eps']
        torch.testing.assert_close(
            a, -0.5 * grad_a / (grad_a.abs() + eps), atol=0.5e-6, rtol=0
        )
        torch.testing.assert_close(
            b, -0.01 * grad_b / (grad_b.abs() + eps), atol=0.01e-6, rtol=0
        )


@pytest.mark.unit
def test_get_optimizer_accepts_module(cfg: Config) -> None:
    """A module is accepted through the ``model_or_params`` keyword."""
    cfg.update_config({'optimizer': 'adamw'})
    model = torch.nn.Linear(2, 1)
    optimizer = get_optimizer(model_or_params=model, cfg=cfg)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['params'][0] is model.weight


@pytest.mark.unit
def test_get_optimizer_unknown_name_raises(cfg: Config) -> None:
    """Unknown optimizer names raise NotImplementedError."""
    cfg.update_config({'optimizer': 'nope'})
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_optimizer(torch.nn.Linear(2, 1), cfg)
