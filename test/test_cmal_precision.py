# Copyright 2026 Google LLC
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

"""Regression tests for CMAL component and likelihood precision."""

import math

import pytest
import torch

from googlehydrology.modelzoo.head import CMAL
from googlehydrology.training.loss import MaskedCMALLoss
from googlehydrology.utils.config import Config


def _loss() -> MaskedCMALLoss:
    return MaskedCMALLoss(
        Config(
            {
                'target_variables': ['streamflow'],
                'predict_last_n': 2,
                'n_distributions': 3,
            }
        )
    )


def _head(
    scale_latent: float = 0.0,
    skew_latent: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> CMAL:
    head = CMAL(n_in=2, n_out=12, n_hidden=2).to(dtype=dtype)
    with torch.no_grad():
        head.fc1.weight.copy_(torch.eye(2, dtype=dtype))
        head.fc1.bias.zero_()
        head.fc2.weight.zero_()
        head.fc2.bias.copy_(
            torch.tensor(
                [0.0] * 3 + [scale_latent] * 3 + [skew_latent] * 3 + [0.0] * 3,
                dtype=dtype,
            )
        )
    return head


@pytest.mark.unit
@pytest.mark.parametrize(
    'dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize('skew_latent', [-100.0, 100.0])
def test_cmal_saturated_skew_has_finite_loss_and_gradients(
    dtype: torch.dtype, skew_latent: float
) -> None:
    head = _head(skew_latent=skew_latent, dtype=dtype)
    prediction = head(torch.ones(1, 2, 2, dtype=dtype))
    value, _ = _loss()(prediction, {'y': torch.ones(1, 2, 1, dtype=dtype)})
    value.backward()

    assert torch.isfinite(value)
    assert torch.all(prediction['tau'] > 0)
    assert torch.all(prediction['tau'] < 1)
    for tensor in prediction.values():
        assert tensor.dtype == torch.promote_types(dtype, torch.float32)
    for parameter in head.parameters():
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('scale_latent', 'skew_latent'), [(0.0, 100.0), (-20.0, 0.0)]
)
def test_cmal_autocast_optimizer_step_stays_finite(
    dtype: torch.dtype, scale_latent: float, skew_latent: float
) -> None:
    head = _head(scale_latent=scale_latent, skew_latent=skew_latent)
    optimizer = torch.optim.SGD(head.parameters(), lr=1e-6)
    x = torch.ones(1, 2, 2)
    y = torch.full((1, 2, 1), 2.0)

    with torch.autocast('cpu', dtype=dtype):
        prediction = head(x)
        value, _ = _loss()(prediction, {'y': y})
    value.backward()

    assert torch.isfinite(value)
    assert value.dtype == torch.float32
    for parameter in head.parameters():
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    for parameter in head.parameters():
        assert torch.isfinite(parameter).all()
    with torch.autocast('cpu', dtype=dtype):
        next_value, _ = _loss()(head(x), {'y': y})
    assert torch.isfinite(next_value)


@pytest.mark.unit
def test_cmal_loss_promotes_before_residual_subtraction() -> None:
    prediction = {
        'mu': torch.full((1, 2, 3), -60000.0, dtype=torch.float16),
        'b': torch.full((1, 2, 3), 10000.0, dtype=torch.float16),
        'tau': torch.full((1, 2, 3), 0.5, dtype=torch.float16),
        'pi': torch.tensor([[[0.25, 0.5, 0.25]]], dtype=torch.float16)
        .expand(1, 2, 3)
        .clone(),
    }
    prediction['mu'].requires_grad_()
    y = torch.full((1, 2, 1), 60000.0, dtype=torch.float16)
    value, _ = _loss()(prediction, {'y': y})
    value.backward()

    # Equal symmetric components reduce to Laplace(location, 2 * scale).
    reference = torch.distributions.Laplace(
        torch.tensor(-60000.0, dtype=torch.float64),
        torch.tensor(20000.0, dtype=torch.float64),
    )
    expected = -reference.log_prob(torch.tensor(60000.0)) - math.log1p(3e-8)
    torch.testing.assert_close(value.double(), expected, rtol=1e-6, atol=0)
    assert torch.isfinite(prediction['mu'].grad).all()


@pytest.mark.unit
def test_cmal_all_missing_half_values_avoid_sum_overflow() -> None:
    mu = torch.full((1, 2, 3), 60000.0, dtype=torch.float16, requires_grad=True)
    prediction = {
        'mu': mu,
        'b': torch.ones_like(mu),
        'tau': torch.full_like(mu, 0.5),
        'pi': torch.full_like(mu, 1.0 / 3),
    }
    value, _ = _loss()(prediction, {'y': torch.full((1, 2, 1), torch.nan)})
    value.backward()
    assert value.item() == 0.0
    torch.testing.assert_close(mu.grad, torch.zeros_like(mu))


@pytest.mark.unit
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cmal_ordinary_values_preserve_precision(dtype: torch.dtype) -> None:
    head = _head(dtype=dtype)
    prediction = head(torch.ones(1, 2, 2, dtype=dtype))
    expected = {
        'mu': 0.0,
        'b': math.log(2) / 2 + 1e-5,
        'tau': 0.500005,
        'pi': 0.33334,
    }
    for key, value in expected.items():
        torch.testing.assert_close(
            prediction[key], torch.full((1, 2, 3), value, dtype=dtype)
        )
    assert set(head.state_dict()) == {
        'fc1.weight',
        'fc1.bias',
        'fc2.weight',
        'fc2.bias',
    }
    value, _ = _loss()(prediction, {'y': torch.zeros(1, 2, 1, dtype=dtype)})
    assert value.dtype == dtype
    assert torch.isfinite(value)
