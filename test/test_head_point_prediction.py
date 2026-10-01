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

"""``point_prediction`` of the model heads."""

import pytest
import torch

from googlehydrology.modelzoo.head import CMAL, BaseHead, Regression
from googlehydrology.utils import cmal_deterministic

BATCH_SIZE = 2
TIME_STEPS = 5
N_DISTRIBUTIONS = 3


def _cmal_outputs(
    n_targets: int, *, requires_grad: bool = False
) -> dict[str, torch.Tensor]:
    """Random, valid CMAL parameters laid out like the head output."""
    generator = torch.Generator().manual_seed(0)
    shape = (BATCH_SIZE, TIME_STEPS, n_targets * N_DISTRIBUTIONS)
    mu = torch.randn(shape, generator=generator)
    b = torch.rand(shape, generator=generator) + 0.1
    tau = 0.8 * torch.rand(shape, generator=generator) + 0.1
    pi = torch.softmax(
        torch.randn(shape, generator=generator).view(
            BATCH_SIZE, TIME_STEPS, n_targets, N_DISTRIBUTIONS
        ),
        dim=-1,
    ).view(shape)
    return {
        'mu': mu.requires_grad_(requires_grad),
        'b': b,
        'tau': tau,
        'pi': pi,
    }


@pytest.mark.unit
def test_base_head_raises() -> None:
    """The base head has no point prediction."""
    with pytest.raises(NotImplementedError, match='BaseHead'):
        BaseHead().point_prediction({'y_hat': torch.zeros(1, 1, 1)})


@pytest.mark.unit
def test_regression_returns_y_hat() -> None:
    """Regression returns the ``y_hat`` tensor itself."""
    head = Regression(n_in=4, n_out=2)
    outputs = head(torch.randn(BATCH_SIZE, TIME_STEPS, 4))
    point = head.point_prediction(outputs)
    assert point is outputs['y_hat']
    assert point.shape == (BATCH_SIZE, TIME_STEPS, 2)


@pytest.mark.unit
@pytest.mark.parametrize('n_targets', [1, 2])
def test_cmal_matches_generate_predictions_mean(n_targets: int) -> None:
    """CMAL returns the mixture mean of ``generate_predictions`` per target."""
    head = CMAL(
        n_in=4,
        n_out=4 * n_targets * N_DISTRIBUTIONS,
        n_distributions=N_DISTRIBUTIONS,
    )
    outputs = _cmal_outputs(n_targets)
    point = head.point_prediction(outputs)
    assert point.shape == (BATCH_SIZE, TIME_STEPS, n_targets)
    assert point.dtype == torch.float32

    for target in range(n_targets):
        start = target * N_DISTRIBUTIONS
        end = start + N_DISTRIBUTIONS
        expected = cmal_deterministic.generate_predictions(
            *(outputs[k][..., start:end] for k in ('mu', 'b', 'tau', 'pi'))
        )[..., 0]
        assert torch.allclose(point[..., target], expected, atol=1e-6)


@pytest.mark.unit
def test_cmal_head_forward_outputs_are_consumable() -> None:
    """``point_prediction`` accepts the head's own forward output."""
    n_targets = 2
    head = CMAL(
        n_in=4,
        n_out=4 * n_targets * N_DISTRIBUTIONS,
        n_distributions=N_DISTRIBUTIONS,
    )
    outputs = head(torch.randn(BATCH_SIZE, TIME_STEPS, 4))
    point = head.point_prediction(outputs)
    assert point.shape == (BATCH_SIZE, TIME_STEPS, n_targets)
    assert torch.isfinite(point).all()
    shifted = head.point_prediction({**outputs, 'mu': outputs['mu'] + 5.0})
    assert torch.allclose(shifted, point + 5.0, atol=1e-5)


@pytest.mark.unit
def test_cmal_without_n_distributions_treats_all_as_one_target() -> None:
    """Without ``n_distributions`` all components form a single mixture."""
    head = CMAL(n_in=4, n_out=4 * N_DISTRIBUTIONS)
    outputs = _cmal_outputs(1)
    point = head.point_prediction(outputs)
    expected = cmal_deterministic.generate_predictions(
        outputs['mu'], outputs['b'], outputs['tau'], outputs['pi']
    )[..., 0]
    assert point.shape == (BATCH_SIZE, TIME_STEPS, 1)
    assert torch.allclose(point[..., 0], expected, atol=1e-6)


@pytest.mark.unit
def test_cmal_point_prediction_is_differentiable() -> None:
    """Gradients flow from the point prediction to ``mu``."""
    head = CMAL(
        n_in=4, n_out=4 * N_DISTRIBUTIONS, n_distributions=N_DISTRIBUTIONS
    )
    outputs = _cmal_outputs(1, requires_grad=True)
    head.point_prediction(outputs).sum().backward()
    grad = outputs['mu'].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    # d mean / d mu = pi for every component.
    assert torch.allclose(grad, outputs['pi'])
