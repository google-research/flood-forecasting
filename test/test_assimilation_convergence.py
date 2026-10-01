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

# ruff: noqa: PLR2004
"""Early stopping of the DA engine: the `_converged` rule and its effect.

`_converged` is tested directly because its per-sequence decision cannot be
observed through ``assimilate`` without re-deriving the rule.
"""

from collections.abc import Callable

import pytest
import torch

from googlehydrology.evaluation.assimilation import _converged
from test.da_helpers import (
    ALL_COMPONENTS,
    assimilate,
    build_model,
    make_data,
    prior_output,
    window_error,
)

NAN = float('nan')


def _col(rows: list[list[float]]) -> torch.Tensor:
    """[B, T] list -> [B, T, 1] tensor."""
    return torch.tensor(rows).unsqueeze(-1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('obs', 'pred', 'y_full', 'expected'),
    [
        pytest.param(
            [[1.0, 2.0, NAN]],
            [[1.0, 2.02, 7.0]],
            [[0.0, 1.0, 2.0, NAN]],
            [True],
            id='last_missing_window_max_0.02/2',
        ),
        pytest.param(
            [[1.0, 2.0, NAN]],
            [[1.5, 2.0, 2.0]],
            [[0.0, 1.0, 2.0, NAN]],
            [False],
            id='last_missing_window_max_0.5/1',
        ),
        pytest.param(
            [[1.0, 2.0, 3.0]],
            [[9.0, 2.0, 3.06]],
            [[0.0, 1.0, 2.0, 3.0]],
            [True],
            id='only_last_step_counts_0.06/3',
        ),
        pytest.param(
            [[1.0, 2.0, 2.0]],
            [[1.0, 2.0, 2.2]],
            [[0.0, 1.0, 2.0, 2.0]],
            [False],
            id='last_step_0.2/2',
        ),
        pytest.param(
            [[10.0, 11.0]],
            [[10.0, 11.1]],
            [[0.0, 10.0, 11.0]],
            [True],
            id='base_level_0:0.1/11',
        ),
        pytest.param(
            [[10.0, 11.0]],
            [[10.0, 11.1]],
            [[NAN, 10.0, 11.0]],
            [False],
            id='base_level_10:0.1/1',
        ),
        pytest.param(
            [[10.0, 11.0]],
            [[10.0, 11.1]],
            [[NAN, NAN, NAN]],
            [True],
            id='all_missing_history_base_level_0',
        ),
        pytest.param(
            [[0.0, 0.0]],
            [[0.0, 0.002]],
            [[0.0, 0.0, 0.0]],
            [True],
            id='zero_flow_floor_0.05:0.002/0.05',
        ),
        pytest.param(
            [[0.0, 0.0]],
            [[0.0, 0.003]],
            [[0.0, 0.0, 0.0]],
            [False],
            id='zero_flow_floor_0.05:0.003/0.05',
        ),
        pytest.param(
            [[-1.0, -0.5]],
            [[-1.0, -0.48]],
            [[-1.0, -1.0, -0.5]],
            [True],
            id='negative_scaled_values_0.02/0.5',
        ),
        pytest.param(
            [[NAN, NAN]],
            [[1.0, 1.0]],
            [[0.0, NAN, NAN]],
            [False],
            id='window_all_missing_never_converges',
        ),
    ],
)
def test_converged(
    obs: list[list[float]],
    pred: list[list[float]],
    y_full: list[list[float]],
    expected: list[bool],
) -> None:
    """Relative error at the last step (or max over the window) <= 0.05."""
    converged = _converged(_col(pred), _col(obs), _col(y_full), tolerance=0.05)

    assert converged.tolist() == expected


@pytest.mark.unit
def test_converged_requires_every_target() -> None:
    """With several targets, all of them must meet the tolerance."""
    obs = torch.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])  # [B=2, W=1, 2 targets]
    pred = torch.tensor([[[1.01, 1.5]], [[1.01, 1.01]]])
    y_full = torch.zeros(2, 1, 2)

    assert _converged(pred, obs, y_full, 0.05).tolist() == [False, True]


@pytest.mark.unit
def test_early_stopping_huge_tolerance_returns_prior(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """If every sequence is converged at the start nothing is optimized."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    prior = prior_output(model, data)

    output = assimilate(model, data, early_stopping_tolerance=1e6)

    for name in ALL_COMPONENTS:
        torch.testing.assert_close(output[name], prior[name], msg=name)


@pytest.mark.unit
def test_early_stopping_tiny_tolerance_has_no_effect(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """A tolerance that is never met gives the same result as no stopping."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)

    plain = assimilate(model, data)
    stopped = assimilate(model, data, early_stopping_tolerance=1e-12)

    torch.testing.assert_close(stopped['y_hat'], plain['y_hat'])


@pytest.mark.unit
def test_early_stopping_freezes_converged_sequences(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """A sequence whose observations equal the prior is never updated."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    data['y'][0] -= 1.0  # Sequence 0: observations equal the prior.
    prior = prior_output(model, data)

    output = assimilate(model, data, early_stopping_tolerance=1e-3)

    for key in ['y_hat', *ALL_COMPONENTS]:
        torch.testing.assert_close(output[key][0], prior[key][0], msg=key)
        assert not torch.equal(output[key][1:], prior[key][1:]), key
    assert (window_error(model, output, data)[1:] < 0.9).all()
