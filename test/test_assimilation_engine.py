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
"""Optimisation behaviour of the data assimilation engine via ``assimilate``.

The targets are the prior prediction plus a constant shift, so the window
error of the prior is ``SHIFT**2`` and assimilation must reduce it.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import xarray as xr

from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config
from test.da_helpers import (
    ALL_COMPONENTS,
    BATCH_SIZE,
    DYNAMIC_COMPONENTS,
    LEAD_TIME,
    SEQ_LENGTH,
    SHIFT,
    WINDOW,
    assimilate,
    build_model,
    make_data,
    prior_output,
    select_rows,
    window_error,
    window_slice,
)

ENGINE_LOGGER = 'googlehydrology.evaluation.assimilation'


@pytest.mark.unit
@pytest.mark.parametrize(
    ('loss', 'head'),
    [
        ('MSE', 'regression'),
        ('NSE', 'regression'),
        ('MSE', 'cmal'),
        ('CMAL', 'cmal'),
    ],
)
def test_objective_matrix(
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
    loss: str,
    head: str,
) -> None:
    """Every supported loss/head pair reduces the window error."""
    model = build_model(tiny_mean_embedding_model, head=head)
    data = make_data(model, tiny_mean_embedding_data)
    prior = prior_output(model, data)
    assert window_error(model, prior, data).mean() == pytest.approx(1.0)

    output = assimilate(model, data, loss=loss)

    point = model.point_prediction(output)
    horizon = point[:, -LEAD_TIME:]
    assert torch.isfinite(horizon).all()
    assert not torch.equal(
        horizon, model.point_prediction(prior)[:, -LEAD_TIME:]
    )
    assert (window_error(model, output, data) < 0.9).all()


@pytest.mark.unit
def test_dynamic_components_only_change_window(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """Dynamic overrides act inside the window only; the rest is intact."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    prior = prior_output(model, data)

    output = assimilate(model, data, assimilation_components=DYNAMIC_COMPONENTS)

    window = window_slice(prior['y_hat'].shape[1])
    torch.testing.assert_close(
        output['y_hat'][:, : window.start], prior['y_hat'][:, : window.start]
    )
    for name in DYNAMIC_COMPONENTS:
        torch.testing.assert_close(
            output[name][:, : window.start], prior[name][:, : window.start]
        )
        torch.testing.assert_close(
            output[name][:, window.stop :], prior[name][:, window.stop :]
        )
        assert not torch.equal(output[name][:, window], prior[name][:, window])
    assert (window_error(model, output, data) < 0.9).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    'nan_at',
    [
        pytest.param(
            {'pr': [SEQ_LENGTH - LEAD_TIME - 2]}, id='group_in_window'
        ),
        pytest.param(
            {
                'pr': [SEQ_LENGTH + 1],
                'tmmn': [SEQ_LENGTH + 1],
                'hres': [SEQ_LENGTH + 1],
            },
            id='all_groups_in_horizon',
        ),
    ],
)
def test_missing_dynamic_inputs(
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
    caplog: pytest.LogCaptureFixture,
    nan_at: dict[str, list[int]],
) -> None:
    """Missing inputs inside the window or horizon do not break assimilation."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data, nan_at=nan_at)
    prior = prior_output(model, data)
    window = window_slice(prior['y_hat'].shape[1])
    assert torch.isfinite(prior['y_hat'][:, window]).all()

    with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
        output = assimilate(model, data)

    assert not caplog.records
    assert torch.isfinite(output['y_hat'][:, window]).all()
    assert (window_error(model, output, data) < 0.9).all()
    for name in DYNAMIC_COMPONENTS:
        assert torch.isfinite(output[name][:, window]).all()


@pytest.mark.unit
def test_sequences_are_independent_of_batch_composition(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """A batched SGD run matches per-sequence runs, even with no-obs rows."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    data['y'][1] = float('nan')  # Nothing to fit: stays at the prior.
    options = {
        'optimizer': 'SGD',
        'epochs': 3,
        'regularization_weight': 0.5,
        'assimilation_components': ['static_embedding', 'forecast_embedding'],
    }

    batched = assimilate(model, data, **options)

    for row in range(data['y'].shape[0]):
        alone = assimilate(model, select_rows(data, [row]), **options)
        for key in ['y_hat', 'static_embedding', 'forecast_embedding']:
            torch.testing.assert_close(
                batched[key][row : row + 1], alone[key], msg=f'{key}[{row}]'
            )
    prior = prior_output(model, data)
    torch.testing.assert_close(batched['y_hat'][1], prior['y_hat'][1])
    assert not torch.equal(batched['y_hat'][0], prior['y_hat'][0])


@pytest.mark.unit
def test_one_nonfinite_sequence_does_not_abort_batch(
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sequence with an overflowing loss is frozen; the others assimilate."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    data['y'][2] = 1e30  # (1e30)**2 overflows float32.
    prior = prior_output(model, data)

    with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
        output = assimilate(model, data)

    warnings = [r for r in caplog.records if 'frozen' in r.getMessage()]
    assert len(warnings) == 1
    assert '1 of 3 sequences' in warnings[0].getMessage()
    assert (window_error(model, output, data)[:2] < 0.9).all()
    for name in ALL_COMPONENTS:
        torch.testing.assert_close(output[name][2], prior[name][2])
        assert not torch.equal(output[name][:2], prior[name][:2])


@pytest.mark.unit
def test_multi_target_cmal(
    tmp_path: Path,
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
) -> None:
    """The CMAL likelihood assimilates every target of a two-target head."""
    options = dict(
        build_model(tiny_mean_embedding_model, head='cmal').cfg.as_dict()
    )
    options['target_variables'] = ['streamflow', 'other']
    xr.Dataset(
        {
            name: ('parameter', [0.0, 1.0, 0.0, 1.0])
            for name in options['target_variables']
        },
        coords={'parameter': ['center', 'scale', 'mean', 'std']},
    ).to_netcdf(tmp_path / 'scaler.nc', engine='scipy')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MeanEmbeddingForecastLSTM(Config(options)).eval()
    data = make_data(model, tiny_mean_embedding_data)
    assert data['y'].shape[-1] == 2

    output = assimilate(model, data, loss='CMAL', epochs=100)

    point = model.point_prediction(output)[:, window_slice(10)]
    per_target = ((point - data['y'][:, 2:6]) ** 2).mean(dim=(0, 1))
    assert (per_target < 1.0).all()
    assert torch.isfinite(point).all()


@pytest.mark.unit
def test_gradient_clipping_bounds_the_update(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """Per-sequence clipping bounds the SGD displacement by epochs*lr*clip."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data, shift=5.0)
    prior = prior_output(model, data)
    options = {'optimizer': 'SGD', 'epochs': 4, 'initial_learning_rate': 0.5}
    clip = 0.1
    window = window_slice(prior['y_hat'].shape[1])

    def displacement(output: dict) -> torch.Tensor:
        parts = [
            (output[name] - prior[name])[:, window]
            if output[name].ndim == 3
            else output[name] - prior[name]
            for name in ALL_COMPONENTS
        ]
        return torch.cat([p.flatten(1) for p in parts], dim=1).norm(dim=1)

    clipped = displacement(
        assimilate(model, data, clip_gradient_norm=clip, **options)
    )
    unclipped = displacement(assimilate(model, data, **options))

    bound = options['epochs'] * options['initial_learning_rate'] * clip
    assert (clipped <= bound * (1 + 1e-5)).all()
    assert (unclipped > bound).all()  # Clipping was active.


@pytest.mark.unit
def test_per_component_learning_rate(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """A tiny per-component learning rate effectively freezes that component."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    prior = prior_output(model, data)

    output = assimilate(
        model,
        data,
        assimilation_components={
            'static_embedding': {'initial_learning_rate': 1e-9},
            'forecast_embedding': {},
        },
    )

    torch.testing.assert_close(
        output['static_embedding'], prior['static_embedding']
    )
    assert not torch.equal(
        output['forecast_embedding'], prior['forecast_embedding']
    )


@pytest.mark.unit
def test_strong_regularization_keeps_prior(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """A large background weight keeps the components near the prior."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    prior = prior_output(model, data)

    def distance(output: dict) -> float:
        return float(sum((output[n] - prior[n]).norm() for n in ALL_COMPONENTS))

    weak = distance(assimilate(model, data, regularization_weight=1e-3))
    strong = distance(assimilate(model, data, regularization_weight=1e3))

    # The normalised background term with weight 1e3 dominates a unit-shift
    # observation term by orders of magnitude; the components barely move.
    assert strong < 0.1 * weak


@pytest.mark.unit
def test_perfect_prior_is_a_fixed_point(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """Observations equal to the prior leave the components unchanged.

    SGD, because Adam turns the round-off gradient (the frozen model's LSTM
    kernel differs from eval mode by ~1e-8) into a learning-rate sized step.
    """
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data, shift=0.0)
    prior = prior_output(model, data)

    output = assimilate(model, data, optimizer='SGD')

    for name in ALL_COMPONENTS:
        torch.testing.assert_close(output[name], prior[name], msg=name)


@pytest.mark.unit
def test_partially_missing_observations(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """Missing observations are skipped; the remaining ones are still fitted."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    data['y'][:, : SEQ_LENGTH - LEAD_TIME - 1] = float('nan')  # Last step only.
    data['y'][0, -LEAD_TIME - 1] = float('nan')  # Row 0 has no observation.
    prior = prior_output(model, data)

    output = assimilate(model, data)

    torch.testing.assert_close(output['y_hat'][0], prior['y_hat'][0])
    assert (window_error(model, output, data)[1:] < 0.9).all()


def _short_output_model(
    builder: Callable, forecast_overlap: int
) -> MeanEmbeddingForecastLSTM:
    """Model whose output (overlap + lead time) is shorter than the targets.

    Without hindcast-only inputs the output only covers the forecast span.
    """
    options = dict(build_model(builder).cfg.as_dict())
    options.update(
        {'forecast_overlap': forecast_overlap, 'hindcast_inputs': ['pr_a']}
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return MeanEmbeddingForecastLSTM(Config(options)).eval()


def _short_output_data(
    model: MeanEmbeddingForecastLSTM, maker: Callable
) -> dict:
    """Targets span ``SEQ_LENGTH`` steps, end-aligned with a shorter output."""
    data = maker(model.cfg, batch_size=BATCH_SIZE)
    span = model.cfg.forecast_overlap + LEAD_TIME
    data['x_d_forecast'] = {
        k: v[:, :span] for k, v in data['x_d_forecast'].items()
    }
    with torch.no_grad():
        prior = model.point_prediction(model(data))
    head = torch.zeros(BATCH_SIZE, SEQ_LENGTH - prior.shape[1], 1)
    data['y'] = torch.cat([head, prior + SHIFT], dim=1)
    return data


@pytest.mark.unit
def test_output_shorter_than_targets(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """The window is end-aligned when the output is shorter than the targets."""
    model = _short_output_model(tiny_mean_embedding_model, WINDOW)
    data = _short_output_data(model, tiny_mean_embedding_data)
    prior = prior_output(model, data)
    assert prior['y_hat'].shape[1] == WINDOW + LEAD_TIME < SEQ_LENGTH
    assert window_error(model, prior, data).mean() == pytest.approx(1.0)

    output = assimilate(model, data)

    assert (window_error(model, output, data) < 0.9).all()


@pytest.mark.unit
def test_window_beyond_short_output_raises(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """A window longer than the observed part of the output is rejected."""
    model = _short_output_model(tiny_mean_embedding_model, WINDOW - 1)
    data = _short_output_data(model, tiny_mean_embedding_data)

    with pytest.raises(ValueError, match='must fit within the model output'):
        assimilate(model, data)
