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

# ruff: noqa: PLR0913, PLR0917
"""Validation, warnings, model-state handling and dtype of the DA engine."""

import logging
from collections.abc import Callable

import pytest
import torch
from torch import nn

from googlehydrology.evaluation.assimilation import Assimilation, _frozen
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config
from test.da_helpers import (
    ALL_COMPONENTS,
    LEAD_TIME,
    WINDOW,
    assimilate,
    build_model,
    da_config,
    make_data,
    prior_output,
)

ENGINE_LOGGER = 'googlehydrology.evaluation.assimilation'
MODEL_LOGGER = 'googlehydrology.modelzoo.mean_embedding_forecast_lstm'
OUTPUT_DROPOUT = 0.3


def _dropout_model(builder: Callable) -> MeanEmbeddingForecastLSTM:
    """Tiny model with output dropout, to observe `_frozen`."""
    options = dict(build_model(builder).cfg.as_dict())
    options['output_dropout'] = OUTPUT_DROPOUT
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return MeanEmbeddingForecastLSTM(Config(options)).eval()


@pytest.mark.unit
@pytest.mark.parametrize(
    ('prepare', 'overrides', 'exc_type', 'match'),
    [
        pytest.param(
            lambda _: None,
            {'assimilation_components': ['not_a_component']},
            ValueError,
            'Unsupported assimilation components',
            id='unknown_component',
        ),
        pytest.param(
            lambda data: data.__setitem__('y', data['y'][:, -(WINDOW + 1) :]),
            {},
            ValueError,
            'exceeds the target sequence length',
            id='window_exceeds_observed_period',
        ),
        pytest.param(
            lambda data: data.pop('per_basin_target_stds'),
            {'loss': 'NSE'},
            ValueError,
            "requires 'per_basin_target_stds'",
            id='nse_without_stds',
        ),
        pytest.param(
            lambda data: data['y'].fill_(1e30),
            {},
            RuntimeError,
            'Every sequence had a non-finite loss',
            id='all_sequences_nonfinite',
        ),
    ],
)
def test_assimilate_rejects(
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
    prepare: Callable[[dict], None],
    overrides: dict,
    exc_type: type[Exception],
    match: str,
) -> None:
    """Invalid components, windows, data and all-non-finite batches raise."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    prepare(data)

    with pytest.raises(exc_type, match=match):
        assimilate(model, data, **overrides)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('prepare', 'overrides', 'reason'),
    [
        pytest.param(
            lambda _: None, {'epochs': 0}, 'epochs is 0', id='zero_epochs'
        ),
        pytest.param(
            lambda data: data['y'][:, :-LEAD_TIME].fill_(float('nan')),
            {},
            'no finite observation',
            id='all_missing',
        ),
    ],
)
def test_prior_is_returned_unchanged(
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
    caplog: pytest.LogCaptureFixture,
    prepare: Callable[[dict], None],
    overrides: dict,
    reason: str,
) -> None:
    """Without epochs or observations the prior is returned and logged."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    prepare(data)
    prior = prior_output(model, data)

    with caplog.at_level(logging.INFO, logger=ENGINE_LOGGER):
        output = assimilate(model, data, **overrides)

    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert reason in caplog.records[0].getMessage()
    for key in ['y_hat', *ALL_COMPONENTS]:
        torch.testing.assert_close(output[key], prior[key], equal_nan=True)


@pytest.mark.unit
def test_single_dynamic_component_warning_comes_from_the_model_once(
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The model warns once per instance; the engine adds no warning."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    assimilation = Assimilation(
        da_config(model, assimilation_components=['forecast_embedding'])
    )

    with caplog.at_level(logging.WARNING):
        assimilation.assimilate(model, data)
        assimilation.assimilate(model, data)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.name for r in warnings] == [MODEL_LOGGER]
    assert 'Assimilating only forecast_embedding' in warnings[0].getMessage()


@pytest.mark.unit
def test_nonfinite_sequences_are_reported_once_per_engine(
    tiny_mean_embedding_model: Callable,
    tiny_mean_embedding_data: Callable,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The frozen-sequence warning is emitted once, not once per batch."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    data['y'][0] = 1e30
    assimilation = Assimilation(da_config(model, epochs=2))

    with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
        assimilation.assimilate(model, data)
        assimilation.assimilate(model, data)

    assert len(caplog.records) == 1
    assert '1 of 3 sequences were frozen' in caplog.records[0].getMessage()


@pytest.mark.unit
def test_model_state_is_restored(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """Weights, mode, dropout and requires_grad are unchanged afterwards."""
    model = _dropout_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    model.head.requires_grad_(requires_grad=False)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}

    assimilate(model, data, epochs=3)

    assert model.training
    assert model.dropout.p == pytest.approx(OUTPUT_DROPOUT)
    assert flags == {n: p.requires_grad for n, p in model.named_parameters()}
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key]), key


@pytest.mark.unit
@pytest.mark.parametrize('training', [True, False])
def test_frozen_only_switches_rnn_modules(
    tiny_mean_embedding_model: Callable, *, training: bool
) -> None:
    """`_frozen` sets RNNs to train mode, zeroes dropout and restores all."""
    model = _dropout_model(tiny_mean_embedding_model)
    model.train(mode=training)
    model.hindcast_lstm.dropout = 0.5
    rnns = [model.hindcast_lstm, model.forecast_lstm]
    others = [
        m
        for m in model.modules()
        if m is not model and not isinstance(m, nn.RNNBase)
    ]

    with _frozen(model):
        assert model.training is training
        assert all(m.training for m in rnns)
        assert all(m.training is training for m in others)
        assert model.dropout.p == 0.0
        assert all(m.dropout == 0.0 for m in rnns)
        assert not any(p.requires_grad for p in model.parameters())

    assert model.training is training
    assert all(m.training is training for m in rnns)
    assert model.dropout.p == pytest.approx(OUTPUT_DROPOUT)
    assert model.hindcast_lstm.dropout == pytest.approx(0.5)
    assert model.forecast_lstm.dropout == 0.0
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.unit
def test_frozen_rejects_batch_norm(tiny_mean_embedding_model: Callable) -> None:
    """Models with BatchNorm cannot be assimilated."""
    model = build_model(tiny_mean_embedding_model)
    model.extra = nn.BatchNorm1d(2)

    with pytest.raises(ValueError, match='BatchNorm1d'), _frozen(model):
        pass


@pytest.mark.unit
def test_dropout_is_disabled_during_assimilation(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """Results are deterministic even if the model was in training mode."""
    model = _dropout_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    model.train()
    assimilation = Assimilation(da_config(model, epochs=5))

    first = assimilation.assimilate(model, data)
    second = assimilation.assimilate(model, data)

    assert torch.equal(first['y_hat'], second['y_hat'])


@pytest.mark.unit
def test_assimilation_runs_in_float32_under_autocast(
    tiny_mean_embedding_model: Callable, tiny_mean_embedding_data: Callable
) -> None:
    """Under an enabled bfloat16 autocast the DA still runs in float32."""
    model = build_model(tiny_mean_embedding_model)
    data = make_data(model, tiny_mean_embedding_data)
    reference = assimilate(model, data, epochs=5)

    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.is_autocast_enabled('cpu')
        output = assimilate(model, data, epochs=5)

    for key in ['y_hat', *ALL_COMPONENTS]:
        assert output[key].dtype == torch.float32, key
        assert torch.isfinite(output[key]).all(), key
        torch.testing.assert_close(output[key], reference[key], msg=key)
