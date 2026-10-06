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

"""NaN handling and gradients of the MeanEmbeddingForecastLSTM DA hooks.

Forward values are checked against ``_reference_forward``, a frozen
re-implementation of the pre-hook model; gradients against central finite
differences in double precision.
"""

from collections.abc import Callable

import pytest
import torch

from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from test.conftest import (
    assert_finite_grads,
    assert_grad_matches_finite_difference,
)

SEQ_LENGTH = 6
LEAD_TIME = 3
BATCH_SIZE = 2
SLICE = (1, 4)
EMBEDDING_KEYS = ('hindcast_embedding', 'forecast_embedding')
# Rows of the hindcast-only / forecast-only groups that are set to NaN.
PARTIAL_NAN = {'streamflow': [2], 'hres': [4]}
ALL_GROUPS = ('pr', 'tmmn', 'streamflow', 'hres')

ModelFactory = Callable[..., MeanEmbeddingForecastLSTM]
DataFactory = Callable[..., dict]
Outputs = dict[str, torch.Tensor]


@pytest.fixture
def model(tiny_mean_embedding_model: ModelFactory) -> MeanEmbeddingForecastLSTM:
    """Tiny double-precision model (finite differences need it)."""
    model = tiny_mean_embedding_model(
        seq_length=SEQ_LENGTH, lead_time=LEAD_TIME
    )
    return model.double()


def _as_double(data: dict) -> dict:
    """Cast every input tensor to double precision."""
    return {
        key: {k: v.double() for k, v in value.items()}
        if isinstance(value, dict)
        else value.double()
        for key, value in data.items()
    }


@pytest.fixture
def make_data(
    model: MeanEmbeddingForecastLSTM, tiny_mean_embedding_data: DataFactory
) -> Callable[..., dict]:
    """Return a factory for double-precision inputs: ``make_data(nan_at)``."""

    def _make(nan_at: dict[str, list[int]] | None = None) -> dict:
        return _as_double(
            tiny_mean_embedding_data(
                model.cfg, batch_size=BATCH_SIZE, nan_at=nan_at
            )
        )

    return _make


def _reference_forward(
    model: MeanEmbeddingForecastLSTM,
    data: dict,
    static_embedding: torch.Tensor | None = None,
) -> Outputs:
    """Regression oracle: the forward pass of the pre-hook model.

    Mirrors ``MeanEmbeddingForecastLSTM.forward`` as of main@3cd6646 using
    only the model's submodules (NaN inputs are fed straight through and
    NaN-padded, exactly like the original). It must NOT be updated alongside
    the model; it exists to prove that the hooks did not change forward values.
    """
    groups = model.config_data
    total_length = SEQ_LENGTH + model.cfg.lead_time
    if static_embedding is None:
        static_embedding = model.static_embedding_fc(data['x_s'])

    def with_static(x: torch.Tensor) -> torch.Tensor:
        repeated = static_embedding.unsqueeze(1).repeat(1, x.shape[1], 1)
        return torch.cat([x, repeated], dim=-1)

    def embed(
        fc: torch.nn.Module, key: str, features: list[str]
    ) -> torch.Tensor:
        x = torch.cat([data[key][e] for e in features], dim=-1)
        out = fc(with_static(x))
        padding = torch.full(
            (out.shape[0], total_length - out.shape[1], out.shape[2]),
            float('nan'),
            dtype=out.dtype,
        )
        return torch.cat([out, padding], dim=1)

    def masked_mean(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.nanmean(torch.stack(tensors, dim=-1), dim=-1)

    hindcast = [
        embed(fc, 'x_d_hindcast', groups.hindcast_inputs_grouped[name])
        for name, fc in model.hindcast_embeddings_fc.items()
    ]
    forecast = [
        embed(fc, 'x_d_forecast', groups.forecast_inputs_grouped[name])
        for name, fc in model.forecast_embeddings_fc.items()
    ]
    shared = [
        embed(fc, 'x_d_forecast', groups.forecast_inputs_grouped[name])
        for name, fc in model.shared_embeddings_fc.items()
    ]
    hindcast_embedding = masked_mean(hindcast + shared)
    forecast_embedding = masked_mean(forecast + shared)
    hindcast_state, _ = model.hindcast_lstm(with_static(hindcast_embedding))
    forecast_state, _ = model.forecast_lstm(
        with_static(torch.cat([forecast_embedding, hindcast_state], dim=-1))
    )
    head = model.head(model.dropout(forecast_state))
    head['hindcast_embedding'] = hindcast_embedding
    head['forecast_embedding'] = forecast_embedding
    return head


def _forward(
    model: MeanEmbeddingForecastLSTM,
    data: dict,
    overrides: dict[str, torch.Tensor] | None = None,
    assimilation_slice: tuple[int, int] | None = None,
) -> Outputs:
    """Forward pass with the data assimilation keyword arguments."""
    return model(
        data,
        assimilation_overrides=overrides,
        assimilation_slice=assimilation_slice,
        return_embeddings=True,
    )


def _assert_equal_where_finite(a: torch.Tensor, b: torch.Tensor) -> None:
    """Assert identical NaN pattern and identical finite values."""
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(a.nan_to_num(), b.nan_to_num())


@pytest.mark.unit
def test_static_override_propagates_to_dynamic_embeddings(
    model: MeanEmbeddingForecastLSTM, make_data: Callable[..., dict]
) -> None:
    """Dynamic embeddings are recomputed from an overridden static embedding."""
    data = make_data()
    start, end = SLICE
    with torch.no_grad():
        out = _forward(model, data)
        static = out['static_embedding'] + 0.5
        static_only = _forward(model, data, {'static_embedding': static})
        reference = _reference_forward(model, data, static_embedding=static)
        override = static_only['hindcast_embedding'][:, start:end] + 1.0
        both = _forward(
            model,
            data,
            {'static_embedding': static, 'hindcast_embedding': override},
            SLICE,
        )
    assert torch.equal(static_only['static_embedding'], static)
    assert torch.equal(static_only['y_hat'], reference['y_hat'])
    for key in EMBEDDING_KEYS:
        assert torch.equal(static_only[key], reference[key]), key
        assert not torch.allclose(static_only[key], out[key]), key
    hindcast = both['hindcast_embedding']
    expected = static_only['hindcast_embedding']
    assert torch.equal(hindcast[:, :start], expected[:, :start])
    assert torch.equal(hindcast[:, start:end], override)
    assert torch.equal(hindcast[:, end:], expected[:, end:])


@pytest.mark.unit
def test_training_path_gradients_with_partially_missing_group(
    model: MeanEmbeddingForecastLSTM, make_data: Callable[..., dict]
) -> None:
    """Plain ``model(data).backward()`` gives finite, correct gradients.

    On ``main`` a NaN row in one input group produced ``0 * NaN = NaN``
    gradients in that group's FC network and in ``static_embedding_fc``
    even though the forward values were finite.
    """
    data = make_data(PARTIAL_NAN)

    def loss() -> torch.Tensor:
        y_hat = model(data)['y_hat']
        return y_hat[~torch.isnan(y_hat)].sum()

    model.zero_grad()
    value = loss()
    assert torch.isfinite(value)
    value.backward()
    for module in (
        model.static_embedding_fc,
        model.hindcast_embeddings_fc,
        model.forecast_embeddings_fc,
        model.shared_embeddings_fc,
    ):
        assert_finite_grads(module)
    assert_finite_grads(model)
    for parameter in (
        next(model.static_embedding_fc.parameters()),
        next(model.hindcast_embeddings_fc['streamflow'].parameters()),
        next(model.forecast_embeddings_fc['hres'].parameters()),
    ):
        index = tuple(0 for _ in parameter.shape)
        assert_grad_matches_finite_difference(loss, parameter, index)


def _all_nan(step: int) -> dict[str, list[int]]:
    """``nan_at`` making every input group missing at ``step``."""
    return {group: [step] for group in ALL_GROUPS}


@pytest.mark.unit
@pytest.mark.parametrize(
    ('name', 'nan_at', 'missing_step'),
    [
        ('hindcast_embedding', None, None),
        ('hindcast_embedding', PARTIAL_NAN, None),
        ('forecast_embedding', None, None),
        ('forecast_embedding', PARTIAL_NAN, None),
        ('hindcast_embedding', _all_nan(SLICE[0] + 1), SLICE[0] + 1),
        ('hindcast_embedding', _all_nan(SLICE[1] + 1), SLICE[1] + 1),
        ('hindcast_embedding', _all_nan(SEQ_LENGTH + 1), SEQ_LENGTH + 1),
        ('forecast_embedding', _all_nan(SLICE[1] + 1), SLICE[1] + 1),
    ],
    ids=[
        'hindcast',
        'hindcast_partial_nan',
        'forecast',
        'forecast_partial_nan',
        'hindcast_missing_inside_slice',
        'hindcast_missing_after_slice',
        'hindcast_missing_in_horizon',
        'forecast_missing_after_slice',
    ],
)
def test_forward_matches_reference_and_override_gradients_are_exact(
    model: MeanEmbeddingForecastLSTM,
    make_data: Callable[..., dict],
    name: str,
    nan_at: dict[str, list[int]] | None,
    missing_step: int | None,
) -> None:
    """Forward equals the pre-hook oracle; override gradients match FD.

    With an all-NaN step every group feeding ``_masked_mean`` is missing, so
    the LSTM input is NaN there. Forward values must match the pre-hook model
    (NaN from that step onwards) while gradients of a loss over the finite
    steps stay finite and exact instead of being poisoned by ``0 * NaN``.
    """
    data = make_data(nan_at)
    start, end = SLICE
    finite_until = (
        SEQ_LENGTH + LEAD_TIME if missing_step is None else missing_step
    )
    with torch.no_grad():
        out = _forward(model, data)
        reference = _reference_forward(model, data)
    for key in ('y_hat', *EMBEDDING_KEYS):
        _assert_equal_where_finite(out[key], reference[key])
    assert torch.isfinite(out['y_hat'][:, :finite_until]).all()
    assert torch.isnan(out['y_hat'][:, finite_until:]).all()

    override = out[name][:, start:end].clone().requires_grad_()
    static = out['static_embedding'].clone().requires_grad_()

    def loss() -> torch.Tensor:
        y_hat = _forward(
            model, data, {name: override, 'static_embedding': static}, SLICE
        )['y_hat']
        return y_hat[:, :finite_until].sum()

    model.zero_grad()
    loss().backward()
    assert_finite_grads([override, static])
    assert_finite_grads(model)
    assert_grad_matches_finite_difference(loss, override, (0, 0, 0))
    assert_grad_matches_finite_difference(loss, override, (1, 1, 2))
    assert_grad_matches_finite_difference(loss, static, (0, 0))


@pytest.mark.unit
def test_disjoint_groups_horizon_is_nan_but_gradients_finite(
    tiny_mean_embedding_model: ModelFactory,
    tiny_mean_embedding_data: DataFactory,
) -> None:
    """Without shared groups the hindcast horizon is all-NaN by construction."""
    model = tiny_mean_embedding_model(
        seq_length=SEQ_LENGTH,
        lead_time=LEAD_TIME,
        hindcast_inputs=('tmmn_a', 'streamflow_lag'),
        forecast_inputs=('pr_a', 'hres_precip'),
    ).double()
    data = _as_double(
        tiny_mean_embedding_data(model.cfg, batch_size=BATCH_SIZE)
    )
    with torch.no_grad():
        out = _forward(model, data)
        reference = _reference_forward(model, data)
    _assert_equal_where_finite(out['y_hat'], reference['y_hat'])
    assert torch.isfinite(out['y_hat'][:, :SEQ_LENGTH]).all()
    assert torch.isnan(out['y_hat'][:, SEQ_LENGTH:]).all()

    override = (
        out['hindcast_embedding'][:, :SEQ_LENGTH].clone().requires_grad_()
    )

    def loss() -> torch.Tensor:
        y_hat = _forward(
            model, data, {'hindcast_embedding': override}, (0, SEQ_LENGTH)
        )['y_hat']
        return y_hat[:, :SEQ_LENGTH].sum()

    model.zero_grad()
    loss().backward()
    assert_finite_grads([override])
    assert_finite_grads(model)
    assert_grad_matches_finite_difference(loss, override, (0, 0, 0))
