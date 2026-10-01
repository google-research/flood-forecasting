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

"""Data assimilation API of MeanEmbeddingForecastLSTM: keys, splicing, errors.

Gradient and NaN-handling tests live in
``test_mean_embedding_assimilation_gradients.py``.
"""

import logging
from collections.abc import Callable

import pytest
import torch

from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)

SEQ_LENGTH = 6
LEAD_TIME = 3
BATCH_SIZE = 2
LAST_OBSERVED = SEQ_LENGTH  # Slices must end at or before this index.
EMBEDDING_KEYS = (
    'static_embedding',
    'hindcast_embedding',
    'forecast_embedding',
)
# Rows of the hindcast-only / forecast-only groups that are set to NaN.
PARTIAL_NAN = {'streamflow': [2], 'hres': [4]}

ModelFactory = Callable[..., MeanEmbeddingForecastLSTM]
DataFactory = Callable[..., dict]
Overrides = dict[str, torch.Tensor]
Outputs = dict[str, torch.Tensor]


@pytest.fixture
def model(tiny_mean_embedding_model: ModelFactory) -> MeanEmbeddingForecastLSTM:
    """Tiny regression model with shared, hindcast- and forecast-only groups."""
    return tiny_mean_embedding_model(seq_length=SEQ_LENGTH, lead_time=LEAD_TIME)


@pytest.fixture
def data(
    model: MeanEmbeddingForecastLSTM, tiny_mean_embedding_data: DataFactory
) -> dict:
    """Random inputs for the default model."""
    return tiny_mean_embedding_data(model.cfg, batch_size=BATCH_SIZE)


def _forward(
    model: MeanEmbeddingForecastLSTM,
    data: dict,
    overrides: Overrides | None = None,
    assimilation_slice: tuple[int, int] | None = None,
) -> Outputs:
    """Forward pass with the data assimilation keyword arguments."""
    return model(
        data,
        assimilation_overrides=overrides,
        assimilation_slice=assimilation_slice,
        return_embeddings=True,
    )


def _assert_heads_equal(a: Outputs, b: Outputs) -> None:
    """Assert all non-embedding outputs are identical."""
    keys = set(a).union(b).difference(EMBEDDING_KEYS)
    assert keys
    for key in keys:
        assert torch.equal(a[key], b[key]), key


@pytest.mark.unit
def test_default_forward_has_no_embedding_keys(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """Without the DA kwargs the output keys are the same as before."""
    assert set(model.supported_assimilation_components) == set(EMBEDDING_KEYS)
    with torch.no_grad():
        out = model(data)
        out_explicit = model(data, return_embeddings=False)
        out_requested = _forward(model, data)
    assert set(out) == set(out_explicit) == {'y_hat'}
    assert set(out_requested) == {'y_hat', *EMBEDDING_KEYS}
    _assert_heads_equal(out, out_requested)
    time_steps = SEQ_LENGTH + LEAD_TIME
    assert out_requested['static_embedding'].shape == (BATCH_SIZE, 4)
    for key in ('hindcast_embedding', 'forecast_embedding'):
        assert out_requested[key].shape == (BATCH_SIZE, time_steps, 8)


@pytest.mark.unit
@pytest.mark.parametrize('head', ['regression', 'cmal'])
@pytest.mark.parametrize(
    'nan_at', [None, PARTIAL_NAN], ids=['finite', 'partial_nan']
)
@pytest.mark.parametrize(
    'overrides_builder',
    [
        lambda _: None,
        lambda _: {},
        lambda out: {'static_embedding': out['static_embedding'].clone()},
    ],
    ids=['none', 'empty', 'static_identity'],
)
def test_identity_overrides_are_no_ops(
    tiny_mean_embedding_model: ModelFactory,
    tiny_mean_embedding_data: DataFactory,
    head: str,
    nan_at: dict[str, list[int]] | None,
    overrides_builder: Callable[[Outputs], Overrides | None],
) -> None:
    """No overrides or feeding back returned embeddings is a plain forward."""
    model = tiny_mean_embedding_model(
        seq_length=SEQ_LENGTH, lead_time=LEAD_TIME, head=head
    )
    data = tiny_mean_embedding_data(
        model.cfg, batch_size=BATCH_SIZE, nan_at=nan_at
    )
    with torch.no_grad():
        plain = model(data)
        returned = _forward(model, data)
        overrides = overrides_builder(returned)
        overridden = _forward(model, data, overrides)
        implicit = model(data, assimilation_overrides=overrides)
    assert EMBEDDING_KEYS[0] not in plain
    _assert_heads_equal(plain, overridden)
    _assert_heads_equal(plain, implicit)
    for key in EMBEDDING_KEYS:
        assert torch.equal(overridden[key], returned[key]), key
        assert torch.isfinite(overridden[key]).all(), key
        # Non-empty overrides return the embeddings even if not requested.
        assert (key in implicit) == bool(overrides), key
    for key in set(plain):
        assert torch.isfinite(plain[key]).all(), key


@pytest.mark.unit
@pytest.mark.parametrize('name', ['hindcast_embedding', 'forecast_embedding'])
@pytest.mark.parametrize(
    ('assimilation_slice', 'lead_time'),
    [
        ((2, 5), LEAD_TIME),
        ((0, LAST_OBSERVED), LEAD_TIME),
        ((LAST_OBSERVED - 3, LAST_OBSERVED), LEAD_TIME),
        (None, 0),
        ((0, SEQ_LENGTH), 0),
    ],
    ids=[
        'inner',
        'full_observed',
        'ends_at_horizon',
        'full_no_lead',
        'explicit_no_lead',
    ],
)
def test_identity_dynamic_override(
    tiny_mean_embedding_model: ModelFactory,
    tiny_mean_embedding_data: DataFactory,
    name: str,
    assimilation_slice: tuple[int, int] | None,
    lead_time: int,
) -> None:
    """Splicing back a slice of a returned dynamic embedding is a no-op."""
    model = tiny_mean_embedding_model(
        seq_length=SEQ_LENGTH, lead_time=lead_time
    )
    data = tiny_mean_embedding_data(model.cfg, batch_size=BATCH_SIZE)
    start, end = assimilation_slice or (0, SEQ_LENGTH + lead_time)
    with torch.no_grad():
        out = _forward(model, data)
        override = out[name][:, start:end].clone()
        overridden = _forward(model, data, {name: override}, assimilation_slice)
    _assert_heads_equal(out, overridden)
    for key in EMBEDDING_KEYS:
        assert torch.equal(out[key], overridden[key]), key


@pytest.mark.unit
def test_slice_override_is_local(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """A slice override only affects outputs from the slice start."""
    start, end = 3, 5
    with torch.no_grad():
        out = _forward(model, data)
        override = out['hindcast_embedding'][:, start:end] + 1.0
        perturbed = _forward(
            model, data, {'hindcast_embedding': override}, (start, end)
        )
    hindcast = perturbed['hindcast_embedding']
    expected = out['hindcast_embedding']
    assert torch.equal(hindcast[:, :start], expected[:, :start])
    assert torch.equal(hindcast[:, start:end], override)
    assert torch.equal(hindcast[:, end:], expected[:, end:])
    assert torch.equal(out['y_hat'][:, :start], perturbed['y_hat'][:, :start])
    assert not torch.allclose(
        out['y_hat'][:, start:], perturbed['y_hat'][:, start:]
    )


def _ov(
    name: str,
    time: tuple[int, int] | None = None,
    *,
    batch: int | None = None,
    features: int | None = None,
    dtype: torch.dtype | None = None,
) -> Callable[[Outputs], Overrides]:
    """Builder selecting a sub-tensor of a returned embedding as override."""

    def build(out: Outputs) -> Overrides:
        # Unknown names reuse the hindcast embedding as a stand-in tensor.
        tensor = out.get(name, out['hindcast_embedding'])
        if time is not None:
            tensor = tensor[:, time[0] : time[1]]
        if batch is not None:
            tensor = tensor[:batch]
        if features is not None:
            tensor = tensor[..., :features]
        if dtype is not None:
            tensor = tensor.to(dtype)
        return {name: tensor}

    return build


@pytest.mark.unit
@pytest.mark.parametrize(
    ('overrides_builder', 'assimilation_slice', 'match'),
    [
        (_ov('hindcast_embedding'), None, 'forecast horizon'),
        (_ov('forecast_embedding'), None, 'forecast horizon'),
        (_ov('forecast_embedding'), (0, 9), 'forecast horizon'),
        (_ov('hindcast_embedding', (4, 7)), (4, 7), 'forecast horizon'),
        (_ov('hindcast_embedding', (1, 3)), None, 'assimilation_slice=.*'),
        (_ov('hindcast_embedding', (1, 3)), (8, 10), 'out of range'),
        (_ov('hindcast_embedding', (1, 3)), (1, 4), 'spans'),
        (_ov('hindcast_embedding', (3, 3)), (3, 3), 'out of range'),
        (_ov('hindcast_embedding', (2, 4)), (4, 2), 'out of range'),
        (_ov('hindcast_embedding', (0, 3)), (-1, 2), 'out of range'),
        (_ov('forecast_embedding', features=3), None, 'incompatible'),
        (_ov('forecast_embedding', (1, 3), batch=1), (1, 3), 'incompatible'),
        (_ov('forecast_embedding', (1, 3), dtype=torch.bool), (1, 3), 'dtype'),
        (
            _ov('hindcast_embedding', (1, 3), dtype=torch.float64),
            (1, 3),
            'dtype',
        ),
        (_ov('static_embedding', batch=1), None, 'static_embedding override'),
        (_ov('static_embedding', dtype=torch.float64), None, 'dtype'),
        (_ov('hidden_state'), None, 'Unsupported.*hindcast_embedding'),
    ],
    ids=[
        'full_hindcast_with_lead',
        'full_forecast_with_lead',
        'explicit_full_slice',
        'slice_into_horizon',
        'missing_slice',
        'slice_out_of_range',
        'slice_length_mismatch',
        'empty_slice',
        'reversed_slice',
        'negative_start',
        'wrong_feature_dim',
        'wrong_batch_size',
        'bool_override',
        'double_override',
        'static_wrong_batch',
        'static_wrong_dtype',
        'unknown_component',
    ],
)
def test_invalid_overrides_raise(
    model: MeanEmbeddingForecastLSTM,
    data: dict,
    overrides_builder: Callable[[Outputs], Overrides],
    assimilation_slice: tuple[int, int] | None,
    match: str,
) -> None:
    """Malformed overrides and slices raise ValueError."""
    with torch.no_grad():
        out = _forward(model, data)
        with pytest.raises(ValueError, match=match):
            _forward(model, data, overrides_builder(out), assimilation_slice)


@pytest.mark.unit
def test_static_override_without_static_inputs_raises(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """Overriding the static embedding requires static inputs."""
    with torch.no_grad():
        out = _forward(model, data)
        with pytest.raises(ValueError, match='no static inputs'):
            _forward(
                model,
                {**data, 'x_s': None},
                {'static_embedding': out['static_embedding']},
            )


@pytest.mark.unit
def test_validate_assimilation_components_rejects_unknown(
    model: MeanEmbeddingForecastLSTM,
) -> None:
    """Unknown names raise and the message lists the supported names."""
    with pytest.raises(ValueError, match='Unsupported') as info:
        model.validate_assimilation_components(['hidden_state', 'cell_state'])
    message = str(info.value)
    assert "['cell_state', 'hidden_state']" in message
    for name in EMBEDDING_KEYS:
        assert name in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ('names', 'warned'),
    [
        ((), None),
        (('static_embedding',), None),
        (('hindcast_embedding', 'forecast_embedding'), None),
        (EMBEDDING_KEYS, None),
        (('hindcast_embedding',), 'hindcast_embedding'),
        (('forecast_embedding', 'static_embedding'), 'forecast_embedding'),
    ],
)
def test_validate_assimilation_components_warns_single_dynamic(
    model: MeanEmbeddingForecastLSTM,
    names: tuple[str, ...],
    warned: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Supported names pass; only a lone dynamic embedding warns, once."""
    with caplog.at_level(logging.WARNING):
        model.validate_assimilation_components(names)
        model.validate_assimilation_components(set(names))
    warnings = [r for r in caplog.records if 'Assimilating only' in r.message]
    if warned is None:
        assert not warnings
        return
    (other,) = {'hindcast_embedding', 'forecast_embedding'} - {warned}
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert f'Assimilating only {warned}' in warnings[0].message
    assert f'shared contribution of {other}' in warnings[0].message


@pytest.mark.unit
@pytest.mark.parametrize('head', ['regression', 'cmal'])
def test_model_point_prediction_matches_head(
    tiny_mean_embedding_model: ModelFactory,
    tiny_mean_embedding_data: DataFactory,
    head: str,
) -> None:
    """``model.point_prediction`` delegates to the head and is [B, T, 1]."""
    model = tiny_mean_embedding_model(
        seq_length=SEQ_LENGTH, lead_time=LEAD_TIME, head=head
    )
    data = tiny_mean_embedding_data(model.cfg, batch_size=BATCH_SIZE)
    with torch.no_grad():
        out = _forward(model, data)
        point = model.point_prediction(out)
    assert point.shape == (BATCH_SIZE, SEQ_LENGTH + LEAD_TIME, 1)
    assert torch.equal(point, model.head.point_prediction(out))
    if head == 'regression':
        assert point is out['y_hat']
    else:
        assert set(out) == {'mu', 'b', 'tau', 'pi', *EMBEDDING_KEYS}
