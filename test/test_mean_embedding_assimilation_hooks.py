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

"""Data assimilation hooks of MeanEmbeddingForecastLSTM."""

from pathlib import Path

import pytest
import torch
import xarray as xr

from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config
from test.test_hot_start import get_base_cfg

SEQ_LENGTH = 6
LEAD_TIME = 3
BATCH_SIZE = 2
HEAD_KEYS = ('y_hat',)
EMBEDDING_KEYS = (
    'static_embedding',
    'hindcast_embedding',
    'forecast_embedding',
)


@pytest.fixture
def model(tmp_path: Path) -> MeanEmbeddingForecastLSTM:
    """Tiny model with shared, hindcast-only and forecast-only groups."""
    options = get_base_cfg(tmp_path)
    embedding = {
        'type': 'fc',
        'hiddens': [8],
        'activation': ['tanh'],
        'dropout': 0.0,
    }
    options.update(
        {
            'model': 'MeanEmbeddingForecastLSTM',
            'seq_length': SEQ_LENGTH,
            'lead_time': LEAD_TIME,
            'forecast_overlap': SEQ_LENGTH,
            'hidden_size': 8,
            'output_dropout': 0.0,
            # 'pr' and 'tmmn' are shared; 'streamflow' is hindcast-only and
            # 'hres' is forecast-only.
            'hindcast_inputs': ['pr_a', 'tmmn_a', 'streamflow_lag'],
            'forecast_inputs': ['pr_a', 'tmmn_a', 'hres_precip'],
            'hindcast_embedding': embedding,
            'forecast_embedding': embedding,
            'statics_embedding': {**embedding, 'hiddens': [4]},
        }
    )
    cfg = Config(options)
    xr.Dataset(
        {'streamflow': ('parameter', [0.0, 1.0, 0.0, 1.0])},
        coords={'parameter': ['center', 'scale', 'mean', 'std']},
    ).to_netcdf(tmp_path / 'scaler.nc', engine='scipy')
    torch.manual_seed(0)
    model = MeanEmbeddingForecastLSTM(cfg)
    model.eval()
    return model


@pytest.fixture
def data(model: MeanEmbeddingForecastLSTM) -> dict:
    """Random inputs; hindcast covers seq_length, forecast the full span."""
    cfg = model.cfg
    generator = torch.Generator().manual_seed(1)
    return {
        'x_d_hindcast': {
            name: torch.rand(BATCH_SIZE, SEQ_LENGTH, 1, generator=generator)
            for name in cfg.hindcast_inputs
        },
        'x_d_forecast': {
            name: torch.rand(
                BATCH_SIZE, SEQ_LENGTH + LEAD_TIME, 1, generator=generator
            )
            for name in cfg.forecast_inputs
        },
        'x_s': torch.rand(
            BATCH_SIZE, len(cfg.static_attributes), generator=generator
        ),
    }


def _reference_forward(
    model: MeanEmbeddingForecastLSTM,
    data: dict,
    static_embedding: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Forward pass as implemented before the assimilation hooks.

    Uses only the model's submodules so it is independent of the helpers that
    the hooks modified.
    """
    groups = model.config_data
    total_length = SEQ_LENGTH + LEAD_TIME
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


def _with(
    data: dict,
    overrides: dict[str, torch.Tensor],
    window: tuple[int, int] | None = None,
) -> dict:
    result = {**data, 'assimilation_overrides': overrides}
    if window is not None:
        result['assimilation_window'] = window
    return result


def _assert_heads_equal(a: dict, b: dict) -> None:
    for key in HEAD_KEYS:
        assert torch.equal(a[key], b[key]), key


@pytest.mark.unit
def test_supported_components(model: MeanEmbeddingForecastLSTM) -> None:
    """The model advertises its overridable embeddings."""
    assert model.supported_assimilation_components == list(EMBEDDING_KEYS)


@pytest.mark.unit
def test_no_overrides_matches_reference(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """Without overrides outputs equal the pre-hook forward pass."""
    with torch.no_grad():
        out = model(data)
        out_empty = model(_with(data, {}))
        reference = _reference_forward(model, data)

    _assert_heads_equal(out, reference)
    _assert_heads_equal(out, out_empty)
    time_steps = SEQ_LENGTH + LEAD_TIME
    assert out['static_embedding'].shape == (BATCH_SIZE, 4)
    assert out['hindcast_embedding'].shape == (BATCH_SIZE, time_steps, 8)
    assert out['forecast_embedding'].shape == (BATCH_SIZE, time_steps, 8)
    for key in ('hindcast_embedding', 'forecast_embedding'):
        assert torch.equal(out[key], reference[key])


@pytest.mark.unit
def test_no_overrides_matches_reference_with_nan_inputs(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """NaN-safe dynamic embeddings do not change forward values."""
    data['x_d_hindcast']['streamflow_lag'][:, 2] = float('nan')
    data['x_d_forecast']['hres_precip'][:, 4] = float('nan')
    with torch.no_grad():
        out = model(data)
        reference = _reference_forward(model, data)
    _assert_heads_equal(out, reference)
    assert torch.isfinite(out['y_hat']).all()


@pytest.mark.unit
@pytest.mark.parametrize('name', EMBEDDING_KEYS)
def test_identity_full_override(
    model: MeanEmbeddingForecastLSTM, data: dict, name: str
) -> None:
    """Feeding back a returned embedding is a no-op."""
    with torch.no_grad():
        out = model(data)
        overridden = model(_with(data, {name: out[name].clone()}))
    _assert_heads_equal(out, overridden)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['hindcast_embedding', 'forecast_embedding'])
def test_identity_window_override(
    model: MeanEmbeddingForecastLSTM, data: dict, name: str
) -> None:
    """Splicing back a window of a returned embedding is a no-op."""
    window = (2, 5)
    with torch.no_grad():
        out = model(data)
        override = out[name][:, window[0] : window[1]].clone()
        overridden = model(_with(data, {name: override}, window))
    _assert_heads_equal(out, overridden)
    assert torch.equal(out[name], overridden[name])


@pytest.mark.unit
def test_window_override_is_local(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """A window override only affects outputs from the window start."""
    start, end = 3, 5
    with torch.no_grad():
        out = model(data)
        override = out['hindcast_embedding'][:, start:end] + 1.0
        perturbed = model(
            _with(data, {'hindcast_embedding': override}, (start, end))
        )
    assert torch.equal(out['y_hat'][:, :start], perturbed['y_hat'][:, :start])
    assert not torch.allclose(
        out['y_hat'][:, start], perturbed['y_hat'][:, start]
    )
    assert not torch.allclose(out['y_hat'][:, -1], perturbed['y_hat'][:, -1])


@pytest.mark.unit
def test_static_override_propagates_to_dynamic_embeddings(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """Dynamic embeddings are computed from an overridden static one."""
    start, end = 2, 4
    with torch.no_grad():
        out = model(data)
        static = out['static_embedding'] + 0.5
        static_only = model(_with(data, {'static_embedding': static}))
        reference = _reference_forward(model, data, static_embedding=static)
        override = static_only['hindcast_embedding'][:, start:end] + 1.0
        both = model(
            _with(
                data,
                {'static_embedding': static, 'hindcast_embedding': override},
                (start, end),
            )
        )

    assert torch.equal(static_only['static_embedding'], static)
    for key in ('hindcast_embedding', 'forecast_embedding'):
        assert torch.equal(static_only[key], reference[key])
        assert not torch.allclose(static_only[key], out[key])
    _assert_heads_equal(static_only, reference)

    hindcast = both['hindcast_embedding']
    expected = static_only['hindcast_embedding']
    assert torch.equal(hindcast[:, :start], expected[:, :start])
    assert torch.equal(hindcast[:, end:], expected[:, end:])
    assert torch.equal(hindcast[:, start:end], override)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['hindcast_embedding', 'forecast_embedding'])
def test_gradient_flows_into_window_override(
    model: MeanEmbeddingForecastLSTM, data: dict, name: str
) -> None:
    """Gradients reach a window override."""
    window = (1, 4)
    with torch.no_grad():
        out = model(data)
    override = out[name][:, window[0] : window[1]].clone().requires_grad_()
    model(_with(data, {name: override}, window))['y_hat'].sum().backward()
    assert override.grad is not None
    assert torch.isfinite(override.grad).all()
    assert override.grad.abs().sum() > 0


@pytest.mark.unit
def test_static_override_gradient_finite_with_nan_inputs(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """Static override gradients stay finite with missing inputs."""
    data['x_d_hindcast']['streamflow_lag'][:, 2] = float('nan')
    data['x_d_forecast']['hres_precip'][:, 4] = float('nan')
    with torch.no_grad():
        out = model(data)
    static = out['static_embedding'].clone().requires_grad_()
    model(_with(data, {'static_embedding': static}))['y_hat'].sum().backward()
    assert static.grad is not None
    assert torch.isfinite(static.grad).all()
    assert static.grad.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.unit
def test_invalid_overrides_raise(
    model: MeanEmbeddingForecastLSTM, data: dict
) -> None:
    """Malformed overrides raise ValueError."""
    with torch.no_grad():
        out = model(data)
    hindcast = out['hindcast_embedding']
    window_override = hindcast[:, 1:3]

    with pytest.raises(ValueError, match='assimilation_window'):
        model(_with(data, {'hindcast_embedding': window_override}))
    with pytest.raises(ValueError, match='out of range'):
        model(
            _with(
                data,
                {'hindcast_embedding': window_override},
                (hindcast.shape[1] - 1, hindcast.shape[1] + 1),
            )
        )
    with pytest.raises(ValueError, match='spans'):
        model(_with(data, {'hindcast_embedding': window_override}, (1, 4)))
    with pytest.raises(ValueError, match='incompatible'):
        model(_with(data, {'forecast_embedding': hindcast[..., :3]}))
    with pytest.raises(ValueError, match='static_embedding override'):
        model(_with(data, {'static_embedding': out['static_embedding'][:1]}))
    with pytest.raises(ValueError, match='Unsupported'):
        model(_with(data, {'hidden_state': hindcast}))
