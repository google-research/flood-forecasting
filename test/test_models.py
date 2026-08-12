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

"""Unit tests for googlehydrology.modelzoo architectures and layers."""

from unittest.mock import MagicMock

import pytest
import torch

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head
from googlehydrology.modelzoo.positional_encoding import PositionalEncoding


@pytest.mark.unit
def test_positional_encoding_concatenate():
    seq_len = 10
    batch_size = 4
    embedding_dim = 16

    pe = PositionalEncoding(
        embedding_dim=embedding_dim,
        position_type='concatenate',
        dropout=0.0,
        max_len=100,
    )
    x = torch.randn(seq_len, batch_size, embedding_dim)
    out = pe(x)
    # Concatenate mode doubles embedding dimension: [seq, batch, 2 * dim]
    assert out.shape == (seq_len, batch_size, embedding_dim * 2)


@pytest.mark.unit
def test_positional_encoding_sum():
    seq_len = 10
    batch_size = 4
    embedding_dim = 16

    pe = PositionalEncoding(
        embedding_dim=embedding_dim,
        position_type='sum',
        dropout=0.1,
        max_len=100,
    )
    x = torch.randn(seq_len, batch_size, embedding_dim)
    out = pe(x)
    # Sum mode keeps embedding dimension: [seq, batch, dim]
    assert out.shape == (seq_len, batch_size, embedding_dim)


@pytest.mark.unit
def test_positional_encoding_invalid_type():
    with pytest.raises(
        RuntimeError, match='Unrecognized positional encoding type'
    ):
        PositionalEncoding(
            embedding_dim=16,
            position_type='invalid_type',
            dropout=0.0,
        )


@pytest.mark.unit
def test_fc_empty_hidden_sizes():
    with pytest.raises(
        ValueError, match='hidden_sizes must at least have one entry'
    ):
        FC(input_size=10, hidden_sizes=[])


@pytest.mark.unit
def test_fc_activations_and_shapes():
    # Single layer linear (hidden_sizes=[output_size])
    fc_linear = FC(input_size=10, hidden_sizes=[5])
    x = torch.randn(4, 10)
    out = fc_linear(x)
    assert out.shape == (4, 5)

    # Multi-layer with activations and dropout
    activations = ['relu', 'tanh', 'sigmoid', 'linear']
    for act in activations:
        fc = FC(
            input_size=10,
            hidden_sizes=[20, 15, 3],
            activation=act,
            dropout=0.1,
            xavier_init=True,
        )
        out = fc(x)
        assert out.shape == (4, 3)

    # Multi-layer with list of activations
    fc_multi_act = FC(
        input_size=10,
        hidden_sizes=[20, 15, 3],
        activation=['relu', 'tanh'],
    )
    out = fc_multi_act(x)
    assert out.shape == (4, 3)

    # Unsupported activation
    with pytest.raises(
        NotImplementedError, match='currently not supported as activation'
    ):
        FC(input_size=10, hidden_sizes=[20, 3], activation='invalid_act')


@pytest.mark.unit
def test_head_regression():
    cfg = MagicMock()
    cfg.head = 'regression'
    cfg.output_activation = 'linear'
    head = get_head(cfg=cfg, n_in=32, n_out=1)

    x = torch.randn(4, 10, 32)  # [batch, seq, in_features]
    out = head(x)
    assert 'y_hat' in out
    assert out['y_hat'].shape == (4, 10, 1)


@pytest.mark.unit
def test_head_cmal():
    cfg = MagicMock()
    cfg.head = 'cmal'
    # n_out for CMAL should match n_targets * 4 * n_distributions
    head = get_head(cfg=cfg, n_in=32, n_out=12, n_hidden=50)

    x = torch.randn(4, 10, 32)
    out = head(x)
    assert 'mu' in out
    assert 'b' in out
    assert 'tau' in out
    assert 'pi' in out

    assert torch.all(out['b'] > 0)
    assert torch.all((out['tau'] > 0) & (out['tau'] < 1))


@pytest.mark.unit
def test_head_invalid_type():
    cfg_empty = MagicMock(head='', model='lstm')
    with pytest.raises(ValueError, match="No 'head' specified"):
        get_head(cfg=cfg_empty, n_in=32, n_out=1)

    cfg_unsupported = MagicMock(head='unknown_head')
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_head(cfg=cfg_unsupported, n_in=32, n_out=1)


@pytest.mark.unit
def test_base_model_methods(monkeypatch):
    mock_sample_fn = MagicMock(return_value={'y_hat': torch.zeros(1)})
    monkeypatch.setattr(
        'googlehydrology.modelzoo.basemodel.sample_pointpredictions',
        mock_sample_fn,
    )
    monkeypatch.setattr(
        'googlehydrology.modelzoo.basemodel.Scaler', MagicMock()
    )

    cfg = MagicMock()
    cfg.target_variables = ['streamflow']
    cfg.head = 'regression'
    cfg.base_run_dir = None
    cfg.run_dir = None
    cfg.is_finetuning = False

    class SimpleModel(BaseModel):
        def forward(self, data):
            return data

    model = SimpleModel(cfg)
    data = {'x': torch.zeros(1)}

    # Test pre_model_hook
    hook_out = model.pre_model_hook(data, is_train=True)
    assert hook_out == data

    # Test sample method
    samples = model.sample(data, n_samples=5)
    assert 'y_hat' in samples
    mock_sample_fn.assert_called_once()
