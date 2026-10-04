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

"""Configured targets and mixture counts determine forecast output width."""

from pathlib import Path

import pytest
import torch
import xarray as xr

from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.training import get_loss_obj
from googlehydrology.utils.config import Config
from test.test_hot_start import get_base_cfg


@pytest.mark.unit
@pytest.mark.parametrize('n_targets', [1, 2])
@pytest.mark.parametrize(
    ('head', 'n_distributions'),
    [('regression', 1)]
    + [
        (head, count)
        for head in ('cmal', 'cmal_deterministic')
        for count in (1, 3, 5)
    ],
)
def test_configured_output_size_and_backward(
    tmp_path: Path,
    head: str,
    n_distributions: int,
    n_targets: int,
) -> None:
    """Run a real configured model and loss for each supported head type."""
    options = get_base_cfg(tmp_path)
    options.update(
        {
            'model': 'MeanEmbeddingForecastLSTM',
            'seq_length': 4,
            'lead_time': 2,
            'forecast_overlap': 4,
            'predict_last_n': 2,
            'head': head,
            'loss': 'mse' if head == 'regression' else 'cmal',
            'n_distributions': n_distributions,
            'target_variables': [
                f'target_{index}' for index in range(n_targets)
            ],
            'output_dropout': 0.0,
        }
    )
    cfg = Config(options)
    # Supply a real scaler file rather than mocking model initialization.
    xr.Dataset(
        {
            name: ('parameter', [0.0, 1.0, 0.0, 1.0])
            for name in cfg.target_variables
        },
        coords={'parameter': ['center', 'scale', 'mean', 'std']},
    ).to_netcdf(tmp_path / 'scaler.nc', engine='scipy')

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = MeanEmbeddingForecastLSTM(cfg)
        data = {
            'x_d_hindcast': {
                name: torch.rand(2, cfg.seq_length, 1)
                for name in cfg.hindcast_inputs
            },
            'x_d_forecast': {
                name: torch.rand(2, cfg.seq_length + cfg.lead_time, 1)
                for name in cfg.forecast_inputs
            },
            'x_s': torch.rand(2, len(cfg.static_attributes)),
            'y': torch.rand(2, cfg.seq_length + cfg.lead_time, n_targets),
        }
        predictions = model(data)
        expected_width = n_targets * (
            1 if head == 'regression' else n_distributions
        )
        expected_keys = (
            {'y_hat'}
            if head == 'regression'
            else {
                'mu',
                'b',
                'tau',
                'pi',
            }
        )
        assert set(predictions) == expected_keys
        for value in predictions.values():
            assert value.shape == (
                2,
                cfg.seq_length + cfg.lead_time,
                expected_width,
            )
            assert torch.isfinite(value).all()
        loss, _ = get_loss_obj(cfg)(predictions, data)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in model.head.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
