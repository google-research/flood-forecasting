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

"""Weighted overlap entries activate the same model outputs as plain names."""

from pathlib import Path

import pytest
import torch
import xarray as xr

from googlehydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from googlehydrology.training import get_loss_obj, get_regularization_obj
from googlehydrology.utils.config import Config
from test.test_hot_start import get_base_cfg


def _config(tmp_path: Path, entries: list, head: str = 'regression') -> Config:
    """Provide a real configuration and scaler for a small model."""
    options = get_base_cfg(tmp_path)
    options.update(
        {
            'seq_length': 4,
            'lead_time': 2,
            'forecast_overlap': 2,
            'predict_last_n': 2,
            'regularization': entries,
            'loss': 'mse',
            'head': head,
            'n_distributions': 3,
            'output_dropout': 0.0,
        }
    )
    xr.Dataset(
        {'streamflow': ('parameter', [0.0, 1.0, 0.0, 1.0])},
        coords={'parameter': ['center', 'scale', 'mean', 'std']},
    ).to_netcdf(tmp_path / 'scaler.nc', engine='scipy')
    return Config(options)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('entries', 'weight'),
    [
        (['forecast_overlap'], 1.0),
        ([['forecast_overlap', 0.5]], 0.5),
        ([('forecast_overlap', 2.0)], 2.0),
        ([['forecast_overlap', 0.0]], 0.0),
    ],
)
def test_weighted_overlap_runs_through_training_loss(
    tmp_path: Path,
    entries: list,
    weight: float,
) -> None:
    """Forward, weighted loss and backward pass all support paired entries."""
    cfg = _config(tmp_path, entries)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = HandoffForecastLSTM(cfg)
        data = {
            'x_d_hindcast': {
                name: torch.rand(2, cfg.seq_length, 1)
                for name in cfg.hindcast_inputs
            },
            'x_d_forecast': {
                name: torch.rand(2, cfg.lead_time + cfg.forecast_overlap, 1)
                for name in cfg.forecast_inputs
            },
            'x_s': torch.rand(2, len(cfg.static_attributes)),
            'y': torch.rand(2, cfg.seq_length, 1),
        }
        prediction = model(data)
        loss_fn = get_loss_obj(cfg)
        loss_fn.set_regularization_terms(get_regularization_obj(cfg))
        total, parts = loss_fn(prediction, data)
        for name in ('y_hindcast_overlap', 'y_forecast_overlap'):
            assert prediction[name].shape == (2, cfg.forecast_overlap, 1)
        expected_overlap = torch.mean(
            (
                prediction['y_hindcast_overlap']
                - prediction['y_forecast_overlap']
            )
            ** 2,
        )
        torch.testing.assert_close(parts['forecast_overlap'], expected_overlap)
        torch.testing.assert_close(
            total, parts['loss'] + weight * expected_overlap
        )
        assert torch.isfinite(total)
        total.backward()
        for head in (model.hindcast_head, model.forecast_head):
            for parameter in head.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize('head', ['cmal', 'cmal_deterministic'])
def test_weighted_overlap_preserves_regression_head_requirement(
    tmp_path: Path,
    head: str,
) -> None:
    """The existing head restriction also applies when a weight is supplied."""
    cfg = _config(tmp_path, [['forecast_overlap', 0.5]], head)
    with pytest.raises(ValueError, match='only works with a regression head'):
        HandoffForecastLSTM(cfg)


@pytest.mark.unit
def test_no_regularization_leaves_overlap_disabled(tmp_path: Path) -> None:
    """The default configuration still leaves overlap outputs disabled."""
    cfg = _config(tmp_path, [])
    with torch.random.fork_rng(devices=[]):
        assert not HandoffForecastLSTM(cfg).overlap_output
