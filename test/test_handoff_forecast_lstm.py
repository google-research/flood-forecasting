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

"""Regression tests for HandoffForecastLSTM feature ordering."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from googlehydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from googlehydrology.utils.config import Config
from test.test_hot_start import get_base_cfg

FeatureBatch = dict[str, torch.Tensor | dict[str, torch.Tensor]]


def _model_and_data(tmp_path: Path) -> tuple[HandoffForecastLSTM, FeatureBatch]:
    cfg = Config(get_base_cfg(tmp_path), dev_mode=True)
    model = HandoffForecastLSTM(cfg)
    model.eval()
    device = next(model.parameters()).device
    batch_size = 2
    data = {
        'x_d_hindcast': {
            name: torch.rand(batch_size, cfg.seq_length, 1, device=device)
            for name in cfg.hindcast_inputs
        },
        'x_d_forecast': {
            name: torch.rand(
                batch_size,
                cfg.lead_time + cfg.forecast_overlap,
                1,
                device=device,
            )
            for name in cfg.forecast_inputs
        },
        'x_s': torch.rand(
            batch_size, len(cfg.static_attributes), device=device
        ),
    }
    return model, data


def _reverse_dynamic_dict_order(data: FeatureBatch) -> FeatureBatch:
    reordered = deepcopy(data)
    for key in ('x_d_hindcast', 'x_d_forecast'):
        features = data[key]
        assert isinstance(features, dict)
        reordered[key] = {
            name: features[name] for name in reversed(list(features))
        }
    return reordered


def test_forward_invariant_to_feature_dict_order(tmp_path: Path) -> None:
    """Forward output is independent of dynamic-feature dict insertion order."""
    with (
        patch('googlehydrology.datautils.scaler.Scaler.load'),
        patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale'),
    ):
        model, data = _model_and_data(tmp_path)
    reordered = _reverse_dynamic_dict_order(data)

    with torch.no_grad():
        expected = model(data)['y_hat']
        actual = model(reordered)['y_hat']

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


def test_save_state_invariant_to_feature_dict_order(tmp_path: Path) -> None:
    """Saved hot-start state is independent of feature dict insertion order."""
    with (
        patch('googlehydrology.datautils.scaler.Scaler.load'),
        patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale'),
    ):
        model, data = _model_and_data(tmp_path)
    reordered = _reverse_dynamic_dict_order(data)
    expected_path = tmp_path / 'expected.npz'
    actual_path = tmp_path / 'actual.npz'

    model.save_state(data, expected_path)
    model.save_state(reordered, actual_path)

    with np.load(expected_path) as expected, np.load(actual_path) as actual:
        assert set(actual.files) == set(expected.files)
        for key in expected.files:
            np.testing.assert_allclose(
                actual[key], expected[key], atol=1e-6, rtol=0.0
            )


def test_forward_reports_missing_dynamic_feature(tmp_path: Path) -> None:
    """A configured feature missing from the batch raises a useful error."""
    with (
        patch('googlehydrology.datautils.scaler.Scaler.load'),
        patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale'),
    ):
        model, data = _model_and_data(tmp_path)
    missing = model.hindcast_inputs[0]
    hindcast = data['x_d_hindcast']
    assert isinstance(hindcast, dict)
    del hindcast[missing]

    message = rf'Missing dynamic features in batch:.*{missing}'
    with pytest.raises(KeyError, match=message):
        model(data)
