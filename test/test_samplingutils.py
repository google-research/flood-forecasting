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

"""Unit tests for googlehydrology.utils.samplingutils."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr

from googlehydrology.utils import samplingutils
from googlehydrology.utils.config import Config


@pytest.fixture
def mock_scaler():
    scaler = MagicMock()
    # Create synthetic scaler xarray Dataset for 'streamflow'
    center = 5.0
    scale = 2.0
    ds = xr.DataArray(
        [center, scale],
        coords={'parameter': ['center', 'scale']},
        dims=['parameter']
    )
    scaler.scaler = {'streamflow': ds}
    return scaler


@pytest.fixture
def mock_model():
    class DummyDropout:
        p = 0.2

    class DummyModel:
        pass

    model = DummyModel()
    model.parameters = lambda: iter([torch.zeros(1)])
    cfg = MagicMock()
    cfg.head = 'cmal'
    cfg.output_dropout = 0.2
    cfg.mc_dropout = False
    cfg.target_variables = ['streamflow']
    cfg.use_frequencies = ['1D']
    cfg.predict_last_n = {'1D': 3}
    cfg.n_distributions = 3
    cfg.negative_sample_handling = 'none'
    cfg.negative_sample_max_retries = 3
    model.cfg = cfg
    model.dropout = DummyDropout()
    return model


@pytest.mark.unit
def test_calc_normalized_zero_thresholds(mock_scaler):
    threshold = samplingutils._calc_normalized_zero_thresholds(
        scaler=mock_scaler,
        targets=['streamflow'],
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    # -center/scale = -5.0 / 2.0 = -2.5
    assert torch.isclose(threshold[0], torch.tensor(-2.5))


@pytest.mark.unit
def test_handle_negative_values_clip():
    cfg = MagicMock(negative_sample_handling='clip')
    values = torch.tensor([-3.0, -1.0, 2.0, 5.0])
    norm_zero = torch.tensor(-2.5)

    result = samplingutils._handle_negative_values(
        cfg=cfg,
        values=values,
        sample_values=lambda ids: torch.zeros_like(ids, dtype=torch.float),
        normalized_zero=norm_zero,
    )
    assert result[0].item() == -2.5
    assert result[1].item() == -1.0
    assert result[2].item() == 2.0


@pytest.mark.unit
def test_handle_negative_values_truncate():
    cfg = MagicMock(
        negative_sample_handling='truncate',
        negative_sample_max_retries=5
    )
    values = torch.tensor([-3.0, 2.0])
    norm_zero = torch.tensor(-2.5)

    # resample function replaces negative values with positive 1.0
    def resample(mask):
        return torch.full((mask.sum(),), 1.0)

    result = samplingutils._handle_negative_values(
        cfg=cfg,
        values=values,
        sample_values=resample,
        normalized_zero=norm_zero,
    )
    assert result[0].item() == 1.0
    assert result[1].item() == 2.0


@pytest.mark.unit
def test_handle_negative_values_invalid_mode():
    cfg = MagicMock(negative_sample_handling='unsupported_mode')
    with pytest.raises(
        NotImplementedError,
        match='not supported for handling negative samples',
    ):
        samplingutils._handle_negative_values(
            cfg=cfg,
            values=torch.tensor([1.0]),
            sample_values=lambda x: x,
            normalized_zero=torch.tensor(0.0),
        )


@pytest.mark.unit
def test_sample_asymmetric_laplacians():
    m = torch.tensor([0.0, 1.0])
    b = torch.tensor([1.0, 2.0])
    t = torch.tensor([0.5, 0.5])
    ids = torch.tensor([True, True])

    sampled = samplingutils._sample_asymmetric_laplacians(ids, m, b, t)
    assert sampled.shape == (2,)
    assert torch.all(torch.isfinite(sampled))


@pytest.mark.unit
def test_sampling_setup_dropout_checks(mock_model):
    mock_model.cfg.mc_dropout = True
    mock_model.dropout.p = 0.0  # Invalid for mc_dropout
    data = {'y': torch.zeros(2, 5, 1)}

    with pytest.raises(
        RuntimeError, match='requires a dropout rate larger than 0.0'
    ):
        samplingutils._SamplingSetup(mock_model, data, head='cmal')

    mock_model.dropout.p = 1.0  # Invalid >= 1.0
    with pytest.raises(RuntimeError, match='maximal dropout-rate is 1'):
        samplingutils._SamplingSetup(mock_model, data, head='cmal')


@pytest.mark.unit
def test_sample_cmal(mock_model, mock_scaler):
    data = {
        'x_d': {'ERA5': torch.zeros(2, 10, 3)},
        'y': torch.zeros(2, 10, 1),
    }
    outputs = {
        'mu': torch.zeros(2, 3, 3),
        'b': torch.ones(2, 3, 3),
        'tau': torch.full((2, 3, 3), 0.5),
        'pi': torch.full((2, 3, 3), 1.0 / 3),
    }

    samples = samplingutils.sample_cmal(
        model=mock_model,
        data=data,
        n_samples=5,
        scaler=mock_scaler,
        outputs=outputs,
    )
    assert 'y_hat' in samples
    # Expected shape: [batch, time, target, n_samples] -> [2, 3, 1, 5]
    assert samples['y_hat'].shape == (2, 3, 1, 5)


@pytest.mark.unit
def test_sample_pointpredictions_dispatch(mock_model, mock_scaler):
    mock_model.cfg.head = 'unsupported_head'
    data = {'y_1D': torch.zeros(2, 5, 1)}
    with pytest.raises(
        NotImplementedError, match='Sampling mode not supported'
    ):
        samplingutils.sample_pointpredictions(
            model=mock_model,
            data=data,
            n_samples=5,
            scaler=mock_scaler,
        )
