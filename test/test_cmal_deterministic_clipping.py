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

from unittest.mock import MagicMock

import pytest
import torch
import xarray as xr

from googlehydrology.utils import samplingutils


@pytest.mark.unit
def test_deterministic_cmal_clips_at_normalized_zero(monkeypatch):
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.cfg.head = 'cmal_deterministic'
    model.cfg.mc_dropout = False
    model.cfg.target_variables = ['streamflow']
    model.cfg.use_frequencies = ['1D']
    model.cfg.predict_last_n = {'1D': 2}
    model.cfg.negative_sample_handling = 'clip'

    scaler = MagicMock()
    scaler.scaler = {
        'streamflow': xr.DataArray(
            [5.0, 2.0],
            coords={'parameter': ['center', 'scale']},
            dims=['parameter'],
        )
    }

    generated = torch.tensor(
        [[[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [-4.0, -2.5, -1.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]]]
    )
    monkeypatch.setattr(
        samplingutils.cmal_deterministic,
        'generate_predictions',
        lambda *args: generated.clone(),
    )

    outputs = {
        'mu': torch.zeros(1, 2, 3),
        'b': torch.ones(1, 2, 3),
        'tau': torch.full((1, 2, 3), 0.5),
        'pi': torch.full((1, 2, 3), 1.0 / 3),
    }
    samples = samplingutils.sample_pointpredictions(
        model,
        {'y': torch.zeros(1, 2, 1)},
        n_samples=10,
        scaler=scaler,
        outputs=outputs,
    )

    assert samples['y_hat'].shape == (1, 2, 1, 10)
    assert samples['y_hat'].min().item() == -2.5
    assert samples['y_hat'][0, 0, 0, 1].item() == -2.0
