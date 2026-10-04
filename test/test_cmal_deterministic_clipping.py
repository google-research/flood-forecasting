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


def _model_with_handling(handling):
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.cfg.head = 'cmal_deterministic'
    model.cfg.mc_dropout = False
    model.cfg.target_variables = ['streamflow']
    model.cfg.use_frequencies = ['1D']
    model.cfg.predict_last_n = {'1D': 2}
    model.cfg.negative_sample_handling = handling
    return model


@pytest.mark.unit
@pytest.mark.parametrize(
    'handling, expected_min',
    [('clip', -2.5), ('none', -4.0), (None, -4.0), ('truncate', -4.0)],
)
def test_deterministic_cmal_negative_handling(monkeypatch, handling, expected_min):
    model = _model_with_handling(handling)

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
    assert samples['y_hat'].min().item() == expected_min
    if handling == 'clip':
        assert samples['y_hat'][0, 0, 0, 1].item() == -2.0


@pytest.mark.unit
def test_deterministic_cmal_rejects_unknown_negative_handling(monkeypatch):
    model = _model_with_handling('bogus')
    scaler = MagicMock()
    scaler.scaler = {
        'streamflow': xr.DataArray(
            [5.0, 2.0],
            coords={'parameter': ['center', 'scale']},
            dims=['parameter'],
        )
    }
    generated = torch.zeros(1, 2, 10)
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

    with pytest.raises(NotImplementedError, match='bogus'):
        samplingutils.sample_pointpredictions(
            model,
            {'y': torch.zeros(1, 2, 1)},
            n_samples=10,
            scaler=scaler,
            outputs=outputs,
        )


@pytest.mark.unit
def test_deterministic_cmal_unmocked_negative_quantiles() -> None:
    """Unmocked generate_predictions preserves or clips negative quantiles."""
    scaler = MagicMock()
    scaler.scaler = {
        'streamflow': xr.DataArray(
            [5.0, 2.0],
            coords={'parameter': ['center', 'scale']},
            dims=['parameter'],
        )
    }
    outputs = {
        'mu': torch.full((1, 2, 3), -4.0),
        'b': torch.ones(1, 2, 3),
        'tau': torch.full((1, 2, 3), 0.5),
        'pi': torch.full((1, 2, 3), 1.0 / 3.0),
    }

    unclipped_model = _model_with_handling('none')
    unclipped = samplingutils.sample_pointpredictions(
        unclipped_model,
        {'y': torch.zeros(1, 2, 1)},
        n_samples=10,
        scaler=scaler,
        outputs=outputs,
    )['y_hat']
    # Median (index 5: mean at 0, q=0.1..0.5 at 1..5) of symmetric mu=-4 is -4.
    torch.testing.assert_close(
        unclipped[:, :, 0, 5], torch.full((1, 2), -4.0), rtol=1e-5, atol=1e-5
    )

    clipped_model = _model_with_handling('clip')
    clipped = samplingutils.sample_pointpredictions(
        clipped_model,
        {'y': torch.zeros(1, 2, 1)},
        n_samples=10,
        scaler=scaler,
        outputs=outputs,
    )['y_hat']
    # Normalized zero threshold is -center / scale = -5.0 / 2.0 = -2.5.
    assert torch.all(clipped >= -2.5)
    torch.testing.assert_close(
        clipped[:, :, 0, 5], torch.full((1, 2), -2.5), rtol=1e-5, atol=1e-5
    )

