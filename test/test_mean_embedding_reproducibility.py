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

"""Keep model initialization independent of Python's string hash seed."""

import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import xarray as xr

import googlehydrology
from googlehydrology.datasetzoo.multimet import (
    _get_products_and_bands_from_feature_strings,
)
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config
from googlehydrology.utils.configutils import flatten_feature_list

# A separate interpreter is required: changing os.environ in-process does not
# change the hash seed that Python selected at startup.
_PROCESS_CODE = """
import sys
from pathlib import Path
import torch
from googlehydrology.utils.config import Config
from googlehydrology.utils.configutils import flatten_feature_list
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.training import get_loss_obj

torch.set_num_threads(1)
cfg = Config(Path(sys.argv[1]))
torch.manual_seed(cfg.seed)
model = MeanEmbeddingForecastLSTM(cfg)
initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
torch.manual_seed(99)
data = {
    'x_s': torch.rand(2, len(cfg.static_attributes)),
    'x_d_hindcast': {
        name: torch.rand(2, cfg.seq_length, 1)
        for name in flatten_feature_list(cfg.hindcast_inputs)
    },
    'x_d_forecast': {
        name: torch.rand(2, cfg.seq_length + cfg.lead_time, 1)
        for name in flatten_feature_list(cfg.forecast_inputs)
    },
    'y': torch.rand(2, cfg.seq_length + cfg.lead_time, 1),
}
predictions = model(data)
loss, _ = get_loss_obj(cfg)(predictions, data)
loss.backward()
gradients = {k: v.grad.detach().clone() for k, v in model.named_parameters()}
torch.save({'weights': initial, 'predictions': predictions,
            'gradients': gradients, 'loss': loss}, sys.argv[2])
"""


def _config(tmp_path: Path, layout: str = 'dict') -> Config:
    """Build real, differently sized shared and period-specific embeddings."""
    hindcast = {
        'zulu': ['zulu_rain', 'zulu_temp'],
        'sharedtwo': ['sharedtwo_rain'],
        'alpha': ['alpha_rain'],
        'sharedone': ['sharedone_rain', 'sharedone_temp'],
        'mike': ['mike_rain'],
    }
    forecast = {
        'omega': ['omega_rain'],
        'sharedone': ['sharedone_rain', 'sharedone_temp'],
        'delta': ['delta_rain', 'delta_temp'],
        'sharedtwo': ['sharedtwo_rain'],
        'bravo': ['bravo_rain'],
    }
    if layout == 'nested':
        hindcast, forecast = list(hindcast.values()), list(forecast.values())
    elif layout == 'flat':
        hindcast = flatten_feature_list(hindcast)
        forecast = flatten_feature_list(forecast)
    embedding = {
        'type': 'fc',
        'hiddens': [4],
        'activation': ['tanh'],
        'dropout': 0.0,
    }
    cfg = Config(
        {
            'model': 'mean_embedding_forecast_lstm',
            'run_dir': str(tmp_path),
            'seed': 17,
            'seq_length': 4,
            'lead_time': 2,
            'forecast_overlap': 4,
            'predict_last_n': 2,
            'hidden_size': 4,
            'head': 'regression',
            'loss': 'mse',
            'target_variables': ['streamflow'],
            'static_attributes': ['area'],
            'hindcast_inputs': hindcast,
            'forecast_inputs': forecast,
            'statics_embedding': embedding.copy(),
            'hindcast_embedding': embedding.copy(),
            'forecast_embedding': embedding.copy(),
            'output_dropout': 0.0,
        }
    )
    xr.Dataset(
        {'streamflow': ('parameter', [0.0, 1.0, 0.0, 1.0])},
        coords={'parameter': ['center', 'scale', 'mean', 'std']},
    ).to_netcdf(tmp_path / 'scaler.nc', engine='scipy')
    return cfg


@pytest.mark.unit
@pytest.mark.parametrize('layout', ['dict', 'nested', 'flat'])
def test_embedding_groups_follow_config_order(
    tmp_path: Path, layout: str
) -> None:
    """Retain config order and instantiate shared groups only once."""
    cfg = _config(tmp_path, layout)
    with torch.random.fork_rng(devices=[]):
        model = MeanEmbeddingForecastLSTM(cfg)
    assert list(model.hindcast_embeddings_fc) == ['zulu', 'alpha', 'mike']
    assert list(model.forecast_embeddings_fc) == ['omega', 'delta', 'bravo']
    assert list(model.shared_embeddings_fc) == ['sharedtwo', 'sharedone']


@pytest.mark.integration
@pytest.mark.parametrize('layout', ['dict', 'nested', 'flat'])
def test_hash_seed_does_not_change_model_or_gradients(
    tmp_path: Path,
    layout: str,
) -> None:
    """Compare weights, predictions, and gradients across interpreters."""
    cfg = _config(tmp_path, layout)
    options = cfg.as_dict()
    options['run_dir'] = str(cfg.run_dir)
    config_path = tmp_path / 'model.yml'
    config_path.write_text(json.dumps(options), encoding='utf-8')
    snapshots = []
    for hash_seed in ('1', '7', '19'):
        output = tmp_path / f'state-{hash_seed}.pt'
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = hash_seed
        # Use the active installation, also when testing an installed wheel.
        env['PYTHONPATH'] = str(
            Path(googlehydrology.__file__).resolve().parents[1]
        )
        subprocess.run(  # noqa: S603 - Fixed interpreter and test code.
            [
                sys.executable,
                '-c',
                _PROCESS_CODE,
                str(config_path),
                str(output),
            ],
            env=env,
            cwd=tmp_path,
            check=True,
            capture_output=True,
            timeout=90,
        )
        snapshots.append(torch.load(output, weights_only=True))
    reference = snapshots[0]
    for actual in snapshots[1:]:
        for section in ('weights', 'predictions', 'gradients'):
            assert list(actual[section]) == list(reference[section])
            for name, value in actual[section].items():
                assert torch.isfinite(value).all(), (section, name)
                assert torch.equal(value, reference[section][name]), (
                    section,
                    name,
                )
        assert torch.equal(actual['loss'], reference['loss'])


@pytest.mark.unit
def test_seed_still_controls_initial_weights(tmp_path: Path) -> None:
    """Different tensor seeds must still select different initial weights."""
    cfg = _config(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = MeanEmbeddingForecastLSTM(cfg).state_dict()
        torch.manual_seed(2)
        second = MeanEmbeddingForecastLSTM(cfg).state_dict()
    assert list(first) == list(second)
    assert any(not torch.equal(first[name], second[name]) for name in first)


@pytest.mark.unit
def test_checkpoint_key_order_does_not_change_loading(tmp_path: Path) -> None:
    """Older checkpoints remain loadable by name, irrespective of key order."""
    cfg = _config(tmp_path)
    with torch.random.fork_rng(devices=[]):
        original = MeanEmbeddingForecastLSTM(cfg)
        restored = MeanEmbeddingForecastLSTM(cfg)
    checkpoint = tmp_path / 'reordered.pt'
    torch.save(
        OrderedDict(reversed(list(original.state_dict().items()))), checkpoint
    )
    restored.load_state_dict(
        torch.load(checkpoint, weights_only=True), strict=True
    )
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


@pytest.mark.unit
def test_all_shared_groups_remain_supported(tmp_path: Path) -> None:
    """Keep the existing shared-only construction and ordering."""
    cfg = _config(tmp_path)
    options = cfg.as_dict()
    options['forecast_inputs'] = options['hindcast_inputs']
    with torch.random.fork_rng(devices=[]):
        model = MeanEmbeddingForecastLSTM(Config(options))
    assert not model.hindcast_embeddings_fc
    assert not model.forecast_embeddings_fc
    assert list(model.shared_embeddings_fc) == list(options['hindcast_inputs'])


@pytest.mark.unit
@pytest.mark.parametrize('use_iterator', [False, True])
def test_product_features_keep_first_seen_order(*, use_iterator: bool) -> None:
    """Guard the already order-preserving feature parser mentioned in #305."""
    features = [
        'hres_temp',
        'era5land_rain',
        'hres_rain',
        'cpc_rain',
        'hres_temp',
    ]
    result = _get_products_and_bands_from_feature_strings(
        iter(features) if use_iterator else features,
    )
    assert list(result.items()) == [
        ('HRES', ['hres_temp', 'hres_rain']),
        ('ERA5_LAND', ['era5land_rain']),
        ('CPC', ['cpc_rain']),
    ]
