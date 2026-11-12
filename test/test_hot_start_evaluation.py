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

"""Integration coverage for saving hot-start states through evaluation."""

import csv
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ruamel.yaml import YAML

from googlehydrology.evaluation.tester import RegressionTester
from googlehydrology.training.train import start_training
from googlehydrology.utils.config import Config
from test import test_integration_pipeline
from test.test_integration_pipeline import _get_base_config_dict

integration_data_env = test_integration_pipeline.integration_data_env


@pytest.fixture(scope='module')
def hot_start_run(
    integration_data_env: dict[str, str],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Train a small real model once using the existing local data fixture."""
    run_root = tmp_path_factory.mktemp('hot_start_runs')
    config = _get_base_config_dict(
        integration_data_env, 'hot_start_evaluation', str(run_root)
    )
    config.update(
        {
            'model': 'handoff_forecast_lstm',
            'state_handoff_network': {
                'type': 'fc',
                'hiddens': [16],
                'activation': ['tanh'],
                'dropout': 0.0,
            },
            'epochs': 1,
            'max_updates_per_epoch': 1,
            'metrics': [],
            'log_n_figures': 0,
            'log_tensorboard': False,
            'validate_n_random_basins': 8,
            'validation_end_date': '03/01/2001',
            'test_end_date': '03/01/2001',
        }
    )
    start_training(Config(config))
    return next(run_root.glob('*/config.yml')).parent


@pytest.mark.integration
@pytest.mark.parametrize(
    ('period', 'save_state', 'save_state_format'),
    [
        ('test', True, 'npz'),
        ('test', True, 'csv'),
        ('validation', True, 'npz'),
        ('train', True, 'npz'),
        ('test', False, 'npz'),
    ],
)
def test_evaluate_saves_reloadable_hot_start_states(
    hot_start_run: Path,
    tmp_path: Path,
    *,
    period: str,
    save_state: bool,
    save_state_format: str,
) -> None:
    """Exercise the public tester, real state archives, and forecast replay."""
    cfg = Config(hot_start_run / 'config.yml')
    cfg.update_config(
        {
            'batch_size': 1,
            'save_state': save_state,
            'save_state_format': save_state_format,
        }
    )
    tester = RegressionTester(cfg, run_dir=hot_start_run, period=period)
    # Keep each case's output separate while using the real trained weights
    # and scaler. The public evaluate method loads the copied checkpoint.
    shutil.copy2(hot_start_run / 'model_epoch001.pt', tmp_path)
    tester.run_dir = tmp_path
    with patch.object(
        tester.model, 'save_state', wraps=tester.model.save_state
    ) as save:
        tester.evaluate(epoch=1, save_results=False)

    state_dir = tmp_path / 'hot_start_states'
    if period == 'train' or not save_state:
        assert not state_dir.exists()
        save.assert_not_called()
        return

    assert save.call_count == len(tester.basins)
    assert {p.name for p in state_dir.glob(f'*.{save_state_format}')} == {
        f'state_{basin}.{save_state_format}' for basin in tester.basins
    }
    expected_keys = {'h_hindcast', 'c_hindcast', 'h_forecast', 'c_forecast'}
    for call in save.call_args_list:
        data, path = call.args
        if save_state_format == 'npz':
            with np.load(path, allow_pickle=False) as state:
                assert set(state.files) == expected_keys
                for value in state.values():
                    assert value.shape == (1, 1, cfg.hidden_size)
                    assert np.isfinite(value).all()
        else:
            with path.open('r', encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert {row['state'] for row in rows} == expected_keys
            assert len(rows) == len(expected_keys) * cfg.hidden_size
            assert all(np.isfinite(float(row['value'])) for row in rows)

        model = tester.model
        model.reset_state()
        with torch.inference_mode():
            cold = model(data)['y_hat'][:, -cfg.lead_time :]
            model.load_state_from_disk(path)
            original_length = model.seq_length
            model.seq_length = 0
            cfg.seq_length = 0
            hot_data = dict(data)
            hot_data['x_d_hindcast'] = {
                key: value[:, :0] for key, value in data['x_d_hindcast'].items()
            }
            hot = model(hot_data)['y_hat'][:, -cfg.lead_time :]
            model.seq_length = original_length
            cfg.seq_length = original_length
        torch.testing.assert_close(hot, cold, rtol=1e-5, atol=1e-6)

    if save_state_format == 'csv':
        # Verify RegressionTester discovers and loads saved CSV states from a
        # directory even when save_state_format stays at its default ('npz').
        hot_cfg = Config(hot_start_run / 'config.yml')
        hot_cfg.update_config(
            {'batch_size': 1, 'hot_start_path': str(state_dir)}
        )
        hot_tester = RegressionTester(
            hot_cfg, run_dir=hot_start_run, period=period
        )
        shutil.copy2(hot_start_run / 'model_epoch001.pt', tmp_path)
        hot_tester.run_dir = tmp_path
        with patch.object(
            hot_tester.model,
            'load_state_from_disk',
            wraps=hot_tester.model.load_state_from_disk,
        ) as load_state:
            hot_tester.evaluate(epoch=1, save_results=False)
        assert load_state.call_count == len(hot_tester.basins)
        assert {c.args[0].name for c in load_state.call_args_list} == {
            f'state_{basin}.csv' for basin in hot_tester.basins
        }


def test_hot_start_template_uses_single_sample_batches() -> None:
    """The example must satisfy the tester's hot-start batch-size constraint."""
    path = (
        Path(__file__).parents[1]
        / 'example-configs/template_hot_start_config.yml'
    )
    with path.open() as stream:
        config = YAML(typ='safe').load(stream)
    assert config['batch_size'] == 1
    assert config['save_state_format'] == 'npz'
