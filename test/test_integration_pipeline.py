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

"""Integration tests for GoogleHydrology training and evaluation pipeline."""

from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from googlehydrology.datasetzoo.caravan import load_caravan_timeseries_together
from googlehydrology.evaluation.evaluate import start_evaluation
from googlehydrology.run import continue_run
from googlehydrology.run import eval_run
from googlehydrology.run import start_run
from googlehydrology.training.train import start_training
from googlehydrology.utils.config import Config


@pytest.fixture(scope='module')
def integration_data_env():
    """Builds a temporary dynamic dataset environment using tutorial NetCDFs."""
    tmp_dir = tempfile.mkdtemp(prefix='hydrology_integration_')
    base_path = Path(__file__).resolve().parent.parent
    nc_dir = base_path / 'tutorial' / 'Caravan-nc'
    train_basin_file = (
        base_path / 'tutorial' / 'basin-lists' / '5-basin-train.txt'
    )
    test_basin_file = (
        base_path / 'tutorial' / 'basin-lists' / '8-basin-test.txt'
    )

    with open(test_basin_file) as f:
        basins = [line.strip() for line in f if line.strip()]

    raw_features = ['total_precipitation_sum', 'temperature_2m_mean']
    ds = load_caravan_timeseries_together(
        nc_dir, basins=basins, target_features=raw_features, csv=False
    )
    ds = ds.rename({
        'total_precipitation_sum': 'era5land_total_precipitation',
        'temperature_2m_mean': 'era5land_temperature_2m',
    })
    lead_times = pd.to_timedelta(np.arange(8), unit='D')
    ds_forecast = ds.expand_dims(lead_time=lead_times).copy()

    dynamics_dir = Path(tmp_dir) / 'dynamics'
    era5_zarr = dynamics_dir / 'ERA5_LAND' / 'timeseries.zarr'
    era5_zarr.parent.mkdir(parents=True, exist_ok=True)
    ds_forecast.to_zarr(era5_zarr, consolidated=True)

    env_info = {
        'tmp_dir': tmp_dir,
        'nc_dir': str(nc_dir.resolve()),
        'dynamics_dir': str(dynamics_dir.resolve()),
        'train_basin_file': str(train_basin_file.resolve()),
        'test_basin_file': str(test_basin_file.resolve()),
    }

    yield env_info

    shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_base_config_dict(
    env_info: dict[str, str], exp_name: str, run_dir: str
) -> dict[str, Any]:
    """Generates standard baseline config dictionary using fixed seed."""
    return {
        'experiment_name': exp_name,
        'run_dir': run_dir,
        'dataset': 'multimet',
        'train_basin_file': env_info['train_basin_file'],
        'validation_basin_file': env_info['train_basin_file'],
        'test_basin_file': env_info['test_basin_file'],
        'targets_data_dir': env_info['nc_dir'],
        'statics_data_dir': env_info['nc_dir'],
        'dynamics_data_dir': env_info['dynamics_dir'],
        'train_start_date': '01/01/2000',
        'train_end_date': '31/12/2000',
        'validation_start_date': '01/01/2001',
        'validation_end_date': '31/12/2001',
        'test_start_date': '01/01/2001',
        'test_end_date': '31/12/2001',
        'hindcast_inputs': {
            'era5land': [
                'era5land_total_precipitation',
                'era5land_temperature_2m',
            ]
        },
        'forecast_inputs': {
            'era5land': [
                'era5land_total_precipitation',
                'era5land_temperature_2m',
            ]
        },
        'static_attributes': ['area', 'p_mean'],
        'target_variables': ['streamflow'],
        'model': 'mean_embedding_forecast_lstm',
        'hidden_size': 16,
        'head': 'regression',
        'output_activation': 'linear',
        'statics_embedding': {
            'type': 'fc',
            'hiddens': [32, 16],
            'activation': ['tanh', 'linear'],
            'dropout': 0.0,
        },
        'hindcast_embedding': {
            'type': 'fc',
            'hiddens': [32, 16],
            'activation': ['tanh', 'linear'],
            'dropout': 0.0,
        },
        'forecast_embedding': {
            'type': 'fc',
            'hiddens': [32, 16],
            'activation': ['tanh', 'linear'],
            'dropout': 0.0,
        },
        'seq_length': 30,
        'lead_time': 7,
        'forecast_overlap': 10,
        'timestep_counter': True,
        'output_dropout': 0.0,
        'device': 'cpu',
        'seed': 42,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'epochs': 2,
        'save_weights_every': 1,
        'batch_size': 32,
        'initial_learning_rate': 0.001,
        'metrics': ['NSE', 'KGE', 'RMSE'],
        'predict_last_n': 8,
        'num_workers': 0,
        'validate_every': 1,
        'validate_n_random_basins': -1,
        'cache': {'enabled': False},
    }


@pytest.mark.integration
def test_mean_embedding_forecast_lstm_regression_pipeline(
    integration_data_env, tmp_path
):
    """Integration test for MeanEmbeddingForecastLSTM with Regression."""
    run_dir = str(tmp_path / 'runs_mean_emb')
    cfg_dict = _get_base_config_dict(
        integration_data_env, 'test_mean_emb_reg', run_dir
    )
    cfg = Config(cfg_dict)

    # 1. Train model from scratch
    start_training(cfg)

    created_runs = list((tmp_path / 'runs_mean_emb').glob('*'))
    assert len(created_runs) == 1
    actual_run_dir = created_runs[0]

    # Verify artifacts and checkpoints
    assert (actual_run_dir / 'config.yml').is_file()
    assert (actual_run_dir / 'output.log').is_file()
    assert (actual_run_dir / 'train_data').is_dir()
    assert (actual_run_dir / 'model_epoch001.pt').is_file()
    assert (actual_run_dir / 'model_epoch002.pt').is_file()

    # 2. Run evaluation on test period
    start_evaluation(cfg=cfg, run_dir=actual_run_dir, epoch=2, period='test')

    test_eval_dir = actual_run_dir / 'test' / 'model_epoch002'
    assert test_eval_dir.is_dir()
    assert (test_eval_dir / 'test_results.zarr').is_dir()
    assert (test_eval_dir / 'test_metrics.csv').is_file()

    # Verify results contents and metrics values
    ds_pred = xr.open_zarr(
        test_eval_dir / 'test_results.zarr', consolidated=False
    )
    assert 'streamflow_sim' in ds_pred
    assert 'streamflow_obs' in ds_pred
    assert not np.all(np.isnan(ds_pred['streamflow_sim'].values))

    # Read metrics summary CSV
    df_metrics = pd.read_csv(test_eval_dir / 'test_metrics.csv')
    assert not df_metrics.empty
    assert 'NSE' in df_metrics.columns
    assert 'KGE' in df_metrics.columns
    assert 'RMSE' in df_metrics.columns
    assert np.all(np.isfinite(df_metrics['NSE'].dropna().values))


@pytest.mark.integration
def test_handoff_forecast_lstm_cmal_pipeline(integration_data_env, tmp_path):
    """End-to-end integration test for HandoffForecastLSTM with CMAL."""
    run_dir = str(tmp_path / 'runs_handoff_cmal')
    cfg_dict = _get_base_config_dict(
        integration_data_env, 'test_handoff_cmal', run_dir
    )
    cfg_dict.update({
        'model': 'handoff_forecast_lstm',
        'head': 'cmal',
        'loss': 'CMAL',
        'n_distributions': 3,
        'n_samples': 5,
        'state_handoff_network': {
            'type': 'fc',
            'hiddens': [32, 16],
            'activation': ['tanh', 'linear'],
            'dropout': 0.0,
        },
        'epochs': 1,
    })
    cfg = Config(cfg_dict)

    # 1. Train CMAL model
    start_training(cfg)

    created_runs = list((tmp_path / 'runs_handoff_cmal').glob('*'))
    assert len(created_runs) == 1
    actual_run_dir = created_runs[0]

    assert (actual_run_dir / 'model_epoch001.pt').is_file()

    # 2. Evaluate CMAL probabilistic model
    start_evaluation(cfg=cfg, run_dir=actual_run_dir, epoch=1, period='test')

    test_eval_dir = actual_run_dir / 'test' / 'model_epoch001'
    assert (test_eval_dir / 'test_results.zarr').is_dir()

    ds_pred = xr.open_zarr(
        test_eval_dir / 'test_results.zarr', consolidated=False
    )
    assert 'streamflow_sim' in ds_pred
    # CMAL predictions must have non-NaN values
    assert np.any(np.isfinite(ds_pred['streamflow_sim'].values))


@pytest.mark.integration
def test_continue_training_and_finetuning_pipeline(
    integration_data_env, tmp_path
):
    """Tests checkpoint resuming and freezing modules for fine-tuning."""
    run_dir = str(tmp_path / 'runs_finetune_base')
    cfg_dict = _get_base_config_dict(
        integration_data_env, 'test_finetune_base', run_dir
    )
    cfg_dict['epochs'] = 1
    cfg = Config(cfg_dict)

    # 1. Initial 1-epoch training
    start_training(cfg)
    actual_run_dir = list((tmp_path / 'runs_finetune_base').glob('*'))[0]
    ckpt_epoch1 = actual_run_dir / 'model_epoch001.pt'
    assert ckpt_epoch1.is_file()

    # Record weights from epoch 1
    weights_epoch1 = torch.load(
        ckpt_epoch1, weights_only=True, map_location='cpu'
    )

    # 2. Test continue_run (train for 1 more epoch)
    continue_cfg_dict = _get_base_config_dict(
        integration_data_env, 'test_finetune_base', str(actual_run_dir.parent)
    )
    continue_cfg_dict.update({
        'base_run_dir': str(actual_run_dir),
        'run_dir': str(actual_run_dir),
        'is_continue_training': True,
        'continue_from_epoch': 1,
        'epochs': 1,
    })
    continue_cfg = Config(continue_cfg_dict)
    start_training(continue_cfg)

    continue_run_dir = actual_run_dir / 'continue_training_from_epoch001'
    assert continue_run_dir.is_dir()
    ckpt_epoch2 = continue_run_dir / 'model_epoch002.pt'
    assert ckpt_epoch2.is_file()
    weights_epoch2 = torch.load(
        ckpt_epoch2, weights_only=True, map_location='cpu'
    )

    # Weights must update from epoch 1 to epoch 2
    weight_diff = False
    for k in weights_epoch1:
        if not torch.equal(weights_epoch1[k], weights_epoch2[k]):
            weight_diff = True
            break
    assert weight_diff, 'Model weights did not change after continue_run'

    # 3. Test finetune with frozen feature embeddings (only train head)
    finetune_dir = str(tmp_path / 'runs_finetune_child')
    finetune_cfg_dict = _get_base_config_dict(
        integration_data_env, 'test_finetuned', finetune_dir
    )
    finetune_cfg_dict.update({
        'base_run_dir': str(actual_run_dir),
        'is_finetuning': True,
        'finetune_modules': ['head'],
        'epochs': 1,
    })
    finetune_cfg = Config(finetune_cfg_dict)
    start_training(finetune_cfg)

    actual_finetune_dir = list((tmp_path / 'runs_finetune_child').glob('*'))[0]
    ckpt_finetuned = actual_finetune_dir / 'model_epoch001.pt'
    assert ckpt_finetuned.is_file()

    weights_finetuned = torch.load(
        ckpt_finetuned, weights_only=True, map_location='cpu'
    )

    # Static embedding and LSTM weights must remain FROZEN
    for k in weights_finetuned:
        if 'static_embedding_fc' in k or 'hindcast_lstm' in k:
            assert torch.equal(weights_finetuned[k], weights_epoch1[k]), (
                f'Frozen parameter {k} changed during finetuning!'
            )


@pytest.mark.integration
def test_inference_mode_without_ground_truth(integration_data_env, tmp_path):
    """Tests evaluation in inference mode when observations are unavailable."""
    run_dir = str(tmp_path / 'runs_infer_mode')
    cfg_dict = _get_base_config_dict(
        integration_data_env, 'test_infer_mode', run_dir
    )
    cfg_dict['epochs'] = 1
    cfg_dict['tester_skip_obs_all_nan'] = False
    cfg = Config(cfg_dict)

    start_training(cfg)
    actual_run_dir = list((tmp_path / 'runs_infer_mode').glob('*'))[0]

    # Evaluate on test period without raising errors when unobserved
    start_evaluation(cfg=cfg, run_dir=actual_run_dir, epoch=1, period='test')
    test_eval_dir = actual_run_dir / 'test' / 'model_epoch001'
    assert (test_eval_dir / 'test_results.zarr').is_dir()


@pytest.mark.integration
def test_numerical_determinism_with_fixed_seed(
    integration_data_env, tmp_path
):
    """Verifies that runs with fixed seeds produce exact numerical equality."""
    run_dir_1 = str(tmp_path / 'runs_determ_1')
    run_dir_2 = str(tmp_path / 'runs_determ_2')

    cfg_dict_1 = _get_base_config_dict(
        integration_data_env, 'test_determ_1', run_dir_1
    )
    cfg_dict_1['epochs'] = 1
    cfg_dict_1['seed'] = 42

    cfg_dict_2 = _get_base_config_dict(
        integration_data_env, 'test_determ_2', run_dir_2
    )
    cfg_dict_2['epochs'] = 1
    cfg_dict_2['seed'] = 42

    cfg_1 = Config(cfg_dict_1)
    cfg_2 = Config(cfg_dict_2)

    start_training(cfg_1)
    start_training(cfg_2)

    actual_dir_1 = list((tmp_path / 'runs_determ_1').glob('*'))[0]
    actual_dir_2 = list((tmp_path / 'runs_determ_2').glob('*'))[0]

    weights_1 = torch.load(
        actual_dir_1 / 'model_epoch001.pt',
        weights_only=True,
        map_location='cpu',
    )
    weights_2 = torch.load(
        actual_dir_2 / 'model_epoch001.pt',
        weights_only=True,
        map_location='cpu',
    )

    # 1. Assert exact identical trained weights across all parameter layers
    assert set(weights_1.keys()) == set(weights_2.keys())
    for k in weights_1:
        assert torch.equal(weights_1[k], weights_2[k]), (
            f'Weights for {k} diverged between deterministic runs!'
        )

    # 2. Evaluate both models and assert identical predicted time series
    start_evaluation(cfg=cfg_1, run_dir=actual_dir_1, epoch=1, period='test')
    start_evaluation(cfg=cfg_2, run_dir=actual_dir_2, epoch=1, period='test')

    ds_pred_1 = xr.open_zarr(
        actual_dir_1 / 'test' / 'model_epoch001' / 'test_results.zarr',
        consolidated=False,
    )
    ds_pred_2 = xr.open_zarr(
        actual_dir_2 / 'test' / 'model_epoch001' / 'test_results.zarr',
        consolidated=False,
    )

    np.testing.assert_allclose(
        ds_pred_1['streamflow_sim'].values,
        ds_pred_2['streamflow_sim'].values,
        rtol=1e-5,
        atol=1e-5,
        err_msg='Evaluated predictions differed between identical seed runs!',
    )


@pytest.mark.integration
def test_run_cli_entrypoints(integration_data_env, tmp_path):
    """Tests the googlehydrology.run CLI entrypoints."""
    run_dir = str(tmp_path / 'runs_cli')
    cfg_dict = _get_base_config_dict(
        integration_data_env, 'test_cli_pipeline', run_dir
    )
    cfg_dict['epochs'] = 1
    cfg = Config(cfg_dict)

    # 1. start_run
    start_run(config=cfg, gpu=-1)
    actual_run_dir = list((tmp_path / 'runs_cli').glob('*'))[0]
    assert (actual_run_dir / 'model_epoch001.pt').is_file()

    # 2. eval_run
    eval_run(
        config=cfg,
        run_dir=actual_run_dir,
        period='test',
        epoch=1,
        gpu=-1,
    )
    assert (
        actual_run_dir / 'test' / 'model_epoch001' / 'test_results.zarr'
    ).is_dir()

    # 3. continue_run
    continue_run(run_dir=actual_run_dir, gpu=-1)
    continue_dir = actual_run_dir / 'continue_training_from_epoch001'
    assert continue_dir.is_dir()

