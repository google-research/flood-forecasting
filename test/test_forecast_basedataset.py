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

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Callable, Dict, List

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo.forecast_basedataset import ForecastDataset
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError

# --- Helper for Config attributes ---
def _get_default_config_attributes(**kwargs):
    """Returns a dictionary of default config attributes."""
    defaults = {
        'base_run_dir': Path('/tmp/test_run'),
        'run_dir': Path('/tmp/test_run'),
        'lead_time': 1,
        'seq_length': 3,
        'predict_last_n': 1,
        'forecast_overlap': 0,
        'static_attributes': ['static_f1'],
        'target_variables': ['target_v1'],
        'hindcast_inputs': ['hindcast_i1'],
        'forecast_inputs': ['forecast_i1'],
        'statics_data_dir': Path('/tmp/data/statics'),
        'dynamics_data_dir': Path('/tmp/data/dynamics'),
        'targets_data_dir': Path('/tmp/data/targets'),
        'nan_handling_method': 'none',
        'timestep_counter': False,
        'use_basin_id_encoding': False,
        'train_start_date': '01/01/2000',
        'train_end_date': '03/01/2000',
        'test_start_date': '01/01/2000',
        'test_end_date': '03/01/2000',
        'validation_start_date': '01/01/2000',
        'validation_end_date': '03/01/2000',
        'loss': 'mse',
        'custom_normalization': {},
        'is_finetuning': False,
    }
    defaults.update(kwargs)
    return defaults

# --- Pytest Fixtures ---

@pytest.fixture
def get_config(tmp_path: Path) -> Callable[[str], Config]:
    """Fixture that provides a function to fetch a run configuration specified by its name.

    The fetched run configuration will use a tmp folder as its run directory.
    This version simulates loading a config without an actual file, by manually
    setting attributes on a Config object.

    Parameters
    ----------
    tmp_path : Path
        Path to the tmp directory to use in the run configuration.

    Returns
    -------
    Callable[[str], Config]
        Function that returns a run configuration.
    """
    def _get_config(name):
        # To satisfy Config's constructor which expects a file, we create a minimal dummy YAML file.
        # Its content will be immediately overridden by manual attribute setting.
        dummy_config_path = tmp_path / f"{name}.yml"
        dummy_config_path.write_text("dataset: dummy\n") # Minimal valid YAML content

        config = Config(dummy_config_path)
        
        # Manually set attributes, overriding any defaults loaded from the dummy file.
        attrs = _get_default_config_attributes(
            base_run_dir=tmp_path / "run_base",
            run_dir=tmp_path / "run",
            train_basin_file=tmp_path / "train_basins.txt",
            validation_basin_file=tmp_path / "validation_basins.txt",
            test_basin_file=tmp_path / "test_basins.txt"
        )
        config.update_config(attrs)

        # Ensure run directories exist for Scaler
        config.base_run_dir.mkdir(parents=True, exist_ok=True)
        config.run_dir.mkdir(parents=True, exist_ok=True)
        
        return config

    return _get_config

@pytest.fixture
def sample_basins():
    """Provides a list of sample basin IDs."""
    return ['basin_01', 'basin_02']

@pytest.fixture
def sample_dates(get_config):
    """Provides a date range for the dataset based on mock_config."""
    cfg = get_config("default") # Get a default config
    # This range needs to be large enough to cover seq_length + sample_dates + lead_time
    start_date = pd.to_datetime(cfg.train_start_date) - pd.Timedelta(days=cfg.seq_length + cfg.lead_time)
    end_date = pd.to_datetime(cfg.train_end_date) + pd.Timedelta(days=cfg.seq_length + cfg.lead_time)
    return pd.date_range(start_date, end_date, freq='D')

@pytest.fixture
def mock_load_data_return(get_config, sample_basins, sample_dates):
    """
    Returns a mock xarray.Dataset that simulates the output of _load_data.
    This dataset is structured to satisfy the requirements of Scaler and validate_samples.
    """
    cfg = get_config("default") # Get a default config
    data_vars = {}
    coords = {
        'basin': sample_basins,
        'date': sample_dates,
    }

    # Add lead_time coordinate if forecast_inputs are present
    if cfg.forecast_inputs:
        coords['lead_time'] = [np.timedelta64(t, 'D') for t in range(1, cfg.lead_time + 1)]

    # Populate data variables based on cfg's feature lists
    for var in cfg.static_attributes:
        data_vars[var] = (('basin',), np.random.rand(len(sample_basins)))
    for var in cfg.hindcast_inputs:
        data_vars[var] = (('basin', 'date'), np.random.rand(len(sample_basins), len(sample_dates)))
    for var in cfg.target_variables:
        data_vars[var] = (('basin', 'date'), np.random.rand(len(sample_basins), len(sample_dates)))
    
    if cfg.forecast_inputs:
        for var in cfg.forecast_inputs:
            data_vars[var] = (('basin', 'date', 'lead_time'), 
                              np.random.rand(len(sample_basins), len(sample_dates), len(coords['lead_time'])))

    return xr.Dataset(data_vars, coords=coords).astype('float32')


# --- Tests for ForecastDataset ---

# Patching `ForecastDataset._load_data` because it's a NotImplementedError in the base class
# and needs to return a concrete Dataset for the rest of __init__ to function.
@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
@patch('neuralhydrology.datasetzoo.forecast_basedataset.Scaler')
def test_forecast_dataset_init_success(
    mock_scaler, mock_load_data, mock_load_basin_file,
    get_config, sample_basins, mock_load_data_return
):
    """
    Tests successful initialization of ForecastDataset.
    """
    # Configure mocks *before* instantiating the dataset
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return
    # Configure the mock Scaler instance that ForecastDataset will create
    mock_scaler_instance = MagicMock()
    mock_scaler.return_value = mock_scaler_instance # When Scaler() is called, return this mock instance
    mock_scaler_instance.scale.return_value = mock_load_data_return, [MagicMock()]
    mock_scaler_instance.save_task.return_value = None # save_task() does nothing
    
    cfg = get_config("default") # Get a default config

    # Instantiate the dataset
    dataset = ForecastDataset(cfg=cfg, is_train=True, period='train')

    # Assertions
    mock_load_basin_file.assert_called_once()
    mock_load_data.assert_called_once()
    mock_scaler_instance.scale.assert_called_once_with(mock_load_data_return)
    mock_scaler_instance.save_task.assert_called_once() # Called if compute_scaler is True
    assert dataset.is_train is True
    assert dataset._period == 'train'
    assert dataset._basins == sample_basins
    assert hasattr(dataset, 'scaler')
    assert dataset._num_samples > 0 # Should have samples if validate_samples works as expected
    assert dataset._dataset is not None

    
@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_len(
    mock_load_data, mock_load_basin_file, get_config, sample_basins, mock_load_data_return
):
    """
    Tests the __len__ method of ForecastDataset.
    """
    cfg = get_config("default") # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    dataset = ForecastDataset(cfg=cfg, is_train=True, period='train')
    
    # The exact number of samples depends on the dates, seq_length, etc.
    # We just need to ensure it's a positive integer.
    assert isinstance(len(dataset), int)
    assert len(dataset) > 0


@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_getitem(
    mock_load_data, mock_load_basin_file, get_config, sample_basins, mock_load_data_return
):
    """
    Tests the __getitem__ method of ForecastDataset.
    """
    cfg = get_config("default") # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    dataset = ForecastDataset(cfg=cfg, is_train=True, period='train')

    # Test valid index
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert 'date' in sample
    assert 'x_s' in sample and isinstance(sample['x_s'], torch.Tensor)
    assert 'x_d_hindcast' in sample and isinstance(sample['x_d_hindcast'], dict)
    assert 'x_d_forecast' in sample and isinstance(sample['x_d_forecast'], dict)
    assert 'y' in sample and isinstance(sample['y'], torch.Tensor)

    # Check shapes (basic check, more detailed checks can be added)
    assert sample['x_s'].shape == (1,) # For a single static feature
    assert sample['y'].shape == (cfg.seq_length, 1) # seq_length, num_target_features
    assert sample['x_d_hindcast']['hindcast_i1'].shape == (cfg.seq_length, 1)
    assert sample['x_d_forecast']['forecast_i1'].shape == (cfg.lead_time, 1)

    # Test IndexError for out-of-bounds
    with pytest.raises(IndexError):
        dataset[len(dataset)]

    # Test ValueError for negative index
    with pytest.raises(ValueError):
        dataset[-1]

    # Test ValueError for non-integer index
    with pytest.raises(ValueError):
        dataset[0.5]


@patch('neuralhydrology.datasetzoo.forecast_basedataset.validate_samples')
@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_no_train_data_error(
    mock_load_data, mock_load_basin_file, mock_validate_samples,
    get_config, sample_basins, mock_load_data_return
):
    """
    Tests that NoTrainDataError is raised when no valid training samples are found.
    This requires mocking `validate_samples` to return an empty mask.
    """
    cfg = get_config("default") # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return
    empty_mask = xr.DataArray(
        np.full((len(sample_basins), len(mock_load_data_return['date'])), False),
        coords={'basin': sample_basins, 'date': mock_load_data_return['date'].values},
        dims=['basin', 'date']
    )
    mock_validate_samples.return_value = (empty_mask, {})

    with pytest.raises(NoTrainDataError):
        ForecastDataset(cfg=cfg, is_train=True, period='train')

@patch('neuralhydrology.datasetzoo.forecast_basedataset.validate_samples')
@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch('neuralhydrology.datasetzoo.forecast_basedataset.Scaler')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_no_evaluation_data_error(
    mock_load_data, mock_scaler, mock_load_basin_file, mock_validate_samples,
    get_config, sample_basins, mock_load_data_return
):
    """
    Tests that NoEvaluationDataError is raised when no valid evaluation samples are found.
    """
    cfg = get_config("default")
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return
    empty_mask = xr.DataArray(
        np.full((len(sample_basins), len(mock_load_data_return['date'])), False),
        coords={'basin': sample_basins, 'date': mock_load_data_return['date'].values},
        dims=['basin', 'date']
    )
    mock_validate_samples.return_value = (empty_mask, {})
    mock_scaler_instance = MagicMock()
    mock_scaler.return_value = mock_scaler_instance # When Scaler() is called, return this mock instance
    mock_scaler_instance.scale.return_value = mock_load_data_return, MagicMock()
    mock_scaler_instance.save_task.return_value = None # save_task() does nothing

    with pytest.raises(NoEvaluationDataError):
        ForecastDataset(cfg=cfg, is_train=False, period='test', compute_scaler=False)

@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_init_period_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError for invalid 'period' argument during initialization.
    """
    cfg = get_config("default") # Get a default config
    with pytest.raises(ValueError, match="'period' must be one of 'train', 'validation' or 'test'"):
        ForecastDataset(cfg=cfg, is_train=True, period='invalid_period')

@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_init_forecast_hindcast_mismatch_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError when only one of `forecast_inputs` or `hindcast_inputs` is supplied.
    """
    # Case 1: No inputs are supplied
    cfg = get_config("default")
    cfg.update_config({'hindcast_inputs': [], 'forecast_inputs': [], 'dynamic_inputs': []})
    with pytest.raises(ValueError, match='Either `hindcast_inputs` or `dynamic_inputs` must be supplied.'):
        ForecastDataset(cfg=cfg, is_train=True, period='train')

    # Case 2: Neither dynamic_inputs nor hindcast_inputs are supplied
    cfg = get_config("default")
    cfg.update_config({'hindcast_inputs': [], 'dynamic_inputs': []})
    with pytest.raises(ValueError, match='Either `hindcast_inputs` or `dynamic_inputs` must be supplied.'):
        ForecastDataset(cfg=cfg, is_train=True, period='train')

@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_init_compute_scaler_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError when compute_scaler is True for validation/test/finetuning periods.
    """
    cfg = get_config("default") # Get a default config
    # Validation period
    with pytest.raises(ValueError, match="Scaler must be loaded \(not computed\) for validation, test, and finetuning."):
        ForecastDataset(cfg=cfg, is_train=False, period='validation', compute_scaler=True)

    # Test period
    with pytest.raises(ValueError, match="Scaler must be loaded \(not computed\) for validation, test, and finetuning."):
        ForecastDataset(cfg=cfg, is_train=False, period='test', compute_scaler=True)

    # Finetuning
    cfg_finetuning = get_config("default")
    cfg_finetuning.is_finetuning = True # Manually set attribute
    with pytest.raises(ValueError, match="Scaler must be loaded \(not computed\) for validation, test, and finetuning."):
        ForecastDataset(cfg=cfg_finetuning, is_train=True, period='train', compute_scaler=True)

@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_nan_handling_method_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError when feature groups are required but not supplied for nan_handling_method.
    """
    # Simulate a scenario where hindcast_inputs are strings, not lists of lists
    cfg_nan_handling = get_config("default")
    cfg_nan_handling.update_config({'hindcast_inputs': ['single_feature']})
    cfg_nan_handling.update_config({'forecast_inputs': ['another_single_feature']})
    cfg_nan_handling.update_config({'nan_handling_method': 'masked_mean'})
    with pytest.raises(ValueError, match='Feature groups are required for masked_mean NaN-handling.'):
        ForecastDataset(cfg=cfg_nan_handling, is_train=True, period='train')

    cfg_nan_handling_attention = get_config("default")
    cfg_nan_handling_attention.update_config({'hindcast_inputs': ['single_feature']})
    cfg_nan_handling_attention.update_config({'forecast_inputs': ['another_single_feature']})
    cfg_nan_handling_attention.update_config({'nan_handling_method': 'attention'})
    with pytest.raises(ValueError, match='Feature groups are required for attention NaN-handling.'):
        ForecastDataset(cfg=cfg_nan_handling_attention, is_train=True, period='train')

    cfg_nan_handling_unioning = get_config("default")
    cfg_nan_handling_unioning.update_config({'hindcast_inputs': ['single_feature']})
    cfg_nan_handling_unioning.update_config({'forecast_inputs': ['another_single_feature']})
    cfg_nan_handling_unioning.update_config({'nan_handling_method': 'unioning'})
    with pytest.raises(ValueError, match='Feature groups are required for unioning NaN-handling.'):
        ForecastDataset(cfg=cfg_nan_handling_unioning, is_train=True, period='train')

@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_per_basin_target_stds(
    mock_load_data, mock_load_basin_file, get_config, sample_basins, mock_load_data_return
):
    """
    Tests that per_basin_target_stds is calculated when loss requires it.
    """
    cfg = get_config("default") # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    cfg_nse = get_config("default")
    cfg_nse.loss = 'nse' # Manually set attribute
    dataset_nse = ForecastDataset(cfg=cfg_nse, is_train=True, period='train')
    sample_nse = dataset_nse[0]
    assert 'per_basin_target_stds' in sample_nse
    assert isinstance(sample_nse['per_basin_target_stds'], torch.Tensor)
    assert sample_nse['per_basin_target_stds'].shape == (1, 1) # 1 basin, 1 target variable

    cfg_weightednse = get_config("default")
    cfg_weightednse.loss = 'weightednse' # Manually set attribute
    dataset_weightednse = ForecastDataset(cfg=cfg_weightednse, is_train=True, period='train')
    sample_weightednse = dataset_weightednse[0]
    assert 'per_basin_target_stds' in sample_weightednse
    assert isinstance(sample_weightednse['per_basin_target_stds'], torch.Tensor)
    assert sample_weightednse['per_basin_target_stds'].shape == (1, 1)

    cfg_mse = get_config("default")
    cfg_mse.loss = 'mse' # Manually set attribute
    dataset_mse = ForecastDataset(cfg=cfg_mse, is_train=True, period='train')
    sample_mse = dataset_mse[0]
    assert 'per_basin_target_stds' not in sample_mse

@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_timestep_counter(
    mock_load_data, mock_load_basin_file, get_config, sample_basins, mock_load_data_return
):
    """
    Tests that timestep counters are added when cfg.timestep_counter is True.
    """
    cfg = get_config("default") # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    cfg_with_counter = get_config("default")
    cfg_with_counter.update_config({'timestep_counter': True}) # Manually set attribute
    dataset = ForecastDataset(cfg=cfg_with_counter, is_train=True, period='train')
    sample = dataset[0]

    assert 'hindcast_counter' in sample['x_d_hindcast']
    assert isinstance(sample['x_d_hindcast']['hindcast_counter'], torch.Tensor)
    assert sample['x_d_hindcast']['hindcast_counter'].shape == (cfg_with_counter.seq_length, 1)

    assert 'forecast_counter' in sample['x_d_forecast']
    assert isinstance(sample['x_d_forecast']['forecast_counter'], torch.Tensor)
    assert sample['x_d_forecast']['forecast_counter'].shape == (cfg_with_counter.lead_time + cfg_with_counter.forecast_overlap, 1)

@patch('neuralhydrology.datasetzoo.forecast_basedataset.load_basin_file')
@patch.object(ForecastDataset, '_load_data')
def test_forecast_dataset_no_forecast_features_renames_key(
    mock_load_data, mock_load_basin_file, get_config, sample_basins, mock_load_data_return
):
    """
    Tests that 'x_d_hindcast' is renamed to 'x_d' if no forecast features are present.
    """
    cfg = get_config("default") # Get a default config
    mock_load_basin_file.return_value = sample_basins
    # Create a config with no forecast inputs
    cfg_no_forecast = get_config("default")
    cfg_no_forecast.update_config({'forecast_inputs': []})
    cfg_no_forecast.update_config({'hindcast_inputs': ['hindcast_i1']})
    
    # Adjust the mock_load_data_return to not include lead_time dim if no forecast_inputs
    data_vars = {}
    coords = {
        'basin': sample_basins,
        'date': mock_load_data_return['date'].values,
    }
    for var in cfg_no_forecast.static_attributes:
        data_vars[var] = (('basin',), np.random.rand(len(sample_basins)))
    for var in cfg_no_forecast.hindcast_inputs:
        data_vars[var] = (('basin', 'date'), np.random.rand(len(sample_basins), len(coords['date'])))
    for var in cfg_no_forecast.target_variables:
        data_vars[var] = (('basin', 'date'), np.random.rand(len(sample_basins), len(coords['date'])))
    
    mock_load_data.return_value = xr.Dataset(data_vars, coords=coords).astype('float32')

    dataset = ForecastDataset(cfg=cfg_no_forecast, is_train=True, period='train')
    sample = dataset[0]

    assert 'x_d' in sample
    assert 'x_d_hindcast' not in sample
    assert 'x_d_forecast' not in sample
