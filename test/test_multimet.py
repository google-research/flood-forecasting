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
import re
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Callable

from googlehydrology.datasetzoo.multimet import Multimet
from googlehydrology.utils.config import Config
from googlehydrology.utils.errors import NoTrainDataError, NoEvaluationDataError


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
        'train_start_date': ['01/01/2000', '01/02/2006'],
        'train_end_date': ['03/01/2000', '03/02/2006'],
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
        dummy_config_path = tmp_path / f'{name}.yml'
        dummy_config_path.write_text(
            'dataset: dummy\n'
        )  # Minimal valid YAML content

        config = Config(dummy_config_path)

        # Manually set attributes, overriding any defaults loaded from the dummy file.
        attrs = _get_default_config_attributes(
            base_run_dir=tmp_path / 'run_base',
            run_dir=tmp_path / 'run',
            train_basin_file=tmp_path / 'train_basins.txt',
            validation_basin_file=tmp_path / 'validation_basins.txt',
            test_basin_file=tmp_path / 'test_basins.txt',
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
    cfg = get_config('default')  # Get a default config
    # This range needs to be large enough to cover seq_length + sample_dates + lead_time
    start_date = pd.to_datetime(cfg.train_start_date[0]) - pd.Timedelta(
        days=cfg.seq_length + cfg.lead_time
    )
    end_date = pd.to_datetime(cfg.train_end_date[-1]) + pd.Timedelta(
        days=cfg.seq_length + cfg.lead_time
    )
    return pd.date_range(start_date, end_date, freq='D')


@pytest.fixture
def mock_load_data_return(get_config, sample_basins, sample_dates):
    """
    Returns a mock xarray.Dataset that simulates the output of _load_data.
    This dataset is structured to satisfy the requirements of Scaler and validate_samples.
    """
    cfg = get_config('default')  # Get a default config
    data_vars = {}
    coords = {
        'basin': sample_basins,
        'date': sample_dates,
    }

    # Add lead_time coordinate if forecast_inputs are present
    if cfg.forecast_inputs:
        coords['lead_time'] = [
            np.timedelta64(t, 'D') for t in range(1, cfg.lead_time + 1)
        ]

    # Populate data variables based on cfg's feature lists
    for var in cfg.static_attributes:
        data_vars[var] = (('basin',), np.random.rand(len(sample_basins)))
    for var in cfg.hindcast_inputs:
        data_vars[var] = (
            ('basin', 'date'),
            np.random.rand(len(sample_basins), len(sample_dates)),
        )
    for var in cfg.target_variables:
        data_vars[var] = (
            ('basin', 'date'),
            np.random.rand(len(sample_basins), len(sample_dates)),
        )

    if cfg.forecast_inputs:
        for var in cfg.forecast_inputs:
            data_vars[var] = (
                ('basin', 'date', 'lead_time'),
                np.random.rand(
                    len(sample_basins),
                    len(sample_dates),
                    len(coords['lead_time']),
                ),
            )

    return xr.Dataset(data_vars, coords=coords).astype('float32')


# --- Tests for Multimet ---


# Patching `Multimet._load_data` because it's a NotImplementedError in the base class
# and needs to return a concrete Dataset for the rest of __init__ to function.
@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
@patch('googlehydrology.datasetzoo.multimet.Scaler')
def test_forecast_dataset_init_success(
    mock_scaler,
    mock_load_data,
    mock_load_basin_file,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests successful initialization of Multimet.
    """
    # Configure mocks *before* instantiating the dataset
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return
    # Configure the mock Scaler instance that Multimet will create
    mock_scaler_instance = MagicMock()
    mock_scaler.return_value = mock_scaler_instance  # When Scaler() is called, return this mock instance
    mock_scaler_instance.scale.return_value = mock_load_data_return
    mock_scaler_instance.check_zero_scale.return_value = None
    mock_scaler_instance.save.return_value = None  # save() does nothing

    cfg = get_config('default')  # Get a default config

    # Instantiate the dataset
    dataset = Multimet(cfg=cfg, is_train=True, period='train')

    # Assertions
    mock_load_basin_file.assert_called_once()
    mock_load_data.assert_called_once()
    mock_scaler_instance.scale.assert_called_once_with(mock_load_data_return)
    mock_scaler_instance.save.assert_called_once()  # When compute_scaler=True
    mock_scaler_instance.check_zero_scale.assert_called_once()

    expected_call_order = ['check_zero_scale', 'save', 'scale']
    actual_call_order = [e[0] for e in mock_scaler_instance.method_calls]
    assert actual_call_order == expected_call_order

    assert dataset.is_train is True
    assert dataset._period == 'train'
    assert dataset._basins == sample_basins
    assert hasattr(dataset, 'scaler')
    assert (
        dataset._num_samples > 0
    )  # Should have samples if validate_samples works as expected
    assert dataset._dataset is not None


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_len(
    mock_load_data,
    mock_load_basin_file,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests the __len__ method of Multimet.
    """
    cfg = get_config('default')  # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    dataset = Multimet(cfg=cfg, is_train=True, period='train')

    # The exact number of samples depends on the dates, seq_length, etc.
    # We just need to ensure it's a positive integer.
    assert isinstance(len(dataset), int)
    assert len(dataset) > 0


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_getitem(
    mock_load_data,
    mock_load_basin_file,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests the __getitem__ method of Multimet.
    """
    cfg = get_config('default')  # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    dataset = Multimet(cfg=cfg, is_train=True, period='train')

    # Test valid index
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert 'date' in sample
    assert 'x_s' in sample
    assert 'x_d_hindcast' in sample and isinstance(sample['x_d_hindcast'], dict)
    assert 'x_d_forecast' in sample and isinstance(sample['x_d_forecast'], dict)
    assert 'y' in sample

    # Assert all values are np.ndarray
    for k, v in sample.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                assert isinstance(v1, np.ndarray), f'{k=} {k1=}'
        else:
            assert isinstance(v, np.ndarray), f'{k=}'

    # Check shapes (basic check, more detailed checks can be added)
    assert sample['x_s'].shape == (1,)  # For a single static feature
    assert sample['y'].shape == (
        cfg.seq_length,
        1,
    )  # seq_length, num_target_features
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


@patch('googlehydrology.datasetzoo.multimet.validate_samples')
@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_no_train_data_error(
    mock_load_data,
    mock_load_basin_file,
    mock_validate_samples,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests that NoTrainDataError is raised when no valid training samples are found.
    This requires mocking `validate_samples` to return an empty mask.
    """
    cfg = get_config('default')  # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return
    empty_mask = xr.DataArray(
        np.full(
            (len(sample_basins), len(mock_load_data_return['date'])), False
        ),
        coords={
            'basin': sample_basins,
            'date': mock_load_data_return['date'].values,
        },
        dims=['basin', 'date'],
    )
    mock_validate_samples.return_value = (empty_mask, {})

    with pytest.raises(NoTrainDataError):
        Multimet(cfg=cfg, is_train=True, period='train')


@patch('googlehydrology.datasetzoo.multimet.validate_samples')
@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch('googlehydrology.datasetzoo.multimet.Scaler')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_no_evaluation_data_error(
    mock_load_data,
    mock_scaler,
    mock_load_basin_file,
    mock_validate_samples,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests that NoEvaluationDataError is raised when no valid evaluation samples are found.
    """
    cfg = get_config('default')
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return
    empty_mask = xr.DataArray(
        np.full(
            (len(sample_basins), len(mock_load_data_return['date'])), False
        ),
        coords={
            'basin': sample_basins,
            'date': mock_load_data_return['date'].values,
        },
        dims=['basin', 'date'],
    )
    mock_validate_samples.return_value = (empty_mask, {})
    mock_scaler_instance = MagicMock()
    mock_scaler.return_value = mock_scaler_instance  # When Scaler() is called, return this mock instance
    mock_scaler_instance.scale.return_value = mock_load_data_return
    mock_scaler_instance.save.return_value = None  # save() does nothing

    with pytest.raises(NoEvaluationDataError):
        Multimet(cfg=cfg, is_train=False, period='test', compute_scaler=False)


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_init_period_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError for invalid 'period' argument during initialization.
    """
    cfg = get_config('default')  # Get a default config
    with pytest.raises(
        ValueError,
        match="'period' must be one of 'train', 'validation' or 'test'",
    ):
        Multimet(cfg=cfg, is_train=True, period='invalid_period')


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_init_forecast_hindcast_mismatch_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError when only one of `forecast_inputs` or `hindcast_inputs` is supplied.
    """
    # Case 1: No inputs are supplied
    cfg = get_config('default')
    cfg.update_config(
        {'hindcast_inputs': [], 'forecast_inputs': []}
    )
    with pytest.raises(
        ValueError,
        match='hindcast_inputs must be supplied.',
    ):
        Multimet(cfg=cfg, is_train=True, period='train')

    # Case 2: Hindcast_inputs are not supplied
    cfg = get_config('default')
    cfg.update_config({'hindcast_inputs': []})
    with pytest.raises(
        ValueError,
        match='hindcast_inputs must be supplied.',
    ):
        Multimet(cfg=cfg, is_train=True, period='train')


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_init_compute_scaler_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError when compute_scaler is True for validation/test/finetuning periods.
    """
    cfg = get_config('default')  # Get a default config
    # Validation period
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Scaler must be loaded (not computed) for validation, test, and finetuning.'
        ),
    ):
        Multimet(
            cfg=cfg, is_train=False, period='validation', compute_scaler=True
        )

    # Test period
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Scaler must be loaded (not computed) for validation, test, and finetuning.'
        ),
    ):
        Multimet(cfg=cfg, is_train=False, period='test', compute_scaler=True)

    # Finetuning
    cfg_finetuning = get_config('default')
    cfg_finetuning.is_finetuning = True  # Manually set attribute
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Scaler must be loaded (not computed) for validation, test, and finetuning.'
        ),
    ):
        Multimet(
            cfg=cfg_finetuning,
            is_train=True,
            period='train',
            compute_scaler=True,
        )


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_nan_handling_method_error(
    mock_load_data, mock_load_basin_file, get_config
):
    """
    Tests ValueError when feature groups are required but not supplied for nan_handling_method.
    """
    # Simulate a scenario where hindcast_inputs are strings, not lists of lists
    cfg_nan_handling = get_config('default')
    cfg_nan_handling.update_config({'hindcast_inputs': ['single_feature']})
    cfg_nan_handling.update_config(
        {'forecast_inputs': ['another_single_feature']}
    )
    cfg_nan_handling.update_config({'nan_handling_method': 'masked_mean'})
    with pytest.raises(
        ValueError,
        match='Feature groups are required for masked_mean NaN-handling.',
    ):
        Multimet(cfg=cfg_nan_handling, is_train=True, period='train')

    cfg_nan_handling_attention = get_config('default')
    cfg_nan_handling_attention.update_config(
        {'hindcast_inputs': ['single_feature']}
    )
    cfg_nan_handling_attention.update_config(
        {'forecast_inputs': ['another_single_feature']}
    )
    cfg_nan_handling_attention.update_config(
        {'nan_handling_method': 'attention'}
    )
    with pytest.raises(
        ValueError,
        match='Feature groups are required for attention NaN-handling.',
    ):
        Multimet(cfg=cfg_nan_handling_attention, is_train=True, period='train')

    cfg_nan_handling_unioning = get_config('default')
    cfg_nan_handling_unioning.update_config(
        {'hindcast_inputs': ['single_feature']}
    )
    cfg_nan_handling_unioning.update_config(
        {'forecast_inputs': ['another_single_feature']}
    )
    cfg_nan_handling_unioning.update_config({'nan_handling_method': 'unioning'})
    with pytest.raises(
        ValueError,
        match='Feature groups are required for unioning NaN-handling.',
    ):
        Multimet(cfg=cfg_nan_handling_unioning, is_train=True, period='train')


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_per_basin_target_stds(
    mock_load_data,
    mock_load_basin_file,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests that per_basin_target_stds is calculated when loss requires it.
    """
    cfg = get_config('default')  # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    cfg_nse = get_config('default')
    cfg_nse.loss = 'nse'  # Manually set attribute
    dataset_nse = Multimet(cfg=cfg_nse, is_train=True, period='train')
    sample_nse = dataset_nse[0]
    assert 'per_basin_target_stds' in sample_nse
    assert isinstance(sample_nse['per_basin_target_stds'], np.ndarray)
    assert sample_nse['per_basin_target_stds'].shape == (
        1,
        1,
    )  # 1 basin, 1 target variable

    cfg_mse = get_config('default')
    cfg_mse.loss = 'mse'  # Manually set attribute
    dataset_mse = Multimet(cfg=cfg_mse, is_train=True, period='train')
    sample_mse = dataset_mse[0]
    assert 'per_basin_target_stds' not in sample_mse


@pytest.mark.unit
@pytest.mark.parametrize(
    ('da_loss', 'expected'), [('NSE', True), ('nse', True), ('MSE', False)]
)
@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_per_basin_target_stds_for_assimilation_loss(
    mock_load_data,
    mock_load_basin_file,
    get_config,
    sample_basins,
    mock_load_data_return,
    da_loss,
    expected,
):
    """An NSE assimilation loss provides the stds like an NSE training loss."""
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return
    cfg = get_config('default')
    cfg.loss = 'mse'
    cfg.update_config(
        {
            'assimilation_config': {
                'assimilation_components': ['static_embedding'],
                'assimilation_window': 1,
                'initial_learning_rate': 0.1,
                'loss': da_loss,
            }
        }
    )

    sample = Multimet(cfg=cfg, is_train=True, period='train')[0]

    assert ('per_basin_target_stds' in sample) is expected


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_timestep_counter(
    mock_load_data,
    mock_load_basin_file,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests that timestep counters are added when cfg.timestep_counter is True.
    """
    cfg = get_config('default')  # Get a default config
    mock_load_basin_file.return_value = sample_basins
    mock_load_data.return_value = mock_load_data_return

    cfg_with_counter = get_config('default')
    cfg_with_counter.update_config(
        {'timestep_counter': True}
    )  # Manually set attribute
    dataset = Multimet(cfg=cfg_with_counter, is_train=True, period='train')
    sample = dataset[0]

    assert 'hindcast_counter' in sample['x_d_hindcast']
    assert isinstance(sample['x_d_hindcast']['hindcast_counter'], np.ndarray)
    assert sample['x_d_hindcast']['hindcast_counter'].shape == (
        cfg_with_counter.seq_length,
        1,
    )

    assert 'forecast_counter' in sample['x_d_forecast']
    assert isinstance(sample['x_d_forecast']['forecast_counter'], np.ndarray)
    assert sample['x_d_forecast']['forecast_counter'].shape == (
        cfg_with_counter.lead_time + cfg_with_counter.forecast_overlap,
        1,
    )


@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
@patch.object(Multimet, '_load_data')
def test_forecast_dataset_no_forecast_features_renames_key(
    mock_load_data,
    mock_load_basin_file,
    get_config,
    sample_basins,
    mock_load_data_return,
):
    """
    Tests that 'x_d_hindcast' is renamed to 'x_d' if no forecast features are present.
    """
    cfg = get_config('default')  # Get a default config
    mock_load_basin_file.return_value = sample_basins
    # Create a config with no forecast inputs
    cfg_no_forecast = get_config('default')
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
        data_vars[var] = (
            ('basin', 'date'),
            np.random.rand(len(sample_basins), len(coords['date'])),
        )
    for var in cfg_no_forecast.target_variables:
        data_vars[var] = (
            ('basin', 'date'),
            np.random.rand(len(sample_basins), len(coords['date'])),
        )

    mock_load_data.return_value = xr.Dataset(data_vars, coords=coords).astype(
        'float32'
    )

    dataset = Multimet(cfg=cfg_no_forecast, is_train=True, period='train')
    sample = dataset[0]

    assert 'x_d' in sample
    assert 'x_d_hindcast' not in sample
    assert 'x_d_forecast' not in sample


def test_product_name_parsing_and_normalization():
    from googlehydrology.datasetzoo.multimet import (
        _canonical_product_name,
        _get_products_and_bands_from_feature_strings,
        _get_products_and_bands_from_features,
        _normalize_product_key,
        _product_name_from_feature,
    )

    assert _normalize_product_key('CHIRPS_GEFS') == 'chirpsgefs'
    assert _normalize_product_key('era5-land') == 'era5land'

    assert _canonical_product_name('chirps_gefs') == 'CHIRPS_GEFS'
    assert _canonical_product_name('chirpsgefs') == 'CHIRPS_GEFS'
    assert _canonical_product_name('ERA5_LAND') == 'ERA5_LAND'
    assert _canonical_product_name('era5land') == 'ERA5_LAND'
    assert _canonical_product_name('graphcast') == 'GRAPHCAST'
    assert _canonical_product_name('custom_prod') == 'CUSTOM_PROD'

    # Feature string parsing with underscores and token-boundary safety
    assert (
        _product_name_from_feature('chirps_gefs_precipitation')
        == 'CHIRPS_GEFS'
    )
    assert _product_name_from_feature('CHIRPS_GEFS_precip') == 'CHIRPS_GEFS'
    assert _product_name_from_feature('era5_land_temperature') == 'ERA5_LAND'
    assert _product_name_from_feature('era5land_temperature') == 'ERA5_LAND'
    assert _product_name_from_feature('cpc_precip') == 'CPC'
    assert _product_name_from_feature('chirps2_precip') == 'CHIRPS2'

    # Flat string feature list
    features = ['chirps_gefs_precip', 'era5_land_temperature', 'cpc_precip']
    pb = _get_products_and_bands_from_feature_strings(features)
    assert pb == {
        'CHIRPS_GEFS': ['chirps_gefs_precip'],
        'ERA5_LAND': ['era5_land_temperature'],
        'CPC': ['cpc_precip'],
    }

    # Nested feature groups (list[list[str]])
    nested_features = [
        ['chirps_gefs_precip', 'cpc_precip'],
        ['era5_land_temperature'],
    ]
    pb_nested = _get_products_and_bands_from_features(nested_features)
    assert pb_nested == {
        'CHIRPS_GEFS': ['chirps_gefs_precip'],
        'CPC': ['cpc_precip'],
        'ERA5_LAND': ['era5_land_temperature'],
    }

    # Dict-formatted inputs (including duplicate canonical keys)
    dict_features = {
        'chirps_gefs': ['chirps_gefs_precip'],
        'era5land': ['era5_land_temperature'],
        'ERA5_LAND': ['era5_land_pressure'],
    }
    pb_dict = _get_products_and_bands_from_features(dict_features)
    assert pb_dict == {
        'CHIRPS_GEFS': ['chirps_gefs_precip'],
        'ERA5_LAND': ['era5_land_temperature', 'era5_land_pressure'],
    }


@patch('googlehydrology.datasetzoo.multimet.load_caravan_attributes')
@patch('googlehydrology.datasetzoo.multimet.load_caravan_timeseries')
@patch('googlehydrology.datasetzoo.multimet._open_zarr')
@patch('googlehydrology.datasetzoo.multimet.load_basin_file')
def test_multimet_dict_inputs_and_missing_band_validation(
    mock_load_basin_file,
    mock_open_zarr,
    mock_load_targets,
    mock_load_statics,
    get_config,
    sample_basins,
):
    mock_load_basin_file.return_value = sample_basins
    dates = pd.date_range('1999-12-25', '2006-03-10', freq='D')
    lead_times = [pd.Timedelta(days=1)]
    rng = np.random.default_rng(42)

    mock_load_statics.return_value = xr.Dataset(
        {
            'static_f1': (
                ('basin',),
                rng.random(len(sample_basins), dtype=np.float32),
            )
        },
        coords={'basin': sample_basins},
    )
    mock_load_targets.return_value = xr.Dataset(
        {
            'target_v1': (
                ('basin', 'date'),
                rng.random(
                    (len(sample_basins), len(dates)), dtype=np.float32
                ),
            )
        },
        coords={'basin': sample_basins, 'date': dates},
    )

    chirps_gefs_ds = xr.Dataset(
        {
            'chirps_gefs_precip': (
                ('basin', 'date', 'lead_time'),
                rng.random(
                    (len(sample_basins), len(dates), len(lead_times)),
                    dtype=np.float32,
                ),
            )
        },
        coords={'basin': sample_basins, 'date': dates, 'lead_time': lead_times},
    )
    era5_land_ds = xr.Dataset(
        {
            'era5_land_temp': (
                ('basin', 'date'),
                rng.random(
                    (len(sample_basins), len(dates)), dtype=np.float32
                ),
            )
        },
        coords={'basin': sample_basins, 'date': dates},
    )

    def fake_open_zarr(path: Path):
        if 'CHIRPS_GEFS' in str(path):
            return chirps_gefs_ds
        if 'ERA5_LAND' in str(path):
            return era5_land_ds
        raise FileNotFoundError(path)

    mock_open_zarr.side_effect = fake_open_zarr

    cfg = get_config('dict_inputs')
    cfg.update_config(
        {
            'hindcast_inputs': {
                'chirps_gefs': ['chirps_gefs_precip'],
                'era5_land': ['era5_land_temp'],
            },
            'forecast_inputs': {
                'chirps_gefs': ['chirps_gefs_precip'],
            },
        }
    )

    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    mock_load_targets.assert_called_once()
    sample = dataset[0]
    assert 'chirps_gefs_precip' in sample['x_d_hindcast']
    assert 'era5_land_temp' in sample['x_d_hindcast']
    assert 'chirps_gefs_precip' in sample['x_d_forecast']

    # Verify missing variable in Zarr store raises ValueError immediately
    cfg_missing = get_config('dict_inputs_missing')
    cfg_missing.update_config(
        {
            'hindcast_inputs': {
                'era5_land': ['era5_land_missing_var'],
            },
            'forecast_inputs': {
                'chirps_gefs': ['chirps_gefs_precip'],
            },
        }
    )
    with pytest.raises(ValueError, match='era5_land_missing_var'):
        Multimet(cfg=cfg_missing, is_train=True, period='train')


def _day_offset_dataset(
    basins: list[str],
    dates: pd.DatetimeIndex,
    lead_times: list[np.timedelta64],
) -> xr.Dataset:
    """Builds a dataset whose values encode the valid date of each entry.

    Each value is the day offset of its valid date from `dates[0]`, so a
    misaligned extraction window shows up as the wrong numbers. Following
    Caravan-MultiMet, a forecast issued on date t with `lead_time = k days` is
    valid on day t + (k - 1).
    """
    day_offsets = np.arange(len(dates), dtype=np.float32)
    forecast_vals = np.stack(
        [
            day_offsets + (lt / np.timedelta64(1, 'D') - 1)
            for lt in lead_times
        ],
        axis=-1,
    )
    n_basins = len(basins)
    static_vals = np.arange(1, n_basins + 1, dtype=np.float32)
    return xr.Dataset(
        {
            'static_f1': (('basin',), static_vals),
            'era5land_2d': (
                ('basin', 'date'),
                np.tile(day_offsets, (n_basins, 1)),
            ),
            'hres_3d': (
                ('basin', 'date', 'lead_time'),
                np.tile(forecast_vals, (n_basins, 1, 1)).astype(np.float32),
            ),
            'target_v1': (
                ('basin', 'date'),
                np.tile(day_offsets * 10.0, (n_basins, 1)).astype(np.float32),
            ),
        },
        coords={'basin': basins, 'date': dates, 'lead_time': lead_times},
    )


def _write_multimet_stores(
    root: Path, cfg: Config, ds: xr.Dataset, basins: list[str]
) -> None:
    """Writes real basin and Zarr stores under `root` and updates `cfg`."""
    statics_dir = root / 'statics'
    targets_dir = root / 'targets'
    dynamics_dir = root / 'dynamics'
    ds[['static_f1']].to_zarr(statics_dir / 'attributes.zarr', mode='w')
    ds[['target_v1']].to_zarr(targets_dir / 'targets.zarr', mode='w')
    ds[['era5land_2d']].drop_vars('lead_time', errors='ignore').to_zarr(
        dynamics_dir / 'ERA5_LAND' / 'timeseries.zarr', mode='w'
    )
    ds[['hres_3d']].to_zarr(
        dynamics_dir / 'HRES' / 'timeseries.zarr', mode='w'
    )
    cfg.train_basin_file.write_text('\n'.join(basins) + '\n')
    cfg.update_config(
        {
            'statics_data_dir': statics_dir,
            'targets_data_dir': targets_dir,
            'dynamics_data_dir': dynamics_dir,
        }
    )


def test_multimet_lead_time_temporal_alignment(
    tmp_path: Path,
    get_config,
):
    """Verifies Caravan-MultiMet temporal alignment end-to-end without mocks.

    With seq_length=3, forecast_overlap=2 and lead_time=2 for issue date D:
    - 2D hindcasts and 3D hindcasts (first lead time) both cover [D-3, D-1].
    - The forecast overlap (first lead time) covers [D-2, D-1], followed by
      the forecast rollout issued on D, valid on [D, D+1].
    - `date` and `y` end at D + lead_time - 1 = D + 1 and cover [D-1, D+1].
    - `union_mapping` fills NaNs across 2D and 3D products using the same
      valid-date alignment.
    """
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('1999-12-25', '2000-01-10', freq='D')
    lead_times = [np.timedelta64(1, 'D'), np.timedelta64(2, 'D')]
    ds = _day_offset_dataset(basins, dates, lead_times)

    # Inject a NaN into the 2D feature on 1999-12-30 (day offset 5) and into
    # the 3D feature at issue date 2000-01-01 (offset 7), lead_time=2D (valid
    # on 2000-01-02, offset 8). Bidirectional union_mapping must restore both.
    ds['era5land_2d'].loc[{'basin': 'basin_01', 'date': '1999-12-30'}] = np.nan
    ds['hres_3d'].loc[
        {
            'basin': 'basin_01',
            'date': '2000-01-01',
            'lead_time': np.timedelta64(2, 'D'),
        }
    ] = np.nan

    cfg = get_config('default')
    _write_multimet_stores(tmp_path / 'stores', cfg, ds, basins)
    identity_norm = {'centering': 'none', 'scaling': 'none'}
    cfg.update_config(
        {
            'seq_length': 3,
            'lead_time': 2,
            'forecast_overlap': 2,
            'predict_last_n': 3,
            'timestep_counter': True,
            'hindcast_inputs': ['era5land_2d', 'hres_3d'],
            'forecast_inputs': ['hres_3d'],
            'union_mapping': {
                'era5land_2d': 'hres_3d',
                'hres_3d': 'era5land_2d',
            },
            'custom_normalization': {
                'era5land_2d': identity_norm,
                'hres_3d': identity_norm,
                'target_v1': identity_norm,
            },
            'train_start_date': ['01/01/2000'],
            'train_end_date': ['02/01/2000'],
        }
    )

    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    assert len(dataset) == 4  # 2 basins x 2 issue dates
    assert dataset.min_lead_time == 1

    # Sample 0: basin_01, issue date D = 2000-01-01 (day offset 7).
    # Sample 1: basin_01, issue date D = 2000-01-02 (day offset 8).
    for sample_idx, d_offset in [(0, 7.0), (1, 8.0)]:
        sample = dataset[sample_idx]
        expected_hindcast = np.arange(
            d_offset - 3.0, d_offset, dtype=np.float32
        )[:, None]
        np.testing.assert_array_equal(
            sample['x_d_hindcast']['era5land_2d'], expected_hindcast
        )
        np.testing.assert_array_equal(
            sample['x_d_hindcast']['hres_3d'], expected_hindcast
        )
        np.testing.assert_array_equal(
            sample['x_d_hindcast']['hindcast_counter'],
            np.zeros((3, 1), dtype=np.int64),
        )

        # Overlap [D-2, D-1] followed by rollout valid on [D, D+1].
        expected_forecast = np.arange(
            d_offset - 2.0, d_offset + 2.0, dtype=np.float32
        )[:, None]
        np.testing.assert_array_equal(
            sample['x_d_forecast']['hres_3d'], expected_forecast
        )
        np.testing.assert_array_equal(
            sample['x_d_forecast']['forecast_counter'],
            np.array([[1], [1], [1], [2]], dtype=np.int64),
        )

        # Targets of length seq_length=3 ending at D + 1 -> [D-1, D, D+1].
        expected_dates = pd.date_range(
            dates[int(d_offset) - 1], dates[int(d_offset) + 1], freq='D'
        ).values
        np.testing.assert_array_equal(sample['date'], expected_dates)
        expected_targets = (
            np.arange(d_offset - 1.0, d_offset + 2.0, dtype=np.float32)[:, None]
            * 10.0
        )
        np.testing.assert_array_equal(sample['y'], expected_targets)


def test_multimet_hindcast_only_alignment(
    tmp_path: Path,
    get_config,
):
    """Without forecast inputs, 2D and 3D hindcasts end on the sample date.

    In a hindcast-only run (`forecast_inputs: []`, `lead_time: 0`) a 3D
    feature used as a hindcast input is loaded from Zarr via `_lead_time_slice`
    and read at its first lead time, which is valid on the issue date, so it
    must line up with the 2D features and with `date` / `y`.
    """
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('1999-12-25', '2000-01-10', freq='D')
    lead_times = [np.timedelta64(1, 'D'), np.timedelta64(2, 'D')]
    ds = _day_offset_dataset(basins, dates, lead_times)

    cfg = get_config('default')
    _write_multimet_stores(tmp_path / 'stores', cfg, ds, basins)
    identity_norm = {'centering': 'none', 'scaling': 'none'}
    cfg.update_config(
        {
            'seq_length': 3,
            'lead_time': 0,
            'forecast_overlap': 0,
            'predict_last_n': 1,
            'hindcast_inputs': ['era5land_2d', 'hres_3d'],
            'forecast_inputs': [],
            'custom_normalization': {
                'era5land_2d': identity_norm,
                'hres_3d': identity_norm,
                'target_v1': identity_norm,
            },
            'train_start_date': ['01/01/2000'],
            'train_end_date': ['02/01/2000'],
        }
    )

    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    assert len(dataset) == 4
    assert dataset.min_lead_time == 0
    sample = dataset[0]

    # Sample date D is 2000-01-01 (day offset 7); window [D-2, D] -> [5, 6, 7].
    expected = np.array([[5.0], [6.0], [7.0]], dtype=np.float32)
    assert 'x_d' in sample
    np.testing.assert_array_equal(sample['x_d']['era5land_2d'], expected)
    np.testing.assert_array_equal(sample['x_d']['hres_3d'], expected)
    np.testing.assert_array_equal(
        sample['date'],
        pd.date_range('1999-12-30', '2000-01-01', freq='D').values,
    )
    np.testing.assert_array_equal(sample['y'], expected * 10.0)


@pytest.mark.parametrize('forecast_inputs', [['hres_3d'], []])
def test_multimet_valid_samples_match_extracted_windows(
    tmp_path: Path,
    get_config,
    forecast_inputs,
):
    """Accepted samples match the exact set of windows with valid data.

    Runs the full unmocked Multimet pipeline (including Zarr loading and
    Scaler) and checks both soundness (no accepted sample has NaN inputs or
    all-NaN targets) and completeness (every (basin, issue_date) whose
    extracted windows are valid is included in the dataset).
    """
    rng = np.random.default_rng(0)
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('1999-11-01', '2000-03-01', freq='D')
    lead_times = [np.timedelta64(k, 'D') for k in (1, 2, 3)]
    ds = _day_offset_dataset(basins, dates, lead_times)
    for name, nan_fraction in [
        ('era5land_2d', 0.03),
        ('hres_3d', 0.01),
        ('target_v1', 0.5),
    ]:
        values = ds[name].values
        values[rng.random(values.shape) < nan_fraction] = np.nan

    seq_length = 5
    predict_last_n = 4
    lead_time = 3 if forecast_inputs else 0
    forecast_overlap = 2 if forecast_inputs else 0
    min_lead = 1 if forecast_inputs else 0

    cfg = get_config('default')
    _write_multimet_stores(tmp_path / 'stores', cfg, ds, basins)
    cfg.update_config(
        {
            'seq_length': seq_length,
            'predict_last_n': predict_last_n,
            'lead_time': lead_time,
            'forecast_overlap': forecast_overlap,
            'hindcast_inputs': ['era5land_2d', 'hres_3d'],
            'forecast_inputs': forecast_inputs,
            'nan_handling_method': 'none',
            'train_start_date': ['01/12/1999'],
            'train_end_date': ['15/02/2000'],
        }
    )

    dataset = Multimet(cfg=cfg, is_train=True, period='train')

    # Independently determine which (basin_idx, issue_date) pairs have valid
    # windows in the raw dataset.
    sample_dates = set(pd.date_range('1999-12-01', '2000-02-15', freq='D'))
    hres_loaded = ds['hres_3d'].isel(lead_time=slice(0, max(lead_time, 1)))
    era5_vals = ds['era5land_2d'].values
    hres_vals = hres_loaded.values
    target_vals = ds['target_v1'].values
    expected_target_end_dates = []
    for b_idx in range(len(basins)):
        for d_idx, d_val in enumerate(dates):
            if d_val not in sample_dates:
                continue
            h_end = d_idx - min_lead
            h_start = h_end - (seq_length - 1)
            if h_start < 0:
                continue
            if np.isnan(era5_vals[b_idx, h_start : h_end + 1]).any():
                continue
            if np.isnan(hres_vals[b_idx, h_start : h_end + 1, :]).any():
                continue
            if forecast_inputs:
                if np.isnan(hres_vals[b_idx, d_idx, :]).any():
                    continue
                ov_start = d_idx - forecast_overlap
                if np.isnan(hres_vals[b_idx, ov_start:d_idx, 0]).any():
                    continue
            t_end = d_idx + lead_time - min_lead
            t_start = t_end - (predict_last_n - 1)
            if np.isnan(target_vals[b_idx, t_start : t_end + 1]).all():
                continue
            expected_target_end_dates.append(
                (b_idx, dates[t_end].to_datetime64())
            )

    assert len(expected_target_end_dates) > 0
    assert len(dataset) == len(expected_target_end_dates)

    actual_target_end_dates = []
    for i in range(len(dataset)):
        sample = dataset[i]
        b_idx = int(dataset._sample_index[i]['basin'])
        actual_target_end_dates.append((b_idx, sample['date'][-1]))
        inputs = sample.get('x_d_hindcast', sample.get('x_d'))
        for name, values in inputs.items():
            assert not np.isnan(values).any(), (i, name)
        for name, values in sample.get('x_d_forecast', {}).items():
            assert not np.isnan(values).any(), (i, name)
        assert not np.isnan(sample['y'][-cfg.predict_last_n :]).all(), i

    assert actual_target_end_dates == expected_target_end_dates


def test_multimet_rejects_unexpected_minimum_lead_time(
    tmp_path: Path,
    get_config,
):
    """The date arithmetic assumes the shortest loaded lead time is 1 day."""
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('1999-12-25', '2000-01-10', freq='D')
    lead_times = [np.timedelta64(2, 'D'), np.timedelta64(3, 'D')]
    ds = _day_offset_dataset(basins, dates, lead_times)

    cfg = get_config('default')
    _write_multimet_stores(tmp_path / 'stores', cfg, ds, basins)
    cfg.update_config(
        {
            'lead_time': 3,
            'hindcast_inputs': ['era5land_2d'],
            'forecast_inputs': ['hres_3d'],
        }
    )

    with pytest.raises(ValueError, match='minimum forecast lead time'):
        Multimet(cfg=cfg, is_train=True, period='train')
