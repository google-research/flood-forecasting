import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from googlehydrology.evaluation.tester import BaseTester
from googlehydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config


def get_base_cfg(tmp_path: Path) -> dict:
    basin_file = tmp_path / 'test_basin.txt'
    with open(basin_file, 'w') as f:
        f.write('us_03338780\n')

    return {
        'model': 'HandoffForecastLSTM',
        'run_dir': str(tmp_path),
        'experiment_name': 'test_hot_start',
        'data_dir': str(tmp_path),
        'test_basin_file': str(basin_file),
        'train_basin_file': str(basin_file),
        'validation_basin_file': str(basin_file),
        'test_start_date': '01/01/2012',
        'test_end_date': '05/01/2012',
        'train_start_date': '01/01/2011',
        'train_end_date': '31/12/2011',
        'validation_start_date': '01/01/2011',
        'validation_end_date': '31/12/2011',
        'seq_length': 30,
        'lead_time': 5,
        'head': 'regression',
        'target_variables': ['streamflow'],
        'hindcast_inputs': ['pr_day_gridmet', 'tmmn_day_gridmet'],
        'forecast_inputs': ['pr_day_gridmet', 'tmmn_day_gridmet'],
        'static_attributes': ['area_gages2', 'elev_mean_x', 'p_mean_x'],
        'hidden_size': 16,
        'forecast_overlap': 2,
        'state_handoff_network': {
            'type': 'fc',
            'hiddens': [16],
            'activation': ['relu'],
            'dropout': 0.0,
        },
        'hindcast_embedding': {
            'type': 'fc',
            'hiddens': [16],
            'activation': ['relu'],
            'dropout': 0.0,
        },
        'forecast_embedding': {
            'type': 'fc',
            'hiddens': [16],
            'activation': ['relu'],
            'dropout': 0.0,
        },
        'statics_embedding': {
            'type': 'fc',
            'hiddens': [16],
            'activation': ['relu'],
            'dropout': 0.0,
        },
        'lazy_load': False,
        'batch_size': 2,
        'epochs': 1,
        'initial_learning_rate': 1e-3,
        'loss': 'nse',
        'optimizer': 'Adam',
        'number_of_basins': 1,
        'nan_handling_method': 'masked_mean',
        'predict_last_n': 0,
    }


@pytest.mark.parametrize('state_ext', ['npz', 'csv'])
@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_handoff_forecast_lstm_hot_start(
    mock_check, mock_load, tmp_path, state_ext
):
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['model'] = 'HandoffForecastLSTM'
    cfg = Config(cfg_dict, dev_mode=True)
    model = HandoffForecastLSTM(cfg)
    model.eval()

    device = next(model.parameters()).device
    batch_size = 2
    data = {}

    data['x_d_hindcast'] = {
        'pr_day_gridmet': torch.rand(
            batch_size, cfg.seq_length, 1, device=device
        ),
        'tmmn_day_gridmet': torch.rand(
            batch_size, cfg.seq_length, 1, device=device
        ),
    }
    data['x_d_forecast'] = {
        'pr_day_gridmet': torch.rand(
            batch_size, cfg.lead_time + cfg.forecast_overlap, 1, device=device
        ),
        'tmmn_day_gridmet': torch.rand(
            batch_size, cfg.lead_time + cfg.forecast_overlap, 1, device=device
        ),
    }
    data['x_s'] = torch.rand(
        batch_size, len(cfg.static_attributes), device=device
    )

    state_path = tmp_path / f'state_handoff.{state_ext}'
    with torch.no_grad():
        model.seq_length = cfg.seq_length
        model.save_state(data, state_path)
        cold_preds = model(data)

    # Hot start run with seq_length=0
    cfg.seq_length = 0
    cfg.hot_start_path = str(state_path)
    model.seq_length = 0
    model.load_state_from_disk(state_path)

    hot_data = {}
    hot_data['x_d_hindcast'] = {
        k: v[:, 0:0, :] for k, v in data['x_d_hindcast'].items()
    }
    hot_data['x_d_forecast'] = data['x_d_forecast']
    hot_data['x_s'] = data['x_s']

    with torch.no_grad():
        hot_preds = model(hot_data)

    cold_last = cold_preds['y_hat'][:, -cfg.lead_time :, :]
    hot_last = hot_preds['y_hat'][:, -cfg.lead_time :, :]
    assert (cold_last - hot_last).abs().max() < 1e-5


@pytest.mark.parametrize('state_ext', ['npz', 'csv'])
@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_mean_embedding_forecast_lstm_hot_start(
    mock_check, mock_load, tmp_path, state_ext
):
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['model'] = 'MeanEmbeddingForecastLSTM'
    cfg_dict['n_distributions'] = 1
    # For MeanEmbedding, forecast_overlap is typically sequence length.
    cfg_dict['forecast_overlap'] = cfg_dict['seq_length']
    cfg = Config(cfg_dict, dev_mode=True)
    model = MeanEmbeddingForecastLSTM(cfg)
    model.eval()

    device = next(model.parameters()).device
    batch_size = 2
    data = {}

    data['x_d_hindcast'] = {
        'pr_day_gridmet': torch.rand(
            batch_size, cfg.seq_length + cfg.lead_time, 1, device=device
        ),
        'tmmn_day_gridmet': torch.rand(
            batch_size, cfg.seq_length + cfg.lead_time, 1, device=device
        ),
    }
    data['x_d_forecast'] = {
        'pr_day_gridmet': torch.rand(
            batch_size, cfg.seq_length + cfg.lead_time, 1, device=device
        ),
        'tmmn_day_gridmet': torch.rand(
            batch_size, cfg.seq_length + cfg.lead_time, 1, device=device
        ),
    }
    data['x_s'] = torch.rand(
        batch_size, len(cfg.static_attributes), device=device
    )

    state_path = tmp_path / f'state_mean.{state_ext}'
    with torch.no_grad():
        model.seq_length = cfg.seq_length
        model.save_state(data, state_path)
        cold_preds = model(data)

    # Hot start run with seq_length=0
    cfg.seq_length = 0
    cfg.hot_start_path = str(state_path)
    model.seq_length = 0
    model.load_state_from_disk(state_path)

    hot_data = {}
    hot_data['x_d_hindcast'] = {
        k: v[:, -(cfg.lead_time) :, :] for k, v in data['x_d_hindcast'].items()
    }
    hot_data['x_d_forecast'] = {
        k: v[:, -(cfg.lead_time) :, :] for k, v in data['x_d_forecast'].items()
    }
    hot_data['x_s'] = data['x_s']

    with torch.no_grad():
        hot_preds = model(hot_data)

    cold_last = cold_preds['y_hat'][:, -cfg.lead_time :, :]
    hot_last = hot_preds['y_hat'][:, -cfg.lead_time :, :]
    assert (cold_last - hot_last).abs().max() < 1e-5


@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_hot_start_preloading(mock_check, mock_load, tmp_path):
    cfg_dict = get_base_cfg(tmp_path)
    cfg = Config(cfg_dict, dev_mode=True)
    model = HandoffForecastLSTM(cfg)
    model.eval()

    device = next(model.parameters()).device
    batch_size = 2
    data = {
        'x_d_hindcast': {
            'pr_day_gridmet': torch.rand(
                batch_size, cfg.seq_length, 1, device=device
            ),
            'tmmn_day_gridmet': torch.rand(
                batch_size, cfg.seq_length, 1, device=device
            ),
        },
        'x_d_forecast': {
            'pr_day_gridmet': torch.rand(
                batch_size,
                cfg.lead_time + cfg.forecast_overlap,
                1,
                device=device,
            ),
            'tmmn_day_gridmet': torch.rand(
                batch_size,
                cfg.lead_time + cfg.forecast_overlap,
                1,
                device=device,
            ),
        },
        'x_s': torch.rand(
            batch_size, len(cfg.static_attributes), device=device
        ),
    }

    state_path = tmp_path / 'state_preload.npz'
    model.save_state(data, state_path)

    # Test preloading into model
    cfg.seq_length = 0
    cfg.hot_start_path = str(state_path)
    model.seq_length = 0
    model.load_state_from_disk(state_path)

    hot_data = {
        'x_d_hindcast': {
            k: v[:, 0:0, :] for k, v in data['x_d_hindcast'].items()
        },
        'x_d_forecast': data['x_d_forecast'],
        'x_s': data['x_s'],
    }
    with torch.no_grad():
        preds = model(hot_data)
    assert preds['y_hat'].shape[1] == cfg.lead_time


def test_hot_start_batch_size_validation(tmp_path):
    import pytest
    from googlehydrology.evaluation.tester import RegressionTester

    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['hot_start_path'] = str(tmp_path / 'dummy_state.npz')
    cfg_dict['batch_size'] = 16
    cfg = Config(cfg_dict, dev_mode=True)

    with pytest.raises(
        ValueError, match='Hot-start inference requires batch_size=1'
    ):
        RegressionTester(cfg, run_dir=tmp_path, init_model=False)


@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_multi_basin_state_isolation(mock_check, mock_load, tmp_path):
    cfg_dict = get_base_cfg(tmp_path)
    cfg = Config(cfg_dict, dev_mode=True)
    model = HandoffForecastLSTM(cfg)

    state_path = tmp_path / 'state_basin_a.npz'
    dummy_data = {
        'x_d_hindcast': {
            'pr_day_gridmet': torch.rand(1, cfg.seq_length, 1),
            'tmmn_day_gridmet': torch.rand(1, cfg.seq_length, 1),
        },
        'x_d_forecast': {
            'pr_day_gridmet': torch.rand(
                1, cfg.lead_time + cfg.forecast_overlap, 1
            ),
            'tmmn_day_gridmet': torch.rand(
                1, cfg.lead_time + cfg.forecast_overlap, 1
            ),
        },
        'x_s': torch.rand(1, len(cfg.static_attributes)),
    }
    model.save_state(dummy_data, state_path)

    # Basin A: Loads state
    model.load_state_from_disk(state_path)
    assert model._preloaded_state is not None

    # Basin B: Resets state between basins
    model.reset_state()
    assert model._preloaded_state is None


@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_cold_start_output_shape(mock_check, mock_load, tmp_path):
    cfg_dict = get_base_cfg(tmp_path)
    cfg = Config(cfg_dict, dev_mode=True)
    model = HandoffForecastLSTM(cfg)
    model.eval()

    device = next(model.parameters()).device
    batch_size = 2
    data = {
        'x_d_hindcast': {
            'pr_day_gridmet': torch.rand(
                batch_size, cfg.seq_length, 1, device=device
            ),
            'tmmn_day_gridmet': torch.rand(
                batch_size, cfg.seq_length, 1, device=device
            ),
        },
        'x_d_forecast': {
            'pr_day_gridmet': torch.rand(
                batch_size,
                cfg.lead_time + cfg.forecast_overlap,
                1,
                device=device,
            ),
            'tmmn_day_gridmet': torch.rand(
                batch_size,
                cfg.lead_time + cfg.forecast_overlap,
                1,
                device=device,
            ),
        },
        'x_s': torch.rand(
            batch_size, len(cfg.static_attributes), device=device
        ),
    }
    with torch.no_grad():
        cold_preds = model(data)

    # In cold start, output must have length == seq_length
    assert cold_preds['y_hat'].shape[1] == cfg.seq_length


@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_forecast_overlap_variations(mock_check, mock_load, tmp_path):
    # Test with forecast_overlap = 0
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['forecast_overlap'] = 0
    cfg = Config(cfg_dict, dev_mode=True)
    model = HandoffForecastLSTM(cfg)
    model.eval()

    device = next(model.parameters()).device
    batch_size = 2
    data = {
        'x_d_hindcast': {
            'pr_day_gridmet': torch.rand(
                batch_size, cfg.seq_length, 1, device=device
            ),
            'tmmn_day_gridmet': torch.rand(
                batch_size, cfg.seq_length, 1, device=device
            ),
        },
        'x_d_forecast': {
            'pr_day_gridmet': torch.rand(
                batch_size, cfg.lead_time, 1, device=device
            ),
            'tmmn_day_gridmet': torch.rand(
                batch_size, cfg.lead_time, 1, device=device
            ),
        },
        'x_s': torch.rand(
            batch_size, len(cfg.static_attributes), device=device
        ),
    }
    state_path = tmp_path / 'state_no_overlap.npz'
    model.save_state(data, state_path)

    cfg.seq_length = 0
    cfg.hot_start_path = str(state_path)
    model.seq_length = 0
    model.load_state_from_disk(state_path)

    hot_data = {
        'x_d_hindcast': {
            k: v[:, 0:0, :] for k, v in data['x_d_hindcast'].items()
        },
        'x_d_forecast': data['x_d_forecast'],
        'x_s': data['x_s'],
    }
    with torch.no_grad():
        hot_preds = model(hot_data)
    assert hot_preds['y_hat'].shape[1] == cfg.lead_time


@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_csv_state_schema_and_mismatched_hidden_sizes(
    _mock_check: MagicMock, _mock_load: MagicMock, tmp_path: Path
) -> None:
    """Verify CSV schema and bit-exact parity with NPZ for mismatched sizes."""
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['model'] = 'HandoffForecastLSTM'
    cfg_dict['hindcast_hidden_size'] = 12
    cfg_dict['forecast_hidden_size'] = 20
    cfg_dict['state_handoff_network']['hiddens'] = [12]
    cfg = Config(cfg_dict, dev_mode=True)
    model = HandoffForecastLSTM(cfg)
    model.eval()

    batch_size = 2
    data = {
        'x_d_hindcast': {
            'pr_day_gridmet': torch.rand(batch_size, cfg.seq_length, 1),
            'tmmn_day_gridmet': torch.rand(batch_size, cfg.seq_length, 1),
        },
        'x_d_forecast': {
            'pr_day_gridmet': torch.rand(
                batch_size, cfg.lead_time + cfg.forecast_overlap, 1
            ),
            'tmmn_day_gridmet': torch.rand(
                batch_size, cfg.lead_time + cfg.forecast_overlap, 1
            ),
        },
        'x_s': torch.rand(batch_size, len(cfg.static_attributes)),
    }

    npz_path = tmp_path / 'state.npz'
    csv_path = tmp_path / 'state.csv'
    with torch.no_grad():
        model.save_state(data, npz_path)
        model.save_state(data, csv_path)

    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ['state', 'layer', 'batch', 'unit', 'value']
    expected_rows = 2 * (1 * batch_size * 12) + 2 * (1 * batch_size * 20)
    assert len(rows) - 1 == expected_rows

    model.load_state_from_disk(npz_path)
    npz_state = model._preloaded_state
    model.load_state_from_disk(csv_path)
    csv_state = model._preloaded_state

    assert set(csv_state.keys()) == set(npz_state.keys())
    for key in npz_state:
        assert csv_state[key].shape == npz_state[key].shape
        assert csv_state[key].dtype == np.float32
        np.testing.assert_array_equal(csv_state[key], npz_state[key])


def test_save_state_format_config_validation(tmp_path: Path) -> None:
    """Validate normalization and error handling for save_state_format."""
    cfg_dict = get_base_cfg(tmp_path)
    cfg = Config(cfg_dict)
    assert cfg.save_state_format == 'npz'

    cfg_csv = Config({**cfg_dict, 'save_state_format': ' CSV '})
    assert cfg_csv.save_state_format == 'csv'

    cfg.save_state_format = 'csv'
    assert cfg.save_state_format == 'csv'

    with pytest.raises(ValueError, match='Unsupported save_state_format'):
        Config({**cfg_dict, 'save_state_format': 'parquet'})

    with pytest.raises(TypeError, match='save_state_format must be a string'):
        cfg.save_state_format = 123  # type: ignore[assignment]


@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_state_file_errors_fail_loudly(
    _mock_check: MagicMock, _mock_load: MagicMock, tmp_path: Path
) -> None:
    """Ensure corrupt, missing, or unsupported state files raise loudly."""
    cfg = Config(get_base_cfg(tmp_path), dev_mode=True)
    model = HandoffForecastLSTM(cfg)

    with pytest.raises(FileNotFoundError, match='state file not found'):
        model.load_state_from_disk(tmp_path / 'missing.csv')

    bad_ext = tmp_path / 'state.json'
    bad_ext.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Unsupported hot-start state file'):
        model.load_state_from_disk(bad_ext)
    with pytest.raises(ValueError, match='Unsupported hot-start state file'):
        model._write_state_file(
            {'h_hindcast': np.zeros((1, 1, 2), dtype=np.float32)}, bad_ext
        )

    header = 'state,layer,batch,unit,value\n'
    bad_cases = [
        ('', 'empty'),
        ('bad,header,cols,here,now\n', 'Invalid header'),
        (header, 'no data rows'),
        (f'{header}h_hindcast,0,0,0\n', 'column count'),
        (f'{header},0,0,0,1.0\n', 'Empty state name'),
        (f'{header}h_hindcast,-1,0,0,1.0\n', 'non-negative'),
        (f'{header}h_hindcast,0,0,0,nan\n', 'finite float'),
        (f'{header}h_hindcast,0,0,0,inf\n', 'finite float'),
        (
            f'{header}h_hindcast,0,0,0,1.0\nh_hindcast,0,0,0,2.0\n',
            'Duplicate coordinate',
        ),
        (
            f'{header}h_hindcast,0,0,0,1.0\nh_hindcast,0,0,2,2.0\n',
            'Incomplete 3D grid',
        ),
    ]
    for idx, (content, err_match) in enumerate(bad_cases):
        bad_csv = tmp_path / f'bad_{idx}.csv'
        bad_csv.write_text(content, encoding='utf-8')
        with pytest.raises(ValueError, match=err_match):
            model.load_state_from_disk(bad_csv)

    nan_csv = tmp_path / 'nan_state.csv'
    with pytest.raises(ValueError, match='non-finite'):
        model._write_state_file(
            {'h_hindcast': np.array([[[np.nan]]], dtype=np.float32)}, nan_csv
        )


def test_resolve_basin_state_path(tmp_path: Path) -> None:
    """Resolve single-file and directory state paths across formats."""
    tester = object.__new__(BaseTester)
    cfg_dict = get_base_cfg(tmp_path)
    tester.cfg = Config(cfg_dict)

    single_csv = tmp_path / 'custom_state.csv'
    single_csv.write_text('state,layer,batch,unit,value\n', encoding='utf-8')
    assert tester._resolve_basin_state_path(single_csv, '0101') == single_csv
    assert (
        tester._resolve_basin_state_path(tmp_path / 'missing.csv', '0101')
        is None
    )

    state_dir = tmp_path / 'states'
    state_dir.mkdir()
    assert tester._resolve_basin_state_path(state_dir, '0101') is None

    bare_csv = state_dir / '0101.csv'
    bare_csv.write_text('', encoding='utf-8')
    assert tester._resolve_basin_state_path(state_dir, '0101') == bare_csv

    prefixed_csv = state_dir / 'state_0101.csv'
    prefixed_csv.write_text('', encoding='utf-8')
    assert tester._resolve_basin_state_path(state_dir, '0101') == prefixed_csv

    prefixed_npz = state_dir / 'state_0101.npz'
    prefixed_npz.write_bytes(b'')
    assert tester._resolve_basin_state_path(state_dir, '0101') == prefixed_npz

    tester.cfg.save_state_format = 'csv'
    assert tester._resolve_basin_state_path(state_dir, '0101') == prefixed_csv



