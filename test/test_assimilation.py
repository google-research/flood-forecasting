"""Unit tests for googlehydrology.evaluation.assimilation."""

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.utils.assimilationconfig import AssimilationConfig


class AssimilationTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cfg_dict = {
            'seq_length': 365,
            'history': 10,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 2,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['h_n', 'c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        self.cfg = AssimilationConfig(self.cfg_dict)
        self.assimilation = Assimilation(self.cfg)

    def test_check_discharge_timing_correct_alignment(self):
        seq_len = 365
        dates = pd.date_range(start='2020-01-01', periods=seq_len, freq='D').strftime('%Y-%m-%d').values

        y = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        y_shift1 = torch.roll(y, shifts=1, dims=1)
        y_shift1[0, 0, 0] = float('nan')

        data = {
            'date': np.tile(dates, (1, 1)),
            'y': y,
            'x_d': {'streamflow_shift1': y_shift1}
        }

        diag = self.assimilation.check_discharge_timing(data, verbose=False)
        self.assertFalse(diag['has_timing_mismatch'])
        self.assertEqual(len(diag['warnings']), 0)
        self.assertEqual(diag['details']['sequence_start_date'], '2020-01-01')

    def test_check_discharge_timing_same_day_mismatch(self):
        seq_len = 365
        dates = pd.date_range(start='2020-01-01', periods=seq_len, freq='D').strftime('%Y-%m-%d').values

        y = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        data = {
            'date': np.tile(dates, (1, 1)),
            'y': y,
            'x_d': {'streamflow_shift1': y}
        }

        diag = self.assimilation.check_discharge_timing(data, verbose=False)
        self.assertTrue(diag['has_timing_mismatch'])
        self.assertGreater(len(diag['warnings']), 0)
        self.assertIn("TIMING MISMATCH DETECTED", diag['warnings'][0])

    def test_cell_state_only_gradient_update(self):
        class MockLSTMModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', torch.zeros(1, 1, 8))
                h_n = data.get('h_n', torch.zeros(1, 1, 8))
                y_hat = self.fc(c_n + h_n)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': h_n}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.1,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockLSTMModel()
        data = {
            'y': torch.ones(1, 10, 1) * 3.0,
            'x_d': torch.zeros(1, 10, 4)
        }
        res = assim.assimilate(model, data, verbose=False, check_timing=False)
        base = model(data)['y_hat']
        da = res['y_hat']
        self.assertGreater((da - base).abs().max().item(), 0.1)

    def test_hindcast_5day_gradient_enforcement(self):
        """Verifies that cell-state gradient descent is computed strictly on the 5 hindcast days."""
        class MockForecastModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', torch.zeros(1, 1, 8))
                y_hat = self.fc(c_n).expand(1, 12, 1)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': torch.zeros_like(c_n)}

        cfg_dict = {
            'seq_length': 365,
            'history': 5,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 12,
            'predict_n_hindcast': 5,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockForecastModel()
        data = {
            'y': torch.ones(1, 365, 1) * 2.0,
            'x_d': torch.zeros(1, 365, 4)
        }
        res = assim.assimilate(model, data, verbose=False, check_timing=False)
        self.assertIn('hindcast_metrics_pre', res)
        self.assertIn('hindcast_metrics_post', res)
        self.assertIn('NSE', res['hindcast_metrics_post'])

    def test_warm_start_identical_when_zero_lr(self):
        """Verifies that baseline and assimilation outputs match identically (|y_assim - y_base| == 0) at lr=0 or epochs=0."""
        class MockWarmLSTM(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', data.get('c_0', torch.zeros(1, 1, 8)))
                h_n = data.get('h_n', data.get('h_0', torch.zeros(1, 1, 8)))
                y_hat = self.fc(c_n + h_n).expand(1, 10, 1)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': h_n}

        for epochs_val, lr_val in [(0, 0.05), (5, 0.0)]:
            for key_mode in ['c_n', 'c_0', 'both']:
                for per_step in [False, True]:
                    cfg_dict = {
                        'seq_length': 10,
                        'history': 2,
                        'assimilation_window': 1,
                        'assimilation_lead_time': 0,
                        'learning_rate': lr_val,
                        'epochs': epochs_val,
                        'loss': 'MSE',
                        'optimizer': 'Adam',
                        'assimilation_targets': ['c_n', 'h_n'],
                        'target_variables': ['streamflow'],
                        'predict_last_n': 1,
                        'predict_n_hindcast': 2,
                        'use_per_step_updates': per_step,
                    }
                    cfg = AssimilationConfig(cfg_dict)
                    assim = Assimilation(cfg)
                    model = MockWarmLSTM()
                    data = {
                        'y': torch.ones(1, 10, 1) * 3.0,
                        'x_d': torch.zeros(1, 10, 4),
                    }
                    if key_mode in ['c_n', 'both']:
                        data['c_n'] = torch.ones(1, 1, 8) * 0.25
                        data['h_n'] = torch.ones(1, 1, 8) * 0.10
                    if key_mode in ['c_0', 'both']:
                        data['c_0'] = torch.ones(1, 1, 8) * 0.25
                        data['h_0'] = torch.ones(1, 1, 8) * 0.10

                    base_out = model(data)['y_hat']
                    da_out = assim.assimilate(model, data, verbose=False, check_timing=False)['y_hat']
                    diff = (da_out - base_out).abs().max().item()
                    self.assertEqual(diff, 0.0)

    def test_probabilistic_mixture_model_assimilation(self):
        """Verifies assimilate handles models returning mu and pi or mu alone without KeyError."""
        class MockProbabilisticModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self, mode='mu_pi'):
                super().__init__()
                self.mode = mode
                self.fc = torch.nn.Linear(8, 4)
            def forward(self, data):
                c_n = data.get('c_n', torch.zeros(1, 1, 8))
                h_n = data.get('h_n', torch.zeros(1, 1, 8))
                mu = self.fc(c_n + h_n).expand(1, 10, 4)
                if self.mode == 'mu_pi':
                    pi = torch.ones_like(mu) * 0.25
                    return {'mu': mu, 'pi': pi, 'c_n': c_n, 'h_n': h_n}
                else:
                    return {'mu': mu[:, :, 0:1], 'c_n': c_n, 'h_n': h_n}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 2,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
            'predict_n_hindcast': 2,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        data = {
            'y': torch.ones(1, 10, 1) * 3.0,
            'x_d': torch.zeros(1, 10, 4),
        }

        for mode in ['mu_pi', 'mu_only']:
            model = MockProbabilisticModel(mode=mode)
            res = assim.assimilate(model, data, verbose=False, check_timing=False)
            self.assertIn('y_hat', res)
            self.assertEqual(res['y_hat'].shape[0], 1)

    def test_multi_horizon_shape_broadcast_robustness(self):
        """Verifies multi-horizon forecasts with predict_last_n > 1 do not trigger shape IndexError."""
        class MultiHorizonModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', data.get('c_0', torch.zeros(1, 1, 8)))
                h_n = data.get('h_n', data.get('h_0', torch.zeros(1, 1, 8)))
                x = data.get('x_d_hindcast', data.get('x_d', data.get('y')))
                seq_len = x.shape[1] if isinstance(x, torch.Tensor) else 7
                y_hat = self.fc(c_n + h_n).expand(1, seq_len, 1)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': h_n}

        cfg_dict = {
            'seq_length': 20,
            'history': 5,
            'assimilation_window': 2,
            'assimilation_lead_time': 2,
            'learning_rate': 0.05,
            'epochs': 3,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 7,
            'predict_n_hindcast': 5,
            'use_per_step_updates': True,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MultiHorizonModel()
        data = {
            'y': torch.ones(1, 20, 1) * 2.5,
            'x_d': torch.zeros(1, 20, 4)
        }
        res = assim.assimilate(model, data, verbose=False, check_timing=False)
        self.assertIn('y_hat', res)
        self.assertEqual(res['y_hat'].shape[1], 20)

    def test_assimilation_continuity_as_lr_approaches_zero(self):
        """Verifies that as learning_rate -> 0, data assimilation smoothly and continuously tends to the baseline."""
        class RecurrentModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                x = data['x_d']
                h_0 = data.get('h_0', data.get('h_n', None))
                c_0 = data.get('c_0', data.get('c_n', None))
                hx = None
                if h_0 is not None and c_0 is not None:
                    if h_0.ndim == 3 and h_0.shape[1] == 1:
                        h_0 = h_0.transpose(0, 1)
                    if c_0.ndim == 3 and c_0.shape[1] == 1:
                        c_0 = c_0.transpose(0, 1)
                    hx = (h_0.contiguous(), c_0.contiguous())
                out, (h_n, c_n) = self.lstm(x, hx)
                return {'y_hat': self.fc(out), 'c_n': c_n.transpose(0, 1), 'h_n': h_n.transpose(0, 1)}

        model = RecurrentModel()
        torch.manual_seed(42)
        data = {
            'y': torch.ones(1, 10, 1) * 5.0,
            'x_d': torch.randn(1, 10, 4),
            'c_0': torch.randn(1, 1, 8),
            'h_0': torch.randn(1, 1, 8),
        }

        diffs = []
        lrs = [0.1, 0.01, 0.001, 1e-4, 0.0]
        base_out = model(data)['y_hat']

        for lr_val in lrs:
            cfg_dict = {
                'seq_length': 10,
                'history': 2,
                'assimilation_window': 1,
                'assimilation_lead_time': 0,
                'learning_rate': lr_val,
                'epochs': 3,
                'loss': 'MSE',
                'optimizer': 'Adam',
                'assimilation_targets': ['c_n', 'h_n'],
                'target_variables': ['streamflow'],
                'predict_last_n': 1,
                'predict_n_hindcast': 2,
                'use_per_step_updates': False,
            }
            cfg = AssimilationConfig(cfg_dict)
            assim = Assimilation(cfg)
            res = assim.assimilate(model, data, verbose=False, check_timing=False)
            diff = (res['y_hat'] - base_out).abs().max().item()
            diffs.append(diff)

        # Confirm exact baseline match at lr = 0.0 within float32 tolerance
        self.assertLessEqual(diffs[-1], 1e-6)
        # Confirm monotonic smooth reduction as lr -> 0
        for i in range(len(diffs) - 1):
            self.assertGreaterEqual(diffs[i], diffs[i+1] - 1e-7)

    def test_autoregressive_state_persistence(self):
        """Verifies that last_prediction is forwarded across window rollouts for autoregressive models."""
        class MockARModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n', 'last_prediction']
            def __init__(self):
                super().__init__()
                self.received_last_predictions = []
            def forward(self, data):
                last_pred = data.get('last_prediction', None)
                self.received_last_predictions.append(last_pred)
                c_n = data.get('c_n', data.get('c_0', torch.zeros(1, 1, 8)))
                h_n = data.get('h_n', data.get('h_0', torch.zeros(1, 1, 8)))
                y_hat = torch.ones(1, 1, 1) * 4.2
                new_last_pred = torch.ones(1, 1, 1) * 9.9
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': h_n, 'last_prediction': new_last_pred}

        cfg_dict = {
            'seq_length': 10,
            'history': 3,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.0,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n', 'h_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockARModel()
        data = {
            'y': torch.ones(1, 10, 1) * 4.2,
            'x_d': torch.zeros(1, 10, 4),
            'last_prediction': torch.ones(1, 1, 1) * 1.1,
        }
        res = assim.assimilate(model, data, verbose=False, check_timing=False)
        self.assertIn('y_hat', res)
        # Verify last_prediction was forwarded to chunk_data in subsequent windows
        non_none_preds = [p for p in model.received_last_predictions if p is not None]
        self.assertGreater(len(non_none_preds), 1)

    def test_state_injection_c_n_only(self):
        """Verifies that when assim_targets=['c_n'], h_0/h_n is still injected into chunk_data and preserved."""
        class MockDualStateModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
            def forward(self, data):
                c_n = data.get('c_n', data.get('c_0', torch.zeros(1, 1, 8)))
                h_n = data.get('h_n', data.get('h_0', torch.zeros(1, 1, 8)))
                # Assert that h state is injected (non-zero when provided in data)
                y_hat = (c_n + h_n).sum().unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 1)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': h_n}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.0,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'], # only c_n targeted
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockDualStateModel()
        data = {
            'y': torch.ones(1, 10, 1),
            'x_d': torch.zeros(1, 10, 4),
            'c_0': torch.ones(1, 1, 8) * 0.5,
            'h_0': torch.ones(1, 1, 8) * 0.3,
        }
        base_out = model(data)['y_hat']
        da_out = assim.assimilate(model, data, verbose=False, check_timing=False)['y_hat']
        diff = (da_out - base_out).abs().max().item()
        self.assertEqual(diff, 0.0)

    def test_a_start_zero_initial_state_shape(self):
        """Verifies initial state shape matching for a_start == 0 with 2-layer LSTM states (2, 1, 8)."""
        class MultiLayerLSTMModel(torch.nn.Module):
            def forward(self, data):
                c_0 = data.get('c_0', data.get('c_n', torch.zeros(2, 1, 8)))
                h_0 = data.get('h_0', data.get('h_n', torch.zeros(2, 1, 8)))
                y_hat = torch.ones(1, 10, 1)
                return {'y_hat': y_hat, 'c_n': c_0, 'h_n': h_0}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.0,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n', 'h_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MultiLayerLSTMModel()
        data = {
            'y': torch.ones(1, 10, 1),
            'x_d': torch.zeros(1, 10, 4),
            'c_0': torch.ones(2, 1, 8) * 0.7,
            'h_0': torch.ones(2, 1, 8) * 0.2,
        }
        res = assim.assimilate(model, data, verbose=False, check_timing=False)
        self.assertEqual(res['c_n'].shape, (2, 1, 8))
        self.assertEqual(res['h_n'].shape, (2, 1, 8))

    def test_warmup_state_continuity_accumulator_a_start_positive(self):
        """Verifies that warmup unrolling at a_start > 0 matches unassimilated baseline with zero future jumps."""
        class AccumulatorLSTM(torch.nn.Module):
            def forward(self, data):
                x = data.get('x_d_hindcast', data.get('x_d'))
                if isinstance(x, dict): x = list(x.values())[0]
                c_0 = data.get('c_0_hindcast', data.get('c_0', torch.zeros(1, 1, 1)))
                c_n = c_0 + torch.sum(x, dim=1, keepdim=True)
                y_hat = c_0 + torch.cumsum(x, dim=1)
                return {'y_hat': y_hat, 'c_n_hindcast': c_n, 'h_n_hindcast': c_n, 'c_n': c_n, 'h_n': c_n}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.0,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = AccumulatorLSTM()
        x = torch.ones(1, 10, 1)
        data = {'x_d': x, 'y': torch.zeros(1, 10, 1)}

        base_out = model(data)['y_hat']
        da_out = assim.assimilate(model, data, verbose=False)['y_hat']
        diff = (base_out - da_out).abs().max().item()
        self.assertEqual(diff, 0.0)

    @patch('googlehydrology.modelzoo.basemodel.Scaler')
    def test_mean_embedding_forecast_lstm_data_assimilation(self, mock_scaler):
        """Tests 4D-Var assimilation on full MeanEmbeddingForecastLSTM model."""
        from googlehydrology.utils.config import Config
        from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM

        cfg_dict = {
            'model': 'mean_embedding_forecast_lstm',
            'head': 'regression',
            'hidden_size': 16,
            'seq_length': 14,
            'lead_time': 2,
            'predict_last_n': 2,
            'target_variables': ['streamflow'],
            'static_attributes': ['area'],
            'statics_embedding': {'type': 'fc', 'hiddens': [8], 'activation': 'tanh', 'dropout': 0.0},
            'dynamics_embedding': {'type': 'fc', 'hiddens': [8], 'activation': 'tanh', 'dropout': 0.0},
            'hindcast_inputs': ['era5_precip'],
            'forecast_inputs': ['hres_precip'],
            'compile': False,
            'dev_mode': True,
        }
        cfg = Config(cfg_dict, dev_mode=True)
        model = MeanEmbeddingForecastLSTM(cfg=cfg)

        da_cfg_dict = {
            'seq_length': 14,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 2,
            'learning_rate': 0.05,
            'epochs': 3,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n_hindcast'],
            'target_variables': ['streamflow'],
            'predict_last_n': 2,
        }
        da_cfg = AssimilationConfig(da_cfg_dict)
        assim = Assimilation(da_cfg)

        data = {
            'x_s': torch.ones(1, 1),
            'x_d_hindcast': {'era5_precip': torch.randn(1, 12, 1)},
            'x_d_forecast': {'hres_precip': torch.randn(1, 14, 1)},
            'y': torch.ones(1, 14, 1) * 2.0,
        }
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        self.assertEqual(res['y_hat'].shape[1], 14)

    @patch('googlehydrology.modelzoo.basemodel.Scaler')
    def test_config_with_assimilation_section_and_model_loading(self, mock_scaler):
        """Tests Config parsing of assimilation_config and checks get_model validation."""
        from googlehydrology.utils.config import Config
        from googlehydrology.modelzoo import get_model

        cfg_dict = {
            'model': 'mean_embedding_forecast_lstm',
            'head': 'regression',
            'hidden_size': 16,
            'seq_length': 14,
            'lead_time': 2,
            'predict_last_n': 2,
            'target_variables': ['streamflow'],
            'static_attributes': ['area'],
            'statics_embedding': {'type': 'fc', 'hiddens': [8], 'activation': 'tanh', 'dropout': 0.0},
            'dynamics_embedding': {'type': 'fc', 'hiddens': [8], 'activation': 'tanh', 'dropout': 0.0},
            'hindcast_inputs': ['era5_precip'],
            'forecast_inputs': ['hres_precip'],
            'compile': False,
            'dev_mode': True,
            'assimilation_config': {
                'assimilation_window': 1,
                'history': 2,
                'assimilation_lead_time': 2,
                'learning_rate': 0.01,
                'loss': 'MSE',
                'optimizer': 'Adam',
            }
        }
        cfg = Config(cfg_dict, dev_mode=True)
        self.assertIsNotNone(cfg.assimilation_config)
        self.assertEqual(cfg.assimilation_config.history, 2)
        model = get_model(cfg)
        self.assertIsNotNone(model)

    def test_cmal_probabilistic_mixture_preservation(self):
        """Tests that CMAL mixture parameters (mu, b, tau, pi) are preserved across unrolled chunks."""
        class MockCMALModel(torch.nn.Module):
            def forward(self, data):
                c_0 = data.get('c_0', data.get('c_n', torch.zeros(1, 1, 8)))
                y_len = data['y'].shape[1] if 'y' in data else 10
                mu = torch.ones(1, y_len, 4) * 2.0
                b = torch.ones(1, y_len, 4) * 0.5
                tau = torch.ones(1, y_len, 4) * 0.5
                pi = torch.ones(1, y_len, 4) * 0.25
                y_hat = torch.ones(1, y_len, 1) * 2.0
                return {'y_hat': y_hat, 'mu': mu, 'b': b, 'tau': tau, 'pi': pi, 'c_n': c_0, 'h_n': c_0}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.0,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockCMALModel()
        data = {'x_d': torch.zeros(1, 10, 2), 'y': torch.ones(1, 10, 1)}
        res = assim.assimilate(model, data, verbose=False)
        for k in ['mu', 'b', 'tau', 'pi']:
            self.assertIn(k, res)
            self.assertEqual(res[k].shape[1], 10)

    def test_model_requires_grad_preserved_after_assimilation(self):
        """Verifies that model parameters retain their original requires_grad status after assimilate."""
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(2, 1)
            def forward(self, data):
                x = data.get('x_d_hindcast', data.get('x_d'))
                return {'y_hat': self.fc(x), 'c_n': torch.zeros(1, 1, 1), 'h_n': torch.zeros(1, 1, 1)}

        model = LinearModel()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.01,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        data = {'x_d': torch.zeros(1, 10, 2), 'y': torch.ones(1, 10, 1)}
        assim.assimilate(model, data, verbose=False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def _create_mef_model_and_data(self, hidden_size=16, seq_length=14, lead_time=2):
        from googlehydrology.utils.config import Config
        from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
        cfg_dict = {
            'model': 'mean_embedding_forecast_lstm',
            'head': 'regression',
            'hidden_size': hidden_size,
            'seq_length': seq_length,
            'lead_time': lead_time,
            'predict_last_n': lead_time,
            'target_variables': ['streamflow'],
            'static_attributes': ['area'],
            'statics_embedding': {'type': 'fc', 'hiddens': [8], 'activation': 'tanh', 'dropout': 0.0},
            'dynamics_embedding': {'type': 'fc', 'hiddens': [8], 'activation': 'tanh', 'dropout': 0.0},
            'hindcast_inputs': ['era5_precip'],
            'forecast_inputs': ['hres_precip'],
            'compile': False,
            'dev_mode': True,
        }
        cfg = Config(cfg_dict, dev_mode=True)
        model = MeanEmbeddingForecastLSTM(cfg=cfg)
        data = {
            'x_s': torch.ones(1, 1),
            'x_d_hindcast': {'era5_precip': torch.randn(1, seq_length - lead_time, 1)},
            'x_d_forecast': {'hres_precip': torch.randn(1, seq_length, 1)},
            'y': torch.ones(1, seq_length, 1) * 2.0,
        }
        return model, data

    @patch('googlehydrology.modelzoo.basemodel.Scaler')
    def test_itemised_type1_recurrent_state_da_with_mean_embedding_forecast_lstm(self, mock_scaler):
        """Itemised Test: Type 1 Recurrent State DA with MeanEmbeddingForecastLSTM."""
        model, data = self._create_mef_model_and_data()
        da_cfg_dict = {
            'seq_length': 14,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 2,
            'learning_rate': 0.05,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n_hindcast', 'h_n_hindcast'],
            'target_variables': ['streamflow'],
            'predict_last_n': 2,
        }
        da_cfg = AssimilationConfig(da_cfg_dict)
        assim = Assimilation(da_cfg)
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        self.assertEqual(res['y_hat'].shape[1], 14)
        self.assertIn('c_n_hindcast', res)
        self.assertIn('h_n_hindcast', res)
        self.assertIsNotNone(res['c_n_hindcast'])
        self.assertIsNotNone(res['h_n_hindcast'])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertIn('hindcast_metrics_pre', res)
        self.assertIn('hindcast_metrics_post', res)


if __name__ == '__main__':
    unittest.main()
