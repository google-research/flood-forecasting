"""Unit tests for googlehydrology.evaluation.assimilation (Embedding Data Assimilation)."""

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.utils.assimilationconfig import AssimilationConfig


class MockEmbeddingModel(torch.nn.Module):
    """Mock model with static and dynamic embeddings for unit testing."""
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        e_stat = data.get('static_embedding', None)
        if e_stat is None:
            e_stat = torch.ones(1, self.hidden_dim)
        e_dyn = data.get('hindcast_embedding', None)
        if e_dyn is None:
            x_d = data.get('x_d', data.get('x_d_hindcast', None))
            if isinstance(x_d, dict):
                x_d = list(x_d.values())[0]
            if x_d is not None:
                e_dyn = x_d
            else:
                e_dyn = torch.zeros(1, 10, self.hidden_dim)
        y_hat = (e_stat.unsqueeze(1) + e_dyn).sum(dim=-1, keepdim=True)
        return {
            'y_hat': y_hat,
            'static_embedding': e_stat,
            'hindcast_embedding': e_dyn,
        }


class AssimilationTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cfg_dict = {
            'seq_length': 365,
            'history': 10,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 2,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['embedded_both'],
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
    def test_itemised_type2_embedded_all_da_with_mean_embedding_forecast_lstm(self, mock_scaler):
        """Itemised Test: Type 2 Embedded DA (embedded_all: static, hindcast, forecast) with MeanEmbeddingForecastLSTM."""
        model, data = self._create_mef_model_and_data()
        da_cfg_dict = {
            'seq_length': 14,
            'history': 2,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 2,
            'learning_rate': 0.05,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['embedded_all'],
            'static_embedding_regularization_weight': 1e-4,
            'hindcast_embedding_regularization_weight': 0.01,
            'target_variables': ['streamflow'],
            'predict_last_n': 2,
        }
        da_cfg = AssimilationConfig(da_cfg_dict)
        assim = Assimilation(da_cfg)
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        self.assertEqual(res['y_hat'].shape[1], 14)
        self.assertIn('static_embedding', res)
        self.assertIn('hindcast_embedding', res)
        self.assertIn('forecast_embedding', res)
        self.assertIsNotNone(res['static_embedding'])
        self.assertIsNotNone(res['hindcast_embedding'])
        self.assertIsNotNone(res['forecast_embedding'])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertIn('hindcast_metrics_pre', res)
        self.assertIn('hindcast_metrics_post', res)

    @patch('googlehydrology.modelzoo.basemodel.Scaler')
    def test_embedded_both_da_with_mean_embedding_forecast_lstm(self, mock_scaler):
        """Itemised Test: Embedded DA (embedded_both: static and hindcast) with MeanEmbeddingForecastLSTM."""
        model, data = self._create_mef_model_and_data()
        da_cfg_dict = {
            'seq_length': 14,
            'history': 2,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 2,
            'learning_rate': 0.05,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['embedded_both'],
            'static_embedding_regularization_weight': 1e-4,
            'hindcast_embedding_regularization_weight': 0.01,
            'target_variables': ['streamflow'],
            'predict_last_n': 2,
        }
        da_cfg = AssimilationConfig(da_cfg_dict)
        assim = Assimilation(da_cfg)
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        self.assertEqual(res['y_hat'].shape[1], 14)
        self.assertIn('static_embedding', res)
        self.assertIn('hindcast_embedding', res)
        self.assertNotIn('forecast_embedding', res)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    @patch('googlehydrology.modelzoo.basemodel.Scaler')
    def test_component_based_da_with_explicit_regularization_weights(self, mock_scaler):
        """Verifies model-agnostic component-based DA using assimilation_components dictionary."""
        model, data = self._create_mef_model_and_data()
        model.eval()

        da_cfg_dict = {
            'seq_length': 14,
            'history': 2,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 2,
            'learning_rate': 0.05,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_components': {
                'static_embedding': 1e-4,
                'hindcast_embedding': 0.02,
            },
            'target_variables': ['streamflow'],
            'predict_last_n': 2,
        }
        da_cfg = AssimilationConfig(da_cfg_dict)
        self.assertIn('static_embedding', da_cfg.assimilation_components)
        self.assertEqual(da_cfg.assimilation_components['static_embedding']['weight'], 1e-4)
        self.assertIn('hindcast_embedding', da_cfg.assimilation_components)
        self.assertEqual(da_cfg.assimilation_components['hindcast_embedding']['weight'], 0.02)

        assim = Assimilation(da_cfg)
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        self.assertIn('static_embedding', res)
        self.assertIn('hindcast_embedding', res)

    @patch('googlehydrology.modelzoo.basemodel.Scaler')
    def test_component_based_da_with_per_component_lr(self, mock_scaler):
        """Verifies component-based DA with per-component learning rates and weights."""
        model, data = self._create_mef_model_and_data()
        model.eval()

        da_cfg_dict = {
            'seq_length': 14,
            'history': 2,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 2,
            'learning_rate': 0.01,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_components': {
                'static_embedding': {'weight': 1e-4, 'learning_rate': 0.02},
                'hindcast_embedding': {'weight': 0.05, 'lr': 0.08},
                'forecast_embedding': {'weight': 0.05, 'lr': 0.08},
            },
            'target_variables': ['streamflow'],
            'predict_last_n': 2,
        }
        da_cfg = AssimilationConfig(da_cfg_dict)
        self.assertEqual(da_cfg.assimilation_components['static_embedding']['lr'], 0.02)
        self.assertEqual(da_cfg.assimilation_components['hindcast_embedding']['lr'], 0.08)

        assim = Assimilation(da_cfg)
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        self.assertIn('static_embedding', res)
        self.assertIn('hindcast_embedding', res)
        self.assertIn('forecast_embedding', res)

    def test_regularization_uses_mean_normalization(self):
        """Verifies that the regularizer uses torch.mean (intensive) rather than torch.sum (extensive)."""
        model = MockEmbeddingModel(hidden_dim=8)
        data = {
            'x_d': torch.zeros(1, 10, 8),
            'y': torch.ones(1, 10, 1) * 5.0,
        }

        da_cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.01,
            'epochs': 3,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_components': {
                'hindcast_embedding': {'weight': 1.0},
            },
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        da_cfg = AssimilationConfig(da_cfg_dict)
        assim = Assimilation(da_cfg)
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        self.assertFalse(torch.isnan(res['y_hat']).any())

    def test_warm_start_identical_when_zero_lr_or_epochs(self):
        """Verifies that baseline and assimilation outputs match identically (|y_assim - y_base| == 0) at lr=0 or epochs=0."""
        model = MockEmbeddingModel(hidden_dim=4)
        data = {
            'x_d': torch.randn(1, 10, 4),
            'y': torch.ones(1, 10, 1) * 3.0,
        }
        base_out = model(data)['y_hat']

        for epochs_val, lr_val in [(0, 0.05), (5, 0.0)]:
            da_cfg_dict = {
                'seq_length': 10,
                'history': 2,
                'assimilation_window_length': 1,
                'assimilation_lead_time': 0,
                'learning_rate': lr_val,
                'epochs': epochs_val,
                'loss': 'MSE',
                'optimizer': 'Adam',
                'assimilation_targets': ['embedded_both'],
                'target_variables': ['streamflow'],
                'predict_last_n': 1,
            }
            da_cfg = AssimilationConfig(da_cfg_dict)
            assim = Assimilation(da_cfg)
            res = assim.assimilate(model, data, verbose=False)
            diff = (res['y_hat'] - base_out).abs().max().item()
            self.assertEqual(diff, 0.0)

    def test_embedding_da_continuity_as_lr_approaches_zero(self):
        """Verifies that as learning_rate -> 0, embedding DA smoothly and continuously tends to the baseline."""
        model = MockEmbeddingModel(hidden_dim=4)
        data = {
            'x_d': torch.randn(1, 10, 4),
            'y': torch.ones(1, 10, 1) * 10.0,
        }
        base_out = model(data)['y_hat']

        diffs = []
        lrs = [0.1, 0.01, 0.001, 1e-4, 0.0]

        for lr_val in lrs:
            da_cfg_dict = {
                'seq_length': 10,
                'history': 2,
                'assimilation_window_length': 1,
                'assimilation_lead_time': 0,
                'learning_rate': lr_val,
                'epochs': 3,
                'loss': 'MSE',
                'optimizer': 'Adam',
                'assimilation_targets': ['embedded_both'],
                'target_variables': ['streamflow'],
                'predict_last_n': 1,
            }
            da_cfg = AssimilationConfig(da_cfg_dict)
            assim = Assimilation(da_cfg)
            res = assim.assimilate(model, data, verbose=False)
            diff = (res['y_hat'] - base_out).abs().max().item()
            diffs.append(diff)

        # Baseline match at lr = 0.0 within float32 tolerance
        self.assertLessEqual(diffs[-1], 1e-6)
        # Smooth reduction as lr -> 0
        for i in range(len(diffs) - 1):
            self.assertGreaterEqual(diffs[i], diffs[i+1] - 1e-7)

    def test_cmal_probabilistic_mixture_preservation(self):
        """Tests that CMAL mixture parameters (mu, b, tau, pi) are preserved across unrolled chunks."""
        class MockCMALEmbeddingModel(torch.nn.Module):
            def forward(self, data):
                y_len = data['y'].shape[1] if 'y' in data else 10
                mu = torch.ones(1, y_len, 4) * 2.0
                b = torch.ones(1, y_len, 4) * 0.5
                tau = torch.ones(1, y_len, 4) * 0.5
                pi = torch.ones(1, y_len, 4) * 0.25
                y_hat = torch.ones(1, y_len, 1) * 2.0
                return {
                    'y_hat': y_hat, 'mu': mu, 'b': b, 'tau': tau, 'pi': pi,
                    'static_embedding': torch.ones(1, 8),
                    'hindcast_embedding': torch.ones(1, y_len, 8),
                }

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.0,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['embedded_both'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockCMALEmbeddingModel()
        data = {'x_d': torch.zeros(1, 10, 2), 'y': torch.ones(1, 10, 1)}
        res = assim.assimilate(model, data, verbose=False)
        for k in ['mu', 'b', 'tau', 'pi']:
            self.assertIn(k, res)
            self.assertEqual(res[k].shape[1], 10)

    def test_model_requires_grad_preserved_after_assimilation(self):
        """Verifies that model parameters retain their original requires_grad status after assimilate."""
        class LinearEmbeddingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(2, 1)
            def forward(self, data):
                x = data.get('x_d_hindcast', data.get('x_d'))
                return {
                    'y_hat': self.fc(x),
                    'static_embedding': torch.ones(1, 4),
                    'hindcast_embedding': torch.ones(1, x.shape[1], 4),
                }

        model = LinearEmbeddingModel()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.01,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['embedded_both'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        data = {'x_d': torch.zeros(1, 10, 2), 'y': torch.ones(1, 10, 1)}
        assim.assimilate(model, data, verbose=False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_autoregressive_state_persistence(self):
        """Verifies that last_prediction is forwarded across window rollouts for autoregressive models."""
        class MockARModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received_last_predictions = []
            def forward(self, data):
                last_pred = data.get('last_prediction', None)
                self.received_last_predictions.append(last_pred)
                y_hat = torch.ones(1, 1, 1) * 4.2
                new_last_pred = torch.ones(1, 1, 1) * 9.9
                return {
                    'y_hat': y_hat,
                    'last_prediction': new_last_pred,
                    'static_embedding': torch.ones(1, 4),
                    'hindcast_embedding': torch.ones(1, 1, 4),
                }

        cfg_dict = {
            'seq_length': 10,
            'history': 3,
            'assimilation_window_length': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.0,
            'epochs': 1,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['embedded_both'],
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
        res = assim.assimilate(model, data, verbose=False)
        self.assertIn('y_hat', res)
        non_none_preds = [p for p in model.received_last_predictions if p is not None]
        self.assertGreater(len(non_none_preds), 1)

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
                'assimilation_window_length': 1,
                'history': 2,
                'assimilation_lead_time': 2,
                'learning_rate': 0.01,
                'loss': 'MSE',
                'optimizer': 'Adam',
                'assimilation_targets': ['embedded_both'],
            }
        }
        cfg = Config(cfg_dict, dev_mode=True)
        self.assertIsNotNone(cfg.assimilation_config)
        self.assertEqual(cfg.assimilation_config.history, 2)
        model = get_model(cfg)
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
