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

"""Unit and integration tests for training-time input feature-group dropout."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.training import get_loss_obj
from googlehydrology.training.basetrainer import BaseTrainer
from googlehydrology.training.input_dropout import InputDropout
from googlehydrology.training.logger import Logger
from googlehydrology.utils.config import Config

pytestmark = pytest.mark.unit


def _make_forecast_batch(
    batch_size: int = 32,
    seq_length: int = 20,
    forecast_overlap: int = 20,
    lead_time: int = 5,
    *,
    include_forecast_only: bool = False,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, object]]:
    hindcast_groups = {
        'hres': ['hres_precip', 'hres_temp'],
        'graphcast': ['graphcast_precip', 'graphcast_temp'],
        'imerg': ['imerg_precip'],
        'cpc': ['cpc_precip'],
    }
    forecast_groups = {
        'hres': ['hres_precip', 'hres_temp'],
        'graphcast': ['graphcast_precip', 'graphcast_temp'],
    }
    if include_forecast_only:
        forecast_groups['gfs'] = ['gfs_precip']

    fcst_len = forecast_overlap + lead_time
    x_h = {
        feat: torch.ones((batch_size, seq_length, 1), dtype=torch.float32)
        for feats in hindcast_groups.values()
        for feat in feats
    }
    x_h['hindcast_counter'] = torch.zeros(
        (batch_size, seq_length, 1), dtype=torch.float32
    )
    x_f = {
        feat: torch.ones((batch_size, fcst_len, 1), dtype=torch.float32)
        for feats in forecast_groups.values()
        for feat in feats
    }
    x_f['forecast_counter'] = torch.arange(
        fcst_len, dtype=torch.float32
    ).view(1, fcst_len, 1).expand(batch_size, fcst_len, 1).clone()

    batch = {
        'x_s': torch.ones((batch_size, 2), dtype=torch.float32),
        'x_d_hindcast': x_h,
        'x_d_forecast': x_f,
        'y': torch.ones((batch_size, seq_length + lead_time, 1)),
    }
    return hindcast_groups, forecast_groups, batch


def _has_single_contiguous_nan_run(mask_1d: torch.Tensor) -> bool:
    """Return True if all True entries in a 1D boolean mask form <= 1 run."""
    indices = torch.nonzero(mask_1d, as_tuple=False).flatten()
    if indices.numel() <= 1:
        return True
    return bool((indices[-1] - indices[0] + 1) == indices.numel())


class TestConfigDropoutValidation:
    """Validate nan_sequence_probability and nan_streak_probability parsing."""

    def test_defaults_to_zero(self):
        cfg = Config({'experiment_name': 'test'})
        assert cfg.nan_sequence_probability == 0.0
        assert cfg.nan_streak_probability == 0.0

    def test_none_values_default_to_zero(self):
        cfg = Config(
            {
                'experiment_name': 'test',
                'nan_sequence_probability': None,
                'nan_streak_probability': None,
            }
        )
        assert cfg.nan_sequence_probability == 0.0
        assert cfg.nan_streak_probability == 0.0

    def test_accepts_scalar_and_per_group_dicts(self):
        cfg = Config(
            {
                'experiment_name': 'test',
                'nan_sequence_probability': 0.1,
                'nan_streak_probability': {'graphcast': 0.2, 'imerg': 0},
            }
        )
        assert cfg.nan_sequence_probability == pytest.approx(0.1)
        assert cfg.nan_streak_probability == {'graphcast': 0.2, 'imerg': 0.0}

    @pytest.mark.parametrize(
        'key', ['nan_sequence_probability', 'nan_streak_probability']
    )
    @pytest.mark.parametrize(
        'bad_value',
        [
            -0.01,
            1.01,
            float('nan'),
            float('inf'),
            True,
            False,
            '0.1',
            [0.1],
            {'': 0.1},
            {'hres': -0.1},
            {'hres': 1.5},
            {'hres': True},
            {123: 0.1},
        ],
    )
    def test_rejects_invalid_values(self, key, bad_value):
        cfg = Config({'experiment_name': 'test', key: bad_value})
        with pytest.raises(ValueError):
            _ = getattr(cfg, key)

    def test_unknown_group_in_dict_raises(self):
        cfg = Config(
            {
                'experiment_name': 'test',
                'hindcast_inputs': {'hres': ['hres_p'], 'cpc': ['cpc_p']},
                'forecast_inputs': {'hres': ['hres_p']},
                'nan_sequence_probability': {'nonexistent': 0.1},
            }
        )
        with pytest.raises(ValueError, match='Unknown feature group'):
            InputDropout.from_config(cfg)

    def test_floodhub_example_config_enables_prod_dropout(self):
        config_path = (
            Path(__file__).resolve().parent.parent
            / 'example-configs'
            / 'floodhub-settings-config.yml'
        )
        cfg = Config(config_path)
        assert cfg.nan_sequence_probability == pytest.approx(0.1)
        assert cfg.nan_streak_probability == pytest.approx(0.1)


class TestInputDropoutBehavior:
    """Validate statistical, structural, and safety invariants of InputDropout."""

    def test_disabled_is_noop(self):
        hind_groups, fcst_groups, batch = _make_forecast_batch()
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.0,
            streak_probabilities=0.0,
            forecast_overlap=20,
        )
        assert not dropout.enabled
        out = dropout(batch)
        assert out is batch

    def test_does_not_mutate_input_tensors_or_counters(self):
        torch.manual_seed(0)
        hind_groups, fcst_groups, batch = _make_forecast_batch()
        orig_h = {k: v.clone() for k, v in batch['x_d_hindcast'].items()}
        orig_f = {k: v.clone() for k, v in batch['x_d_forecast'].items()}

        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.5,
            streak_probabilities=0.5,
            forecast_overlap=20,
        )
        out = dropout(batch)

        for k, v in orig_h.items():
            torch.testing.assert_close(batch['x_d_hindcast'][k], v)
        for k, v in orig_f.items():
            torch.testing.assert_close(batch['x_d_forecast'][k], v)
        torch.testing.assert_close(
            out['x_d_hindcast']['hindcast_counter'],
            orig_h['hindcast_counter'],
        )
        torch.testing.assert_close(
            out['x_d_forecast']['forecast_counter'],
            orig_f['forecast_counter'],
        )

    def test_sequence_dropout_drops_whole_windows_with_expected_rates(self):
        torch.manual_seed(42)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=2000, seq_length=12, forecast_overlap=12, lead_time=4
        )
        # Pin hres to 0.0 (matching prod v2 NON_MASKABLE_FEATURE_PROVIDERS)
        # so the remaining groups drop independently at their exact rates.
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities={
                'graphcast': 0.2,
                'imerg': 0.3,
                'cpc': 0.4,
            },
            streak_probabilities=0.0,
            forecast_overlap=12,
        )
        out = dropout(batch)

        # Pinned hres is never dropped.
        assert not torch.isnan(out['x_d_hindcast']['hres_precip']).any()
        assert not torch.isnan(out['x_d_forecast']['hres_precip']).any()

        # Multi-feature group features share identical NaN masks.
        assert torch.equal(
            torch.isnan(out['x_d_hindcast']['graphcast_precip']),
            torch.isnan(out['x_d_hindcast']['graphcast_temp']),
        )
        assert torch.equal(
            torch.isnan(out['x_d_forecast']['graphcast_precip']),
            torch.isnan(out['x_d_forecast']['graphcast_temp']),
        )

        # Each dropped sample is either 0% or 100% NaN across the window.
        for feat, expected in [
            ('graphcast_precip', 0.2),
            ('imerg_precip', 0.3),
            ('cpc_precip', 0.4),
        ]:
            nan_mask = torch.isnan(out['x_d_hindcast'][feat]).squeeze(-1)
            row_all_nan = nan_mask.all(dim=1)
            row_any_nan = nan_mask.any(dim=1)
            assert torch.equal(row_all_nan, row_any_nan)
            assert row_all_nan.float().mean().item() == pytest.approx(
                expected, abs=0.03
            )

        # Forecast horizon [12:16] is dropped independently at rate 0.2.
        fcst_nan = torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, 12:]
        ).squeeze(-1)
        assert torch.equal(fcst_nan.all(dim=1), fcst_nan.any(dim=1))
        fcst_rate = fcst_nan.all(dim=1).float().mean().item()
        assert fcst_rate == pytest.approx(0.2, abs=0.03)

        # Hindcast and forecast horizon drops are statistically independent.
        hind_dropped = (
            torch.isnan(out['x_d_hindcast']['graphcast_precip'])
            .squeeze(-1)
            .all(dim=1)
        )
        both_rate = (hind_dropped & fcst_nan.all(dim=1)).float().mean().item()
        assert both_rate == pytest.approx(0.2 * 0.2, abs=0.02)

    def test_streak_dropout_creates_contiguous_subperiods(self):
        torch.manual_seed(7)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=500, seq_length=30, forecast_overlap=30, lead_time=7
        )
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.0,
            streak_probabilities={'graphcast': 0.3, 'imerg': 0.3},
            forecast_overlap=30,
        )
        out = dropout(batch)

        for feat in ['graphcast_precip', 'imerg_precip']:
            nan_mask = torch.isnan(out['x_d_hindcast'][feat]).squeeze(-1)
            has_drop = nan_mask.any(dim=1)
            assert has_drop.float().mean().item() == pytest.approx(
                0.3, abs=0.05
            )
            # Not all dropped streaks are full-sequence drops.
            assert not nan_mask[has_drop].all(dim=1).all()
            for row in nan_mask:
                assert _has_single_contiguous_nan_run(row)

        # Forecast horizon streaks are also contiguous and occur around 0.3.
        fcst_nan = torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, 30:]
        ).squeeze(-1)
        assert fcst_nan.any(dim=1).float().mean().item() == pytest.approx(
            0.3, abs=0.05
        )
        for row in fcst_nan:
            assert _has_single_contiguous_nan_run(row)

    @pytest.mark.parametrize('forecast_overlap', [0, 8, 20])
    def test_overlap_is_synchronized_between_hindcast_and_forecast(
        self, forecast_overlap
    ):
        torch.manual_seed(11)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=64,
            seq_length=20,
            forecast_overlap=forecast_overlap,
            lead_time=5,
        )
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.4,
            streak_probabilities=0.4,
            forecast_overlap=forecast_overlap,
        )
        out = dropout(batch)

        if forecast_overlap > 0:
            for feat in [
                'hres_precip',
                'hres_temp',
                'graphcast_precip',
                'graphcast_temp',
            ]:
                hind_tail = torch.isnan(
                    out['x_d_hindcast'][feat][:, -forecast_overlap:]
                )
                fcst_prefix = torch.isnan(
                    out['x_d_forecast'][feat][:, :forecast_overlap]
                )
                assert torch.equal(hind_tail, fcst_prefix)

    @pytest.mark.parametrize('include_forecast_only', [False, True])
    @pytest.mark.parametrize('lead_time', [1, 5])
    def test_survivor_invariant_holds_at_probability_one(
        self, include_forecast_only, lead_time
    ):
        torch.manual_seed(123)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=64,
            seq_length=15,
            forecast_overlap=15,
            lead_time=lead_time,
            include_forecast_only=include_forecast_only,
        )
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=1.0,
            streak_probabilities=1.0,
            forecast_overlap=15,
        )
        out = dropout(batch)

        # Every hindcast (b, t) retains at least one valid hindcast group.
        hind_valid = torch.stack(
            [
                ~torch.isnan(out['x_d_hindcast'][feats[0]]).squeeze(-1)
                for feats in hind_groups.values()
            ],
            dim=0,
        )
        assert (hind_valid.sum(dim=0) >= 1).all()

        # Every forecast (b, t) retains at least one valid forecast group.
        fcst_valid = torch.stack(
            [
                ~torch.isnan(out['x_d_forecast'][feats[0]]).squeeze(-1)
                for feats in fcst_groups.values()
            ],
            dim=0,
        )
        assert (fcst_valid.sum(dim=0) >= 1).all()

        # Shared groups retain at least one valid group across the horizon
        # (protecting masked-mean hindcast embeddings where hindcast-only
        # groups are NaN-padded).
        shared_valid = torch.stack(
            [
                ~torch.isnan(out['x_d_forecast'][fcst_groups[g][0]][:, 15:])
                .squeeze(-1)
                for g in ['hres', 'graphcast']
            ],
            dim=0,
        )
        assert (shared_valid.sum(dim=0) >= 1).all()

    def test_survivor_invariant_respects_preexisting_nans(self):
        torch.manual_seed(19)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=32, seq_length=10, forecast_overlap=10, lead_time=4
        )
        # Make graphcast NaN on steps 0..4 and hres NaN on steps 5..9 in both
        # hindcast and forecast overlap, and graphcast NaN on horizon 10..11.
        for feat in ['graphcast_precip', 'graphcast_temp']:
            batch['x_d_hindcast'][feat][:, :5] = float('nan')
            batch['x_d_forecast'][feat][:, :5] = float('nan')
            batch['x_d_forecast'][feat][:, 10:12] = float('nan')
        for feat in ['hres_precip', 'hres_temp']:
            batch['x_d_hindcast'][feat][:, 5:10] = float('nan')
            batch['x_d_forecast'][feat][:, 5:10] = float('nan')
            batch['x_d_forecast'][feat][:, 12:14] = float('nan')

        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=1.0,
            streak_probabilities=1.0,
            forecast_overlap=10,
        )
        out = dropout(batch)

        # Sole surviving forecast provider at each timestep must be preserved.
        assert not torch.isnan(out['x_d_forecast']['hres_precip'][:, :5]).any()
        assert not torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, 5:10]
        ).any()
        assert not torch.isnan(
            out['x_d_forecast']['hres_precip'][:, 10:12]
        ).any()
        assert not torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, 12:14]
        ).any()

    def test_single_forecast_group_protects_forecast_while_dropping_hindcast(
        self,
    ):
        torch.manual_seed(5)
        hind_groups = {
            'hres': ['hres_precip'],
            'imerg': ['imerg_precip'],
            'cpc': ['cpc_precip'],
        }
        fcst_groups = {'hres': ['hres_precip']}
        batch = {
            'x_d_hindcast': {
                'hres_precip': torch.ones((32, 10, 1)),
                'imerg_precip': torch.ones((32, 10, 1)),
                'cpc_precip': torch.ones((32, 10, 1)),
            },
            'x_d_forecast': {'hres_precip': torch.ones((32, 14, 1))},
        }
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=1.0,
            streak_probabilities=0.0,
            forecast_overlap=10,
        )
        out = dropout(batch)

        # Sole forecast group hres is protected everywhere.
        assert not torch.isnan(out['x_d_forecast']['hres_precip']).any()
        assert not torch.isnan(out['x_d_hindcast']['hres_precip']).any()
        # Both hindcast-only groups can still be dropped completely.
        assert torch.isnan(out['x_d_hindcast']['imerg_precip']).all()
        assert torch.isnan(out['x_d_hindcast']['cpc_precip']).all()

    def test_unbiased_survival_across_symmetric_groups(self):
        torch.manual_seed(31)
        hind_groups = {'a': ['a_p'], 'b': ['b_p']}
        fcst_groups = {'a': ['a_p'], 'b': ['b_p']}
        batch = {
            'x_d_hindcast': {
                'a_p': torch.ones((2000, 6, 1)),
                'b_p': torch.ones((2000, 6, 1)),
            },
            'x_d_forecast': {
                'a_p': torch.ones((2000, 10, 1)),
                'b_p': torch.ones((2000, 10, 1)),
            },
        }
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=1.0,
            streak_probabilities=0.0,
            forecast_overlap=6,
        )
        out = dropout(batch)
        a_survives = (~torch.isnan(out['x_d_hindcast']['a_p'])[:, 0, 0]).float()
        b_survives = (~torch.isnan(out['x_d_hindcast']['b_p'])[:, 0, 0]).float()
        assert torch.equal(a_survives + b_survives, torch.ones_like(a_survives))
        assert a_survives.mean().item() == pytest.approx(0.5, abs=0.04)

    def test_hindcast_only_batch_with_x_d_key(self):
        torch.manual_seed(8)
        groups = {'a': ['a_p'], 'b': ['b_p'], 'c': ['c_p']}
        batch = {
            'x_d': {
                'a_p': torch.ones((16, 10, 1)),
                'b_p': torch.ones((16, 10, 1)),
                'c_p': torch.ones((16, 10, 1)),
            }
        }
        dropout = InputDropout(
            hindcast_groups=groups,
            forecast_groups={},
            sequence_probabilities=1.0,
            streak_probabilities=1.0,
        )
        out = dropout(batch)
        valid = torch.stack(
            [~torch.isnan(out['x_d'][f'{g}_p']).squeeze(-1) for g in groups],
            dim=0,
        )
        assert (valid.sum(dim=0) == 1).all()

    def test_reproducible_with_torch_seed(self):
        hind_groups, fcst_groups, batch = _make_forecast_batch()
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.3,
            streak_probabilities=0.3,
            forecast_overlap=20,
        )
        torch.manual_seed(999)
        out1 = dropout(batch)
        torch.manual_seed(999)
        out2 = dropout(batch)
        for key in ('x_d_hindcast', 'x_d_forecast'):
            for feat in out1[key]:
                assert torch.equal(
                    torch.isnan(out1[key][feat]), torch.isnan(out2[key][feat])
                )


class TestTrainerAndModelIntegration:
    """Verify model capability flags and end-to-end BaseTrainer integration."""

    def test_model_supports_missing_inputs_flags(self):
        assert BaseModel.supports_missing_inputs is False
        assert HandoffForecastLSTM.supports_missing_inputs is False
        assert MeanEmbeddingForecastLSTM.supports_missing_inputs is True

    @pytest.mark.parametrize('compiled_wrapper', [False, True])
    def test_trainer_rejects_model_without_missing_input_support(
        self, tmp_path, compiled_wrapper
    ):
        cfg = Config(
            {
                'experiment_name': 'reject_test',
                'run_dir': str(tmp_path),
                'model': 'handoff_forecast_lstm',
                'hindcast_inputs': {'hres': ['hres_p'], 'cpc': ['cpc_p']},
                'forecast_inputs': {'hres': ['hres_p']},
                'static_attributes': ['area'],
                'target_variables': ['streamflow'],
                'seq_length': 10,
                'lead_time': 3,
                'forecast_overlap': 10,
                'predict_last_n': 4,
                'batch_size': 4,
                'epochs': 1,
                'loss': 'mse',
                'optimizer': 'Adam',
                'initial_learning_rate': 1e-3,
                'head': 'regression',
                'hidden_size': 8,
                'nan_sequence_probability': 0.1,
                'log_tensorboard': False,
                'validate_every': None,
            }
        )
        fake_model = SimpleNamespace(
            supports_missing_inputs=False,
            to=lambda _dev: fake_model,
        )
        if compiled_wrapper:
            wrapped = SimpleNamespace(
                _orig_mod=fake_model,
                to=lambda _dev: wrapped,
            )
            returned_model = wrapped
        else:
            returned_model = fake_model

        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.cfg = cfg
        trainer.device = torch.device('cpu')
        trainer._input_dropout = None
        trainer._get_dataset = lambda compute_scaler: [1]  # noqa: ARG005
        trainer._get_data_loader = lambda ds: ds  # noqa: ARG005
        trainer._get_model = lambda: returned_model

        with pytest.raises(ValueError, match='does not support missing inputs'):
            trainer.initialize_training()

    @patch('googlehydrology.datautils.scaler.Scaler.load')
    @patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
    def test_train_epoch_with_mean_embedding_and_cmal_stays_finite(
        self, _mock_check, _mock_load, tmp_path
    ):
        torch.manual_seed(0)
        embedding = {
            'type': 'fc',
            'hiddens': [8],
            'activation': ['tanh'],
            'dropout': 0.0,
        }
        cfg = Config(
            {
                'experiment_name': 'dropout_train_test',
                'run_dir': str(tmp_path),
                'model': 'mean_embedding_forecast_lstm',
                'hindcast_inputs': {
                    'hres': ['hres_precip', 'hres_temp'],
                    'graphcast': ['graphcast_precip', 'graphcast_temp'],
                    'imerg': ['imerg_precip'],
                    'cpc': ['cpc_precip'],
                },
                'forecast_inputs': {
                    'hres': ['hres_precip', 'hres_temp'],
                    'graphcast': ['graphcast_precip', 'graphcast_temp'],
                },
                'static_attributes': ['area', 'p_mean'],
                'target_variables': ['streamflow'],
                'statics_embedding': embedding,
                'hindcast_embedding': embedding,
                'forecast_embedding': embedding,
                'seq_length': 10,
                'lead_time': 4,
                'forecast_overlap': 10,
                'predict_last_n': 5,
                'hidden_size': 8,
                'head': 'cmal',
                'n_distributions': 3,
                'loss': 'cmal',
                'optimizer': 'Adam',
                'initial_learning_rate': 1e-3,
                'clip_gradient_norm': 1.0,
                'log_loss_every_nth_update': 1,
                'nan_sequence_probability': 1.0,
                'nan_streak_probability': 1.0,
            }
        )
        model = MeanEmbeddingForecastLSTM(cfg)
        _, _, batch = _make_forecast_batch(
            batch_size=8, seq_length=10, forecast_overlap=10, lead_time=4
        )
        # Positive targets for CMAL.
        batch['y'] = torch.rand((8, 14, 1)) + 0.1

        seen_nan_in_forward = False
        orig_hook = model.pre_model_hook

        def spy_hook(data, is_train):
            nonlocal seen_nan_in_forward
            if any(
                torch.isnan(v).any() for v in data['x_d_forecast'].values()
            ) or any(
                torch.isnan(v).any() for v in data['x_d_hindcast'].values()
            ):
                seen_nan_in_forward = True
            return orig_hook(data, is_train=is_train)

        model.pre_model_hook = spy_hook

        logger = Logger(
            MagicMock(
                log_interval=1,
                run_dir=tmp_path,
                img_log_dir=tmp_path,
                save_git_diff=False,
            )
        )
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.cfg = cfg
        trainer.device = torch.device('cpu')
        trainer.model = model
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer.scaler = torch.amp.GradScaler('cpu', enabled=False)
        trainer.loss_obj = get_loss_obj(cfg)
        trainer.experiment_logger = logger
        trainer.loader = [batch, batch]
        trainer.noise_sampler_y = None
        trainer._input_dropout = InputDropout.from_config(cfg)
        trainer._max_updates_per_epoch = 0
        trainer._disable_pbar = True
        trainer._allow_subsequent_nan_losses = 0

        trainer._train_epoch(epoch=1)

        assert seen_nan_in_forward
        avg_loss = logger.summarise()['avg_loss']
        assert torch.isfinite(torch.tensor(avg_loss))
        assert all(torch.isfinite(p).all() for p in model.parameters())

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA is unavailable'
    )
    @patch('googlehydrology.datautils.scaler.Scaler.load')
    @patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
    def test_cuda_amp_train_epoch_with_input_dropout(
        self, _mock_check, _mock_load, tmp_path
    ):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        embedding = {
            'type': 'fc',
            'hiddens': [8],
            'activation': ['tanh'],
            'dropout': 0.0,
        }
        cfg = Config(
            {
                'experiment_name': 'cuda_dropout_test',
                'run_dir': str(tmp_path),
                'model': 'mean_embedding_forecast_lstm',
                'hindcast_inputs': {
                    'hres': ['hres_precip', 'hres_temp'],
                    'graphcast': ['graphcast_precip', 'graphcast_temp'],
                    'imerg': ['imerg_precip'],
                    'cpc': ['cpc_precip'],
                },
                'forecast_inputs': {
                    'hres': ['hres_precip', 'hres_temp'],
                    'graphcast': ['graphcast_precip', 'graphcast_temp'],
                },
                'static_attributes': ['area', 'p_mean'],
                'target_variables': ['streamflow'],
                'statics_embedding': embedding,
                'hindcast_embedding': embedding,
                'forecast_embedding': embedding,
                'seq_length': 10,
                'lead_time': 4,
                'forecast_overlap': 10,
                'predict_last_n': 5,
                'hidden_size': 8,
                'head': 'cmal',
                'n_distributions': 3,
                'loss': 'cmal',
                'optimizer': 'Adam',
                'initial_learning_rate': 1e-3,
                'clip_gradient_norm': 1.0,
                'log_loss_every_nth_update': 1,
                'nan_sequence_probability': 0.5,
                'nan_streak_probability': 0.5,
            }
        )
        device = torch.device('cuda:0')
        model = MeanEmbeddingForecastLSTM(cfg).to(device)
        _, _, batch = _make_forecast_batch(
            batch_size=16, seq_length=10, forecast_overlap=10, lead_time=4
        )
        batch['y'] = torch.rand((16, 14, 1)) + 0.1

        logger = Logger(
            MagicMock(
                log_interval=1,
                run_dir=tmp_path,
                img_log_dir=tmp_path,
                save_git_diff=False,
            )
        )
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.cfg = cfg
        trainer.device = device
        trainer.model = model
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer.scaler = torch.amp.GradScaler('cuda', enabled=True)
        trainer.loss_obj = get_loss_obj(cfg).to(device)
        trainer.experiment_logger = logger
        trainer.loader = [batch, batch]
        trainer.noise_sampler_y = None
        trainer._input_dropout = InputDropout.from_config(cfg)
        trainer._max_updates_per_epoch = 0
        trainer._disable_pbar = True
        trainer._allow_subsequent_nan_losses = 0

        trainer._train_epoch(epoch=1)

        avg_loss = logger.summarise()['avg_loss']
        assert torch.isfinite(torch.tensor(avg_loss))
        assert all(torch.isfinite(p).all() for p in model.parameters())
