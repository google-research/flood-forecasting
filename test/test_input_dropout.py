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

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from googlehydrology.evaluation.tester import BaseTester
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
    x_f['forecast_counter'] = (
        torch.arange(fcst_len, dtype=torch.float32)
        .view(1, fcst_len, 1)
        .expand(batch_size, fcst_len, 1)
        .clone()
    )

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


def _make_mean_embedding_config(tmp_path: Path, **overrides: object) -> Config:
    embedding = {
        'type': 'fc',
        'hiddens': [8],
        'activation': ['tanh'],
        'dropout': 0.0,
    }
    raw: dict[str, object] = {
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
        'log_tensorboard': False,
        'validate_every': None,
        'compile': False,
    }
    raw.update(overrides)
    return Config(raw)


def _make_trainer_stub(
    cfg: Config,
    *,
    device: torch.device,
    model: object = None,
    loader: object = None,
) -> BaseTrainer:
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.cfg = cfg
    trainer.device = device
    trainer.model = model
    trainer.optimizer = None
    trainer.loss_obj = None
    trainer.experiment_logger = None
    trainer.loader = loader
    trainer.validator = None
    trainer.noise_sampler_y = None
    trainer._target_mean = None
    trainer._target_std = None
    trainer._max_updates_per_epoch = 0
    trainer._disable_pbar = True
    trainer._allow_subsequent_nan_losses = 0
    return trainer


@pytest.mark.unit
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

    def test_accepts_scalar_numpy_and_per_group_dicts(self):
        cfg = Config(
            {
                'experiment_name': 'test',
                'nan_sequence_probability': np.float32(0.1),
                'nan_streak_probability': {'graphcast': 0.2, 'imerg': 0},
            }
        )
        assert cfg.nan_sequence_probability == pytest.approx(0.1)
        assert cfg.nan_streak_probability == {'graphcast': 0.2, 'imerg': 0.0}

    def test_dump_and_reload_round_trip_with_dict_and_override(self, tmp_path):
        cfg = Config(
            {
                'experiment_name': 'roundtrip',
                'hindcast_inputs': {'hres': ['hres_p'], 'cpc': ['cpc_p']},
                'forecast_inputs': {'hres': ['hres_p']},
                'nan_sequence_probability': {'cpc': 0.15},
                'nan_streak_probability': 0.1,
            }
        )
        cfg.dump_config(tmp_path)
        reloaded = Config(tmp_path / 'config.yml')
        assert reloaded.nan_sequence_probability == {'cpc': 0.15}
        assert reloaded.nan_streak_probability == pytest.approx(0.1)

        # Finetune/continue override can disable inherited dropout via None/0.0.
        reloaded.update_config(
            {'nan_sequence_probability': None, 'nan_streak_probability': 0.0}
        )
        assert reloaded.nan_sequence_probability == 0.0
        assert reloaded.nan_streak_probability == 0.0

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
            float('-inf'),
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

    def test_from_config_accepts_flat_and_nested_list_inputs(self):
        cfg_flat = Config(
            {
                'experiment_name': 'flat_list',
                'hindcast_inputs': ['hres_precip', 'cpc_precip'],
                'forecast_inputs': ['hres_precip', 'graphcast_precip'],
                'nan_sequence_probability': 0.1,
            }
        )
        dropout_flat = InputDropout.from_config(cfg_flat)
        assert dropout_flat.enabled
        assert set(dropout_flat.sequence_probabilities) == {
            'hres',
            'cpc',
            'graphcast',
        }

        cfg_nested = Config(
            {
                'experiment_name': 'nested_list',
                'hindcast_inputs': [['hres_precip'], ['cpc_precip']],
                'forecast_inputs': [['hres_precip'], ['graphcast_precip']],
                'nan_streak_probability': 0.2,
            }
        )
        dropout_nested = InputDropout.from_config(cfg_nested)
        assert dropout_nested.enabled
        assert len(dropout_nested.streak_probabilities) == 3

    @pytest.mark.parametrize(
        'bad_prob',
        [-0.5, 1.5, float('nan'), float('inf'), True, {'': 0.1}, {'a': 2.0}],
    )
    def test_direct_constructor_rejects_invalid_probabilities(self, bad_prob):
        with pytest.raises(ValueError):
            InputDropout(
                hindcast_groups={'a': ['a_p'], 'b': ['b_p']},
                forecast_groups={},
                sequence_probabilities=bad_prob,
            )

    def test_nonzero_scalar_with_empty_groups_raises(self):
        with pytest.raises(ValueError, match='no input feature groups'):
            InputDropout(
                hindcast_groups={},
                forecast_groups={},
                sequence_probabilities=0.1,
            )

    def test_overlapping_feature_across_groups_raises(self):
        with pytest.raises(ValueError, match='pairwise disjoint'):
            InputDropout(
                hindcast_groups={
                    'imerg': ['imerg_precip', 'era5_temp'],
                    'cpc': ['cpc_precip', 'era5_temp'],
                },
                forecast_groups={},
                sequence_probabilities=0.1,
            )

        with pytest.raises(ValueError, match='pairwise disjoint'):
            InputDropout(
                hindcast_groups={'hres': ['hres_p'], 'cpc': ['cpc_p']},
                forecast_groups={
                    'hres': ['hres_p', 'shared_temp'],
                    'graphcast': ['gc_p', 'shared_temp'],
                },
                streak_probabilities=0.1,
            )

        # Disabled dropout does not reject existing configs with shared features.
        disabled = InputDropout(
            hindcast_groups={
                'imerg': ['imerg_precip', 'era5_temp'],
                'cpc': ['cpc_precip', 'era5_temp'],
            },
            forecast_groups={},
            sequence_probabilities=0.0,
        )
        assert not disabled.enabled


@pytest.mark.unit
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

        # Confirm dropout actually occurred (guards against silent no-op).
        assert any(
            torch.isnan(v).any()
            for k, v in out['x_d_hindcast'].items()
            if k != 'hindcast_counter'
        )
        assert any(
            torch.isnan(v).any()
            for k, v in out['x_d_forecast'].items()
            if k != 'forecast_counter'
        )

        # Original tensors and counters are untouched.
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
        # Pin hres to 0.0 (unlisted groups default to 0.0) so the remaining
        # groups drop independently at their nominal rates.
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

        # Pinned hres is never dropped and its tensor is not cloned needlessly.
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
                expected, abs=0.045
            )

        # Forecast horizon [12:16] is dropped independently at rate 0.2.
        fcst_nan = torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, 12:]
        ).squeeze(-1)
        assert torch.equal(fcst_nan.all(dim=1), fcst_nan.any(dim=1))
        fcst_rate = fcst_nan.all(dim=1).float().mean().item()
        assert fcst_rate == pytest.approx(0.2, abs=0.045)

        # Hindcast and forecast horizon drops are statistically independent.
        hind_dropped = (
            torch.isnan(out['x_d_hindcast']['graphcast_precip'])
            .squeeze(-1)
            .all(dim=1)
        )
        both_rate = (hind_dropped & fcst_nan.all(dim=1)).float().mean().item()
        assert both_rate == pytest.approx(0.2 * 0.2, abs=0.025)

    def test_streak_dropout_creates_contiguous_subperiods(self):
        torch.manual_seed(7)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=2000, seq_length=30, forecast_overlap=30, lead_time=7
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
                0.3, abs=0.045
            )
            assert not nan_mask[has_drop].all(dim=1).all()
            for row in nan_mask:
                assert _has_single_contiguous_nan_run(row)

        # Forecast horizon streaks are also contiguous and occur around 0.3.
        fcst_nan = torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, 30:]
        ).squeeze(-1)
        assert fcst_nan.any(dim=1).float().mean().item() == pytest.approx(
            0.3, abs=0.045
        )
        for row in fcst_nan:
            assert _has_single_contiguous_nan_run(row)

    def test_streak_length_and_right_weighted_coverage_distribution(self):
        """Pin exact streak length mean and right-weighted timestep coverage."""
        torch.manual_seed(17)
        window_len = 20
        batch_size = 4000
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=batch_size,
            seq_length=window_len,
            forecast_overlap=window_len,
            lead_time=7,
        )
        # Pin hres so imerg streaks are never altered by the survivor guard.
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.0,
            streak_probabilities={'imerg': 1.0},
            forecast_overlap=window_len,
        )
        out = dropout(batch)
        nan_mask = torch.isnan(out['x_d_hindcast']['imerg_precip']).squeeze(-1)

        # Exact expected length for start ~ U{0..L-1}, dur ~ U{1..L} clamped:
        # E[min(D, L - S)] = (L + 1)(2L + 1) / (6L) for L=20 -> 7.175.
        expected_mean_len = (
            (window_len + 1) * (2 * window_len + 1) / (6 * window_len)
        )
        measured_mean_len = nan_mask.sum(dim=1).float().mean().item()
        assert measured_mean_len == pytest.approx(expected_mean_len, abs=0.35)

        # Strictly interior streaks exist (rules out suffix-only streaks) and
        # streaks with length > 1 exist (rules out single-step streaks).
        ends_before_tail = ~nan_mask[:, -1]
        assert ends_before_tail.float().mean().item() > 0.4
        assert (nan_mask.sum(dim=1) > 1).float().mean().item() > 0.8

        # Coverage is right-weighted: P(t=0) = 1/L = 0.05 vs
        # P(t=L-1) = (L + 1)/(2L) = 0.525.
        cov_first = nan_mask[:, 0].float().mean().item()
        cov_last = nan_mask[:, -1].float().mean().item()
        assert cov_first == pytest.approx(1.0 / window_len, abs=0.015)
        assert cov_last == pytest.approx(
            (window_len + 1) / (2 * window_len), abs=0.03
        )
        assert cov_last > 4.0 * cov_first

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
        else:
            # With overlap=0, the entire forecast tensor is sampled as horizon
            # and still satisfies the forecast survivor invariant.
            fcst_valid = torch.stack(
                [
                    ~torch.isnan(out['x_d_forecast'][feats[0]]).squeeze(-1)
                    for feats in fcst_groups.values()
                ],
                dim=0,
            )
            assert (fcst_valid.sum(dim=0) >= 1).all()
            assert (fcst_valid.sum(dim=0) < len(fcst_groups)).any()

    def test_forecast_overlap_exceeding_window_length_raises(self):
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=8, seq_length=10, forecast_overlap=10, lead_time=4
        )
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.5,
            forecast_overlap=12,
        )
        with pytest.raises(ValueError, match='forecast_overlap'):
            dropout(batch)

    def test_forecast_only_group_overlap_and_horizon_are_independent(self):
        torch.manual_seed(23)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=2000,
            seq_length=12,
            forecast_overlap=12,
            lead_time=5,
            include_forecast_only=True,
        )
        # Drop only forecast-only group gfs; hres and graphcast stay pinned.
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities={'gfs': 0.3},
            streak_probabilities=0.0,
            forecast_overlap=12,
        )
        out = dropout(batch)
        overlap_drop = (
            torch.isnan(out['x_d_forecast']['gfs_precip'][:, :12])
            .squeeze(-1)
            .all(dim=1)
        )
        horizon_drop = (
            torch.isnan(out['x_d_forecast']['gfs_precip'][:, 12:])
            .squeeze(-1)
            .all(dim=1)
        )
        assert overlap_drop.float().mean().item() == pytest.approx(
            0.3, abs=0.045
        )
        assert horizon_drop.float().mean().item() == pytest.approx(
            0.3, abs=0.045
        )
        assert (overlap_drop & horizon_drop).float().mean().item() == (
            pytest.approx(0.09, abs=0.03)
        )

    @pytest.mark.parametrize('include_forecast_only', [False, True])
    @pytest.mark.parametrize('lead_time', [0, 1, 5])
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

        hind_valid = torch.stack(
            [
                ~torch.isnan(out['x_d_hindcast'][feats[0]]).squeeze(-1)
                for feats in hind_groups.values()
            ],
            dim=0,
        )
        assert (hind_valid.sum(dim=0) >= 1).all()

        fcst_valid = torch.stack(
            [
                ~torch.isnan(out['x_d_forecast'][feats[0]]).squeeze(-1)
                for feats in fcst_groups.values()
            ],
            dim=0,
        )
        assert (fcst_valid.sum(dim=0) >= 1).all()

        if lead_time > 0:
            shared_valid = torch.stack(
                [
                    ~torch.isnan(
                        out['x_d_forecast'][fcst_groups[g][0]][:, 15:]
                    ).squeeze(-1)
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

    def test_asymmetric_preexisting_nan_in_shared_group_preserves_survivor(
        self,
    ):
        """A shared group NaN only in x_d_forecast prefix cannot be sole survivor."""
        torch.manual_seed(29)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=64, seq_length=10, forecast_overlap=10, lead_time=4
        )
        # hres is valid in x_d_hindcast, but already NaN across x_d_forecast[:10].
        batch['x_d_forecast']['hres_precip'][:, :10] = float('nan')

        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=1.0,
            streak_probabilities=1.0,
            forecast_overlap=10,
        )
        out = dropout(batch)

        # graphcast is the only valid group in the forecast overlap prefix, so
        # it must survive at every (b, t) in x_d_forecast[:, :10].
        assert not torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, :10]
        ).any()
        assert not torch.isnan(
            out['x_d_forecast']['graphcast_temp'][:, :10]
        ).any()

    def test_partial_feature_nan_and_all_nan_timestep_handling(self):
        torch.manual_seed(37)
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=32, seq_length=10, forecast_overlap=10, lead_time=4
        )
        # Partial-feature NaN: only hres_temp is NaN on steps 0..4 (making hres
        # invalid there), so graphcast must survive on steps 0..4.
        batch['x_d_hindcast']['hres_temp'][:, :5] = float('nan')
        batch['x_d_forecast']['hres_temp'][:, :5] = float('nan')
        # All-NaN timestep at t=9 across all groups stays NaN without crashing.
        for feat in batch['x_d_hindcast']:
            if feat != 'hindcast_counter':
                batch['x_d_hindcast'][feat][:, 9] = float('nan')
        for feat in batch['x_d_forecast']:
            if feat != 'forecast_counter':
                batch['x_d_forecast'][feat][:, 9] = float('nan')

        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=1.0,
            streak_probabilities=1.0,
            forecast_overlap=10,
        )
        out = dropout(batch)

        assert not torch.isnan(
            out['x_d_forecast']['graphcast_precip'][:, :5]
        ).any()
        assert torch.isnan(out['x_d_forecast']['hres_precip'][:, 9]).all()
        assert torch.isnan(out['x_d_forecast']['graphcast_precip'][:, 9]).all()

    def test_drop_rate_unaffected_by_preexisting_nans(self):
        """Already-NaN groups are not counted as valid drops or survivors."""
        torch.manual_seed(41)
        hind_groups = {'a': ['a_p'], 'b': ['b_p'], 'c': ['c_p']}
        batch = {
            'x_d': {
                'a_p': torch.ones((2000, 8, 1)),
                'b_p': torch.ones((2000, 8, 1)),
                'c_p': torch.ones((2000, 8, 1)),
            }
        }
        # Half of group 'b' is already NaN; group 'c' is always valid & pinned.
        batch['x_d']['b_p'][:1000] = float('nan')
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups={},
            sequence_probabilities={'a': 0.5, 'b': 1.0, 'c': 0.0},
        )
        out = dropout(batch)

        # In the first 1000 rows, 'b' is already NaN while 'c' is valid; 'a'
        # must still drop at its nominal 0.5 rate and never drop 'c'.
        a_drop_first_half = (
            torch.isnan(out['x_d']['a_p'][:1000, 0, 0]).float().mean().item()
        )
        assert a_drop_first_half == pytest.approx(0.5, abs=0.06)
        assert not torch.isnan(out['x_d']['c_p']).any()

    def test_single_forecast_group_protects_forecast_while_dropping_hindcast(
        self, caplog
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
        with caplog.at_level(logging.WARNING):
            dropout = InputDropout(
                hindcast_groups=hind_groups,
                forecast_groups=fcst_groups,
                sequence_probabilities=1.0,
                streak_probabilities=0.0,
                forecast_overlap=10,
            )
        assert 'forecast_inputs has only 1 feature group' in caplog.text
        out = dropout(batch)

        assert not torch.isnan(out['x_d_forecast']['hres_precip']).any()
        assert not torch.isnan(out['x_d_hindcast']['hres_precip']).any()
        assert torch.isnan(out['x_d_hindcast']['imerg_precip']).all()
        assert torch.isnan(out['x_d_hindcast']['cpc_precip']).all()

    def test_single_group_everywhere_warns_and_returns_unchanged(self, caplog):
        batch = {
            'x_d_hindcast': {'hres_p': torch.ones((8, 6, 1))},
            'x_d_forecast': {'hres_p': torch.ones((8, 10, 1))},
        }
        with caplog.at_level(logging.WARNING):
            dropout = InputDropout(
                hindcast_groups={'hres': ['hres_p']},
                forecast_groups={'hres': ['hres_p']},
                sequence_probabilities=1.0,
                forecast_overlap=6,
            )
        assert 'fewer than 2 groups' in caplog.text
        assert dropout.enabled
        out = dropout(batch)
        assert out is batch

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
        assert a_survives.mean().item() == pytest.approx(0.5, abs=0.045)

    @pytest.mark.parametrize('batch_key', ['x_d', 'x_d_hindcast'])
    def test_hindcast_only_batch_with_x_d_or_x_d_hindcast(self, batch_key):
        torch.manual_seed(8)
        groups = {'a': ['a_p'], 'b': ['b_p'], 'c': ['c_p']}
        batch = {
            batch_key: {
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
            [
                ~torch.isnan(out[batch_key][f'{g}_p']).squeeze(-1)
                for g in groups
            ],
            dim=0,
        )
        assert (valid.sum(dim=0) == 1).all()

    @pytest.mark.parametrize(('batch_size', 'seq_len'), [(0, 5), (1, 1)])
    def test_edge_batch_and_sequence_shapes(self, batch_size, seq_len):
        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=batch_size,
            seq_length=seq_len,
            forecast_overlap=seq_len,
            lead_time=1,
        )
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=1.0,
            streak_probabilities=1.0,
            forecast_overlap=seq_len,
        )
        out = dropout(batch)
        if batch_size == 0:
            assert out is batch
        else:
            hind_valid = torch.stack(
                [
                    ~torch.isnan(out['x_d_hindcast'][feats[0]]).squeeze(-1)
                    for feats in hind_groups.values()
                ],
                dim=0,
            )
            assert (hind_valid.sum(dim=0) == 1).all()

    def test_non_3d_feature_tensor_raises_clear_error(self):
        dropout = InputDropout(
            hindcast_groups={'a': ['a_p'], 'b': ['b_p']},
            forecast_groups={},
            sequence_probabilities=0.5,
        )
        batch = {
            'x_d': {'a_p': torch.ones((4, 10)), 'b_p': torch.ones((4, 10))}
        }
        with pytest.raises(ValueError, match='3D'):
            dropout(batch)

    def test_zero_host_device_syncs_via_torch_dispatch(self):
        """Ensure InputDropout executes no .item() / scalar host-device syncs."""

        class _SyncCounter(TorchDispatchMode):
            def __init__(self) -> None:
                super().__init__()
                self.sync_ops: list[str] = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                op_name = str(func)
                if '_local_scalar_dense' in op_name or 'aten.item' in op_name:
                    self.sync_ops.append(op_name)
                return func(*args, **(kwargs or {}))

        hind_groups, fcst_groups, batch = _make_forecast_batch(
            batch_size=32, seq_length=20, forecast_overlap=20, lead_time=7
        )
        dropout = InputDropout(
            hindcast_groups=hind_groups,
            forecast_groups=fcst_groups,
            sequence_probabilities=0.1,
            streak_probabilities=0.1,
            forecast_overlap=20,
        )
        # Warm cache on first call, then check steady-state zero syncs.
        _ = dropout(batch)
        with _SyncCounter() as counter:
            _ = dropout(batch)
        assert counter.sync_ops == []

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


@pytest.mark.integration
class TestTrainerAndModelIntegration:
    """Verify model capability flags and end-to-end BaseTrainer integration."""

    def test_model_supports_missing_inputs_flags(self):
        assert BaseModel.supports_missing_inputs is False
        assert HandoffForecastLSTM.supports_missing_inputs is False
        assert MeanEmbeddingForecastLSTM.supports_missing_inputs is True

    def test_trainer_rejects_unsupported_model_before_loading_dataset(
        self, tmp_path
    ):
        cfg = Config(
            {
                'experiment_name': 'fail_fast_test',
                'run_dir': str(tmp_path),
                'model': 'handoff_forecast_lstm',
                'hindcast_inputs': {'hres': ['hres_p'], 'cpc': ['cpc_p']},
                'forecast_inputs': {'hres': ['hres_p']},
                'nan_sequence_probability': 0.1,
            }
        )
        trainer = _make_trainer_stub(cfg, device=torch.device('cpu'))

        def fail_if_called(**_kwargs):
            raise AssertionError(
                'Dataset must not be loaded for unsupported model'
            )

        trainer._get_dataset = fail_if_called

        with pytest.raises(ValueError, match='does not support missing inputs'):
            trainer.initialize_training()

    @pytest.mark.parametrize('compiled_wrapper', [False, True])
    def test_trainer_rejects_custom_model_without_missing_input_support(
        self, tmp_path, compiled_wrapper
    ):
        cfg = _make_mean_embedding_config(
            tmp_path, nan_sequence_probability=0.1
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

        trainer = _make_trainer_stub(cfg, device=torch.device('cpu'))
        trainer._get_dataset = lambda compute_scaler: [1]  # noqa: ARG005
        trainer._get_data_loader = lambda ds: ds  # noqa: ARG005
        trainer._get_model = lambda: returned_model

        with pytest.raises(ValueError, match='does not support missing inputs'):
            trainer.initialize_training()

    @pytest.mark.parametrize('compiled_wrapper', [False, True])
    @patch('googlehydrology.datautils.scaler.Scaler.load')
    @patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
    def test_initialize_training_wires_input_dropout_for_supported_model(
        self, _mock_check, _mock_load, tmp_path, compiled_wrapper
    ):
        """Verify initialize_training sets _input_dropout (including _orig_mod)."""
        cfg = _make_mean_embedding_config(
            tmp_path,
            nan_sequence_probability=0.1,
            nan_streak_probability=0.1,
        )
        real_model = MeanEmbeddingForecastLSTM(cfg)
        if compiled_wrapper:
            # Wrapper deliberately omits supports_missing_inputs so unwrapping
            # _orig_mod is required for this positive case to succeed.
            wrapped = SimpleNamespace(
                _orig_mod=real_model,
                to=lambda _dev: wrapped,
                parameters=real_model.parameters,
            )
            returned_model = wrapped
        else:
            returned_model = real_model

        trainer = _make_trainer_stub(cfg, device=torch.device('cpu'))
        trainer._get_dataset = lambda compute_scaler: [1]  # noqa: ARG005
        trainer._get_data_loader = lambda ds: ds  # noqa: ARG005
        trainer._get_model = lambda: returned_model
        trainer._get_optimizer = lambda: torch.optim.Adam(
            real_model.parameters(), lr=1e-3
        )
        trainer._set_regularization = lambda: None

        trainer.initialize_training()

        assert trainer._input_dropout is not None
        assert trainer._input_dropout.enabled
        assert trainer._input_dropout.sequence_probabilities['hres'] == (
            pytest.approx(0.1)
        )

    @patch('googlehydrology.datautils.scaler.Scaler.load')
    @patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
    def test_initialize_training_clears_dropout_when_disabled(
        self, _mock_check, _mock_load, tmp_path
    ):
        cfg = _make_mean_embedding_config(
            tmp_path,
            nan_sequence_probability=0.0,
            nan_streak_probability=0.0,
        )
        real_model = MeanEmbeddingForecastLSTM(cfg)
        trainer = _make_trainer_stub(cfg, device=torch.device('cpu'))
        trainer._input_dropout = MagicMock()
        trainer._get_dataset = lambda compute_scaler: [1]  # noqa: ARG005
        trainer._get_data_loader = lambda ds: ds  # noqa: ARG005
        trainer._get_model = lambda: real_model
        trainer._get_optimizer = lambda: torch.optim.Adam(
            real_model.parameters(), lr=1e-3
        )
        trainer._set_regularization = lambda: None

        trainer.initialize_training()
        assert trainer._input_dropout is None

    def test_tester_never_applies_input_dropout(self):
        assert not hasattr(BaseTester, '_input_dropout')

    @patch('googlehydrology.datautils.scaler.Scaler.load')
    @patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
    def test_train_epoch_with_mean_embedding_and_cmal_stays_finite(
        self, _mock_check, _mock_load, tmp_path
    ):
        torch.manual_seed(0)
        cfg = _make_mean_embedding_config(
            tmp_path,
            nan_sequence_probability=1.0,
            nan_streak_probability=1.0,
        )
        model = MeanEmbeddingForecastLSTM(cfg)
        _, _, batch = _make_forecast_batch(
            batch_size=8, seq_length=10, forecast_overlap=10, lead_time=4
        )
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

        trainer = _make_trainer_stub(
            cfg, device=torch.device('cpu'), model=model
        )
        trainer._get_dataset = lambda compute_scaler: [batch, batch]  # noqa: ARG005
        trainer._get_data_loader = lambda ds: ds
        trainer._get_model = lambda: model
        trainer._set_regularization = lambda: None

        # Wire via initialize_training() end-to-end rather than injecting manually.
        trainer.initialize_training()
        trainer.experiment_logger = Logger(
            MagicMock(
                log_interval=1,
                run_dir=tmp_path,
                img_log_dir=tmp_path,
                save_git_diff=False,
            )
        )
        trainer._train_epoch(epoch=1)

        assert seen_nan_in_forward
        avg_loss = trainer.experiment_logger.summarise()['avg_loss']
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
        cfg = _make_mean_embedding_config(
            tmp_path,
            nan_sequence_probability=0.5,
            nan_streak_probability=0.5,
        )
        device = torch.device('cuda:0')
        model = MeanEmbeddingForecastLSTM(cfg).to(device)
        _, _, batch = _make_forecast_batch(
            batch_size=16, seq_length=10, forecast_overlap=10, lead_time=4
        )
        batch['y'] = torch.rand((16, 14, 1)) + 0.1

        trainer = _make_trainer_stub(cfg, device=device, model=model)
        trainer._get_dataset = lambda compute_scaler: [batch, batch]  # noqa: ARG005
        trainer._get_data_loader = lambda ds: ds
        trainer._get_model = lambda: model
        trainer._set_regularization = lambda: None
        trainer.initialize_training()
        trainer.experiment_logger = Logger(
            MagicMock(
                log_interval=1,
                run_dir=tmp_path,
                img_log_dir=tmp_path,
                save_git_diff=False,
            )
        )

        trainer._train_epoch(epoch=1)

        avg_loss = trainer.experiment_logger.summarise()['avg_loss']
        assert torch.isfinite(torch.tensor(avg_loss))
        assert all(torch.isfinite(p).all() for p in model.parameters())
