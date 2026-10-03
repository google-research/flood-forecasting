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

"""Regression tests for gradient clipping diagnostics and training behavior."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from googlehydrology.training.basetrainer import BaseTrainer
from googlehydrology.training.logger import Logger

pytestmark = pytest.mark.unit


@pytest.fixture
def logger(tmp_path):
    """Create the real experiment logger without dataset configuration."""
    cfg = MagicMock(
        log_interval=1,
        run_dir=tmp_path,
        img_log_dir=tmp_path,
        save_git_diff=False,
    )
    return Logger(cfg)


def _scalars(writer):
    return {
        call.args[0].removeprefix('train/gradient_clipping/'): call.args[1]
        for call in writer.add_scalar.call_args_list
        if call.args[0].startswith('train/gradient_clipping/')
    }


@pytest.mark.parametrize('tensorboard', [False, True])
def test_summary_counts_preclip_norms_and_percentiles(
    logger, caplog, tensorboard
):
    logger.writer = MagicMock() if tensorboard else None
    logger.log_step(loss=2.0)
    norms = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    with caplog.at_level(logging.INFO):
        logger.log_gradient_norms(norms, clip_gradient_norm=1.0, epoch=7)

    assert 'gradient clipped 2/5 finite steps (40.0%)' in caplog.text
    assert 'median norm 1' in caplog.text
    assert 'threshold 1' in caplog.text
    assert logger.update == 1
    assert logger.epoch == 0
    assert logger.summarise() == {'avg_loss': 2.0}
    if tensorboard:
        stats = _scalars(logger.writer)
        assert stats['checked_steps'] == 5
        assert stats['finite_steps'] == 5
        assert stats['clipped_steps'] == 2
        assert stats['clipped_fraction'] == pytest.approx(0.4)
        assert stats['nonfinite_steps'] == 0
        assert stats['threshold'] == 1.0
        assert stats['norm_median'] == 1.0
        assert stats['norm_p90'] == pytest.approx(3.2)
        assert stats['norm_p99'] == pytest.approx(3.92)
        gradient_calls = [
            call
            for call in logger.writer.add_scalar.call_args_list
            if call.args[0].startswith('train/gradient_clipping/')
        ]
        assert all(call.args[2] == 7 for call in gradient_calls)


@pytest.mark.parametrize('norms', [[], [np.nan, np.inf, -np.inf]])
def test_no_finite_norms_do_not_report_zero_percent(logger, caplog, norms):
    logger.writer = MagicMock()
    with caplog.at_level(logging.INFO):
        logger.log_gradient_norms(np.array(norms), 1.0, epoch=1)
    assert 'no finite gradient norms' in caplog.text
    assert '0.0%' not in caplog.text
    stats = _scalars(logger.writer)
    assert stats['checked_steps'] == len(norms)
    assert stats['nonfinite_steps'] == len(norms)
    assert stats['finite_steps'] == stats['clipped_steps'] == 0
    assert 'clipped_fraction' not in stats
    assert 'norm_median' not in stats


def test_nonfinite_norms_are_reported_separately(logger, caplog):
    logger.writer = MagicMock()
    with caplog.at_level(logging.INFO):
        logger.log_gradient_norms(
            np.array([0.5, 2.0, np.nan, np.inf, -np.inf]), 1.0, epoch=1
        )
    assert 'gradient clipped 1/2 finite steps (50.0%)' in caplog.text
    assert '3 non-finite / 5 checked steps' in caplog.text
    stats = _scalars(logger.writer)
    assert stats['nonfinite_steps'] == 3
    assert stats['norm_median'] == pytest.approx(1.25)


def test_tensorboard_events_use_explicit_epoch_without_polluting_validation(
    logger, tmp_path
):
    logger.log_interval = 0
    logger.start_tb()
    logger.log_gradient_norms(np.array([0.5, 2.0]), 1.0, epoch=4)
    logger.valid().log_step(NSE=0.75)
    assert logger.summarise() == {'NSE': 0.75}
    assert logger.update == 0
    logger.stop_tb()
    events = EventAccumulator(str(tmp_path)).Reload()
    samples = events.Scalars('train/gradient_clipping/clipped_fraction')
    assert [(sample.step, sample.value) for sample in samples] == [(4, 0.5)]
    assert not any(
        'gradient' in tag
        for tag in events.Tags()['scalars']
        if tag.startswith('valid/')
    )


class _LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))

    def pre_model_hook(self, data, is_train):
        return data

    def forward(self, data):
        return torch.dot(self.weight, data['gradient'])


def _make_trainer(logger, gradients, threshold=1.0, scaled=False):
    """Exercise the actual epoch loop with analytically known gradients."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.cfg = SimpleNamespace(
        clip_gradient_norm=threshold, log_loss_every_nth_update=100
    )
    trainer.device = torch.device('cpu')
    trainer.model = _LinearModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler('cpu', enabled=scaled, init_scale=8.0)
    trainer.loss_obj = lambda prediction, data: (
        prediction,
        {'loss': prediction},
    )
    trainer.experiment_logger = logger
    trainer.loader = [
        {'gradient': torch.tensor(gradient, dtype=torch.float32)}
        for gradient in gradients
    ]
    trainer.noise_sampler_y = None
    trainer._max_updates_per_epoch = 0
    trainer._disable_pbar = True
    trainer._allow_subsequent_nan_losses = 10
    return trainer


@pytest.mark.parametrize('scaled', [False, True])
@pytest.mark.parametrize('threshold', [None, 0.0, 1.0, 100.0])
def test_epoch_diagnostics_preserve_optimizer_updates(
    logger, caplog, threshold, scaled
):
    gradients = [[0.0, 0.0], [0.3, 0.4], [3.0, 4.0], [0.0, 1.0]]
    trainer = _make_trainer(logger, gradients, threshold, scaled)
    logger.writer = MagicMock()

    reference = _LinearModel()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    for gradient in gradients:
        optimizer.zero_grad()
        reference({'gradient': torch.tensor(gradient)}).backward()
        if threshold is not None:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), threshold)
        optimizer.step()

    with caplog.at_level(logging.INFO):
        trainer._train_epoch(epoch=3)

    torch.testing.assert_close(
        trainer.model.weight, reference.weight, rtol=0, atol=0
    )
    assert logger.update == 2  # Loss is sampled; gradient statistics are not.
    stats = _scalars(logger.writer)
    if threshold is None:
        assert not stats
        assert 'gradient clipped' not in caplog.text
    else:
        assert stats['checked_steps'] == 4
        assert stats['clipped_steps'] == {0.0: 3, 1.0: 1, 100.0: 0}[threshold]
        assert stats['norm_median'] == pytest.approx(0.75)


def test_epoch_counters_reset_and_nan_losses_are_not_checked(logger):
    trainer = _make_trainer(logger, [[0.0, 2.0], [float('nan'), 0.0]])
    logger.writer = MagicMock()
    trainer._train_epoch(epoch=1)
    assert _scalars(logger.writer)['checked_steps'] == 1
    assert _scalars(logger.writer)['clipped_fraction'] == 1.0
    trainer.loader = [{'gradient': torch.tensor([0.0, 0.5])}]
    logger.writer.reset_mock()
    trainer._train_epoch(epoch=2)
    assert _scalars(logger.writer)['checked_steps'] == 1
    assert _scalars(logger.writer)['clipped_fraction'] == 0.0


@pytest.mark.parametrize('gradients', [[], [[float('nan'), 0.0]]])
def test_epoch_without_checked_steps(logger, caplog, gradients):
    trainer = _make_trainer(logger, gradients)
    with caplog.at_level(logging.INFO):
        trainer._train_epoch(epoch=1)
    assert 'no finite gradient norms' in caplog.text


def test_epoch_respects_update_limit(logger):
    trainer = _make_trainer(logger, [[0.0, 0.5], [0.0, 2.0], [0.0, 4.0]])
    trainer._max_updates_per_epoch = 2
    logger.writer = MagicMock()
    trainer._train_epoch(epoch=1)
    stats = _scalars(logger.writer)
    assert stats['checked_steps'] == 2
    assert stats['clipped_fraction'] == 0.5


def test_double_precision_norms_are_not_rounded_before_comparison(logger):
    trainer = _make_trainer(logger, [])
    trainer.model.double()
    trainer.loader = [
        {'gradient': torch.tensor([0.0, 1.0 + 1e-10], dtype=torch.float64)}
    ]
    logger.writer = MagicMock()
    trainer._train_epoch(epoch=1)
    stats = _scalars(logger.writer)
    assert stats['clipped_steps'] == 1
    assert stats['norm_median'] > 1.0


class _MultipleParametersModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor(0.0))
        self.second = torch.nn.Parameter(torch.tensor(0.0))
        self.unused = torch.nn.Parameter(torch.tensor(0.0))
        self.frozen = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def pre_model_hook(self, data, is_train):
        return data

    def forward(self, data):
        return (
            self.first * data['gradient'][0] + self.second * data['gradient'][1]
        )


def test_norm_combines_all_parameters_and_ignores_missing_gradients(logger):
    trainer = _make_trainer(logger, [[3.0, 4.0]])
    trainer.model = _MultipleParametersModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    logger.writer = MagicMock()
    trainer._train_epoch(epoch=1)
    stats = _scalars(logger.writer)
    assert stats['norm_median'] == 5.0
    assert stats['clipped_steps'] == 1
    assert trainer.model.first.item() == pytest.approx(-0.06)
    assert trainer.model.second.item() == pytest.approx(-0.08)
    assert trainer.model.unused.grad is None
    assert trainer.model.frozen.grad is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_unscale_and_preclip_norm_collection(logger):
    trainer = _make_trainer(logger, [[0.3, 0.4], [3.0, 4.0]])
    trainer.device = torch.device('cuda')
    trainer.model.to(trainer.device)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler('cuda', init_scale=8.0)
    logger.writer = MagicMock()
    trainer._train_epoch(epoch=1)
    stats = _scalars(logger.writer)
    assert stats['checked_steps'] == 2
    assert stats['clipped_fraction'] == 0.5
    assert stats['norm_median'] == pytest.approx(2.75)
    torch.testing.assert_close(
        trainer.model.weight.detach().cpu(), torch.tensor([-0.09, -0.12])
    )


class _NonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gradient):
        ctx.gradient = gradient
        return value.clone()

    @staticmethod
    def backward(ctx, output_gradient):
        return output_gradient * ctx.gradient, None


@pytest.mark.parametrize('gradient', [float('inf'), float('nan')])
def test_grad_scaler_overflow_is_separate_from_finite_clipping(
    logger, gradient
):
    trainer = _make_trainer(logger, [[0.0, 2.0]], scaled=True)
    trainer.loss_obj = lambda prediction, data: (
        _NonfiniteGradient.apply(prediction, gradient),
        {'loss': prediction},
    )
    logger.writer = MagicMock()
    trainer._train_epoch(epoch=1)
    torch.testing.assert_close(trainer.model.weight, torch.zeros(2))
    assert trainer.scaler.get_scale() == 4.0
    stats = _scalars(logger.writer)
    assert stats['checked_steps'] == stats['nonfinite_steps'] == 1
    assert stats['clipped_steps'] == 0
    assert 'clipped_fraction' not in stats
