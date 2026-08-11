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

"""Unit tests for googlehydrology.run and googlehydrology.run_scheduler CLI."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from googlehydrology import run, run_scheduler


@pytest.mark.unit
def test_run_get_args_valid_modes():
    # Train mode
    with patch.object(
        sys, 'argv', ['run.py', 'train', '--config-file', 'test_config.yml']
    ):
        args = run._get_args()
        assert args['mode'] == 'train'
        assert args['config_file'] == 'test_config.yml'

    # Continue training mode
    with patch.object(
        sys, 'argv', ['run.py', 'continue_training', '--run-dir', '/tmp/run']
    ):
        args = run._get_args()
        assert args['mode'] == 'continue_training'
        assert args['run_dir'] == '/tmp/run'

    # Evaluate mode
    with patch.object(
        sys,
        'argv',
        ['run.py', 'evaluate', '--run-dir', '/tmp/run', '--period', 'test'],
    ):
        args = run._get_args()
        assert args['mode'] == 'evaluate'
        assert args['period'] == 'test'

    # Infer mode
    with patch.object(
        sys, 'argv', ['run.py', 'infer', '--run-dir', '/tmp/run']
    ):
        args = run._get_args()
        assert args['mode'] == 'infer'


@pytest.mark.unit
def test_run_get_args_missing_required_args():
    # Train missing config file
    with patch.object(sys, 'argv', ['run.py', 'train']):
        with pytest.raises(ValueError, match='Missing path to config file'):
            run._get_args()

    # Continue training missing run dir
    with patch.object(sys, 'argv', ['run.py', 'continue_training']):
        with pytest.raises(
            ValueError, match='Missing path to run directory file'
        ):
            run._get_args()

    # Evaluate missing run dir
    with patch.object(sys, 'argv', ['run.py', 'evaluate']):
        with pytest.raises(ValueError, match='Missing path to run directory'):
            run._get_args()


@pytest.mark.unit
def test_run_dispatch_start_run():
    cfg = MagicMock()
    with patch('googlehydrology.run.start_training') as mock_train:
        run.start_run(config=cfg, gpu=0)
        assert cfg.device == 'cuda:0'
        mock_train.assert_called_once_with(cfg)

        run.start_run(config=cfg, gpu=-1)
        assert cfg.device == 'cpu'


@pytest.mark.unit
def test_run_dispatch_eval_run():
    cfg = MagicMock()
    with patch('googlehydrology.run.start_evaluation') as mock_eval:
        run.eval_run(
            config=cfg,
            run_dir=Path('/tmp/run'),
            period='test',
            epoch=1,
            gpu=-1,
        )
        assert cfg.device == 'cpu'
        mock_eval.assert_called_once()


@pytest.mark.unit
def test_run_scheduler_get_args(tmp_path):
    # Valid arguments
    with patch.object(sys, 'argv', [
        'schedule-runs', 'train',
        '--directory', str(tmp_path),
        '--gpu-ids', '0', '1',
        '--runs-per-gpu', '2'
    ]):
        args = run_scheduler._get_args()
        assert args['mode'] == 'train'
        assert args['directory'] == tmp_path
        assert args['gpu_ids'] == [0, 1]
        assert args['runs_per_gpu'] == 2

    # Non-existent directory
    with patch.object(sys, 'argv', [
        'schedule-runs', 'train',
        '--directory', str(tmp_path / 'nonexistent'),
        '--gpu-ids', '0',
        '--runs-per-gpu', '1'
    ]):
        with pytest.raises(ValueError, match='No folder at'):
            run_scheduler._get_args()
