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

"""Unit tests for googlehydrology.training.logger."""

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from googlehydrology.training.logger import Logger


@pytest.fixture
def mock_config(tmp_path):
    cfg = MagicMock()
    cfg.log_interval = 1
    cfg.run_dir = tmp_path
    cfg.img_log_dir = tmp_path / 'img_log'
    cfg.img_log_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_git_diff = False
    cfg.update_config = MagicMock()
    cfg.dump_config = MagicMock()
    return cfg


@pytest.mark.unit
def test_logger_lifecycle_and_summarise(mock_config, tmp_path):
    logger = Logger(mock_config)
    assert logger.epoch == 0
    assert logger.update == 0

    # Test train mode logging
    logger.train()
    assert logger._train is True
    assert logger.tag == 'train'
    logger.log_step(loss=1.5, reg=0.1)
    logger.log_step(loss=1.1, reg=0.1)

    train_summary = logger.summarise()
    assert logger.epoch == 1
    assert 'avg_loss' in train_summary
    assert np.isclose(train_summary['avg_loss'], 1.3)

    # Test valid mode logging
    logger.valid()
    assert logger._train is False
    assert logger.tag == 'valid'
    # Per-basin validation loss passes tuples (loss, n_samples)
    logger.log_step(val_loss=(0.8, 10))
    logger.log_step(val_loss=(0.4, 10))
    # Other metrics pass scalar lists
    logger.log_step(NSE=0.75)
    logger.log_step(NSE=0.85)

    val_summary = logger.summarise()
    assert 'avg_val_loss' in val_summary
    assert np.isclose(val_summary['avg_val_loss'], 0.6)
    assert 'NSE' in val_summary
    assert np.isclose(val_summary['NSE'], 0.8)


@pytest.mark.unit
def test_logger_figures(mock_config, tmp_path):
    logger = Logger(mock_config)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    logger.log_figures(
        figures=[fig],
        freq='1D',
        preamble='test',
        suffix='hydrograph.png',
    )
    img_files = list((tmp_path / 'img_log').glob('*'))
    assert len(img_files) > 0
    plt.close(fig)
