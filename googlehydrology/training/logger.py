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

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Union

import matplotlib as mpl
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from googlehydrology.__about__ import __version__
from googlehydrology.utils.config import Config
from googlehydrology.utils.logging_utils import get_git_hash, save_git_diff


class Logger(object):
    """Class that logs runs to tensorboard and saves plots to disk.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        self._train = True
        self.log_interval = cfg.log_interval
        self.log_dir = cfg.run_dir
        self._img_log_dir = cfg.img_log_dir

        # get git commit hash if folder is a git repository
        cfg.update_config({'commit_hash': get_git_hash()})

        # save git diff to file if branch is dirty
        if cfg.save_git_diff:
            save_git_diff(cfg.run_dir)

        # Additionally, the package version is stored in the config
        cfg.update_config({'package_version': __version__})

        # store a copy of the config into the run folder
        cfg.dump_config(folder=self.log_dir)

        self.epoch = 0
        self.update = 0
        self._metrics = defaultdict(list)
        self.writer = None

    @property
    def tag(self):
        return 'train' if self._train else 'valid'

    def train(self) -> 'Logger':
        """Set logging to training period.

        Returns
        -------
        Logger
            The Logger instance, set to training mode.
        """
        self._train = True
        return self

    def valid(self) -> 'Logger':
        """Set logging to validation period.

        Returns
        -------
        Logger
            The Logger instance, set to validation mode.
        """
        self._train = False
        return self

    def start_tb(self):
        """Start tensorboard logging."""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def stop_tb(self):
        """Stop tensorboard logging."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log_figures(
        self,
        figures: list[mpl.figure.Figure],
        freq: str,
        preamble: str = '',
        period: str = 'validation',
        suffix: str = '',
    ):
        """Log matplotlib figures as to disk.

        Parameters
        ----------
        figures : list[mpl.figure.Figure]
            List of figures to save.
        freq : str
            Prediction frequency of the figures.
        preamble : str, optional
            Prefix to prepend to the figures' file names.
        period : str
            'validation' or 'test'
        suffix : str
            Appended to figure filenames
        """
        do_log_figures(
            self.writer,
            self._img_log_dir,
            self.epoch,
            figures,
            freq,
            preamble,
            period,
            suffix,
        )

    def log_step(self, **kwargs):
        """Log the results of a single step within an epoch.

        Parameters
        ----------
        **kwargs
            Key-value pairs of metric names and values.
        """
        for k, v in kwargs.items():
            self._metrics[k].append(v)

        if not self._train:
            return

        self.update += 1

        if self.log_interval <= 0 or self.writer is None:
            return

        if self.update % self.log_interval == 0:
            tag = self.tag
            for k, v in kwargs.items():
                self.writer.add_scalar('/'.join([tag, k]), v, self.update)

    def summarise(self) -> float | dict[str, float]:
        """ "Log the results of the entire training or validation epoch.

        Returns
        -------
        float | dict[str, float]
            Average loss if training is summarized, else a dict mapping metric names to median metric values.
        """
        value = {}
        # summarize statistics of training epoch
        if self._train:
            self.epoch += 1

            # summarize training
            for k, v in self._metrics.items():
                mean = np.nanmean(v) if v else np.nan
                value[f'avg_{k}'] = mean

                if self.writer is not None:
                    self.writer.add_scalar(
                        '/'.join([self.tag, f'avg_{k}']), mean, self.epoch
                    )

        # summarize validation
        else:
            for k, v in self._metrics.items():
                if v and isinstance(v[0], tuple):
                    # The only tuple that is passed is the per basin validation loss, which is a list of tuples, where
                    # each element is defined as (basin loss, number of batches). The aggregate across basins is
                    # weighted by the number of batches per basin, to approximate the training loss computation.
                    v_not_nan = [
                        (loss, samples)
                        for loss, samples in v
                        if not np.isnan(loss)
                    ]
                    num_samples = sum(samples for _, samples in v_not_nan)
                    if num_samples > 0:
                        weighted_loss = sum(
                            loss * samples / num_samples
                            for loss, samples in v_not_nan
                        )
                    else:
                        weighted_loss = np.nan
                    value[f'avg_{k}'] = weighted_loss
                    if self.writer is not None:
                        self.writer.add_scalar(
                            '/'.join([self.tag, f'avg_{k}']),
                            weighted_loss,
                            self.epoch,
                        )
                else:
                    # All other metrics are lists of float values
                    means = np.nanmean(v) if v else np.nan
                    medians = np.nanmedian(v) if v else np.nan
                    value[k] = medians
                    if self.writer is not None:
                        self.writer.add_scalar(
                            '/'.join([self.tag, f'mean_{k.lower()}']),
                            means,
                            self.epoch,
                        )
                        self.writer.add_scalar(
                            '/'.join([self.tag, f'median_{k.lower()}']),
                            medians,
                            self.epoch,
                        )

        # clear buffer
        self._metrics = defaultdict(list)

        return value


def do_log_figures(
    writer: SummaryWriter | None,
    img_log_dir: Path,
    epoch: int,
    figures: list[mpl.figure.Figure],
    freq: str,
    preamble: str = '',
    period: str = 'validation',
    suffix: str = '',
):
    if writer is not None:
        writer.add_figure(
            f'{period}/timeseries/{freq}', figures, global_step=epoch
        )

    for figure in figures:
        figure.savefig(
            Path(
                img_log_dir,
                preamble + f'_{period}_freq{freq}_epoch{epoch}_{suffix}',
            ),
            dpi=300,
        )
