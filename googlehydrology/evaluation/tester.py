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

import itertools
import logging
import pickle
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.cuda
from torch.amp import autocast
import xarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from googlehydrology.datasetzoo import get_dataset
from googlehydrology.datautils.utils import (
    get_frequency_factor,
    load_basin_file,
    sort_frequencies,
)
from googlehydrology.evaluation import plots
from googlehydrology.evaluation.metrics import (
    calculate_metrics,
    get_available_metrics,
)
from googlehydrology.evaluation.utils import (
    load_basin_id_encoding,
    metrics_to_dataframe,
    BasinBatchSampler,
    get_samples_indexes,
)
from googlehydrology.modelzoo import get_model
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.training import get_loss_obj, get_regularization_obj
from googlehydrology.training.logger import Logger, do_log_figures
from googlehydrology.utils.config import Config, TesterSamplesReduction
from googlehydrology.utils.errors import AllNaNError, NoEvaluationDataError

LOGGER = logging.getLogger(__name__)


class BaseTester(object):
    """Base class to run inference on a model.

    Use subclasses of this class to evaluate a trained model on its train, test, or validation period.
    For regression settings, `RegressionTester` is used; for uncertainty prediction, `UncertaintyTester`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}, optional
        The period to evaluate, by default 'test'.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = 'test',
        init_model: bool = True,
    ):
        self.cfg = cfg
        self.run_dir = run_dir
        self.init_model = init_model
        if period in ['train', 'validation', 'test']:
            self.period = period
        else:
            raise ValueError(
                f'Invalid period {period}. Must be one of ["train", "validation", "test"]'
            )

        # determine device
        self._set_device()

        if self.init_model:
            self.model = get_model(cfg).to(self.device)

        self._disable_pbar = cfg.verbose == 0

        # pre-initialize variables, defined in class methods
        self.basins = None

        # initialize loss object to compute the loss of the evaluation data
        self.loss_obj = get_loss_obj(cfg)
        self.loss_obj.set_regularization_terms(
            get_regularization_obj(cfg=self.cfg)
        )

        self._load_run_data()

        self.dataset = self._get_dataset_all()

        self.exclude_basins = set(self._calc_exclude_basins())

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith('cuda'):
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(
                        f'This machine does not have GPU #{gpu_id} '
                    )
                else:
                    self.device = torch.device(self.cfg.device)
            elif self.cfg.device == 'mps':
                if torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    raise RuntimeError('MPS device is not available.')
            else:
                self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

    def _load_run_data(self):
        """Load run specific data from run directory"""

        # get list of basins
        self.basins = load_basin_file(
            getattr(self.cfg, f'{self.period}_basin_file')
        )

    def _get_weight_file(self, epoch: int | None):
        """Get file path to weight file"""
        if epoch is None:
            weight_file = sorted(list(self.run_dir.glob('model_epoch*.pt')))[-1]
        else:
            weight_file = self.run_dir / f'model_epoch{str(epoch).zfill(3)}.pt'

        return weight_file

    def _load_weights(self, epoch: int = None):
        """Load weights of a certain (or the last) epoch into the model."""
        weight_file = self._get_weight_file(epoch)

        LOGGER.info(f'Using the model weights from {weight_file}')
        self.model.load_state_dict(
            torch.load(weight_file, map_location=self.device, weights_only=True)
        )

    def _get_dataset_all(self) -> Dataset:
        """Get dataset for all basin."""
        return get_dataset(
            cfg=self.cfg,
            is_train=False,
            period=self.period,
            basin=None,
            compute_scaler=False,
        )

    def _get_dataset(self, basin: str) -> Dataset:
        """Get dataset for a single basin."""
        return get_dataset(
            cfg=self.cfg,
            is_train=False,
            period=self.period,
            basin=basin,
            compute_scaler=False,
        )

    def evaluate(
        self,
        epoch: int = None,
        save_results: bool = True,
        save_all_output: bool = False,
        metrics: list | dict = [],
        model: torch.nn.Module = None,
        experiment_logger: Logger = None,
    ) -> dict:
        """Evaluate the model.

        Parameters
        ----------
        epoch : int, optional
            Define a specific epoch to evaluate. By default, the weights of the last epoch are used.
        save_results : bool, optional
            If True, stores the evaluation results in the run directory. By default, True.
        save_all_output : bool, optional
            If True, stores all of the model output in the run directory. By default, False.
        metrics : list | dict, optional
            List of metrics to compute during evaluation. Can also be a dict that specifies per-target metrics
        model : torch.nn.Module, optional
            If a model is passed, this is used for validation.
        experiment_logger : Logger, optional
            Logger can be passed during training to log metrics
        """
        if model is None:
            if self.init_model:
                self._load_weights(epoch=epoch)
                model = self.model
            else:
                raise RuntimeError(
                    'No model was initialized for the evaluation'
                )

        # during validation, depending on settings, only evaluate on a random subset of basins
        basins = set(self.basins) - self.exclude_basins
        if self.period == 'validation':
            if len(basins) > self.cfg.validate_n_random_basins:
                basins = set(
                    random.sample(
                        list(basins), k=self.cfg.validate_n_random_basins
                    )
                )

        # force model to train-mode when doing mc-dropout evaluation
        if self.cfg.mc_dropout:
            model.train()
        else:
            model.eval()

        batch_sampler = BasinBatchSampler(
            sample_index=self.dataset._sample_index,
            batch_size=self.cfg.batch_size,
            basins_indexes=set(
                get_samples_indexes(self.basins, samples=list(basins))
            ),
        )
        loader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=self.dataset.collate_fn,
            pin_memory=True,  # avoid 1 of 2 mem copies to gpu
        )

        max_figures = min(
            self.cfg.validate_n_random_basins,
            self.cfg.log_n_figures,
            len(basins),
        )
        basins_for_figures = random.sample(list(basins), k=max_figures)

        eval_data_it = self._evaluate(
            model, loader, self.dataset.frequencies, save_all_output, basins
        )
        pbar = tqdm(
            eval_data_it,
            file=sys.stdout,
            disable=self._disable_pbar,
            total=len(basins),
        )
        if self.period == 'validation':
            pbar.set_description('# Validation')
        else:
            pbar.set_description(
                '# Inference' if self.cfg.inference_mode else '# Evaluation'
            )

        self._ensure_no_previous_results_saved(epoch)

        metrics_results = {}

        for basin_data in pbar:
            results = {}

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            basin = basin_data['basin']
            y_hat = basin_data['preds']
            y = basin_data['obs']
            dates = basin_data['dates']
            all_losses = basin_data['mean_losses']

            # log loss of this basin plus number of samples in the logger to compute epoch aggregates later
            if experiment_logger is not None:
                experiment_logger.log_step(
                    **{k: (v, len(loader)) for k, v in all_losses.items()}
                )

            predict_last_n = self.cfg.predict_last_n
            seq_length = self.cfg.seq_length
            # if predict_last_n/seq_length are int, there's only one frequency
            if isinstance(predict_last_n, int):
                predict_last_n = {self.dataset.frequencies[0]: predict_last_n}
            if isinstance(seq_length, int):
                seq_length = {self.dataset.frequencies[0]: seq_length}
            lowest_freq = sort_frequencies(self.dataset.frequencies)[0]

            for freq in self.dataset.frequencies:
                if predict_last_n[freq] == 0:
                    continue  # this frequency is not being predicted
                results.setdefault(freq, {})

                # Create data_vars dictionary for the xarray.Dataset
                data_vars = self._create_xarray_data_vars(y_hat[freq], y[freq])

                # freq_range are the steps of the current frequency at each lowest-frequency step
                frequency_factor = int(get_frequency_factor(lowest_freq, freq))

                # Create coords dictionary for the xarray.Dataset. 'date' can be directly infered from the dates
                # dictionary. We index the sample by the date of the last timestep of the sequence. The 'time_step'
                # index that specifies the position in the output sequence (relative to the end) can be inferred by
                # computing the timedelta of the dates. To account for predict_last_n > 1 and multi-freq stuff, we
                # need to add the frequency factor and remove 1 (to start at zero). If this is a forecast model,
                # `date` should refer to the issue dates and the `time_step` coordinates should be positive for
                # positive lead times (negative for any lookback into the hindcast).
                time_step_coords = (
                    (
                        (dates[freq][0, :] - dates[freq][0, -1])
                        / pd.Timedelta(freq)
                    ).astype(np.int64)
                    + frequency_factor
                    - 1
                )
                date_coords = dates[lowest_freq][:, -1]
                # TODO (future) : As in all of the forecast models (but not `Multimet`), this assumes
                # that all lead times are present from 1 to `self.dataset.lead_time`.
                if (
                    hasattr(self.dataset, 'lead_time')
                    and self.dataset.lead_time
                ):
                    time_step_coords += self.dataset.lead_time
                    date_coords = dates[lowest_freq][
                        :, -self.dataset.lead_time - 1
                    ]
                coords = {'date': date_coords, 'time_step': time_step_coords}
                xr = xarray.Dataset(data_vars=data_vars, coords=coords)
                xr = xr.reindex(
                    {
                        'date': pd.DatetimeIndex(
                            pd.date_range(
                                xr['date'].values[0],
                                xr['date'].values[-1],
                                freq=lowest_freq,
                            ),
                            name='date',
                        )
                    }
                )
                xr = self.dataset.scaler.unscale(xr)
                results[freq]['xr'] = xr

                # create datetime range at the current frequency
                freq_date_range = pd.date_range(
                    start=dates[lowest_freq][0, -1],
                    end=dates[freq][-1, -1],
                    freq=freq,
                )
                # remove datetime steps that are not being predicted from the datetime range
                mask = np.ones(frequency_factor).astype(bool)
                mask[: -predict_last_n[freq]] = False
                freq_date_range = freq_date_range[
                    np.tile(mask, len(xr['date']))
                ]

                # only warn once per freq
                if frequency_factor < predict_last_n[freq] and basin == next(
                    iter(basins)
                ):
                    tqdm.write(
                        f'Metrics for {freq} are calculated over last {frequency_factor} elements only. '
                        f'Ignoring {predict_last_n[freq] - frequency_factor} predictions per sequence.'
                    )

                if metrics:
                    for target_variable in self.cfg.target_variables:
                        # stack dates and time_steps so we don't just evaluate every 24h when use_frequencies=[1D, 1h]
                        obs = (
                            xr.isel(
                                time_step=slice(
                                    -predict_last_n[freq],
                                    -predict_last_n[freq] + 1,
                                )
                            )
                            .stack(datetime=['date', 'time_step'])
                            .drop_vars({'datetime', 'date', 'time_step'})[
                                f'{target_variable}_obs'
                            ]
                        )
                        obs['datetime'] = freq_date_range
                        # check if there are observations for this period
                        if obs.notnull().any():
                            sim = (
                                xr.isel(
                                    time_step=slice(
                                        -predict_last_n[freq],
                                        -predict_last_n[freq] + 1,
                                    )
                                )
                                .stack(datetime=['date', 'time_step'])
                                .drop_vars({'datetime', 'date', 'time_step'})[
                                    f'{target_variable}_sim'
                                ]
                            )
                            sim['datetime'] = freq_date_range

                            # clip negative predictions to zero, if variable is listed in config 'clip_target_to_zero'
                            if target_variable in self.cfg.clip_targets_to_zero:
                                sim = xarray.where(sim < 0, 0, sim)

                            if 'samples' in sim.dims:
                                match self.cfg.tester_sample_reduction:
                                    case TesterSamplesReduction.MEAN:
                                        sim = sim.mean(dim="samples")
                                    case TesterSamplesReduction.MEDIAN:
                                        sim = sim.median(dim='samples')
                                    case _:
                                        msg = f'Supported {self.cfg.tester_sample_reduction=}'
                                        raise KeyError(msg)

                            var_metrics = (
                                metrics
                                if isinstance(metrics, list)
                                else metrics[target_variable]
                            )
                            if 'all' in var_metrics:
                                var_metrics = get_available_metrics()
                            try:
                                values = calculate_metrics(
                                    obs,
                                    sim,
                                    metrics=var_metrics,
                                    resolution=freq,
                                )
                            except AllNaNError as err:
                                msg = (
                                    f'Basin {basin} '
                                    + (
                                        f'{target_variable} '
                                        if len(self.cfg.target_variables) > 1
                                        else ''
                                    )
                                    + (
                                        f'{freq} '
                                        if len(self.dataset.frequencies) > 1
                                        else ''
                                    )
                                    + str(err)
                                )
                                LOGGER.warning(msg)
                                values = {
                                    metric: np.nan for metric in var_metrics
                                }

                            # add variable identifier to metrics if needed
                            if len(self.cfg.target_variables) > 1:
                                values = {
                                    f'{target_variable}_{key}': val
                                    for key, val in values.items()
                                }
                            # add frequency identifier to metrics if needed
                            if len(self.dataset.frequencies) > 1:
                                values = {
                                    f'{key}_{freq}': val
                                    for key, val in values.items()
                                }
                            if experiment_logger is not None:
                                experiment_logger.log_step(**values)
                            results[freq].update(values)

            if basin in basins_for_figures:
                self._create_and_log_figures(
                    basin, results, experiment_logger, epoch or -1
                )

            self._save_incremental_results(
                basin,
                results=results,
                states=basin_data['all_output'] if save_all_output else {},
                save_results=save_results,
                save_all_output=save_all_output,
                epoch=epoch,
            )

            if metrics and not experiment_logger:
                for freq, freq_metrics in results.items():
                    for name, metric in freq_metrics.items():
                        if name == 'xr':
                            continue
                        metrics_results.setdefault(freq, {}).setdefault(
                            name, []
                        ).append(metric)

        if metrics and not experiment_logger:
            for freq, freq_metrics in metrics_results.items():
                for name, metric in freq_metrics.items():
                    median = np.nanmedian(metric)
                    LOGGER.info('%s %s median=%f', freq, name, median)

    def _calc_exclude_basins(self) -> Iterator[str]:
        if not self.cfg.tester_skip_obs_all_nan:
            return

        period_start, period_end = (
            self.cfg.test_start_date,
            self.cfg.test_end_date,
        )
        if self.period == 'validation':
            period_start, period_end = (
                self.cfg.validation_start_date,
                self.cfg.validation_end_date,
            )

        # TODO(future): this may be optimized to work vectorically via xarray on all
        # basins at once.
        for basin in self.basins:
            basin_ds = self.dataset._dataset.sel(basin=basin)
            # Calculate all-nan ranges
            diffs = np.diff(
                basin_ds.streamflow.isnull(), prepend=[0], append=[0]
            )
            (starts,), (ends,) = np.where(diffs == 1), np.where(diffs == -1)

            nan_date_starts = basin_ds.date.data[starts]
            nan_date_ends = basin_ds.date.data[ends - 1]
            for start, end in zip(period_start, period_end):
                if np.any((nan_date_starts <= start) & (nan_date_ends >= end)):
                    yield basin

    def _create_and_log_figures(
        self,
        basin: str,
        results: dict,
        experiment_logger: Logger | None,
        epoch: int,
    ):
        for target_var in self.cfg.target_variables:
            for freq in results:
                xr = results[freq]['xr']
                obs = xr[f'{target_var}_obs'].values
                sim = xr[f'{target_var}_sim'].values
                # clip negative predictions to zero, if variable is listed in config 'clip_target_to_zero'
                if target_var in self.cfg.clip_targets_to_zero:
                    sim = xarray.where(sim < 0, 0, sim)
                figures = [
                    self._get_plots(
                        obs,
                        sim,
                        title=f'{target_var} - Basin {basin} - Epoch {epoch} - Frequency {freq}',
                    )[0],
                ]
                # make sure the preamble is a valid file name
                preamble = re.sub(r'[^A-Za-z0-9\._\-]+', '', target_var)
                if experiment_logger:
                    experiment_logger.log_figures(
                        figures, freq, preamble, self.period, basin
                    )
                else:
                    do_log_figures(
                        None,
                        self.cfg.img_log_dir,
                        epoch,
                        figures,
                        freq,
                        preamble,
                        self.period,
                        basin,
                    )

    def _ensure_no_previous_results_saved(self, epoch: int | None = None):
        parent_directory = self._parent_directory_for_results(epoch)

        zarr_stores_to_remove = [
            parent_directory / f'{self.period}_results.zarr',
            parent_directory / f'{self.period}_all_output.zarr',
        ]
        for zarr_store in zarr_stores_to_remove:
            shutil.rmtree(zarr_store, ignore_errors=True)

        metrics_csv_path = parent_directory / f'{self.period}_metrics.csv'
        if metrics_csv_path.exists():
            metrics_csv_path.unlink()

    def _save_incremental_results(
        self,
        basin: str,
        *,
        results: dict,
        states: dict,
        save_results: bool,
        save_all_output: bool,
        epoch: int | None,
    ):
        """Store results in various formats to disk.

        Developer note: We cannot store the time series data (the xarray objects) as netCDF file but have to use
        pickle as a wrapper. The reason is that netCDF files have special constraints on the characters/symbols that can
        be used as variable names. However, for convenience we will store metrics, if calculated, in a separate csv-file.
        """
        parent_directory = self._parent_directory_for_results(epoch)

        # save metrics any time this function is called, as long as they exist
        if self.cfg.metrics and results:
            metrics_list = self.cfg.metrics
            if isinstance(metrics_list, dict):
                metrics_list = list(set(metrics_list.values()))
            if 'all' in metrics_list:
                metrics_list = get_available_metrics()
            df = metrics_to_dataframe(
                {basin: results}, metrics_list, self.cfg.target_variables
            )
            metrics_file = parent_directory / f'{self.period}_metrics.csv'
            df.to_csv(metrics_file, mode='a', header=not metrics_file.exists())

        # store all results in a zarr store
        if (
            results
            and save_results
            and self.cfg.inference_mode
            and self.period == 'test'
        ):
            result_file = parent_directory / f'{self.period}_results.zarr'

            dss = (
                freq_results['xr'].assign_coords(freq=freq)
                for freq, freq_results in results.items()
            )
            ds = xarray.concat(dss, dim='freq').expand_dims(basin=[basin])
            ds = _ensure_unicode_or_bytes_are_strings(ds)

            if result_file.exists():
                ds.to_zarr(result_file, append_dim='basin', consolidated=False)
            else:
                ds.to_zarr(result_file, mode='w', consolidated=False)

        # store all model output in a zarr store
        if (
            states
            and save_all_output
            and self.cfg.inference_mode
            and self.period == 'test'
        ):
            result_file = parent_directory / f'{self.period}_all_output.zarr'

            # TODO(future): setup dims by name instead of by order.
            data_vars = {
                key: (tuple(f'{key}_dim_{i}' for i in range(value.ndim)), value)
                for key, value in states.items()
            }
            ds = xarray.Dataset(data_vars).expand_dims(basin=[basin])
            ds = _ensure_unicode_or_bytes_are_strings(ds)

            if result_file.exists():
                ds.to_zarr(result_file, append_dim='basin', consolidated=False)
            else:
                ds.to_zarr(result_file, mode='w', consolidated=False)

    def _parent_directory_for_results(self, epoch: int | None = None):
        # determine parent directory name and create if needed
        weight_file = self._get_weight_file(epoch=epoch)
        parent_directory = self.run_dir / self.period / weight_file.stem
        parent_directory.mkdir(parents=True, exist_ok=True)
        return parent_directory

    def _evaluate(
        self,
        model: BaseModel,
        loader: DataLoader,
        frequencies: list[str],
        save_all_output: bool = False,
        basins: set[str] = set(),
    ):
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {
                frequencies[0]: predict_last_n
            }  # if predict_last_n is int, there's only one frequency

        with torch.inference_mode():
            basin_samples = itertools.groupby(
                loader, lambda data: data['basin_index'][0].item()
            )
            for basin_index, samples in basin_samples:
                basin = loader.dataset._basins[basin_index]
                if basin not in basins:
                    continue

                preds = {}
                obs = {}
                dates = {}
                all_output = {}
                losses = []
                mean_losses = {}

                for data in samples:
                    for key in data:
                        if key.startswith('x_d'):
                            data[key] = {
                                k: v.to(self.device)
                                for k, v in data[key].items()
                            }
                        elif not key.startswith('date'):
                            data[key] = data[key].to(self.device)

                    with autocast(
                        self.device.type, enabled=(self.device.type == 'cuda')
                    ):
                        data = model.pre_model_hook(data, is_train=False)
                        predictions, loss = self._get_predictions_and_loss(
                            model, data
                        )

                    if save_all_output:
                        for key, value in predictions.items():
                            if value is not None and type(value) != dict:
                                all_output.setdefault(key, []).append(value)

                    for freq in frequencies:
                        if predict_last_n[freq] == 0:
                            continue  # no predictions for this frequency
                        freq_key = '' if len(frequencies) == 1 else f'_{freq}'
                        y_hat_sub, y_sub = self._subset_targets(
                            model,
                            data,
                            predictions,
                            predict_last_n[freq],
                            freq_key,
                        )
                        # Date subsetting is universal across all models and thus happens here.
                        date_sub = data[f'date{freq_key}'][
                            :, -predict_last_n[freq] :
                        ]

                        if freq not in preds:
                            preds[freq] = y_hat_sub
                            obs[freq] = y_sub.detach().to('cpu', non_blocking=False)
                            dates[freq] = date_sub
                        else:
                            preds[freq] = torch.cat((preds[freq], y_hat_sub), 0)
                            obs[freq] = torch.cat((obs[freq], y_sub), 0)
                            dates[freq] = np.concatenate(
                                (dates[freq], date_sub), axis=0
                            )

                    losses.append(loss)

                # concatenate all output variables (currently a dict-of-dicts) into a single-level dict
                for key, list_of_data in all_output.items():
                    all_output[key] = (
                        torch.concatenate(list_of_data, 0)
                        .detach()
                        .to('cpu', non_blocking=True)
                    )

                # set to NaN explicitly if all losses are NaN to avoid RuntimeWarning
                if len(losses) == 0:
                    mean_losses['loss'] = np.nan
                else:
                    for loss_name in losses[0].keys():
                        loss_values = [loss[loss_name] for loss in losses]
                        mean_losses[loss_name] = (
                            np.nanmean(loss_values)
                            if not np.all(np.isnan(loss_values))
                            else np.nan
                        )

                yield {
                    'basin': basin,
                    'preds': preds,
                    'obs': obs,
                    'dates': dates,
                    'all_output': all_output,
                    'losses': losses,
                    'mean_losses': mean_losses,
                }

    def _get_predictions_and_loss(
        self, model: BaseModel, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, float]:
        predictions = model(data)
        _, all_losses = self.loss_obj(predictions, data)
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: int,
        freq: str,
    ):
        raise NotImplementedError

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        raise NotImplementedError


class RegressionTester(BaseTester):
    """Tester class to run inference on a regression model.

    Use the `evaluate` method of this class to evaluate a trained model on its train, test, or validation period.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = 'test',
        init_model: bool = True,
    ):
        super(RegressionTester, self).__init__(cfg, run_dir, period, init_model)

    def _subset_targets(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: np.ndarray,
        freq: str,
    ):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f'{var}_obs'] = (('date', 'time_step'), y[:, :, i])
            data[f'{var}_sim'] = (('date', 'time_step'), y_hat[:, :, i])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.regression_plot(qobs, qsim, title)


class UncertaintyTester(BaseTester):
    """Tester class to run inference on an uncertainty model.

    Use the `evaluate` method of this class to evaluate a trained model on its train, test, or validation period.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = 'test',
        init_model: bool = True,
    ):
        super(UncertaintyTester, self).__init__(
            cfg, run_dir, period, init_model
        )

    def _get_predictions_and_loss(
        self, model: BaseModel, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, float]:
        LOGGER.debug('getting model outputs')
        outputs = model(data)
        LOGGER.debug('getting model losses')
        _, all_losses = self.loss_obj(outputs, data)
        LOGGER.debug('getting model predictions (sample)')
        predictions = model.sample(data, self.cfg.n_samples, outputs=outputs)
        LOGGER.debug('getting model eval')
        model.eval()
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: int,
        freq: str = None,
    ):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f'{var}_obs'] = (('date', 'time_step'), y[:, :, i])
            data[f'{var}_sim'] = (
                ('date', 'time_step', 'samples'),
                y_hat[:, :, i, :],
            )
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.uncertainty_plot(qobs, qsim, title)


def _ensure_unicode_or_bytes_are_strings(ds: xarray.Dataset):
    updates = {
        name: coord.astype('O')
        for name, coord in ds.coords.items()
        if coord.dtype.kind in ('U', 'S')
    }
    return ds.assign_coords(updates)
