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

"""Integration tests that perform full runs."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import approx

from googlehydrology.datasetzoo import caravan
from googlehydrology.evaluation.evaluate import start_evaluation
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.training.basetrainer import BaseTrainer
from googlehydrology.training.train import start_training
from googlehydrology.utils.config import Config
from test import Fixture


def test_tutorial_finetune_modules_are_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the tutorial modules resolve to trainable model parameters."""
    monkeypatch.setattr(
        'googlehydrology.modelzoo.basemodel.Scaler',
        lambda **_kwargs: None,
    )
    config = Config(Path('tutorial/model-runs/5-basin-example/config.yml'))
    config.update_config(Path('tutorial/configs/finetune-config.yml'))
    model = MeanEmbeddingForecastLSTM(config)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.cfg = config
    trainer.model = model
    trainer._freeze_model_parts()

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any(name.startswith('static_embedding_fc.') for name in trainable_names)
    assert any(name.startswith('head.') for name in trainable_names)
    assert all(
        name.startswith(('static_embedding_fc.', 'head.'))
        for name in trainable_names
    )


def test_forecast_daily_regression(
    get_config: Fixture[Callable[[str], dict]],
    forecast_model: Fixture[str],
    forecast_config_updates: Fixture[Callable[[str], dict]],
    lazy_load: Fixture[bool],
):
    """Test regression training and evaluation for daily predictions.

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]
        Method that returns a run configuration to test.
    forecast_model : Fixture[str]
        Model to test.
    forecast_config : Fixture[str]
        Updates the config with model-specific parameters.
    """
    # Currently only supports testing with Multimet.
    config = get_config('forecast')
    config.update_config(forecast_config_updates(forecast_model))
    config.lazy_load = lazy_load

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    nan_basin = 'camelsaus_102101A'
    nan_dates = pd.date_range(*get_test_start_end_dates(config))
    index = pd.MultiIndex.from_product(
        [[nan_basin], nan_dates], 
        names=['basin', 'date']
    )
    nan_discharge = pd.DataFrame(
        data=np.nan, 
        index=index, 
        columns=['streamflow']
    )
    _check_results(config, nan_basin, nan_discharge)  # No valid data.
    _check_results(config, 'lamah_1145')
    _check_results(config, 'hysets_01075000')


def _check_results(config: Config, basin: str, discharge: pd.Series = None):
    """Perform basic sanity checks of model predictions.

    Checks that the results file has the correct date range, that the
    observed discharge in the file is correct, and that there are no
    NaN predictions.

    Parameters
    ----------
    config : Config
        The run configuration used to produce the results
    basin : str
        Id of a basin for which to check the results
    discharge : pd.Series, optional
        If provided, will check that the stored discharge obs match this series.
        Else, will compare to the discharge loaded from disk.
    """
    test_start_date, test_end_date = get_test_start_end_dates(config)

    # TODO (current) :: Remove debugging comments.
    results = get_basin_results(config.run_dir, 1).sel(basin=basin)
    assert pd.to_datetime(results['date'].values[0]) == test_start_date.floor(
        'D'
    )
    assert pd.to_datetime(results['date'].values[-1]) == test_end_date.floor(
        'D'
    )

    full_discharge = None
    if discharge is None:
        discharge_ds = caravan.load_caravan_timeseries_together(
            config.data_dir, [basin], config.target_variables, csv=False
        )
        discharge = discharge_ds.to_dataframe()
        full_discharge = discharge.loc[basin, 'streamflow']
        discharge_ds.close()

    target = config.target_variables[0]
    if hasattr(config, 'lead_time') and full_discharge is not None:
        issue_dates = pd.to_datetime(results['date'].values)
        for ts in results['time_step'].values:
            if ts < 1:
                continue
            obs_ts = results[f'{target}_obs'].sel(time_step=ts).squeeze().values
            valid_dates = issue_dates + pd.Timedelta(days=int(ts) - 1)
            expected_ts = full_discharge.reindex(valid_dates).values
            assert obs_ts == approx(expected_ts, nan_ok=True)

    if hasattr(config, 'lead_time'):
        # time_step=1 is the first (1-day) lead time, valid on the issue date.
        results = results.sel(time_step=1).squeeze()
    else:
        results = results.isel(time_step=-1)

    results_array = results[f'{target}_obs'].values
    idx = pd.IndexSlice
    discharge_slice = discharge.loc[
        idx[basin, test_start_date:test_end_date], 'streamflow'
    ]
    discharge_array = discharge_slice.values

    assert discharge_array == approx(results_array, nan_ok=True)

    # CAMELS forcings have no NaNs, so there should be no NaN predictions
    assert not pd.isna(results[f'{target}_sim']).any()


def test_forecast_short_predict_last_n(
    get_config: Fixture[Callable[[str], dict]],
    forecast_config_updates: Fixture[Callable[[str], dict]],
):
    """Evaluation works when fewer target steps than lead times are predicted.

    Results must still be indexed by forecast issue date, with `time_step=k`
    holding the observation valid `k - 1` days after the issue date.
    """
    config = get_config('forecast')
    config.update_config(forecast_config_updates('handoff_forecast_lstm'))
    config.update_config({'predict_last_n': 3, 'lead_time': 7})

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    basin = 'lamah_1145'
    test_start_date, test_end_date = get_test_start_end_dates(config)
    results = get_basin_results(config.run_dir, 1).sel(basin=basin)
    np.testing.assert_array_equal(results['time_step'].values, [5, 6, 7])
    assert pd.to_datetime(results['date'].values[0]) == test_start_date.floor(
        'D'
    )
    assert pd.to_datetime(results['date'].values[-1]) == test_end_date.floor(
        'D'
    )

    discharge_ds = caravan.load_caravan_timeseries_together(
        config.data_dir, [basin], config.target_variables, csv=False
    )
    discharge = discharge_ds.to_dataframe().loc[basin, 'streamflow']
    discharge_ds.close()

    target = config.target_variables[0]
    for time_step in (5, 6, 7):
        observed = results[f'{target}_obs'].sel(time_step=time_step).squeeze()
        valid_dates = pd.to_datetime(results['date'].values) + pd.Timedelta(
            days=time_step - 1
        )
        expected = discharge.reindex(valid_dates).values
        assert observed.values == approx(expected, nan_ok=True)


def get_test_start_end_dates(
    config: Config,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    test_start_date = pd.to_datetime(
        config.test_start_date[0], format='%d/%m/%Y'
    )
    test_end_date = pd.to_datetime(
        config.test_end_date[0], format='%d/%m/%Y'
    ) + pd.Timedelta(days=1, seconds=-1)

    return test_start_date, test_end_date


def get_basin_results(run_dir: Path, epoch: int) -> xr.Dataset:
    epoch_str = f'model_epoch{str(epoch).zfill(3)}'
    target_path = run_dir / 'test' / epoch_str / 'test_results.zarr'
    if not target_path.exists():
        matches = list(run_dir.glob(f'**/{epoch_str}/test_results.zarr'))
        if not matches:
            matches = list(run_dir.glob('**/test_results.zarr'))
        if len(matches) != 1:
            pytest.fail(f'Results file not found in {run_dir}.')
        target_path = matches[0]
    with xr.open_zarr(str(target_path), consolidated=False) as ds:
        return ds.load()
