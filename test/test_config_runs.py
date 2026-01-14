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
from pandas.tseries.frequencies import to_offset
from pytest import approx

from googlehydrology.datasetzoo import caravan
from googlehydrology.evaluation.evaluate import start_evaluation
from googlehydrology.training.train import start_training
from googlehydrology.utils.config import Config
from test import Fixture


def test_forecast_daily_regression(
    get_config: Fixture[Callable[[str], dict]],
    forecast_model: Fixture[str],
    forecast_config_updates: Fixture[Callable[[str], dict]],
    lazy_data: Fixture[bool],
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
    config.lazy_data = lazy_data

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

    Checks that the results file has the correct date range, that the observed discharge in the file is correct, and
    that there are no NaN predictions.

    Parameters
    ----------
    config : Config
        The run configuration used to produce the results
    basin : str
        Id of a basin for which to check the results
    discharge : pd.Series, optional
        If provided, will check that the stored discharge obs match this series. Else, will compare to the discharge
        loaded from disk.
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

    if discharge is None:
        discharge = caravan.load_caravan_timeseries_together(
            config.data_dir, [basin], config.target_variables, csv=False
        ).to_dataframe()

    if hasattr(config, 'lead_time'):
        results = results.isel(time_step=0).squeeze()
    else:
        results = results.isel(time_step=-1)

    results_array = results[f'{config.target_variables[0]}_obs'].values
    idx = pd.IndexSlice
    discharge_slice = discharge.loc[idx[basin, test_start_date:test_end_date], 'streamflow']
    discharge_array = discharge_slice.values

    assert discharge_array == approx(results_array, nan_ok=True)

    # CAMELS forcings have no NaNs, so there should be no NaN predictions
    assert not pd.isna(results[f'{config.target_variables[0]}_sim']).any()


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
    results_file = list(
        run_dir.glob(f'test/model_epoch{str(epoch).zfill(3)}/test_results.zarr')
    )
    if len(results_file) != 1:
        pytest.fail('Results file not found.')
    return xr.open_zarr(str(results_file[0]), consolidated=False)
