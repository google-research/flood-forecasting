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

from pathlib import Path
from typing import Union, Dict, Callable

import pytest

from googlehydrology.utils.config import Config
from test import Fixture


def pytest_addoption(parser):
    parser.addoption('--smoke-test',
                     action='store_true',
                     default=False,
                     help='Skips some tests for faster execution. Out of the single-timescale '
                     'models and forcings, only test cudalstm on forcings that include daymet.')


@pytest.fixture
def get_config(tmpdir: Fixture[str]) -> Fixture[Callable[[str], dict]]:
    """Fixture that provides a function to fetch a run configuration specified by its name.

    The fetched run configuration will use a tmp folder as its run directory.

    Parameters
    ----------
    tmpdir : Fixture[str]
        Name of the tmp directory to use in the run configuration.

    Returns
    -------
    Fixture[Callable[[str], dict]]
        Function that returns a run configuration.
    """

    def _get_config(name):
        config_file = Path(f'./test/test_configs/{name}.test.yml')
        if not config_file.is_file():
            raise ValueError(f'Test config file not found at {config_file}.')
        config = Config(config_file)
        config.run_dir = Path(tmpdir)
        return config

    return _get_config


@pytest.fixture
def forecast_config_updates() -> Fixture[Callable[[str], dict]]:
    """Fixture that provides a function to update forecast model configs for model-specific parameters.

    Returns
    -------
    Fixture[Callable[[str], dict]]
        Function that returns an update dict.
    """
    def _forecast_config_updates(forecast_model):
        update_dict = {'model': forecast_model.lower()}
        if forecast_model.lower() == 'stacked_forecast_lstm':
            update_dict['forecast_overlap'] = 20
        if forecast_model.lower() == 'handoff_forecast_lstm':
            update_dict['forecast_overlap'] = 10
            update_dict['regularization'] = ['forecast_overlap']
        return update_dict
    
    return _forecast_config_updates


@pytest.fixture(params=['customlstm', 'ealstm', 'cudalstm', 'gru'])
def single_timescale_model(request) -> str:
    """Fixture that provides models that support predicting only a single timescale.

    Returns
    -------
    str
        Name of the single-timescale model.
    """
    if request.config.getoption('--smoke-test') and request.param != 'cudalstm':
        pytest.skip('--smoke-test skips this test.')
    return request.param


@pytest.fixture(params=['handoff_forecast_lstm', 'sequential_forecast_lstm',
                        'stacked_forecast_lstm', 'multihead_forecast_lstm'])
def forecast_model(request) -> str:
    """Fixture that provides models that support predicting only a single timescale.

    Returns
    -------
    str
        Name of the single-timescale model.
    """
    if request.config.getoption('--smoke-test') and request.param != 'handoff_forecast_lstm':
        pytest.skip('--smoke-test skips this test.')
    return request.param


@pytest.fixture(params=[('daymet', ['prcp(mm/day)', 'tmax(C)']), ('nldas', ['PRCP(mm/day)', 'Tmax(C)']),
                        ('maurer', ['PRCP(mm/day)', 'Tmax(C)']), ('maurer_extended', ['prcp(mm/day)', 'tmax(C)']),
                        (['daymet',
                          'nldas'], ['prcp(mm/day)_daymet', 'tmax(C)_daymet', 'PRCP(mm/day)_nldas', 'Tmax(C)_nldas'])],
                ids=lambda param: str(param[0]))
def single_timescale_forcings(request) -> Dict[str, Union[str, list[str]]]:
    """Fixture that provides daily forcings.

    Returns
    -------
    Dict[str, Union[str, list[str]]]
        Dictionary ``{'forcings': <name of the forcings set>, 'variables': <list of forcings variables>}``.
    """
    if request.config.getoption('--smoke-test') and 'daymet' not in request.param[0]:
        pytest.skip('--smoke-test skips this test.')
    return {'forcings': request.param[0], 'variables': request.param[1]}


@pytest.fixture(params=['mtslstm', 'odelstm'])
def multi_timescale_model(request) -> str:
    """Fixture that provides multi-timescale models.

    Returns
    -------
    str
        Name of the multi-timescale model.
    """
    return request.param


@pytest.fixture(params=[('camels_us', ['QObs(mm/d)'])], ids=lambda param: param[0])
def daily_dataset(request) -> Dict[str, list[str]]:
    """Fixture that provides daily datasets.

    Returns
    -------
    Dict[str, list[str]]
        Dictionary ``{'dataset: <name of the dataset>, 'target': <list of target variables>}``.
    """
    if request.config.getoption('--smoke-test') and request.param[0] != 'camels_us':
        pytest.skip('--smoke-test skips this test.')
    return {'dataset': request.param[0], 'target': request.param[1]}


@pytest.fixture(params=["cudalstm"])
def custom_lstm_supported_models(request) -> str:
    """Fixture that provides the models that are supported to be copied into the `CustomLSTM`.

    Returns
    -------
    str
        Name of the model.
    """
    return request.param
