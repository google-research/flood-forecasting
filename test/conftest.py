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

import gc
import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pytest

from googlehydrology.utils.config import Config
from test import Fixture


@pytest.fixture(autouse=True)
def cleanup_resources_after_test():
    """Closes all logging handlers, open matplotlib figures, and forces GC."""
    yield
    # Close & remove FileHandlers from root logger to avoid Windows file locks.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)
    # Close any open matplotlib figures
    plt.close('all')
    # Force garbage collection to release unreferenced file descriptors
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        '--smoke-test',
        action='store_true',
        default=False,
        help=(
            'Skips some tests for faster execution. Out of single-timescale '
            'models/forcings, only test cudalstm on daymet.'
        ),
    )


@pytest.fixture
def get_config(tmpdir: Fixture[str]) -> Fixture[Callable[[str], dict]]:
    """Provides a function to fetch a run config specified by name.

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
        config_file = (
            Path(__file__).parent / 'test_configs' / f'{name}.test.yml'
        )
        if not config_file.is_file():
            raise ValueError(f'Test config file not found at {config_file}.')
        config = Config(config_file)
        config.run_dir = Path(tmpdir)
        return config

    return _get_config


@pytest.fixture
def forecast_config_updates() -> Fixture[Callable[[str], dict]]:
    """Provides a function to update forecast model configs.

    Returns
    -------
    Fixture[Callable[[str], dict]]
        Function that returns an update dict.
    """

    def _forecast_config_updates(forecast_model):
        update_dict = {'model': forecast_model.lower()}
        if forecast_model.lower() == 'handoff_forecast_lstm':
            update_dict['forecast_overlap'] = 10
            update_dict['regularization'] = ['forecast_overlap']
        if forecast_model.lower() == 'mean_embedding_forecast_lstm':
            update_dict['forecast_overlap'] = 30
            update_dict['hindcast_inputs'] = [
                'era5land_total_precipitation',
                'graphcast_total_precipitation',
            ]
        return update_dict

    return _forecast_config_updates


@pytest.fixture(
    params=['handoff_forecast_lstm', 'mean_embedding_forecast_lstm']
)
def forecast_model(request) -> str:
    """Fixture that provides single-timescale forecast models.

    Returns
    -------
    str
        Name of the single-timescale model.
    """
    if (
        request.config.getoption('--smoke-test')
        and request.param != 'handoff_forecast_lstm'
    ):
        pytest.skip('--smoke-test skips this test.')
    return request.param


@pytest.fixture(params=[True, False])
def lazy_load(request) -> bool:
    return request.param


@pytest.fixture(
    params=[
        ('daymet', ['prcp(mm/day)', 'tmax(C)']),
        ('nldas', ['PRCP(mm/day)', 'Tmax(C)']),
        ('maurer', ['PRCP(mm/day)', 'Tmax(C)']),
        ('maurer_extended', ['prcp(mm/day)', 'tmax(C)']),
        (
            ['daymet', 'nldas'],
            [
                'prcp(mm/day)_daymet',
                'tmax(C)_daymet',
                'PRCP(mm/day)_nldas',
                'Tmax(C)_nldas',
            ],
        ),
    ],
    ids=lambda param: str(param[0]),
)
def single_timescale_forcings(request) -> dict[str, str | list[str]]:
    """Fixture that provides daily forcings.

    Returns
    -------
    dict[str, str | list[str]]
        Dict ``{'forcings': <name>, 'variables': <list of variables>}``.
    """
    if (
        request.config.getoption('--smoke-test')
        and 'daymet' not in request.param[0]
    ):
        pytest.skip('--smoke-test skips this test.')
    return {'forcings': request.param[0], 'variables': request.param[1]}


@pytest.fixture(
    params=[('camels_us', ['QObs(mm/d)'])], ids=lambda param: param[0]
)
def daily_dataset(request) -> dict[str, list[str]]:
    """Fixture that provides daily datasets.

    Returns
    -------
    dict[str, list[str]]
        Dict ``{'dataset: <name>, 'target': <list of target variables>}``.
    """
    if (
        request.config.getoption('--smoke-test')
        and request.param[0] != 'camels_us'
    ):
        pytest.skip('--smoke-test skips this test.')
    return {'dataset': request.param[0], 'target': request.param[1]}
