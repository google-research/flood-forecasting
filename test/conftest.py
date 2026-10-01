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
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch._dynamo
import xarray as xr

from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config
from googlehydrology.utils.configutils import group_features_list
from test import Fixture
from test.test_hot_start import get_base_cfg

torch._dynamo.config.suppress_errors = True


def _cleanup_all_open_resources():
    """Closes all logging handlers, open matplotlib figures, and forces GC."""
    for handler in list(logging.root.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.root.removeHandler(handler)
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
    plt.close('all')
    try:
        from googlehydrology.datasetzoo.multimet import _open_zarr

        _open_zarr.cache_clear()
    except Exception:
        pass
    gc.collect()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Run resource cleanup before any fixtures are torn down."""
    _cleanup_all_open_resources()


@pytest.fixture(autouse=True)
def cleanup_resources_after_test():
    """Closes all logging handlers, open matplotlib figures, and forces GC."""
    yield
    _cleanup_all_open_resources()


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
def get_config(tmp_path: Fixture[Path]) -> Fixture[Callable[[str], dict]]:
    """Provides a function to fetch a run config specified by name.

    The fetched run configuration will use a tmp folder as its run directory.

    Parameters
    ----------
    tmp_path : Fixture[Path]
        Tmp directory to use in the run configuration.

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
        config.run_dir = tmp_path
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


@pytest.fixture
def tiny_mean_embedding_model(
    tmp_path: Fixture[Path],
) -> Fixture[Callable[..., MeanEmbeddingForecastLSTM]]:
    """Return a factory for a tiny, deterministic MeanEmbeddingForecastLSTM.

    The default inputs give one shared group (``pr``, ``tmmn``), one
    hindcast-only group (``streamflow``) and one forecast-only group
    (``hres``). The model is seeded (without touching the global RNG state)
    and in eval mode.

    Parameters
    ----------
    tmp_path : Fixture[Path]
        Tmp directory used as run directory (a dummy scaler is written there).

    Returns
    -------
    Fixture[Callable[..., MeanEmbeddingForecastLSTM]]
        Factory ``(seq_length=6, lead_time=3, hindcast_inputs=...,
        forecast_inputs=..., head='regression', n_distributions=3)
        -> MeanEmbeddingForecastLSTM``. ``n_distributions`` is only used by
        the ``'cmal'`` head.
    """

    def _build(  # noqa: PLR0913
        *,
        seq_length: int = 6,
        lead_time: int = 3,
        hindcast_inputs: tuple[str, ...] = ('pr_a', 'tmmn_a', 'streamflow_lag'),
        forecast_inputs: tuple[str, ...] = ('pr_a', 'tmmn_a', 'hres_precip'),
        head: str = 'regression',
        n_distributions: int = 3,
    ) -> MeanEmbeddingForecastLSTM:
        options = get_base_cfg(tmp_path)
        embedding = {
            'type': 'fc',
            'hiddens': [8],
            'activation': ['tanh'],
            'dropout': 0.0,
        }
        options.update(
            {
                'model': 'MeanEmbeddingForecastLSTM',
                'seq_length': seq_length,
                'lead_time': lead_time,
                'forecast_overlap': seq_length,
                'hidden_size': 8,
                'output_dropout': 0.0,
                'head': head,
                'n_distributions': n_distributions,
                'hindcast_inputs': list(hindcast_inputs),
                'forecast_inputs': list(forecast_inputs),
                'hindcast_embedding': embedding,
                'forecast_embedding': embedding,
                'statics_embedding': {**embedding, 'hiddens': [4]},
            }
        )
        cfg = Config(options)
        xr.Dataset(
            {'streamflow': ('parameter', [0.0, 1.0, 0.0, 1.0])},
            coords={'parameter': ['center', 'scale', 'mean', 'std']},
        ).to_netcdf(tmp_path / 'scaler.nc', engine='scipy')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = MeanEmbeddingForecastLSTM(cfg)
        model.eval()
        return model

    return _build


@pytest.fixture
def tiny_mean_embedding_data() -> Fixture[Callable[..., dict]]:
    """Return a factory for random inputs of a tiny MeanEmbeddingForecastLSTM.

    Returns
    -------
    Fixture[Callable[..., dict]]
        Factory ``(cfg, batch_size=2, nan_at=None) -> data`` where hindcast
        inputs cover ``cfg.seq_length`` steps and forecast inputs the full
        ``cfg.seq_length + cfg.lead_time`` span. Values are seeded from a
        private generator. ``nan_at`` maps an input group (e.g. ``'pr'``) or
        a single feature (e.g. ``'pr_a'``) to the time indices that are set
        to NaN in every matching hindcast and forecast feature; indices
        beyond a feature's length are ignored.
    """

    def _make(
        cfg: Config,
        *,
        batch_size: int = 2,
        nan_at: dict[str, list[int]] | None = None,
    ) -> dict:
        generator = torch.Generator().manual_seed(1)
        data = {
            'x_d_hindcast': {
                name: torch.rand(
                    batch_size, cfg.seq_length, 1, generator=generator
                )
                for name in cfg.hindcast_inputs
            },
            'x_d_forecast': {
                name: torch.rand(
                    batch_size,
                    cfg.seq_length + cfg.lead_time,
                    1,
                    generator=generator,
                )
                for name in cfg.forecast_inputs
            },
            'x_s': torch.rand(
                batch_size, len(cfg.static_attributes), generator=generator
            ),
        }
        for key, steps in (nan_at or {}).items():
            matched = False
            for inputs, features in (
                (cfg.hindcast_inputs, data['x_d_hindcast']),
                (cfg.forecast_inputs, data['x_d_forecast']),
            ):
                names = group_features_list(inputs).get(key, [])
                names = names or [name for name in inputs if name == key]
                for name in names:
                    matched = True
                    tensor = features[name]
                    for step in steps:
                        if step < tensor.shape[1]:
                            tensor[:, step] = float('nan')
            if not matched:
                msg = f'nan_at: {key!r} is neither an input group nor feature'
                raise ValueError(msg)
        return data

    return _make


def assert_finite_grads(
    tensors_or_module: torch.nn.Module | Iterable[torch.Tensor],
) -> None:
    """Assert every (populated) gradient is finite and at least one exists."""
    if isinstance(tensors_or_module, torch.nn.Module):
        tensors = list(tensors_or_module.parameters())
    else:
        tensors = list(tensors_or_module)
    grads = [t.grad for t in tensors if t.grad is not None]
    assert grads, 'no gradients were populated'
    for grad in grads:
        assert torch.isfinite(grad).all()


def assert_grad_matches_finite_difference(
    loss_fn: Callable[[], torch.Tensor],
    tensor: torch.Tensor,
    index: tuple[int, ...],
    eps: float = 1e-6,
    atol: float = 1e-5,
) -> None:
    """Assert ``tensor.grad[index]`` equals a central finite difference.

    ``tensor.grad`` must already be populated by a backward pass of the same
    loss. Use double precision for the default tolerances.
    """
    assert tensor.grad is not None, 'tensor has no gradient'
    with torch.no_grad():
        original = tensor[index].item()
        tensor[index] = original + eps
        plus = loss_fn().item()
        tensor[index] = original - eps
        minus = loss_fn().item()
        tensor[index] = original
    expected = (plus - minus) / (2 * eps)
    assert tensor.grad[index].item() == pytest.approx(expected, abs=atol)
