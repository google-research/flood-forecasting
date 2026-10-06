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

"""Tests for the data assimilation config and its run-config integration."""

# ruff: noqa: PLR2004  # literal expected values are clearer in tests

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from googlehydrology.training import get_loss_obj, get_optimizer
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.utils.config import Config

_GetConfig = Callable[[str], Config]

# Keys the loss reads; given explicitly when no parent run config is used.
_PARENT = {
    'seq_length': 30,
    'predict_last_n': 8,
    'target_variables': ['streamflow'],
}
_N_DISTRIBUTIONS = 3


def _da_dict(**overrides: object) -> dict:
    cfg = {
        'assimilation_components': ['static_embedding', 'hindcast_embedding'],
        'assimilation_window': 10,
        'initial_learning_rate': 0.01,
    }
    cfg.update(overrides)
    return cfg


def _acfg(**overrides: object) -> AssimilationConfig:
    return AssimilationConfig(_da_dict(**overrides), parent_cfg=_PARENT)


def _opts(weight: float, lr: float | None = None) -> dict:
    """Build the normalized options of one assimilation component."""
    return {'regularization_weight': weight, 'initial_learning_rate': lr}


def _run_config(get_config: _GetConfig, **da_overrides: object) -> Config:
    """Forecast test config (tmp run_dir) with a DA block added."""
    cfg = get_config('forecast')
    cfg.update_config({'assimilation_config': _da_dict(**da_overrides)})
    return cfg


@pytest.mark.unit
def test_defaults() -> None:
    """A minimal DA config is a Config with the documented defaults."""
    acfg = _acfg()

    assert isinstance(acfg, Config)
    assert acfg.assimilation_window == 10
    assert acfg.initial_learning_rate == 0.01
    assert acfg.epochs == 100
    assert acfg.regularization_weight == 0.0
    assert acfg.optimizer == 'Adam'
    assert acfg.loss == 'MSE'
    assert acfg.clip_gradient_norm is None
    assert acfg.early_stopping_tolerance is None
    assert acfg.regularization == ['bg_embedding']
    assert acfg.lead_time == 0
    assert acfg.seq_length == 30
    # predict_last_n is the window, not the parent's training value (8).
    assert acfg.predict_last_n == acfg.assimilation_window == 10


@pytest.mark.unit
@pytest.mark.parametrize(
    ('overrides', 'attr', 'expected'),
    [
        ({'epochs': 5}, 'epochs', 5),
        ({'optimizer': 'SGD'}, 'optimizer', 'SGD'),
        ({'loss': 'nse'}, 'loss', 'nse'),
        ({'loss': 'CMAL'}, 'loss', 'CMAL'),
        ({'loss': 'cmal'}, 'loss', 'cmal'),
        ({'clip_gradient_norm': 1}, 'clip_gradient_norm', 1),
        ({'early_stopping_tolerance': 0.05}, 'early_stopping_tolerance', 0.05),
        ({'regularization_weight': 0.5}, 'regularization_weight', 0.5),
        ({'assimilation_window': 12}, 'predict_last_n', 12),
    ],
)
def test_optional_keys(overrides: dict, attr: str, expected: object) -> None:
    """Optional DA keys are exposed as given."""
    assert getattr(_acfg(**overrides), attr) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ('components', 'expected'),
    [
        # List form: every component gets the top-level defaults.
        (
            ['static_embedding', 'hindcast_embedding'],
            {
                'static_embedding': _opts(0.5),
                'hindcast_embedding': _opts(0.5),
            },
        ),
        # Dict form: per-component options override the defaults, and
        # integer options (YAML ``1``) are normalized to float.
        (
            {
                'static_embedding': {
                    'regularization_weight': 2,
                    'initial_learning_rate': 1,
                },
                'hindcast_embedding': None,
                'forecast_embedding': {'initial_learning_rate': 0.2},
            },
            {
                'static_embedding': _opts(2.0, 1.0),
                'hindcast_embedding': _opts(0.5),
                'forecast_embedding': _opts(0.5, 0.2),
            },
        ),
    ],
)
def test_components_are_normalized(
    components: list | dict, expected: dict
) -> None:
    """Both component forms map every component to float options."""
    acfg = _acfg(regularization_weight=0.5, assimilation_components=components)
    normalized = acfg.assimilation_components

    assert normalized == expected
    assert all(
        v is None or isinstance(v, float)
        for opts in normalized.values()
        for v in opts.values()
    )


@pytest.mark.unit
def test_components_are_copied() -> None:
    """Mutating the returned components does not affect the config."""
    acfg = _acfg()
    components = acfg.assimilation_components
    components['static_embedding']['regularization_weight'] = 9

    assert acfg.assimilation_components['static_embedding'] == _opts(0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('window', 'parent'),
    [
        (23, {'seq_length': 30, 'lead_time': 7}),
        (30, {'seq_length': 30}),
        (1000, None),
        (1000, {'lead_time': 7}),
    ],
)
def test_window_within_observed_steps_accepted(
    window: int, parent: dict | None
) -> None:
    """Windows up to seq_length - lead_time (any, without seq_length) pass."""
    acfg = AssimilationConfig(_da_dict(assimilation_window=window), parent)

    assert acfg.assimilation_window == window


@pytest.mark.unit
@pytest.mark.parametrize(
    ('window', 'parent'),
    [
        (1000, {'seq_length': 30, 'lead_time': 7}),
        (24, {'seq_length': 30, 'lead_time': 7}),
        (31, {'seq_length': 30}),
    ],
)
def test_window_exceeding_observed_steps_rejected(
    window: int, parent: dict
) -> None:
    """Windows above seq_length - lead_time raise, naming the observed count."""
    with pytest.raises(ValueError, match='observed target steps'):
        AssimilationConfig(_da_dict(assimilation_window=window), parent)


_TOP_LEVEL_INVALID = [
    ({'assimilation_targets': ['x']}, r"'assimilation_targets'.*Allowed"),
    ({'regularization': ['bg_embedding']}, 'not recognized'),
    ({'assimilate': True}, 'not recognized'),
    ({'seq_length': 10}, 'not recognized'),
    ({'learning_rate_strategy': 'ConstantLR'}, 'not recognized'),
    ({'assimilation_lead_time': 1}, 'not recognized'),
    ({'assimilation_window': 0}, 'positive integer'),
    ({'assimilation_window': None}, 'positive integer'),
    ({'assimilation_window': 2.5}, 'positive integer'),
    ({'initial_learning_rate': 0}, 'initial_learning_rate'),
    ({'initial_learning_rate': None}, 'initial_learning_rate'),
    ({'initial_learning_rate': float('nan')}, 'finite'),
    ({'initial_learning_rate': float('inf')}, 'finite'),
    ({'epochs': -1}, 'epochs'),
    ({'regularization_weight': -1.0}, 'regularization_weight'),
    ({'regularization_weight': float('nan')}, 'finite'),
    ({'regularization_weight': float('inf')}, 'finite'),
    ({'optimizer': 1}, 'optimizer'),
    # RMSE is excluded: its batch-level sqrt couples the sequences' losses.
    ({'loss': 'RMSE'}, 'not supported'),
    ({'loss': 'CMALLoss'}, 'not supported'),
    ({'loss': None}, 'not supported'),
    ({'clip_gradient_norm': 0}, 'clip_gradient_norm'),
    ({'clip_gradient_norm': float('nan')}, 'finite'),
    ({'clip_gradient_norm': float('inf')}, 'finite'),
    ({'early_stopping_tolerance': 0}, 'early_stopping_tolerance'),
    ({'early_stopping_tolerance': -0.1}, 'early_stopping_tolerance'),
    ({'early_stopping_tolerance': 'x'}, 'early_stopping_tolerance'),
    ({'early_stopping_tolerance': float('nan')}, 'finite'),
    ({'early_stopping_tolerance': float('inf')}, 'finite'),
]
# Rows are (assimilation_components value, match).
_COMPONENT_INVALID = [
    ([], 'assimilation_components'),
    ({}, 'assimilation_components'),
    (None, 'assimilation_components'),
    (3, 'assimilation_components'),
    ({'a': 'x'}, 'assimilation_components'),
    ({'a': {'lr': 1}}, 'lr'),
    ({'a': {'regularization_weight': -1}}, 'regularization_weight'),
    ({'a': {'regularization_weight': True}}, 'regularization_weight'),
    ({'a': {'regularization_weight': float('inf')}}, 'regularization_weight'),
    ({'a': {'initial_learning_rate': 0}}, 'initial_learning_rate'),
    ({'a': {'initial_learning_rate': '0.5'}}, 'initial_learning_rate'),
    ({'a': {'initial_learning_rate': float('nan')}}, 'initial_learning_rate'),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        *_TOP_LEVEL_INVALID,
        *[
            ({'assimilation_components': value}, match)
            for value, match in _COMPONENT_INVALID
        ],
    ],
)
def test_invalid_values_rejected(overrides: dict, match: str) -> None:
    """Unknown keys and invalid values raise a ValueError."""
    with pytest.raises(ValueError, match=match):
        _acfg(**overrides)


@pytest.mark.unit
def test_missing_learning_rate_is_not_inherited() -> None:
    """The DA learning rate must be given even if the parent has one."""
    da_cfg = _da_dict()
    del da_cfg['initial_learning_rate']

    with pytest.raises(ValueError, match='initial_learning_rate'):
        AssimilationConfig(
            da_cfg,
            parent_cfg={**_PARENT, **da_cfg, 'initial_learning_rate': 0.1},
        )


@pytest.mark.unit
def test_run_config_inherits_keys(get_config: _GetConfig) -> None:
    """Run-config keys are inherited, except the training and DA keys."""
    cfg = _run_config(get_config)
    cfg.update_config({'loss': 'CMALLoss', 'head': 'cmal'})
    acfg = cfg.assimilation_config

    assert isinstance(acfg, AssimilationConfig)
    assert acfg.lead_time == cfg.lead_time == 7
    assert acfg.seq_length == cfg.seq_length == 30
    assert acfg.target_variables == cfg.target_variables == ['streamflow']
    assert acfg.head == cfg.head == 'cmal'
    assert cfg.predict_last_n == 8
    assert acfg.predict_last_n == acfg.assimilation_window == 10
    # Training keys with DA-specific values are not inherited.
    assert (cfg.clip_gradient_norm, acfg.clip_gradient_norm) == (1, None)
    assert (cfg.loss, acfg.loss) == ('CMALLoss', 'MSE')
    assert (cfg.epochs, acfg.epochs) == (1, 100)
    # The run config keeps the user-given dict unchanged.
    assert 'seq_length' not in cfg.as_dict()['assimilation_config']


@pytest.mark.unit
def test_run_config_da_keys_take_precedence(get_config: _GetConfig) -> None:
    """DA keys override the run config; DA defaults apply when not given."""
    cfg = _run_config(get_config, epochs=5, optimizer='SGD')
    acfg = cfg.assimilation_config

    assert (acfg.epochs, cfg.epochs) == (5, 1)
    assert (acfg.optimizer, cfg.optimizer) == ('SGD', 'Adam')
    assert (acfg.initial_learning_rate, cfg.initial_learning_rate) == (
        0.01,
        0.001,
    )
    assert (acfg.regularization, cfg.regularization) == (['bg_embedding'], [])


@pytest.mark.unit
def test_run_config_assimilate_flag(get_config: _GetConfig) -> None:
    """Without a DA block there is no DA config; the flag can be set."""
    cfg = get_config('forecast')

    assert cfg.assimilation_config is None
    assert cfg.assimilate is False

    cfg.assimilate = True

    assert cfg.assimilate is True
    assert cfg.as_dict()['assimilate'] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ('da_block', 'match'),
    [
        (['x'], 'must be a dict'),
        ('not a dict', 'must be a dict'),
        (_da_dict(assimilation_window=0), 'positive integer'),
    ],
)
def test_run_config_invalid_block_raises_on_access(
    get_config: _GetConfig, da_block: object, match: str
) -> None:
    """An invalid DA block raises on access, not on run-config creation."""
    cfg = get_config('forecast')
    cfg.update_config({'assimilation_config': da_block})

    with pytest.raises(ValueError, match=match):
        _ = cfg.assimilation_config


@pytest.mark.unit
def test_run_config_assimilation_config_reflects_changes(
    get_config: _GetConfig,
) -> None:
    """The DA config is rebuilt on access and follows config updates."""
    cfg = _run_config(get_config)
    assert cfg.assimilation_config.seq_length == 30

    cfg.update_config({'seq_length': 20})
    assert cfg.assimilation_config.seq_length == 20

    cfg.update_config({'seq_length': 12})
    with pytest.raises(ValueError, match='observed target steps'):
        _ = cfg.assimilation_config


@pytest.mark.unit
def test_dump_config_round_trip(get_config: _GetConfig, tmp_path: Path) -> None:
    """The DA block survives dump_config and reload unchanged."""
    cfg = _run_config(
        get_config,
        assimilation_components={
            'hindcast_embedding': {'regularization_weight': 0.3}
        },
    )
    cfg.assimilate = True
    cfg.dump_config(tmp_path)

    reloaded = Config(tmp_path / 'config.yml')

    assert reloaded.assimilate is True
    assert (
        reloaded.as_dict()['assimilation_config']
        == cfg.as_dict()['assimilation_config']
    )
    assert reloaded.assimilation_config.assimilation_components == {
        'hindcast_embedding': _opts(0.3)
    }
    assert reloaded.assimilation_config.lead_time == 7


def _prediction(head: str, out: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map raw model outputs to the prediction dict of the given head."""
    if head == 'regression':
        return {'y_hat': out}
    # CMAL: four tensors of shape [B, T, n_targets * n_distributions] with
    # scale b > 0, asymmetry tau in (0, 1) and mixture weights pi summing to 1.
    mu, b, tau, pi = out.chunk(4, dim=-1)
    return {
        'mu': mu,
        'b': torch.nn.functional.softplus(b) + 1e-3,
        'tau': torch.sigmoid(tau),
        'pi': torch.softmax(pi, dim=-1),
    }


@pytest.mark.unit
@pytest.mark.parametrize('optimizer_name', ['SGD', 'Adam'])
@pytest.mark.parametrize(
    ('loss_name', 'head', 'n_outputs'),
    [
        ('MSE', 'regression', 1),
        ('NSE', 'regression', 1),
        ('cmal', 'cmal', 4 * _N_DISTRIBUTIONS),
    ],
)
def test_factories_run_forward_backward_step(
    get_config: _GetConfig,
    loss_name: str,
    head: str,
    n_outputs: int,
    optimizer_name: str,
) -> None:
    """Loss and optimizer built from the DA config complete one update."""
    # NOTE: get_regularization_obj(acfg) is not exercised here because the
    # 'bg_embedding' regularization term lands in a separate PR.
    cfg = _run_config(
        get_config,
        loss=loss_name,
        optimizer=optimizer_name,
        initial_learning_rate=0.5,
    )
    # n_distributions is only read by the CMAL loss.
    cfg.update_config({'head': head, 'n_distributions': _N_DISTRIBUTIONS})
    acfg = cfg.assimilation_config
    loss_fn = get_loss_obj(acfg)
    model = torch.nn.Linear(3, n_outputs)
    optimizer = get_optimizer(model, acfg)
    before = model.weight.detach().clone()

    # The loss evaluates the last predict_last_n == assimilation_window steps,
    # so it is given observed-period tensors (forecast horizon dropped).
    n_observed = acfg.seq_length - acfg.lead_time
    assert acfg.predict_last_n == acfg.assimilation_window <= n_observed
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, n_observed, 3, generator=gen)
    data = {
        'y': torch.randn(2, n_observed, 1, generator=gen),
        'per_basin_target_stds': torch.rand(2, 1, 1, generator=gen) + 0.5,
    }
    total_loss, _ = loss_fn(_prediction(head, model(x)), data)
    total_loss.backward()
    optimizer.step()

    assert optimizer.param_groups[0]['lr'] == 0.5  # the DA lr, not 0.001
    assert torch.isfinite(total_loss)
    assert torch.isfinite(model.weight.grad).all()
    assert not torch.equal(model.weight, before)


@pytest.mark.unit
def test_cmal_loss_requires_n_distributions_from_run_config(
    get_config: _GetConfig,
) -> None:
    """A CMAL DA loss needs n_distributions on the run config."""
    acfg = _run_config(get_config, loss='CMAL').assimilation_config

    assert acfg.loss == 'CMAL'
    with pytest.raises(ValueError, match='n_distributions'):
        get_loss_obj(acfg)


@pytest.mark.unit
def test_assimilation_example_config_parses() -> None:
    """The example DA config parses; DA is off by default."""
    path = (
        Path(__file__).parent.parent
        / 'example-configs'
        / 'camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml'
    )
    cfg = Config(path)
    assert not cfg.assimilate
    acfg = cfg.assimilation_config
    assert set(acfg.assimilation_components) == {
        'static_embedding',
        'hindcast_embedding',
        'forecast_embedding',
    }
    assert acfg.lead_time == cfg.lead_time
    assert acfg.seq_length == cfg.seq_length
    assert acfg.assimilation_window <= cfg.seq_length - cfg.lead_time
    assert acfg.early_stopping_tolerance == 0.05
    # The DA loss is NSE while training uses CMAL: the datasets provide
    # `per_basin_target_stds` for either (see test_multimet).
    assert acfg.loss == 'NSE'
    assert cfg.loss == 'CMALLoss'
