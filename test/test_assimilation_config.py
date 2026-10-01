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

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from googlehydrology.training import get_loss_obj, loss
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.utils.config import Config

_TEST_CONFIG = Path(__file__).parent / 'test_configs' / 'forecast.test.yml'


def _da_dict(**overrides) -> dict:
    cfg = {
        'assimilation_components': ['static_embedding', 'hindcast_embedding'],
        'assimilation_window': 10,
        'initial_learning_rate': 0.01,
        'seq_length': 30,
        'predict_last_n': 8,
        'target_variables': ['streamflow'],
    }
    cfg.update(overrides)
    return cfg


def _run_config(**da_overrides) -> Config:
    yaml = YAML(typ='safe')
    with _TEST_CONFIG.open() as fp:
        cfg = yaml.load(fp)
    cfg['model'] = 'handoff_forecast_lstm'
    cfg['assimilation_config'] = {
        'assimilation_components': ['hindcast_embedding'],
        'assimilation_window': 10,
        'initial_learning_rate': 0.01,
        **da_overrides,
    }
    return Config(cfg)


def test_valid_dict_defaults():
    acfg = AssimilationConfig(_da_dict())

    assert acfg.assimilation_window == 10
    assert acfg.assimilation_lead_time == 0
    assert acfg.epochs == 100
    assert acfg.regularization_weight == 0.0
    assert acfg.initial_learning_rate == 0.01
    assert acfg.optimizer == 'Adam'
    assert acfg.loss == 'MSE'
    assert acfg.clip_gradient_norm is None
    assert acfg.early_stopping_tolerance is None
    assert acfg.target_loss_weights is None
    assert acfg.no_loss_frequencies == []
    assert acfg.regularization == ['bg_embedding']


def test_input_dict_is_not_modified():
    d = _da_dict(assimilation_components={'a': {'regularization_weight': 2}})
    acfg = AssimilationConfig(d)
    acfg.as_dict()['assimilation_components']['a']['regularization_weight'] = 0
    acfg.assimilation_components['a']['regularization_weight'] = 0

    assert d['assimilation_components'] == {'a': {'regularization_weight': 2}}
    assert acfg.assimilation_components['a']['regularization_weight'] == 2.0


def test_from_yaml_path(tmp_path):
    path = tmp_path / 'da.yml'
    with path.open('w') as fp:
        YAML().dump(_da_dict(epochs=5), fp)

    acfg = AssimilationConfig(path)

    assert acfg.epochs == 5
    assert acfg.as_dict() == _da_dict(epochs=5)


def test_invalid_input_type():
    with pytest.raises(ValueError, match='Cannot create'):
        AssimilationConfig('not a dict')


def test_components_list_form_uses_defaults():
    acfg = AssimilationConfig(_da_dict(regularization_weight=0.5))

    assert acfg.assimilation_components == {
        'static_embedding': {
            'regularization_weight': 0.5,
            'initial_learning_rate': None,
        },
        'hindcast_embedding': {
            'regularization_weight': 0.5,
            'initial_learning_rate': None,
        },
    }


def test_components_dict_form_overrides_defaults():
    acfg = AssimilationConfig(
        _da_dict(
            regularization_weight=0.5,
            assimilation_components={
                'static_embedding': {
                    'regularization_weight': 2,
                    'initial_learning_rate': 0.1,
                },
                'hindcast_embedding': None,
                'forecast_embedding': {'initial_learning_rate': 0.2},
            },
        )
    )

    assert acfg.assimilation_components == {
        'static_embedding': {
            'regularization_weight': 2.0,
            'initial_learning_rate': 0.1,
        },
        'hindcast_embedding': {
            'regularization_weight': 0.5,
            'initial_learning_rate': None,
        },
        'forecast_embedding': {
            'regularization_weight': 0.5,
            'initial_learning_rate': 0.2,
        },
    }


@pytest.mark.parametrize('components', [[], {}, None])
def test_empty_components_rejected(components):
    with pytest.raises(ValueError, match='assimilation_components'):
        AssimilationConfig(_da_dict(assimilation_components=components))


def test_duplicate_components_rejected():
    with pytest.raises(ValueError, match='duplicates'):
        AssimilationConfig(_da_dict(assimilation_components=['a', 'a']))


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValueError, match=r"'assimilation_targets'.*Allowed"):
        AssimilationConfig(_da_dict(assimilation_targets=['x']))


@pytest.mark.parametrize('key', ['regularization', 'assimilate'])
def test_non_configurable_keys_rejected(key):
    with pytest.raises(ValueError, match='not recognized'):
        AssimilationConfig(_da_dict(**{key: ['bg_embedding']}))


def test_unknown_component_key_rejected():
    with pytest.raises(ValueError, match=r"'lr'.*static_embedding"):
        AssimilationConfig(
            _da_dict(assimilation_components={'static_embedding': {'lr': 1}})
        )


@pytest.mark.parametrize('loss_name', ['CMAL', 'CMALLoss', 'GMM'])
def test_unsupported_loss_rejected(loss_name):
    with pytest.raises(ValueError, match='not supported'):
        AssimilationConfig(_da_dict(loss=loss_name))


@pytest.mark.parametrize('loss_name', ['MSE', 'nse', 'RMSE'])
def test_supported_losses(loss_name):
    assert AssimilationConfig(_da_dict(loss=loss_name)).loss == loss_name


@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        ({'assimilation_lead_time': -1}, 'non-negative'),
        ({'assimilation_lead_time': 30}, 'smaller than seq_length'),
        ({'assimilation_lead_time': 25}, 'must not exceed'),
        ({'assimilation_window': 31}, 'must not exceed'),
        ({'assimilation_window': 0}, 'positive integer'),
        ({'assimilation_window': None}, 'mandatory'),
        ({'assimilation_window': 2.5}, 'positive integer'),
        ({'epochs': -1}, 'epochs'),
        ({'regularization_weight': -1.0}, 'regularization_weight'),
        ({'clip_gradient_norm': 0}, 'clip_gradient_norm'),
        ({'early_stopping_tolerance': 0}, 'early_stopping_tolerance'),
        ({'early_stopping_tolerance': -0.1}, 'early_stopping_tolerance'),
        ({'early_stopping_tolerance': 'x'}, 'early_stopping_tolerance'),
        ({'seq_length': {'1D': 30}}, 'single-frequency'),
        ({'seq_length': None}, 'seq_length'),
        ({'predict_last_n': None}, 'predict_last_n'),
        ({'target_variables': None}, 'target_variables'),
    ],
)
def test_invalid_values_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        AssimilationConfig(_da_dict(**overrides))


def test_window_and_lead_time_at_limit():
    acfg = AssimilationConfig(
        _da_dict(assimilation_window=20, assimilation_lead_time=10)
    )

    assert acfg.assimilation_window + acfg.assimilation_lead_time == 30


def test_early_stopping_tolerance():
    acfg = AssimilationConfig(_da_dict(early_stopping_tolerance=0.05))

    assert acfg.early_stopping_tolerance == 0.05


@pytest.mark.parametrize(
    'key',
    [
        'history',
        'learning_rate_strategy',
        'learning_rate_drop_factor',
        'learning_rate_epochs_drop',
    ],
)
def test_removed_keys_rejected(key):
    with pytest.raises(ValueError, match='not recognized'):
        AssimilationConfig(_da_dict(**{key: 1}))


@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        ({'initial_learning_rate': 0}, 'initial_learning_rate'),
        ({'initial_learning_rate': None}, 'initial_learning_rate'),
        (
            {'assimilation_components': {'a': {'initial_learning_rate': -1}}},
            'a.initial_learning_rate',
        ),
    ],
)
def test_invalid_learning_rate_keys(overrides, match):
    with pytest.raises(ValueError, match=match):
        AssimilationConfig(_da_dict(**overrides))


def test_get_loss_obj_accepts_assimilation_config():
    assert isinstance(
        get_loss_obj(AssimilationConfig(_da_dict())), loss.MaskedMSELoss
    )
    assert isinstance(
        get_loss_obj(AssimilationConfig(_da_dict(loss='NSE'))),
        loss.MaskedNSELoss,
    )


def test_run_config_without_assimilation_config():
    cfg = Config(_TEST_CONFIG)

    assert cfg.assimilation_config is None
    assert cfg.assimilate is False


def test_run_config_assimilate_setter():
    cfg = Config(_TEST_CONFIG)
    cfg.assimilate = True

    assert cfg.assimilate is True
    assert cfg.as_dict()['assimilate'] is True


def test_run_config_inherits_keys():
    cfg = _run_config()
    acfg = cfg.assimilation_config

    assert isinstance(acfg, AssimilationConfig)
    assert acfg.seq_length == cfg.seq_length == 30
    assert acfg.predict_last_n == cfg.predict_last_n == 8
    assert acfg.target_variables == cfg.target_variables == ['streamflow']
    # The run config keeps the user-given (non-inherited) dict.
    assert 'seq_length' not in cfg.as_dict()['assimilation_config']


def test_run_config_explicit_keys_take_precedence():
    cfg = _run_config(seq_length=20, predict_last_n=4)

    assert cfg.assimilation_config.seq_length == 20
    assert cfg.assimilation_config.predict_last_n == 4
    assert cfg.seq_length == 30


def test_run_config_validates_assimilation_config_lazily():
    # Like other run-config keys, the DA config is validated on access.
    cfg = _run_config(assimilation_lead_time=30)

    with pytest.raises(ValueError, match='smaller than seq_length'):
        _ = cfg.assimilation_config


def test_run_config_rejects_non_dict_assimilation_config():
    cfg = Config({'assimilation_config': ['x'], 'seq_length': 30})

    with pytest.raises(ValueError, match='must be a dict'):
        _ = cfg.assimilation_config


def test_run_config_assimilation_config_reflects_changes():
    cfg = _run_config()
    assert cfg.assimilation_config.assimilation_window == 10

    cfg.update_config({'seq_length': 12, 'predict_last_n': 4})

    assert cfg.assimilation_config.seq_length == 12
    assert cfg.assimilation_config.predict_last_n == 4
    cfg.update_config(
        {
            'assimilation_config': {
                'assimilation_components': ['hindcast_embedding'],
                'assimilation_window': 13,
                'initial_learning_rate': 0.01,
            }
        }
    )
    with pytest.raises(ValueError, match='must not exceed'):
        _ = cfg.assimilation_config


def test_update_config_adds_assimilation_config():
    cfg = Config(_TEST_CONFIG)
    cfg.update_config(
        {
            'assimilate': True,
            'assimilation_config': {
                'assimilation_components': ['hindcast_embedding'],
                'assimilation_window': 10,
                'initial_learning_rate': 0.01,
            },
        }
    )

    assert cfg.assimilate is True
    assert cfg.assimilation_config.seq_length == 30
    assert cfg.assimilation_config.assimilation_window == 10


def test_dump_config_round_trip(tmp_path):
    cfg = _run_config(
        assimilation_components={
            'hindcast_embedding': {'regularization_weight': 0.3}
        },
        assimilation_lead_time=8,
    )
    cfg.assimilate = True
    cfg.dump_config(tmp_path)

    reloaded = Config(tmp_path / 'config.yml')

    assert reloaded.assimilate is True
    assert (
        reloaded.as_dict()['assimilation_config']
        == cfg.as_dict()['assimilation_config']
    )
    assert (
        reloaded.assimilation_config.as_dict()
        == cfg.assimilation_config.as_dict()
    )
    assert reloaded.assimilation_config.assimilation_components == {
        'hindcast_embedding': {
            'regularization_weight': 0.3,
            'initial_learning_rate': None,
        }
    }
