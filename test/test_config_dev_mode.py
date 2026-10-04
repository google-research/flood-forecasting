# Copyright 2026 Google LLC
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

"""Explicitly disabling development mode must retain strict config checks."""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from ruamel.yaml import YAML

from googlehydrology.utils.config import Config

HIDDEN_SIZE = 8
SEED = 42


def _input(
    tmp_path: Path,
    source: str,
    values: dict,
    filename: str = 'input.yml',
) -> Path | dict:
    """Exercise both dictionary input and the real YAML parser."""
    if source == 'dict':
        return deepcopy(values)
    path = tmp_path / filename
    with path.open('w') as stream:
        YAML().dump(values, stream)
    return path


@pytest.mark.unit
@pytest.mark.parametrize('source', ['dict', 'yaml'])
def test_explicit_false_keeps_normal_parsing(
    tmp_path: Path, source: str
) -> None:
    """Accept False without changing path, date, or normal option parsing."""
    values = {
        'dev_mode': False,
        'hidden_size': HIDDEN_SIZE,
        'seed': SEED,
        'train_start_date': '01/02/2020',
        'data_dir': str(tmp_path / 'data'),
        'package_version': 'fixture',
        'commit_hash': 'fixture',
    }
    cfg = Config(_input(tmp_path, source, values))
    assert cfg.as_dict()['dev_mode'] is False
    assert cfg.hidden_size == HIDDEN_SIZE
    assert cfg.seed == SEED
    assert cfg.train_start_date == [pd.Timestamp('2020-02-01')]
    assert cfg.data_dir == tmp_path / 'data'
    without_flag = {
        key: value for key, value in values.items() if key != 'dev_mode'
    }
    expected = Config(_input(tmp_path, source, without_flag, 'without.yml'))
    actual = cfg.as_dict().copy()
    actual.pop('dev_mode')
    assert actual == expected.as_dict()


@pytest.mark.unit
@pytest.mark.parametrize('source', ['dict', 'yaml'])
@pytest.mark.parametrize('explicit_false', [False, True])
def test_strict_mode_rejects_unknown_keys(
    tmp_path: Path,
    source: str,
    *,
    explicit_false: bool,
) -> None:
    """A disabled flag is recognized, while a misspelled setting still fails."""
    values = {'hidden_size': HIDDEN_SIZE, 'hiddden_size': 9}
    if explicit_false:
        values['dev_mode'] = False
    with pytest.raises(ValueError, match='hiddden_size') as error:
        Config(_input(tmp_path, source, values))
    assert (
        str(error.value) == "['hiddden_size'] are not recognized config keys."
    )


@pytest.mark.unit
@pytest.mark.parametrize('source', ['dict', 'yaml'])
@pytest.mark.parametrize(
    ('stored_flag', 'constructor_flag'),
    [(True, False), (False, True), (True, True)],
)
def test_enabled_development_mode_keeps_existing_precedence(
    tmp_path: Path,
    source: str,
    *,
    stored_flag: bool,
    constructor_flag: bool,
) -> None:
    """Either existing opt-in mechanism can still permit extension keys."""
    values = {'dev_mode': stored_flag, 'custom_extension': 42}
    cfg = Config(_input(tmp_path, source, values), dev_mode=constructor_flag)
    assert cfg.as_dict() == values


@pytest.mark.unit
@pytest.mark.parametrize('source', ['dict', 'yaml'])
def test_updates_can_explicitly_disable_development_mode(
    tmp_path: Path,
    source: str,
) -> None:
    """Use the real update path without changing unmentioned settings."""
    cfg = Config({'hidden_size': 16, 'seed': SEED, 'dev_mode': True})
    cfg.update_config(
        _input(
            tmp_path, source, {'hidden_size': HIDDEN_SIZE, 'dev_mode': False}
        )
    )
    assert cfg.hidden_size == HIDDEN_SIZE
    assert cfg.seed == SEED
    assert cfg.as_dict()['dev_mode'] is False
    before = deepcopy(cfg.as_dict())
    with pytest.raises(ValueError, match='custom_extension'):
        cfg.update_config({'custom_extension': 1})
    assert cfg.as_dict() == before


@pytest.mark.unit
@pytest.mark.parametrize('source', ['dict', 'yaml'])
def test_invalid_updates_do_not_change_existing_values(
    tmp_path: Path,
    source: str,
) -> None:
    """A failed strict update must not partially change the live config."""
    cfg = Config({'hidden_size': 16, 'seed': SEED})
    before = deepcopy(cfg.as_dict())
    values = {
        'hidden_size': HIDDEN_SIZE,
        'dev_mode': False,
        'unknown_option': 1,
    }
    with pytest.raises(ValueError, match='unknown_option'):
        cfg.update_config(_input(tmp_path, source, values))
    assert cfg.as_dict() == before


@pytest.mark.unit
@pytest.mark.parametrize('source', ['dict', 'yaml'])
def test_disabled_flag_survives_dump_and_reload(
    tmp_path: Path, source: str
) -> None:
    """Reload a saved False flag without bypassing config checks."""
    values = {
        'dev_mode': False,
        'hidden_size': HIDDEN_SIZE,
        'train_start_date': '01/02/2020',
        'run_dir': str(tmp_path / 'run'),
    }
    # The explicit constructor override also permits creation on old code.
    cfg = Config(_input(tmp_path, source, values), dev_mode=True)
    cfg.dump_config(tmp_path, 'saved.yml')
    restored = Config(tmp_path / 'saved.yml')
    assert restored.as_dict() == cfg.as_dict()
    assert restored.as_dict()['dev_mode'] is False
    original_bytes = (tmp_path / 'saved.yml').read_bytes()
    with pytest.raises(FileExistsError):
        restored.dump_config(tmp_path, 'saved.yml')
    assert (tmp_path / 'saved.yml').read_bytes() == original_bytes


@pytest.mark.unit
@pytest.mark.parametrize('literal', ['false', 'False', 'FALSE'])
def test_yaml_false_spellings_remain_booleans(
    tmp_path: Path, literal: str
) -> None:
    """Load standard YAML boolean spellings through Config itself."""
    path = tmp_path / 'config.yml'
    path.write_text(f'dev_mode: {literal}\nhidden_size: 8\n')
    cfg = Config(path)
    assert cfg.as_dict()['dev_mode'] is False
