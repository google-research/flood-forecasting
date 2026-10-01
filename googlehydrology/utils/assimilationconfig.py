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

"""Configuration for gradient-based data assimilation (DA).

DA optimizes a set of model-internal components (e.g. embeddings) for each
forecast issue time so that the model output matches observations within an
assimilation window preceding the issue time. The optimized components are
regularized towards their background (un-assimilated) values.
"""

import copy
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

# Keys of a run config that the DA config inherits if not given explicitly.
INHERITED_KEYS = (
    'seq_length',
    'predict_last_n',
    'target_variables',
    'target_loss_weights',
    'no_loss_frequencies',
)


class AssimilationConfig:
    """Read and validate a data assimilation configuration.

    The DA config is usually given as the nested ``assimilation_config`` key of
    a run configuration (see :py:class:`googlehydrology.utils.config.Config`),
    which fills in the keys listed in ``INHERITED_KEYS`` from the run config if
    they are not specified explicitly. The object also exposes the attributes
    read by :py:func:`googlehydrology.training.get_loss_obj` and
    :py:func:`googlehydrology.training.get_regularization_obj`, so it can be
    passed to them in place of a run config.

    Parameters
    ----------
    yml_path_or_dict : Path | dict
        Either a path to a .yml file or a dictionary of DA configuration values.
        A dictionary is deep-copied and not modified.

    Raises
    ------
    ValueError
        If the input is neither a Path nor a dict, contains unrecognized keys,
        misses required keys, or contains invalid values.
    """

    _COMPONENT_KEYS = ('regularization_weight', 'initial_learning_rate')
    _SUPPORTED_LOSSES = ('MSE', 'NSE', 'RMSE')
    # Properties that are derived and must not be set by the user.
    _NON_CONFIGURABLE_KEYS = ('regularization',)

    def __init__(self, yml_path_or_dict: Path | dict):
        if isinstance(yml_path_or_dict, Path):
            cfg = AssimilationConfig._read_config(yml_path_or_dict)
        elif isinstance(yml_path_or_dict, dict):
            cfg = copy.deepcopy(yml_path_or_dict)
        else:
            raise ValueError(
                'Cannot create an assimilation config from input of type '
                f'{type(yml_path_or_dict)}.'
            )
        AssimilationConfig._check_cfg_keys(cfg)
        self._cfg = cfg
        self._components = self._parse_components()
        self._validate()

    def as_dict(self) -> dict:
        """Return a copy of the DA configuration as a dictionary.

        Returns
        -------
        dict
            The DA configuration, including keys inherited from the run config.
        """
        return copy.deepcopy(self._cfg)

    @staticmethod
    def allowed_keys() -> list[str]:
        """Return the sorted list of keys accepted in a DA configuration."""
        return sorted(
            p
            for p in dir(AssimilationConfig)
            if isinstance(getattr(AssimilationConfig, p), property)
            and p not in AssimilationConfig._NON_CONFIGURABLE_KEYS
        )

    @staticmethod
    def _read_config(yml_path: Path) -> dict:
        if not yml_path.exists():
            raise FileNotFoundError(yml_path)
        with yml_path.open('r') as fp:
            cfg = YAML(typ='safe').load(fp)
        if not isinstance(cfg, dict):
            raise ValueError(f'{yml_path} does not contain a mapping.')
        return cfg

    @staticmethod
    def _check_cfg_keys(cfg: dict) -> None:
        allowed = AssimilationConfig.allowed_keys()
        unknown_keys = sorted(str(k) for k in cfg if k not in allowed)
        if unknown_keys:
            raise ValueError(
                f'{unknown_keys} are not recognized assimilation config keys. '
                f'Allowed keys are: {allowed}.'
            )

    def _get_required(self, key: str) -> Any:  # noqa: ANN401
        if self._cfg.get(key) is None:
            raise ValueError(
                f'{key} is mandatory in the assimilation config but missing.'
            )
        return self._cfg[key]

    def _parse_components(self) -> dict[str, dict[str, float | None]]:
        raw = self._get_required('assimilation_components')
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            if len(set(raw)) != len(raw):
                raise ValueError(
                    f'assimilation_components contains duplicates: {raw}.'
                )
            raw = dict.fromkeys(raw)
        if not isinstance(raw, dict):
            raise ValueError(
                'assimilation_components must be a list of component names or '
                f'a dict of name -> options, got {type(raw)}.'
            )
        if not raw:
            raise ValueError('assimilation_components must not be empty.')

        components = {}
        for name, raw_options in raw.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f'Invalid assimilation component name: {name!r}.'
                )
            options = {} if raw_options is None else raw_options
            if not isinstance(options, dict):
                raise ValueError(
                    f'Options of assimilation component {name!r} must be a '
                    f'dict, got {type(options)}.'
                )
            unknown = sorted(
                str(k) for k in options if k not in self._COMPONENT_KEYS
            )
            if unknown:
                raise ValueError(
                    f'{unknown} are not recognized options of assimilation '
                    f'component {name!r}. Allowed options are: '
                    f'{list(self._COMPONENT_KEYS)}.'
                )
            weight = options.get('regularization_weight')
            if weight is None:
                weight = self.regularization_weight
            _check_number(f'{name}.regularization_weight', weight, minimum=0)
            lr = options.get('initial_learning_rate')
            if lr is not None:
                _check_number(
                    f'{name}.initial_learning_rate', lr, positive=True
                )
            components[name] = {
                'regularization_weight': float(weight),
                'initial_learning_rate': None if lr is None else float(lr),
            }
        return components

    def _validate(self) -> None:
        seq_length = self.seq_length
        if not _is_int(seq_length) or seq_length < 1:
            raise ValueError(
                'seq_length of the assimilation config must be a positive '
                f'integer (single-frequency runs only), got {seq_length!r}.'
            )
        # Required by the loss; fail early if missing.
        self._get_required('predict_last_n')
        self._get_required('target_variables')

        window = self.assimilation_window
        if not _is_int(window) or window < 1:
            raise ValueError(
                'assimilation_window must be a positive integer, got '
                f'{window!r}.'
            )
        lead_time = self.assimilation_lead_time
        if not _is_int(lead_time) or lead_time < 0:
            raise ValueError(
                'assimilation_lead_time must be a non-negative integer, got '
                f'{lead_time!r}.'
            )
        if lead_time >= seq_length:
            raise ValueError(
                f'assimilation_lead_time ({lead_time}) must be smaller than '
                f'seq_length ({seq_length}).'
            )
        if window > seq_length - lead_time:
            raise ValueError(
                f'assimilation_window ({window}) must not exceed seq_length - '
                f'assimilation_lead_time ({seq_length} - {lead_time} = '
                f'{seq_length - lead_time}).'
            )

        if not _is_int(self.epochs) or self.epochs < 0:
            raise ValueError(
                f'epochs must be a non-negative integer, got {self.epochs!r}.'
            )
        _check_number(
            'regularization_weight', self.regularization_weight, minimum=0
        )
        _check_number(
            'initial_learning_rate', self.initial_learning_rate, positive=True
        )
        if not isinstance(self.optimizer, str):
            raise ValueError(
                f'optimizer must be a string, got {self.optimizer!r}.'
            )
        if (
            not isinstance(self.loss, str)
            or self.loss.upper() not in self._SUPPORTED_LOSSES
        ):
            raise ValueError(
                f'loss {self.loss!r} is not supported for data assimilation. '
                f'Supported losses are: {list(self._SUPPORTED_LOSSES)}.'
            )
        if self.clip_gradient_norm is not None:
            _check_number(
                'clip_gradient_norm', self.clip_gradient_norm, positive=True
            )
        if self.early_stopping_tolerance is not None:
            _check_number(
                'early_stopping_tolerance',
                self.early_stopping_tolerance,
                positive=True,
            )

    # --- DA-specific keys ----------------------------------------------------

    @property
    def assimilation_components(self) -> dict[str, dict[str, float | None]]:
        """Components to optimize, mapped to their normalized options.

        Every entry has the keys ``regularization_weight`` (defaults to the
        top-level ``regularization_weight``) and ``initial_learning_rate``
        (``None`` means the top-level ``initial_learning_rate`` is used).
        Component names are not validated here; the DA engine validates them
        against the model.
        """
        return copy.deepcopy(self._components)

    @property
    def regularization_weight(self) -> float:
        """Default weight of the background term for all components."""
        return self._cfg.get('regularization_weight', 0.0)

    @property
    def assimilation_window(self) -> int:
        """Number of time steps before the issue time used for DA."""
        return self._get_required('assimilation_window')

    @property
    def assimilation_lead_time(self) -> int:
        """Number of trailing time steps (forecast horizon) excluded from DA."""
        return self._cfg.get('assimilation_lead_time', 0)

    @property
    def epochs(self) -> int:
        """Number of DA optimization steps per sample."""
        return self._cfg.get('epochs', 100)

    @property
    def initial_learning_rate(self) -> float:
        """Constant DA learning rate (default for all components)."""
        return self._get_required('initial_learning_rate')

    @property
    def optimizer(self) -> str:
        """Name of the DA optimizer, as in the training config."""
        return self._cfg.get('optimizer', 'Adam')

    @property
    def loss(self) -> str:
        """DA loss, one of 'MSE', 'NSE', 'RMSE'."""
        return self._cfg.get('loss', 'MSE')

    @property
    def clip_gradient_norm(self) -> float | None:
        """Max gradient norm during DA; None disables clipping."""
        return self._cfg.get('clip_gradient_norm', None)

    @property
    def early_stopping_tolerance(self) -> float | None:
        """Relative error at which a sequence stops being optimized.

        A sequence is converged once the relative error of its prediction at
        the last step of the assimilation window is <= this tolerance; its
        components are then frozen, and DA of the batch ends once all of its
        sequences converged. None disables early stopping.
        """
        return self._cfg.get('early_stopping_tolerance', None)

    # --- Keys read by the loss (usually inherited from the run config) -------

    @property
    def seq_length(self) -> int:
        """Input sequence length of the model (inherited)."""
        return self._get_required('seq_length')

    @property
    def predict_last_n(self) -> int | dict[str, int]:
        """Number of predicted time steps (inherited)."""
        return self._get_required('predict_last_n')

    @property
    def target_variables(self) -> list[str]:
        """Target variables (inherited)."""
        return self._get_required('target_variables')

    @property
    def target_loss_weights(self) -> list[float] | None:
        """Per-target loss weights (inherited); None weighs equally."""
        return self._cfg.get('target_loss_weights', None)

    @property
    def no_loss_frequencies(self) -> list[str]:
        """Frequencies excluded from the loss (inherited)."""
        value = self._cfg.get('no_loss_frequencies', None)
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @property
    def regularization(self) -> list[str]:
        """Regularization terms; always the background (prior) term."""
        return ['bg_embedding']


def _is_int(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, int) and not isinstance(value, bool)


def _check_number(
    name: str,
    value: Any,  # noqa: ANN401
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number, got {value!r}.')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive, got {value!r}.')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}.')
