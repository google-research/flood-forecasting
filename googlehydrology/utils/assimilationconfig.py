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

The DA settings are given as the nested ``assimilation_config`` block of a run
configuration. :py:class:`AssimilationConfig` is a
:py:class:`~googlehydrology.utils.config.Config` built from the run config
overlaid with that block, so all run-config keys (``lead_time``,
``seq_length``, ``target_variables``, ...) are inherited, while the training
keys ``epochs``, ``initial_learning_rate``, ``optimizer``, ``loss`` and
``clip_gradient_norm`` are *not* inherited: they take the values given in the
DA block, or the DA defaults. ``predict_last_n`` is *not* inherited either: on
the DA config it equals ``assimilation_window``, so that a loss built from it
evaluates the assimilation window rather than the forecast horizon.

If the run config defines ``seq_length``, the window is validated against the
number of observed target steps of a sequence:
``1 <= assimilation_window <= seq_length - lead_time``.

Example::

    assimilate: True
    assimilation_config:
      # Components to optimize. Either a list of names ...
      # assimilation_components: [hindcast_embedding, static_embedding]
      # ... or a dict of name -> per-component options (all optional).
      assimilation_components:
        hindcast_embedding:
          regularization_weight: 0.5
          initial_learning_rate: 0.05
        static_embedding:
      assimilation_window: 10  # time steps before the issue time
      initial_learning_rate: 0.01  # required; DA default for all components
      regularization_weight: 0.1  # optional, default 0.0
      # Optional, default None (disabled). A sequence is converged when the
      # relative error at the last window step, |pred - obs| / max(obs - y_min,
      # 0.05) in the scaled target space, is <= the tolerance; see
      # AssimilationConfig.early_stopping_tolerance for the exact rule.
      early_stopping_tolerance: 0.05
      epochs: 50  # optional, default 100 (DA steps per sample)
      optimizer: Adam  # optional, default Adam
      loss: MSE  # optional, default MSE; one of MSE, NSE, CMAL
      clip_gradient_norm: 1.0  # optional, default None (no clipping)
"""

import math
from typing import Annotated

import pydantic
import pydantic.dataclasses

from googlehydrology.utils.config import Config

# Training keys that have DA-specific values. They are never inherited from
# the run config: the DA block sets them, or the DA defaults apply.
_TRAINING_KEYS = (
    'epochs',
    'initial_learning_rate',
    'optimizer',
    'loss',
    'clip_gradient_norm',
)
# Keys accepted in the ``assimilation_config`` block.
_DA_KEYS = (
    'assimilation_components',
    'assimilation_window',
    'regularization_weight',
    'early_stopping_tolerance',
    *_TRAINING_KEYS,
)
# Compared case-insensitively, as in ``get_loss_obj``. CMAL requires the
# run config to define ``n_distributions`` (as for a CMAL training loss).
_SUPPORTED_LOSSES = ('MSE', 'NSE', 'CMAL')

# Strict numeric option types: YAML ``1`` and ``0.5`` are accepted, but bools,
# strings and non-finite floats are rejected (as for the top-level keys).
_NonNegativeNumber = (
    Annotated[pydantic.StrictFloat, pydantic.Field(ge=0, allow_inf_nan=False)]
    | Annotated[pydantic.StrictInt, pydantic.Field(ge=0)]
    | None
)
_PositiveNumber = (
    Annotated[pydantic.StrictFloat, pydantic.Field(gt=0, allow_inf_nan=False)]
    | Annotated[pydantic.StrictInt, pydantic.Field(gt=0)]
    | None
)


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, config=pydantic.ConfigDict(extra='forbid')
)
class AssimilationComponentSpec:
    """Per-component options of ``assimilation_components``.

    Both options are optional finite numbers (int or float, not bool or str):
    ``regularization_weight`` must be >= 0 and ``initial_learning_rate`` > 0.
    """

    regularization_weight: _NonNegativeNumber = None
    initial_learning_rate: _PositiveNumber = None


_COMPONENTS_ADAPTER = pydantic.TypeAdapter(dict[str, AssimilationComponentSpec])


class AssimilationConfig(Config):
    """Run configuration overlaid with the ``assimilation_config`` block.

    Usually obtained via :py:attr:`Config.assimilation_config`. Being a
    ``Config``, it can be passed to
    :py:func:`googlehydrology.training.get_loss_obj`,
    :py:func:`googlehydrology.training.get_optimizer` and
    :py:func:`googlehydrology.training.get_regularization_obj`.

    Parameters
    ----------
    da_cfg : dict
        The ``assimilation_config`` block.
    parent_cfg : dict | None, optional
        The run configuration (as dict) the block is nested in. Keys of the
        block take precedence over keys of the run configuration.

    Raises
    ------
    ValueError
        If ``da_cfg`` is not a dict, contains unrecognized keys, misses
        required keys, contains invalid values, or if ``assimilation_window``
        exceeds the observed target steps ``seq_length - lead_time`` of the
        run configuration (when ``seq_length`` is given).
    """

    def __init__(self, da_cfg: dict, parent_cfg: dict | None = None):
        """Build and validate the DA config; see the class docstring."""
        if not isinstance(da_cfg, dict):
            # ValueError (not TypeError) for consistency with Config.
            raise ValueError(  # noqa: TRY004
                f'assimilation_config must be a dict, got {type(da_cfg)}.'
            )
        unknown = sorted(str(k) for k in da_cfg if k not in _DA_KEYS)
        if unknown:
            raise ValueError(
                f'{unknown} are not recognized assimilation config keys. '
                f'Allowed keys are: {list(_DA_KEYS)}.'
            )
        # The nested block and the flag are not meaningful on the DA config,
        # and the training values of the DA-specific keys must not leak in
        # (e.g. the training loss or learning rate must not become the DA
        # loss or learning rate).
        skip = {'assimilation_config', 'assimilate', *_TRAINING_KEYS}
        cfg = {k: v for k, v in (parent_cfg or {}).items() if k not in skip}
        cfg.update(da_cfg)
        super().__init__(cfg, dev_mode=True)
        self._components = self._parse_components(da_cfg)
        self._validate(da_cfg)

    @staticmethod
    def _parse_components(
        da_cfg: dict,
    ) -> dict[str, AssimilationComponentSpec]:
        raw = da_cfg.get('assimilation_components')
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            raw = dict.fromkeys(raw)
        if not isinstance(raw, dict):
            raise ValueError(  # noqa: TRY004
                'assimilation_components must be a list of component names or '
                f'a dict of name -> options, got {raw!r}.'
            )
        if not raw:
            raise ValueError('assimilation_components must not be empty.')
        raw = {name: {} if spec is None else spec for name, spec in raw.items()}
        try:
            return _COMPONENTS_ADAPTER.validate_python(raw)
        except pydantic.ValidationError as ex:
            raise ValueError(f'Invalid assimilation_components: {ex}') from ex

    def _validate(self, da_cfg: dict) -> None:
        self._validate_window(da_cfg.get('assimilation_window'))
        lr = da_cfg.get('initial_learning_rate')
        if lr is None:
            raise ValueError(
                'initial_learning_rate is mandatory in the assimilation config '
                '(it is not inherited from the training learning rate).'
            )
        _check_number('initial_learning_rate', lr, positive=True)
        _check_number(
            'regularization_weight', self.regularization_weight, minimum=0
        )
        if not _is_int(self.epochs) or self.epochs < 0:
            raise ValueError(
                f'epochs must be a non-negative integer, got {self.epochs!r}.'
            )
        if not isinstance(self.optimizer, str):
            raise ValueError(  # noqa: TRY004
                f'optimizer must be a string, got {self.optimizer!r}.'
            )
        if (
            not isinstance(self.loss, str)
            or self.loss.upper() not in _SUPPORTED_LOSSES
        ):
            raise ValueError(
                f'loss {self.loss!r} is not supported for data assimilation. '
                f'Supported losses are: {list(_SUPPORTED_LOSSES)}.'
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

    def _validate_window(self, window: object) -> None:
        if not _is_int(window) or window < 1:
            raise ValueError(
                'assimilation_window must be a positive integer, got '
                f'{window!r}.'
            )
        # The bound can only be checked if the run config defines the sequence
        # length (it is optional on a stand-alone DA block). Multi-frequency
        # dict values are not supported by DA and are left to the engine.
        seq_length = self._cfg.get('seq_length')
        lead_time = self.lead_time
        if not _is_int(seq_length) or not _is_int(lead_time):
            return
        n_observed = seq_length - lead_time
        if window > n_observed:
            raise ValueError(
                f'assimilation_window ({window}) must not exceed the number '
                f'of observed target steps per sequence, seq_length - '
                f'lead_time = {seq_length} - {lead_time} = {n_observed}.'
            )

    # --- DA-specific keys ----------------------------------------------------

    @property
    def assimilation_components(self) -> dict[str, dict[str, float | None]]:
        """Components to optimize, mapped to their normalized options.

        Every entry has the keys ``regularization_weight`` (defaults to the
        top-level ``regularization_weight``) and ``initial_learning_rate``
        (``None`` means the top-level ``initial_learning_rate`` is used).
        Numeric options are normalized to float. Component names are not
        validated here; the DA engine validates them against the model.
        """
        default_weight = float(self.regularization_weight)
        return {
            name: {
                'regularization_weight': (
                    default_weight
                    if spec.regularization_weight is None
                    else float(spec.regularization_weight)
                ),
                'initial_learning_rate': (
                    None
                    if spec.initial_learning_rate is None
                    else float(spec.initial_learning_rate)
                ),
            }
            for name, spec in self._components.items()
        }

    @property
    def assimilation_window(self) -> int:
        """Number of time steps before the issue time used for DA."""
        return self._get_value_verbose('assimilation_window')

    @property
    def regularization_weight(self) -> float:
        """Default weight of the background term for all components."""
        return self._cfg.get('regularization_weight', 0.0)

    @property
    def early_stopping_tolerance(self) -> float | None:
        """Relative error at which a sequence stops being optimized.

        None disables early stopping. Otherwise, a sequence is converged when
        the relative error at the last step of the assimilation window,
        ``|pred - obs| / max(obs - y_min, 0.05)``, is <= this tolerance, where
        ``y_min`` is the minimum of that sequence's observed target over the
        full sequence (0 if all observations are missing) and all values are
        in the model's scaled target space. If the observation or the
        prediction at the last window step is missing, the maximum relative
        error over the valid window steps is used instead; a sequence without
        any valid window step never converges. Converged sequences are frozen
        for the remaining optimization steps, and DA of a batch ends once all
        of its sequences converged.
        """
        return self._cfg.get('early_stopping_tolerance', None)

    # --- Config keys with DA-specific defaults -------------------------------

    @property
    def predict_last_n(self) -> int:
        """The assimilation window; not inherited from the run config.

        A loss built from this config (e.g. via ``get_loss_obj``) evaluates the
        last ``predict_last_n`` steps of the tensors it is given. For DA these
        must be the last ``assimilation_window`` observed steps, so callers
        must pass observed-period tensors (i.e. with the forecast horizon
        dropped). Inheriting the training value instead would silently fit
        the forecast horizon, so the window is used as a fail-safe default.
        """
        return self.assimilation_window

    @property
    def epochs(self) -> int:
        """Number of DA optimization steps per sample."""
        return self._cfg.get('epochs', 100)

    @property
    def optimizer(self) -> str:
        """Name of the DA optimizer, as in the training config."""
        return self._cfg.get('optimizer', 'Adam')

    @property
    def loss(self) -> str:
        """DA loss, one of 'MSE', 'NSE', 'CMAL' (case-insensitive)."""
        return self._cfg.get('loss', 'MSE')

    @property
    def regularization(self) -> list[str]:
        """Regularization terms; always the background (prior) term."""
        return ['bg_embedding']


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_number(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004
            f'{name} must be a number, got {value!r}.'
        )
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number, got {value!r}.')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive, got {value!r}.')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}.')
