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


import csv
import math
import os
import re
import stat
import uuid
import zipfile
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, TextIO

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.datautils.scaler import Scaler
from googlehydrology.utils.config import Config
from googlehydrology.utils.samplingutils import sample_pointpredictions

_STATE_TENSOR_NDIM = 3
_STATE_CSV_COLUMNS: tuple[str, ...] = (
    'state',
    'layer',
    'batch',
    'unit',
    'value',
)
_NON_NEG_INT_RE = re.compile(r'^[0-9]+\Z')
_FINITE_FLOAT_RE = re.compile(
    r'^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z'
)
_LSTM_STATE_KEY_PAIRS: tuple[
    tuple[str, tuple[str, str], tuple[str, str]], ...
] = (
    ('hindcast', ('h_hindcast', 'h_hind'), ('c_hindcast', 'c_hind')),
    ('forecast', ('h_forecast', 'h_fore'), ('c_forecast', 'c_fore')),
)
_ALLOWED_LSTM_STATE_KEYS: frozenset[str] = frozenset(
    key
    for _, h_keys, c_keys in _LSTM_STATE_KEY_PAIRS
    for key in (*h_keys, *c_keys)
)


class BaseModel(nn.Module):
    """Abstract base model class, don't use this class for model training.

    Use subclasses of this class for training/evaluating different models, e.g. use `CudaLSTM` for training a standard
    LSTM model or `EA-LSTM` for training an Entity-Aware-LSTM. Refer to  :doc:`Documentation/Modelzoo </usage/models>`
    for a full list of available models and how to integrate a new model.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = []

    # Names of internal tensors that a data assimilation (DA) procedure may
    # read from the forward output dict and override via the keyword argument
    # `assimilation_overrides={name: tensor}` of `forward`.
    # Models that do not support DA leave this empty.
    supported_assimilation_components: ClassVar[tuple[str, ...]] = ()

    # Whether the model can consume dynamic input tensors containing NaN values
    # (e.g. via masked mean embeddings), enabling training-time input dropout.
    # InputDropout guarantees that every timestep with at least one valid group
    # before dropout retains at least one valid group afterward in each active
    # pool (hindcast window, forecast overlap/horizon, and shared-only groups
    # over the forecast horizon when forecast-only groups also exist). Timesteps
    # that were already all-NaN in the source data remain NaN.
    supports_missing_inputs: ClassVar[bool] = False

    def __init__(self, cfg: Config):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.output_size = len(cfg.target_variables)
        if cfg.head.lower() in ['cmal', 'cmal_deterministic']:
            self.output_size *= 4 * cfg.n_distributions
        self._scaler = Scaler(
            scaler_dir=(cfg.base_run_dir if cfg.is_finetuning else cfg.run_dir),
            calculate_scaler=False,
        )
        self._preloaded_state = None

    def sample(
        self,
        data: dict[str, torch.Tensor],
        n_samples: int,
        *,
        outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Provides point prediction samples from a probabilistic model.

        This function wraps the `sample_pointpredictions` function, which provides different point sampling functions
        for the different uncertainty estimation approaches. There are also options to handle negative point prediction
        samples that arise while sampling from the uncertainty estimates. They can be controlled via the configuration.


        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of point predictions that ought ot be sampled form the model.
        outputs, optional
            Model forward result

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled point predictions
        """
        return sample_pointpredictions(
            self, data, n_samples, self._scaler, outputs=outputs
        )

    def forward(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        *,
        assimilation_overrides: dict[str, torch.Tensor] | None = None,
        assimilation_slice: tuple[int, int] | None = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Perform a forward pass.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.
        assimilation_overrides : dict[str, torch.Tensor] | None, optional
            Optional data assimilation (DA) hook: tensors replacing the
            internal components named in `supported_assimilation_components`.
            Models that do not support DA may ignore this argument.
        assimilation_slice : tuple[int, int] | None, optional
            Optional DA hook: ``(start, end)`` time indices on the model time
            axis where a partial-length dynamic override is spliced in. Models
            that do not support DA may ignore this argument.
        return_embeddings : bool, optional
            Optional DA hook: if True, models that support DA additionally
            return their assimilable components in the output dict.

        Returns
        -------
        dict[str, torch.Tensor]
            Model output and potentially any intermediate states and activations as a dictionary.
        """
        raise NotImplementedError

    def validate_assimilation_components(self, names: Iterable[str]) -> None:
        """Raise ValueError for names not in supported_assimilation_components.

        Parameters
        ----------
        names : Iterable[str]
            Names of the components a data assimilation procedure intends to
            override via ``forward(assimilation_overrides=...)``.
        """
        unknown = sorted(
            set(names).difference(self.supported_assimilation_components)
        )
        if unknown:
            msg = (
                f'Model {self.cfg.model!r} does not support data assimilation '
                f'with the configured assimilation_components: '
                f'Unsupported assimilation components {unknown}; '
                f'supported: {list(self.supported_assimilation_components)}'
            )
            raise ValueError(msg)

    def point_prediction(
        self, outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Deterministic prediction ``[B, T, n_targets]`` from forward outputs.

        Models delegate to their head (see ``BaseHead.point_prediction``).
        Used by data assimilation and other consumers that need a single value
        per time step irrespective of the head type.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Output dict of ``forward``.

        Returns
        -------
        torch.Tensor
            Point prediction of shape ``[B, T, n_targets]``.
        """
        raise NotImplementedError

    def save_state(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        path: str | Path,
    ) -> None:
        """Save the hot start state of the model.

        Default implementation is a no-op for models that do not support state persistence.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary containing input features as key-value pairs.
        path : str | Path
            The file path where the state should be saved (.npz or .csv format).
        """
        pass

    def load_state_from_disk(self, path: str | Path) -> None:
        """Pre-load a hot start state file from disk into memory.

        Default implementation is a no-op for models that do not support state persistence.

        Parameters
        ----------
        path : str | Path
            Path to the state file to load (.npz or .csv format).
        """
        pass

    def reset_state(self) -> None:
        """Reset in-memory hot-start state between basins."""
        self._preloaded_state = None

    @staticmethod
    def _validate_csv_state_array(state_name: str, arr: np.ndarray) -> None:
        """Validate a single state array before CSV serialization."""
        if not state_name or state_name != state_name.strip():
            raise ValueError(
                'Hot-start CSV state name must be non-empty without '
                f'leading/trailing whitespace, got {state_name!r}.'
            )
        if not (
            np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.integer)
        ):
            raise ValueError(
                f'Hot-start CSV state array {state_name!r} must have a real '
                f'numeric dtype, got {arr.dtype}.'
            )
        if arr.ndim != _STATE_TENSOR_NDIM:
            raise ValueError(
                f'Hot-start CSV state array {state_name!r} must be 3D '
                f'(layer, batch, unit), got shape {arr.shape}.'
            )
        if arr.size == 0:
            raise ValueError(
                f'Hot-start CSV state array {state_name!r} has an empty '
                f'dimension (shape {arr.shape}) and cannot be written to CSV.'
            )
        with np.errstate(over='ignore'):
            arr_f32 = arr.astype(np.float32, copy=False)
        if not (np.isfinite(arr).all() and np.isfinite(arr_f32).all()):
            raise ValueError(
                f'Hot-start state array {state_name!r} contains non-finite '
                'or float32-overflowing values and cannot be written to CSV.'
            )

    @classmethod
    def _validate_csv_state(
        cls, state: dict[str, np.ndarray], target: Path
    ) -> None:
        """Validate state arrays before serializing to CSV."""
        if not state:
            raise ValueError(
                'Hot-start CSV state dictionary is empty and cannot be '
                f'written to {target}.'
            )
        for state_name, arr in state.items():
            cls._validate_csv_state_array(state_name, arr)

    @staticmethod
    def _write_csv_rows(handle: TextIO, state: dict[str, np.ndarray]) -> None:
        """Write validated 3D state arrays to an open CSV text stream."""
        writer = csv.writer(handle)
        writer.writerow(_STATE_CSV_COLUMNS)
        for state_name, arr in state.items():
            n_layers, n_batch, n_units = arr.shape
            for layer_idx in range(n_layers):
                for batch_idx in range(n_batch):
                    row_values = arr[layer_idx, batch_idx]
                    for unit_idx in range(n_units):
                        writer.writerow(
                            [
                                state_name,
                                layer_idx,
                                batch_idx,
                                unit_idx,
                                repr(float(row_values[unit_idx])),
                            ]
                        )

    @classmethod
    def _write_state_file(
        cls, state: dict[str, np.ndarray], path: str | Path
    ) -> None:
        """Atomically write 3D LSTM state tensors to ``.npz`` or ``.csv``."""
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix not in {'.npz', '.csv'}:
            raise ValueError(
                f'Unsupported hot-start state file extension {target.suffix!r} '
                f'for {target}. Expected ".npz" or ".csv".'
            )
        write_target = target.resolve()
        if write_target.is_dir():
            raise IsADirectoryError(
                f'Hot-start state path is a directory: {target}'
            )
        if suffix == '.csv':
            cls._validate_csv_state(state, target)

        existing_mode = (
            stat.S_IMODE(write_target.stat().st_mode)
            if os.name == 'posix' and write_target.is_file()
            else None
        )
        tmp_path = write_target.with_name(
            f'.{write_target.name}.{uuid.uuid4().hex}.tmp'
        )
        try:
            if suffix == '.npz':
                with tmp_path.open('xb') as handle:
                    np.savez_compressed(handle, **state)
            else:
                with tmp_path.open('x', encoding='utf-8', newline='') as handle:
                    cls._write_csv_rows(handle, state)
            if existing_mode is not None:
                tmp_path.chmod(existing_mode)
            tmp_path.replace(write_target)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _parse_csv_state_row(
        row: list[str], *, line_no: int, source: Path
    ) -> tuple[str, tuple[int, int, int], float]:
        """Validate and parse a single CSV state row."""
        if len(row) != len(_STATE_CSV_COLUMNS):
            raise ValueError(
                f'Invalid column count on line {line_no} of {source}: '
                f'expected {len(_STATE_CSV_COLUMNS)}, got {len(row)}.'
            )
        state_name = row[0].strip()
        if not state_name:
            raise ValueError(f'Empty state name on line {line_no} of {source}.')

        raw_indices = (
            ('layer', row[1].strip()),
            ('batch', row[2].strip()),
            ('unit', row[3].strip()),
        )
        parsed_indices: list[int] = []
        for col_name, raw_idx in raw_indices:
            if not _NON_NEG_INT_RE.match(raw_idx):
                raise ValueError(
                    f'Invalid non-negative integer for {col_name!r} '
                    f'on line {line_no} of {source}: {raw_idx!r}.'
                )
            try:
                parsed_indices.append(int(raw_idx))
            except ValueError as exc:
                raise ValueError(
                    f'Integer index out of range for {col_name!r} '
                    f'on line {line_no} of {source}: {exc}'
                ) from exc
        coord = (parsed_indices[0], parsed_indices[1], parsed_indices[2])

        value_s = row[4].strip()
        if not _FINITE_FLOAT_RE.match(value_s):
            raise ValueError(
                f'Invalid finite float value on line {line_no} of '
                f'{source}: {value_s!r}.'
            )
        val = float(value_s)
        if not math.isfinite(val):
            raise ValueError(
                f'Non-finite float value on line {line_no} of '
                f'{source}: {value_s!r}.'
            )
        return state_name, coord, val

    @staticmethod
    def _build_csv_state_arrays(
        entries: dict[str, dict[tuple[int, int, int], float]],
        max_dims: dict[str, list[int]],
        source: Path,
    ) -> dict[str, np.ndarray]:
        """Convert validated coordinate maps into dense 3D float32 arrays."""
        state: dict[str, np.ndarray] = {}
        for state_name, state_map in entries.items():
            shape = tuple(d + 1 for d in max_dims[state_name])
            expected_count = shape[0] * shape[1] * shape[2]
            if len(state_map) != expected_count:
                expected_str = (
                    str(expected_count) if expected_count < 10**18 else '>1e18'
                )
                raise ValueError(
                    f'Incomplete 3D grid for state {state_name!r} in {source}: '
                    f'expected {expected_str} entries for shape '
                    f'{shape}, got {len(state_map)}.'
                )
            arr = np.empty(shape, dtype=np.float32)
            with np.errstate(over='ignore'):
                for coord, val in state_map.items():
                    arr[coord] = val
            if not np.isfinite(arr).all():
                raise ValueError(
                    f'Overflow to non-finite float32 for state {state_name!r} '
                    f'in {source}.'
                )
            state[state_name] = arr
        return state

    @classmethod
    def _read_csv_state_file(cls, source: Path) -> dict[str, np.ndarray]:
        """Read and validate a tidy CSV hot-start state file."""
        entries: dict[str, dict[tuple[int, int, int], float]] = {}
        max_dims: dict[str, list[int]] = {}
        try:
            with source.open('r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header is None:
                    raise ValueError(
                        f'Hot-start state CSV file is empty: {source}'
                    )
                if tuple(col.strip() for col in header) != _STATE_CSV_COLUMNS:
                    raise ValueError(
                        f'Invalid header {header!r} in hot-start state CSV '
                        f'{source}. Expected {list(_STATE_CSV_COLUMNS)!r}.'
                    )
                for row in reader:
                    if not row or (len(row) == 1 and not row[0].strip()):
                        continue
                    line_no = reader.line_num
                    state_name, coord, val = cls._parse_csv_state_row(
                        row, line_no=line_no, source=source
                    )
                    state_map = entries.setdefault(state_name, {})
                    if coord in state_map:
                        raise ValueError(
                            f'Duplicate coordinate {coord} for state '
                            f'{state_name!r} on line {line_no} of {source}.'
                        )
                    state_map[coord] = val
                    dims = max_dims.setdefault(state_name, [0, 0, 0])
                    dims[0] = max(dims[0], coord[0])
                    dims[1] = max(dims[1], coord[1])
                    dims[2] = max(dims[2], coord[2])
        except csv.Error as exc:
            raise ValueError(
                f'Malformed hot-start state CSV {source}: {exc}'
            ) from exc

        if not entries:
            raise ValueError(
                f'Hot-start state CSV file contains no data rows: {source}'
            )

        return cls._build_csv_state_arrays(entries, max_dims, source)

    @staticmethod
    def _read_npz_state_file(source: Path) -> dict[str, np.ndarray]:
        """Read and validate a ``.npz`` hot-start state archive."""
        try:
            loaded = np.load(source, allow_pickle=False)
        except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
            raise ValueError(
                f'Invalid or corrupted hot-start NPZ file {source}: {exc}'
            ) from exc
        if not hasattr(loaded, 'files'):
            raise ValueError(
                f'Hot-start state file {source} is not a valid NPZ archive '
                f'(loaded {type(loaded).__name__}).'
            )
        try:
            with loaded as archive:
                arrays = {key: archive[key] for key in archive.files}
        except (
            zipfile.BadZipFile,
            zlib.error,
            EOFError,
            OSError,
            ValueError,
        ) as exc:
            raise ValueError(
                f'Invalid or corrupted hot-start NPZ file {source}: {exc}'
            ) from exc
        if not arrays:
            raise ValueError(
                f'Hot-start state NPZ file contains no arrays: {source}'
            )
        return arrays

    @classmethod
    def _read_state_file(cls, path: str | Path) -> dict[str, np.ndarray]:
        """Load hot-start state arrays from a ``.npz`` or ``.csv`` file."""
        source = Path(path)
        if source.is_dir():
            raise IsADirectoryError(
                f'Hot-start state path is a directory: {source}'
            )
        if not source.is_file():
            raise FileNotFoundError(f'Hot-start state file not found: {source}')

        suffix = source.suffix.lower()
        if suffix == '.npz':
            return cls._read_npz_state_file(source)
        if suffix == '.csv':
            return cls._read_csv_state_file(source)
        raise ValueError(
            f'Unsupported hot-start state file extension {source.suffix!r} '
            f'for {source}. Expected ".npz" or ".csv".'
        )

    @staticmethod
    def _get_state_slot(
        state: dict[str, np.ndarray],
        canonical: str,
        alias: str,
        path: str | Path,
    ) -> np.ndarray | None:
        """Fetch a canonical state array or its legacy alias, rejecting both."""
        if canonical in state and alias in state:
            raise ValueError(
                f'Ambiguous hot-start state in {path}: both {canonical!r} and '
                f'legacy alias {alias!r} are present.'
            )
        if canonical in state:
            return state[canonical]
        return state.get(alias)

    @classmethod
    def _validate_lstm_hot_start_state(
        cls, state: dict[str, np.ndarray], path: str | Path
    ) -> dict[str, np.ndarray]:
        """Validate hindcast/forecast LSTM state keys and paired shapes."""
        unknown = sorted(set(state) - _ALLOWED_LSTM_STATE_KEYS)
        if unknown:
            raise ValueError(
                f'Unrecognized hot-start state key(s) {unknown} in {path}. '
                'Expected hindcast/forecast h and c states '
                '(h_hindcast, c_hindcast, h_forecast, c_forecast).'
            )
        for group, (h_canon, h_alias), (
            c_canon,
            c_alias,
        ) in _LSTM_STATE_KEY_PAIRS:
            h_arr = cls._get_state_slot(state, h_canon, h_alias, path)
            c_arr = cls._get_state_slot(state, c_canon, c_alias, path)
            if h_arr is None or c_arr is None:
                raise ValueError(
                    f'Missing required {group} hot-start state in {path}: '
                    f'expected both {h_canon!r} and {c_canon!r}.'
                )
            if h_arr.shape != c_arr.shape:
                raise ValueError(
                    f'Mismatched shapes for {group} hot-start state in {path}: '
                    f'{h_canon!r} has shape {h_arr.shape}, '
                    f'{c_canon!r} has shape {c_arr.shape}.'
                )
        return state

    def pre_model_hook(
        self, data: dict[str, torch.Tensor], is_train: bool
    ) -> dict[str, torch.Tensor]:
        """A function to execute before the model in training, validation and test.
        The beahvior can be adapted depending on the run configuration and the provided arguments.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs and labels y.
        is_train : bool
            Defines if the hook is executed in train mode or in validation/test mode.

        Returns
        -------
        data : dict[str, torch.Tensor]
            The modified (or unmodified) data that are used for the training or evaluation.
        """
        # here one can implement additional pre model hooks e.g. based on head
        return data
