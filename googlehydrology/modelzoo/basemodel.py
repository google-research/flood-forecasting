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
import re
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

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
_NON_NEG_INT_RE = re.compile(r'^[0-9]+$')
_FINITE_FLOAT_RE = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')


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
    def _write_state_file(
        state: dict[str, np.ndarray], path: str | Path
    ) -> None:
        """Write 3D LSTM state tensors to a ``.npz`` or ``.csv`` file."""
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix == '.npz':
            np.savez_compressed(target, **state)
            return
        if suffix != '.csv':
            raise ValueError(
                f'Unsupported hot-start state file extension {target.suffix!r} '
                f'for {target}. Expected ".npz" or ".csv".'
            )

        for state_name, arr in state.items():
            if arr.ndim != _STATE_TENSOR_NDIM:
                raise ValueError(
                    f'Hot-start CSV state array {state_name!r} must be 3D '
                    f'(layer, batch, unit), got shape {arr.shape}.'
                )
            if not np.isfinite(arr).all():
                raise ValueError(
                    f'Hot-start state array {state_name!r} contains non-finite '
                    'values and cannot be written to CSV.'
                )

        with target.open('w', encoding='utf-8', newline='') as handle:
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
        for col_name, raw_idx in raw_indices:
            if not _NON_NEG_INT_RE.match(raw_idx):
                raise ValueError(
                    f'Invalid non-negative integer for {col_name!r} '
                    f'on line {line_no} of {source}: {raw_idx!r}.'
                )
        coord = (
            int(raw_indices[0][1]),
            int(raw_indices[1][1]),
            int(raw_indices[2][1]),
        )

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

    @classmethod
    def _read_csv_state_file(cls, source: Path) -> dict[str, np.ndarray]:
        """Read and validate a tidy CSV hot-start state file."""
        entries: dict[str, dict[tuple[int, int, int], float]] = {}
        max_dims: dict[str, list[int]] = {}
        with source.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError(f'Hot-start state CSV file is empty: {source}')
            if tuple(col.strip() for col in header) != _STATE_CSV_COLUMNS:
                raise ValueError(
                    f'Invalid header {header!r} in hot-start state CSV '
                    f'{source}. Expected {list(_STATE_CSV_COLUMNS)!r}.'
                )
            for line_no, row in enumerate(reader, start=2):
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

        if not entries:
            raise ValueError(
                f'Hot-start state CSV file contains no data rows: {source}'
            )

        state: dict[str, np.ndarray] = {}
        for state_name, state_map in entries.items():
            shape = tuple(d + 1 for d in max_dims[state_name])
            expected_count = shape[0] * shape[1] * shape[2]
            if len(state_map) != expected_count:
                raise ValueError(
                    f'Incomplete 3D grid for state {state_name!r} in {source}: '
                    f'expected {expected_count} entries for shape '
                    f'{shape}, got {len(state_map)}.'
                )
            arr = np.empty(shape, dtype=np.float32)
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
    def _read_state_file(cls, path: str | Path) -> dict[str, np.ndarray]:
        """Load hot-start state arrays from a ``.npz`` or ``.csv`` file."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f'Hot-start state file not found: {source}')

        suffix = source.suffix.lower()
        if suffix == '.npz':
            with np.load(source, allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}
        if suffix == '.csv':
            return cls._read_csv_state_file(source)
        raise ValueError(
            f'Unsupported hot-start state file extension {source.suffix!r} '
            f'for {source}. Expected ".npz" or ".csv".'
        )

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
