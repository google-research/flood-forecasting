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


from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.datautils.scaler import Scaler
from googlehydrology.utils.config import Config
from googlehydrology.utils.samplingutils import sample_pointpredictions


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
            The file path where the state should be saved (npz recommended).
        """
        pass

    def load_state_from_disk(self, path: str | Path) -> None:
        """Pre-load a hot start state archive from disk into memory.

        Default implementation is a no-op for models that do not support state persistence.

        Parameters
        ----------
        path : str | Path
            Path to the state file to load.
        """
        pass

    def reset_state(self) -> None:
        """Reset in-memory hot-start state between basins."""
        self._preloaded_state = None

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
