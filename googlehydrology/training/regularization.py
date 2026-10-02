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


import torch

from googlehydrology.utils.config import Config


class BaseRegularization(torch.nn.Module):
    """Base class for regularization terms.

    Regularization terms subclass this class by implementing the `forward` method.

    Parameters
    ----------
    cfg: Config
        The run configuration.
    name: str
        The name of the regularization term.
    weight: float, optional.
        The weight of the regularization term. Default: 1.
    """

    def __init__(self, cfg: Config, name: str, weight: float = 1.0):
        super(BaseRegularization, self).__init__()
        self.cfg = cfg
        self.name = name
        self.weight = weight

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        other_model_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculate the regularization term.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Dictionary of predicted variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        ground_truth : dict[str, torch.Tensor]
            Dictionary of ground truth variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        other_model_data : dict[str, torch.Tensor]
            Dictionary of all remaining keys-value pairs in the prediction dictionary that are not directly linked to
            the model predictions but can be useful for regularization purposes, e.g. network internals, weights etc.

        Returns
        -------
        torch.Tensor
            The regularization value.
        """
        raise NotImplementedError


class ForecastOverlapMSERegularization(BaseRegularization):
    """Squared error regularization for penalizing differences between hindcast and forecast models.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(ForecastOverlapMSERegularization, self).__init__(
            cfg, name='forecast_overlap', weight=weight
        )

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        other_model_output: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate the squared difference between hindcast and forecast model during overlap.

        Does not work with multi-frequency models.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Not used.
        ground_truth : dict[str, torch.Tensor]
            Not used.
        other_model_output : dict[str, dict[str, torch.Tensor]]
            Dictionary containing ``y_forecast_overlap`` and ``y_hindcast_overlap``, which are
            both dictionaries containing keys to relevant model outputs.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations between overlapping portions of hindcast and forecast models.

        Raises
        ------
        ValueError if y_hindcast_overlap or y_forecast_overlap is not present in model output.
        """
        loss = 0
        if 'y_hindcast_overlap' not in other_model_output:
            raise ValueError(
                'y_hindcast_overlap is not present in the model output.'
            )
        if 'y_forecast_overlap' not in other_model_output:
            raise ValueError(
                'y_forecast_overlap is not present in the model output.'
            )
        hindcast = other_model_output['y_hindcast_overlap']
        forecast = other_model_output['y_forecast_overlap']
        loss += torch.mean((hindcast - forecast) ** 2)
        return loss


class BackgroundEmbeddingRegularization(BaseRegularization):
    """Background (prior) term for variational data assimilation.

    In gradient-based (variational) data assimilation, latent model components
    (e.g. embeddings or initial states) are optimized so that the model better
    fits recent observations. This term keeps the optimized components close to
    the values the model produced without assimilation (the background), which
    regularizes the otherwise ill-posed inversion:

    ``sum_c w_c * mean((optimized_c - baseline_c) ** 2)``

    The tensors are passed through the ``other_model_data`` argument of
    `BaseLoss.forward`, which merges them into the third argument of this
    module.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    weight : float, optional
        Global weight of the regularization term. Default: 1.
    name : str, optional
        Name of the regularization term. Default: 'bg_embedding'.
    """

    def __init__(
        self, cfg: Config, weight: float = 1.0, name: str = 'bg_embedding'
    ):
        super().__init__(cfg, name=name, weight=weight)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        other_model_data: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate the weighted squared deviation from the background.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Not used.
        ground_truth : dict[str, torch.Tensor]
            Not used.
        other_model_data : dict[str, dict[str, torch.Tensor]]
            Dictionary that may contain ``optimized_components`` (name -> tensor
            being optimized), ``baseline_components`` (name -> unassimilated
            tensor of the same shape) and ``component_weights`` (name -> float,
            default 1). Components without a baseline are skipped.

        Returns
        -------
        torch.Tensor
            Sum over components of ``w * mean((optimized - baseline) ** 2)``, or
            a zero tensor if there is nothing to regularize.
        """
        optimized = other_model_data.get('optimized_components') or {}
        baseline = other_model_data.get('baseline_components') or {}
        weights = other_model_data.get('component_weights') or {}

        loss = None
        device = None
        for comp_name, opt in optimized.items():
            device = opt.device
            base = baseline.get(comp_name)
            if base is None:
                continue
            term = float(weights.get(comp_name, 1.0)) * torch.mean(
                (opt - base) ** 2
            )
            loss = term if loss is None else loss + term
        if loss is None:
            return torch.zeros((), dtype=torch.float32, device=device)
        return loss
