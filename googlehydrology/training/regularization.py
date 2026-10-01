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
    regularizes the otherwise ill-posed inversion.

    For every component ``c`` (batch dimension first), the squared departure
    from the background is averaged over the valid non-batch entries and
    divided by the detached mean squared magnitude of the background of the
    same sequence plus a small epsilon, i.e. a diagonal background-error
    scaling that is constant within a sequence::

        valid_c[i]     = isfinite(optimized_c[i]) & isfinite(baseline_c[i])
        dep_c[i]       = mean_{valid}((optimized_c[i] - baseline_c[i]) ** 2)
        scale_c[i]     = mean_{valid}(baseline_c[i] ** 2) + eps
        term           = sum_c w_c * mean_{i in V_c}(dep_c[i] / scale_c[i])

    where ``V_c`` is the set of sequences with at least one valid entry.

    Non-finite entries (e.g. NaN embeddings the model produces for missing
    input steps) are masked out of both the departure and the scale, so they
    neither poison the term nor receive a gradient. Sequences without a single
    valid entry are skipped in the mean over sequences; if no sequence of a
    component is valid, that component contributes zero.

    The normalisation makes the term, and therefore ``regularization_weight``,
    dimensionless and comparable across components of different magnitude and
    across models. Baselines are detached so no gradient flows into them.
    ``_EPS`` (1e-3) is deliberately not tiny: it prevents the normalisation
    from amplifying departures from backgrounds with a near-zero norm, where
    the ratio would otherwise blow up.

    The term is computed in float32 regardless of the input dtype and cast
    back to the dtype of the optimized tensors at the end: in float16 the
    ratio ``departure / scale`` overflows to ``inf`` for moderately large
    departures and the masked sums lose precision.

    Inputs are strict: ``other_model_data`` must contain non-empty
    ``optimized_components`` and ``baseline_components`` dictionaries, every
    optimized component needs a baseline, every component needs a batch
    dimension (``ndim >= 2``), and the shapes must match exactly. Violations
    raise ``ValueError`` instead of silently contributing zero. Components
    whose weight is exactly zero are skipped before their tensors are touched,
    so they can never contribute (not even a NaN).

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

    _EPS = 1e-3
    # Batch dimension first plus at least one feature dimension.
    _MIN_NDIM = 2

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
        """Calculate the normalised squared departure from the background.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Not used.
        ground_truth : dict[str, torch.Tensor]
            Not used.
        other_model_data : dict[str, dict[str, torch.Tensor]]
            Dictionary that must contain ``optimized_components`` (name ->
            tensor being optimized, batch dimension first) and
            ``baseline_components`` (name -> unassimilated tensor of the same
            shape). It may contain ``component_weights`` (name -> float,
            default 1).

        Returns
        -------
        torch.Tensor
            Sum over components of ``w * mean_i(departure[i] / scale[i])``
            over the sequences ``i`` with at least one finite entry, computed
            in float32 and cast to the dtype of the first optimized tensor.

        Raises
        ------
        ValueError
            If ``optimized_components`` or ``baseline_components`` are missing
            or empty, if an optimized component has no baseline, if a
            component has no batch dimension (``ndim < 2``), or if the shapes
            of an optimized component and its baseline differ.
        """
        optimized = other_model_data.get('optimized_components')
        baseline = other_model_data.get('baseline_components')
        weights = other_model_data.get('component_weights') or {}
        if not optimized or not baseline:
            raise ValueError(
                'BackgroundEmbeddingRegularization requires non-empty '
                "'optimized_components' and 'baseline_components'."
            )

        dtype = next(iter(optimized.values())).dtype
        loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=next(iter(optimized.values())).device,
        )
        for comp_name, opt in optimized.items():
            w = float(weights.get(comp_name, 1.0))
            if w == 0.0:
                # Skip before touching the tensors so a zero-weight component
                # can never contribute, not even 0 * NaN.
                continue
            if comp_name not in baseline:
                raise ValueError(f'No baseline for component {comp_name!r}.')
            base = baseline[comp_name]
            if opt.ndim < self._MIN_NDIM:
                raise ValueError(
                    f'{comp_name}: expected a batch dimension first'
                    f' (ndim >= {self._MIN_NDIM}), got shape'
                    f' {tuple(opt.shape)}. For a 1-D tensor the per-sequence'
                    ' mean would reduce over the batch.'
                )
            if opt.shape != base.shape:
                raise ValueError(
                    f'{comp_name}: optimized shape {tuple(opt.shape)} != '
                    f'baseline shape {tuple(base.shape)}.'
                )
            # float32: the ratio overflows in float16.
            opt32 = opt.float()
            base32 = base.detach().float()
            # Mask non-finite entries (e.g. NaN embeddings for missing input
            # steps). The inputs are replaced *before* any arithmetic so that
            # NaNs never enter the graph: masking the squared difference
            # afterwards (whether by ``* mask`` or ``torch.where``) would
            # still back-propagate ``0 * NaN = NaN`` into the optimized tensor.
            valid = torch.isfinite(opt32) & torch.isfinite(base32)
            zero = torch.zeros((), dtype=torch.float32, device=opt32.device)
            opt32 = torch.where(valid, opt32, zero)
            base32 = torch.where(valid, base32, zero)
            n_valid = valid.flatten(1).sum(1)
            denom = n_valid.clamp(min=1)
            departure = ((opt32 - base32) ** 2).flatten(1).sum(1) / denom
            scale = (base32**2).flatten(1).sum(1) / denom + self._EPS
            has_valid = n_valid > 0
            if not torch.any(has_valid):
                # No valid entry in any sequence: contributes zero but keeps
                # the graph connected to the optimized tensor.
                loss = loss + w * (departure.sum() * 0.0)
                continue
            loss = loss + w * torch.mean((departure / scale)[has_valid])
        return loss.to(dtype)
