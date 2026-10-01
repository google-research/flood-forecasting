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

import numbers
from collections import defaultdict

import numpy as np
import torch

from googlehydrology.training.regularization import BaseRegularization
from googlehydrology.utils.config import Config

ONE_OVER_2PI_SQUARED = 1.0 / np.sqrt(2.0 * np.pi)


class BaseLoss(torch.nn.Module):
    """Base loss class.

    All losses extend this class by implementing `_get_loss`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    prediction_keys : list[str]
        List of keys that will be predicted. During the forward pass, the passed `prediction` dict
        must contain these keys. Note that the keys listed here should be without frequency identifier.
    ground_truth_keys : list[str]
        List of ground truth keys that will be needed to compute the loss. During the forward pass, the
        passed `data` dict must contain these keys. Note that the keys listed here should be without
        frequency identifier.
    additional_data : list[str], optional
        Additional list of keys that will be taken from `data` in the forward pass to compute the loss.
        For instance, this parameter can be used to pass the variances that are needed to compute an NSE.
    output_size_per_target : int, optional
        Number of model outputs (per element in `prediction_keys`) connected to a single target variable, by default 1.
        For example for regression, one output (last dimension in `y_hat`) maps to one target variable. For mixture
        models (e.g. CMAL) the number of outputs per target corresponds to the number of distributions
        (`n_distributions`).
    per_sequence : bool, optional
        Keyword-only. If True, losses that support it average over the valid
        (non-NaN) observations of each sequence first and then over the
        sequences that have at least one valid observation, so every sequence
        gets the same total weight regardless of how many observations it
        has. This is what variational data assimilation needs, where each
        sequence is an independent inversion. Default False: the pooled mean
        over all valid observations of the batch, as used for training.
        Stored as ``self._per_sequence``; losses that do not support it
        ignore it.
    """

    def __init__(  # noqa: PLR0913
        self,
        cfg: Config,
        prediction_keys: list[str],
        ground_truth_keys: list[str],
        additional_data: list[str] = None,
        output_size_per_target: int = 1,
        *,
        per_sequence: bool = False,
    ):
        super(BaseLoss, self).__init__()
        self._per_sequence = per_sequence
        self._predict_last_n = _get_predict_last_n(cfg)
        # Frequency names of the run (if any) so that a dict override on a
        # single-frequency run can be validated against them.
        self._frequency_names = set(cfg.use_frequencies)
        if isinstance(cfg.predict_last_n, dict):
            self._frequency_names.update(cfg.predict_last_n)
        self._frequencies = [
            f
            for f in self._predict_last_n.keys()
            if f not in cfg.no_loss_frequencies
        ]
        self._output_size_per_target = output_size_per_target

        self._regularization_terms = []

        # names of ground truth and prediction keys to be unpacked and subset to predict_last_n items.
        self._prediction_keys = prediction_keys
        self._ground_truth_keys = ground_truth_keys

        # subclasses can use this list to register inputs to be unpacked during the forward call
        # and passed as kwargs to _get_loss() without subsetting.
        self._additional_data = []
        if additional_data is not None:
            self._additional_data = additional_data

        # all classes allow per-target weights for multi-target settings. By default, all targets are weighted equally
        if cfg.target_loss_weights is None:
            weights = torch.tensor(
                [
                    1 / len(cfg.target_variables)
                    for _ in range(len(cfg.target_variables))
                ]
            )
        else:
            if len(cfg.target_loss_weights) == len(cfg.target_variables):
                weights = torch.tensor(cfg.target_loss_weights)
            else:
                raise ValueError(
                    'Number of weights must be equal to the number of target variables'
                )
        self._target_weights = weights

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        data: dict[str, torch.Tensor],
        predict_last_n: int | dict[str, int] | None = None,
        other_model_data: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the loss.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Dictionary of predictions for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        data : dict[str, torch.Tensor]
            Dictionary of ground truth data for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        predict_last_n : int | dict[str, int] | None, optional
            Overrides the config's ``predict_last_n`` for this call only. An
            int applies to all frequencies; a dict maps frequencies to values
            (missing frequencies keep the config value). Values must be
            integers ``>= 1`` (``bool`` is rejected). On single-frequency runs
            a dict must use the key ``''`` or the run's frequency name. This
            lets callers outside training (e.g. gradient-based data
            assimilation) evaluate the loss over a different window. None uses
            the config value.

            The value is a suffix slice ``[:, -n:, :]`` of the raw prediction
            and target tensors as passed in. In forecast setups the last
            ``lead_time`` steps of both tensors are the forecast horizon, so a
            caller that wants the loss over the last ``n`` *observed* steps
            must drop the horizon from the tensors before calling (the DA
            engine does this).
        other_model_data : dict[str, torch.Tensor] | None, optional
            Extra entries merged into the dict passed as third argument to the
            regularization modules, on top of the non-prediction entries of
            `prediction` (entries here take precedence). None adds nothing.

        Returns
        -------
        torch.Tensor
            The overall calculated loss.
        dict[str, torch.Tensor]
            The individual components of the loss (e.g., regularization terms). 'total_loss' contains the overall loss.
        """
        # unpack loss-specific additional arguments
        kwargs = {key: data[key] for key in self._additional_data}

        predict_last_n = self._resolve_predict_last_n(predict_last_n)

        losses = []
        prediction_sub, ground_truth_sub = {}, {}
        for freq in self._frequencies:
            if predict_last_n[freq] == 0:
                continue  # no predictions for this frequency
            freq_suffix = '' if freq == '' else f'_{freq}'

            # apply predict_last_n and mask for all outputs of this frequency at once
            freq_pred, freq_gt = self._subset_in_time(
                {
                    key: prediction[f'{key}{freq_suffix}']
                    for key in self._prediction_keys
                },
                {
                    key: data[f'{key}{freq_suffix}']
                    for key in self._ground_truth_keys
                },
                predict_last_n[freq],
            )

            # remember subsets for multi-frequency component
            prediction_sub.update(
                {
                    f'{key}{freq_suffix}': freq_pred[key]
                    for key in freq_pred.keys()
                }
            )
            ground_truth_sub.update(
                {f'{key}{freq_suffix}': freq_gt[key] for key in freq_gt.keys()}
            )

            for n_target, weight in enumerate(self._target_weights):
                # subset the model outputs and ground truth corresponding to this particular target
                target_pred, target_gt = self._subset_target(
                    freq_pred, freq_gt, n_target
                )

                # model hook to subset additional data, which might be different for different losses
                kwargs_sub = self._subset_additional_data(kwargs, n_target)

                loss = self._get_loss(target_pred, target_gt, **kwargs_sub)
                losses.append(loss * weight)

        loss = torch.sum(torch.stack(losses))
        total_loss = loss.clone()
        all_losses = defaultdict(lambda: 0)
        all_losses['loss'] = loss
        reg_other = {
            k: v
            for k, v in prediction.items()
            if k not in self._prediction_keys
        }
        if other_model_data is not None:
            reg_other.update(other_model_data)
        for reg_module in self._regularization_terms:
            reg_out = reg_module(prediction_sub, ground_truth_sub, reg_other)
            total_loss += reg_module.weight * reg_out
            # One name may appear multiple times. We add all regularizations of the same name for logging purposes.
            all_losses[reg_module.name] += reg_out
        all_losses['total_loss'] = total_loss
        return total_loss, all_losses

    def _resolve_predict_last_n(
        self, predict_last_n: int | dict[str, int] | None
    ) -> dict[str, int]:
        """Return the per-frequency predict_last_n, applying a call override.

        The config-derived default (``None``) is returned untouched and may
        contain 0 for frequencies without predictions. Override values are
        validated: they must be integers (``numbers.Integral`` but not
        ``bool``) and ``>= 1``.
        """
        if predict_last_n is None:
            return self._predict_last_n
        if not isinstance(predict_last_n, dict):
            value = self._validate_predict_last_n_value(predict_last_n)
            return dict.fromkeys(self._predict_last_n, value)
        resolved = dict(self._predict_last_n)
        if list(resolved) == ['']:
            # single-frequency runs omit the frequency identifier
            if len(predict_last_n) != 1:
                raise ValueError(
                    'predict_last_n override for a single-frequency run must'
                    f' have exactly one entry, got {predict_last_n!r}.'
                )
            ((freq, value),) = predict_last_n.items()
            if freq != '' and freq not in self._frequency_names:
                known = sorted(self._frequency_names) or ['']
                raise ValueError(
                    f'predict_last_n override for unknown frequency {freq!r}.'
                    f' Known frequencies: {known}.'
                )
            resolved[''] = self._validate_predict_last_n_value(value)
            return resolved
        for freq, value in predict_last_n.items():
            if freq not in resolved:
                raise ValueError(
                    f'predict_last_n override for unknown frequency {freq!r}.'
                    f' Known frequencies: {list(resolved)}.'
                )
            resolved[freq] = self._validate_predict_last_n_value(value)
        return resolved

    @staticmethod
    def _validate_predict_last_n_value(value: object) -> int:
        """Return ``value`` as int if it is a non-bool integer ``>= 1``."""
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            # Repo convention: ValueError also for type problems.
            raise ValueError(  # noqa: TRY004
                'predict_last_n override values must be integers, got'
                f' {value!r} of type {type(value).__name__}.'
            )
        if value < 1:
            raise ValueError(
                f'predict_last_n override values must be >= 1, got {value}.'
            )
        return int(value)

    @staticmethod
    def _subset_in_time(
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        predict_last_n: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        ground_truth_sub = {
            key: gt[:, -predict_last_n:, :] for key, gt in ground_truth.items()
        }
        prediction_sub = {
            key: pred[:, -predict_last_n:, :]
            for key, pred in prediction.items()
        }

        return prediction_sub, ground_truth_sub

    def _subset_target(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        n_target: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # determine which output neurons correspond to the n_target target variable
        start = n_target * self._output_size_per_target
        end = (n_target + 1) * self._output_size_per_target
        prediction_sub = {
            key: pred[:, :, start:end] for key, pred in prediction.items()
        }

        # subset target by slicing to keep shape [bs, seq, 1]
        ground_truth_sub = {
            key: gt[:, :, n_target : n_target + 1]
            for key, gt in ground_truth.items()
        }

        return prediction_sub, ground_truth_sub

    @staticmethod
    def _subset_additional_data(
        additional_data: dict[str, torch.Tensor], n_target: int
    ) -> dict[str, torch.Tensor]:
        # by default, nothing happens
        return additional_data

    def _get_loss(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        **kwargs,
    ):
        raise NotImplementedError

    def set_regularization_terms(
        self, regularization_modules: list[BaseRegularization]
    ):
        """Register the passed regularization terms to be added to the loss function.

        Parameters
        ----------
        regularization_modules : list[BaseRegularization]
            List of regularization functions to be added to the loss during `forward`.
        """
        self._regularization_terms = regularization_modules


class MaskedMSELoss(BaseLoss):
    """Mean squared error loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y``.

    By default the squared errors are pooled over all valid observations of the
    batch (training). With ``per_sequence=True`` they are averaged within each
    sequence first and then over the sequences that have at least one valid
    observation, so every sequence gets the same total weight regardless of
    how many observations it has (variational data assimilation).

    Parameters
    ----------
    cfg : Config
        The run configuration.
    per_sequence : bool, optional
        Per-sequence reduction as described above. Default False.
    """

    def __init__(self, cfg: Config, *, per_sequence: bool = False):
        super(MaskedMSELoss, self).__init__(
            cfg,
            prediction_keys=['y_hat'],
            ground_truth_keys=['y'],
            per_sequence=per_sequence,
        )

    def _get_loss(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        **kwargs,
    ):
        mask = ~torch.isnan(ground_truth['y'])
        if not torch.any(mask):
            return prediction['y_hat'].sum() * 0.0
        if self._per_sequence:
            y = _fill_masked(ground_truth['y'], mask)
            squared_error = (prediction['y_hat'] - y) ** 2
            return 0.5 * _per_sequence_mean(squared_error, mask)
        loss = 0.5 * torch.mean(
            (prediction['y_hat'][mask] - ground_truth['y'][mask]) ** 2
        )
        return loss


class MaskedRMSELoss(BaseLoss):
    """Root mean squared error loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(MaskedRMSELoss, self).__init__(
            cfg, prediction_keys=['y_hat'], ground_truth_keys=['y']
        )

    def _get_loss(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        **kwargs,
    ):
        mask = ~torch.isnan(ground_truth['y'])
        if not torch.any(mask):
            return prediction['y_hat'].sum() * 0.0
        loss = torch.sqrt(
            0.5
            * torch.mean(
                (prediction['y_hat'][mask] - ground_truth['y'][mask]) ** 2
            )
        )
        return loss


class MaskedNSELoss(BaseLoss):
    """Basin-averaged Nash--Sutcliffe Model Efficiency Coefficient loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y`` and ``per_basin_target_stds``.

    A description of the loss function is available in [#]_.

    By default the scaled squared errors are pooled over all valid observations
    of the batch (training). With ``per_sequence=True`` they are averaged
    within each sequence first and then over the sequences that have at least
    one valid observation, so every sequence gets the same total weight
    regardless of how many observations it has (variational data
    assimilation).

    Parameters
    ----------
    cfg : Config
        The run configuration.
    eps: float, optional
        Small constant for numeric stability.
    per_sequence : bool, optional
        Per-sequence reduction as described above. Default False.

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
       universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
       *Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019
    """

    def __init__(
        self, cfg: Config, eps: float = 0.1, *, per_sequence: bool = False
    ):
        super(MaskedNSELoss, self).__init__(
            cfg,
            prediction_keys=['y_hat'],
            ground_truth_keys=['y'],
            additional_data=['per_basin_target_stds'],
            per_sequence=per_sequence,
        )
        self.eps = eps

    def _get_loss(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        **kwargs,
    ):
        mask = ~torch.isnan(ground_truth['y'])
        if not torch.any(mask):
            return prediction['y_hat'].sum() * 0.0
        if self._per_sequence:
            y = _fill_masked(ground_truth['y'], mask)
            stds = kwargs['per_basin_target_stds'].expand_as(
                prediction['y_hat']
            )
            stds = _fill_masked(stds, mask)
            weights = 1 / (stds + self.eps) ** 2
            scaled_error = weights * (prediction['y_hat'] - y) ** 2
            return _per_sequence_mean(scaled_error, mask)
        y_hat = prediction['y_hat'][mask]
        y = ground_truth['y'][mask]
        per_basin_target_stds = kwargs['per_basin_target_stds']
        # expand dimension 1 to predict_last_n
        per_basin_target_stds = per_basin_target_stds.expand_as(
            prediction['y_hat']
        )[mask]

        squared_error = (y_hat - y) ** 2
        weights = 1 / (per_basin_target_stds + self.eps) ** 2
        scaled_loss = weights * squared_error
        return torch.mean(scaled_loss)

    @staticmethod
    def _subset_additional_data(
        additional_data: dict[str, torch.Tensor], n_target: int
    ) -> dict[str, torch.Tensor]:
        # here we need to subset the per_basin_target_stds. We slice to keep the shape of [bs, seq, 1]
        return {
            key: value[:, :, n_target : n_target + 1]
            for key, value in additional_data.items()
        }


class MaskedCMALLoss(BaseLoss):
    """Average negative log-likelihood for a model that uses the CMAL head.

    The loss is averaged over observed timesteps. If all target timesteps are
    missing, it returns a differentiable zero.

    By default the negative log-likelihoods are pooled over all observed
    timesteps of the batch (training). With ``per_sequence=True`` they are
    averaged within each sequence first and then over the sequences that have
    at least one observation, so every sequence gets the same total weight
    regardless of how many observations it has (variational data
    assimilation).

    Parameters
    ----------
    cfg : Config
        The run configuration.
    eps : float, optional
        Small constant for numeric stability.
    per_sequence : bool, optional
        Per-sequence reduction as described above. Default False.
    """

    def __init__(
        self, cfg: Config, eps: float = 1e-8, *, per_sequence: bool = False
    ):
        super(MaskedCMALLoss, self).__init__(
            cfg,
            prediction_keys=['mu', 'b', 'tau', 'pi'],
            ground_truth_keys=['y'],
            output_size_per_target=cfg.n_distributions,
            per_sequence=per_sequence,
        )
        self.eps = eps  # stability epsilon

    def _get_loss(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        **kwargs,
    ):
        y = ground_truth['y'].squeeze(-1)
        mask = ~torch.isnan(y)
        if not torch.any(mask):
            return prediction['mu'].sum() * 0.0

        if self._per_sequence:
            nll = -self._log_likelihood(
                _fill_masked(y, mask).unsqueeze(-1),
                prediction['mu'],
                prediction['b'],
                prediction['tau'],
                prediction['pi'],
            )
            return _per_sequence_mean(nll, mask)

        y = y[mask].unsqueeze(-1)
        m = prediction['mu'][mask]
        b = prediction['b'][mask]
        t = prediction['tau'][mask]
        p = prediction['pi'][mask]

        return -torch.mean(self._log_likelihood(y, m, b, t, p))

    def _log_likelihood(
        self,
        y: torch.Tensor,
        m: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Log-likelihood of ``y`` under the mixture, mixture dim reduced."""
        error = y - m
        log_like = (
            torch.log(t)
            + torch.log(1.0 - t)
            - torch.log(b)
            - torch.max(t * error, (t - 1.0) * error) / b
        )
        log_weights = torch.log(p + self.eps)

        return torch.logsumexp(log_weights + log_like, dim=-1)


def _fill_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace entries where ``mask`` is False by zero.

    Used to take NaN targets out of the computation *before* any arithmetic:
    masking a NaN product afterwards would still back-propagate
    ``0 * NaN = NaN`` into the prediction.
    """
    return torch.where(mask, values, values.new_zeros(()))


def _per_sequence_mean(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Average ``values`` within each sequence, then over sequences.

    ``values`` and ``mask`` share the same shape with the batch dimension
    first. Entries where ``mask`` is False must already be finite (see
    `_fill_masked`). Each sequence is averaged over its own valid entries and
    the result is averaged over the sequences that have at least one valid
    entry, so every sequence gets the same total weight. The caller guarantees
    that at least one entry is valid.
    """
    n_valid = mask.flatten(1).sum(1)
    masked = torch.where(mask, values, values.new_zeros(()))
    per_sequence = masked.flatten(1).sum(1) / n_valid.clamp(min=1)
    return torch.mean(per_sequence[n_valid > 0])


def _get_predict_last_n(cfg: Config) -> dict:
    predict_last_n = cfg.predict_last_n
    if isinstance(predict_last_n, int):
        predict_last_n = {'': predict_last_n}
    if len(predict_last_n) == 1:
        predict_last_n = {
            '': list(predict_last_n.values())[0]
        }  # if there's only one frequency, we omit its identifier
    return predict_last_n
