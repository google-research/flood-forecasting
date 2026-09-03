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

import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.utils.cmal_deterministic import generate_predictions
from googlehydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)
CMAL_MEDIAN_INDEX = 5


def ensure_y_hat(pred: Any, use_median: bool = True) -> Dict[str, Any]:
    """Ensures prediction dictionary contains 'y_hat' using CMAL median (q0.5) or mean."""
    if not isinstance(pred, dict):
        return {'y_hat': pred}
    res = dict(pred)
    if (
        'mu' in res
        and 'pi' in res
        and 'b' in res
        and 'tau' in res
        and isinstance(res['mu'], torch.Tensor)
    ):
        y_mean = calc_cmal_mean(res['mu'], res['b'], res['tau'], res['pi'])
        if use_median:
            try:
                cmal_summary = generate_predictions(
                    res['mu'], res['b'], res['tau'], res['pi']
                )
                y_med = cmal_summary[..., CMAL_MEDIAN_INDEX : CMAL_MEDIAN_INDEX + 1]
                # Element-wise NaN/Inf masking: substitute mean for non-finite entries
                invalid_mask = torch.isnan(y_med) | torch.isinf(y_med)
                if invalid_mask.any():
                    LOGGER.warning(
                        'CMAL median prediction contained non-finite values (NaN/Inf);'
                        ' masking invalid entries with calc_cmal_mean.'
                    )
                    res['y_hat'] = torch.where(invalid_mask, y_mean, y_med)
                else:
                    res['y_hat'] = y_med
                return res
            except (RuntimeError, ArithmeticError, ValueError) as e:
                LOGGER.warning(
                    f'CMAL generate_predictions raised exception ({e}); falling back to'
                    ' calc_cmal_mean.'
                )
        res['y_hat'] = y_mean
        return res
    if 'y_hat' not in res:
        if 'mu' in res and 'pi' in res:
            res['y_hat'] = torch.sum(res['pi'] * res['mu'], dim=-1, keepdim=True)
        elif 'mu' in res:
            mu = res['mu']
            if isinstance(mu, torch.Tensor):
                res['y_hat'] = mu if mu.ndim == 3 else mu.unsqueeze(-1)
            elif isinstance(mu, np.ndarray):
                res['y_hat'] = mu if mu.ndim == 3 else np.expand_dims(mu, -1)
    return res


def get_head(
    cfg: Config, n_in: int, n_out: int, *, n_hidden: int = 100
) -> nn.Module:
    """Get specific head module, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.
    n_hidden : int
        Size of the hidden layer.

    Returns
    -------
    nn.Module
        The model head, as specified in the run configuration.
    """
    if cfg.head.lower() == 'regression':
        head = Regression(
            n_in=n_in, n_out=n_out, activation=cfg.output_activation
        )
    elif cfg.head.lower() in ['cmal', 'cmal_deterministic']:
        head = CMAL(n_in=n_in, n_out=n_out, n_hidden=n_hidden)
    elif cfg.head.lower() == '':
        raise ValueError(
            f"No 'head' specified in the config but is required for {cfg.model}"
        )
    else:
        raise NotImplementedError(
            f'{cfg.head} not implemented or not linked in `get_head()`'
        )

    return head


class Regression(nn.Module):
    """Single-layer regression head with different output activations.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation : str, optional
        Output activation function. Can be specified in the config using the `output_activation` argument. Supported
        are {'linear', 'relu', 'softplus'}. If not specified (or an unsupported activation function is specified), will
        default to 'linear' activation.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = 'linear'):
        super(Regression, self).__init__()

        # TODO: Add multi-layer support
        layers = [nn.Linear(n_in, n_out)]
        if activation != 'linear':
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'softplus':
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(
                    f"## WARNING: Ignored output activation {activation} and used 'linear' instead."
                )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Perform a forward pass on the Regression head.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the model predictions in the 'y_hat' key.
        """
        return {'y_hat': self.net(x)}


def calc_cmal_mean(
    mu: torch.Tensor,
    b: torch.Tensor,
    tau: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Calculates the exact expected value (mean) of a CMAL mixture distribution.

    E[X] = sum( pi_k * ( mu_k + b_k * (1 - 2*tau_k) / (tau_k * (1 - tau_k)) ) )
    """
    denom = tau * (1.0 - tau) + eps
    shift = b * (1.0 - 2.0 * tau) / denom
    shift = torch.clamp(shift, min=-10.0, max=10.0)
    return torch.sum(pi * (mu + shift), dim=-1, keepdim=True)


class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians.

    An mixture density network with Laplace distributions as components.

    The CMAL-head uses an additional hidden layer to give it more expressiveness (same as a GMM-head).
    CMAL is better suited for many hydrological settings as it handles asymmetries with more ease. However, it is also
    more brittle than GMM and can more often throw exceptions. Details for CMAL can be found in [#]_.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 4 times the number of components.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing:
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(CMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Perform a CMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the CMAL components.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary, containing the mixture component parameters, weights, and expected value ('y_hat').
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = (
            self._softplus(b_latent) + self._eps
        )  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * torch.softmax(
            p_latent, dim=-1
        ) + self._eps  # sum(pi) = 1 & pi > 0

        y_hat = calc_cmal_mean(m, b, t, p)

        return {'mu': m, 'b': b, 'tau': t, 'pi': p, 'y_hat': y_hat}

