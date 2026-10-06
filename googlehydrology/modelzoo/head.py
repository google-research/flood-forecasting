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

import torch
import torch.nn as nn

from googlehydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


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
        head = CMAL(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_distributions=cfg.n_distributions,
        )
    elif cfg.head.lower() == '':
        raise ValueError(
            f"No 'head' specified in the config but is required for {cfg.model}"
        )
    else:
        raise NotImplementedError(
            f'{cfg.head} not implemented or not linked in `get_head()`'
        )

    return head


class BaseHead(nn.Module):
    """Base class of all model heads."""

    def point_prediction(
        self, outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Deterministic prediction derived from the head outputs.

        Returns a single value per time step and target, ``[B, T, n_targets]``,
        derived from the head outputs; used by data assimilation and other
        consumers that need a single value per step. For probabilistic heads
        this is the mean of the predicted distribution. The result stays
        differentiable w.r.t. the head outputs.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Output dict of ``forward``.

        Returns
        -------
        torch.Tensor
            Point prediction of shape ``[B, T, n_targets]``.
        """
        msg = f'{type(self).__name__} does not implement point_prediction'
        raise NotImplementedError(msg)


class Regression(BaseHead):
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

    def point_prediction(
        self, outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return the regression output ``outputs['y_hat']``."""
        return outputs['y_hat']


class CMAL(BaseHead):
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
    n_distributions : int | None, optional
        Number of mixture components per target. Only needed by
        ``point_prediction`` to separate the targets; if None, all components
        are assumed to belong to a single target.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing:
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 100,
        n_distributions: int | None = None,
    ):
        """Create the two-layer CMAL head; see the class docstring."""
        super(CMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.n_distributions = n_distributions

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
            Dictionary, containing the mixture component parameters and weights; where the key 'mu'stores the means,
            the key 'b' the scale parameters, the key 'tau' the skewness parameters, and the key 'pi' the weights).
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

        return {'mu': m, 'b': b, 'tau': t, 'pi': p}

    def point_prediction(
        self, outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Mean of the predicted mixture, ``[B, T, n_targets]``.

        Uses the closed-form mean of each asymmetric Laplacian component (same
        formula as the mean column of
        ``cmal_deterministic.generate_predictions``) and is differentiable;
        no quantile search is involved. The head outputs are laid out as
        ``[B, T, n_targets * n_distributions]`` with the components of each
        target contiguous.
        """
        mu, b, tau, pi = (outputs[k] for k in ('mu', 'b', 'tau', 'pi'))
        n_distributions = self.n_distributions or mu.shape[-1]
        shape = (*mu.shape[:-1], -1, n_distributions)
        mu, b, tau, pi = (e.reshape(shape) for e in (mu, b, tau, pi))
        pi = pi / pi.sum(dim=-1, keepdim=True)
        tau = torch.clamp(tau, min=1e-6, max=1.0 - 1e-6)
        means = mu + b * (1 - 2 * tau) / (tau * (1 - tau))
        return torch.sum(pi * means, dim=-1)
