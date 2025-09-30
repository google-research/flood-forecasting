"""Generates representative point predictions from CMAL head parameters.

This method is a deterministic alternative to random sampling CMAL, where there's
memory constraints on GPU and/or CPU. It takes 10 representative points from the
predictive dist, and 32 binary search iterations for quantiles. 10 points are
9 quantiles from 0.1 to 0.9 and a statistical mean of mixture dist.

When n_samples is low, this algorithm should serve as a better approximation.
"""

from typing import Callable

import torch


def generate_predictions(
    mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, pi: torch.Tensor
) -> torch.Tensor:
    """Generates predictions from a CMAL head: the dist mean followed by 9 quantiles.

    Calculates mean of mixture dists and quantiles as a summary of predicting dist.

    Args:
        mu: location parameter
        b: scale parameter
        tau: asymmetry parameter
        pi: mixture weights

    Returns:
        Summary dist where last dim has the dist mean followed by calculated quantiles.
    """
    # https://www.tandfonline.com/doi/abs/10.1080/03610920500199018
    means = mu + b * tau * (1 - tau) * (1 / tau**2 - 1 / (1 - tau) ** 2)
    mean = torch.unsqueeze(torch.sum(pi * means, dim=-1), dim=-1)
    quantiles = _mixture_params_to_quantiles(mu, b, tau, pi)
    # Returned tensor, in last dimension, has the distribution mean followed by
    # the calculated quantiles.
    return torch.concat([mean, quantiles], dim=-1)


def _cdf(
    x: torch.Tensor, mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Computes the Cumulative Distribution Function (CDF) at x for the dists."""
    return torch.where(
        x <= mu,
        tau * torch.exp((1 - tau) * (x - mu) / b),
        1 - (1 - tau) * torch.exp(-tau * (x - mu) / b),
    )


def _ppf(
    quantile: torch.Tensor, mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Computes the Percent Point Function (PPF) = the quantile value.

    PPF is inverse of CDF.
    """
    return torch.where(
        quantile <= tau,
        mu + (b / (1 - tau)) * torch.log(quantile / tau),
        mu - (b / tau) * torch.log((1 - quantile) / (1 - tau)),
    )


def _search_quantile(
    mixture_cdf_fn, quantile: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    """Binary searches for the quantile of a mixture dist."""
    # k shape: batch_size X sequence_length X num_kernels X len(quantile).
    k = 0.5 * (high + low)
    for _ in range(32):
        low = torch.where(mixture_cdf_fn(k) < quantile, k, low)
        high = torch.where(mixture_cdf_fn(k) > quantile, k, high)
        k = 0.5 * (high + low)

    # shape: batch_size X sequence_length X len(quantile).
    return torch.squeeze(k, dim=2)  # get rid of the kernels axis.


def _get_mixture_cdf_fn(
    mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, pi: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a func that calcs CDF of the mixture dist.

    The CDF is the weighted sum of the CDFs of the components.
    """
    return lambda x: torch.sum(_cdf(x, mu, b, tau) * pi, dim=2, keepdim=True)


def _calc_mixture_quantile(
    quantile: torch.Tensor,
    mu: torch.Tensor,
    b: torch.Tensor,
    tau: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """Calculates a quantile for the mixture of asymmetric laplace dists."""
    kernels_quantile_values = _ppf(quantile, mu, b, tau)
    # The high and low limits for the binary search are determined by the higest
    # and lowest quantiles among all mixture components (on the second axis).
    low, _ = torch.min(kernels_quantile_values, dim=2, keepdim=True)
    high, _ = torch.max(kernels_quantile_values, dim=2, keepdim=True)
    mixture_cdf_fn = _get_mixture_cdf_fn(mu, b, tau, pi)
    return _search_quantile(mixture_cdf_fn, quantile, low, high)


def _mixture_params_to_quantiles(
    mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, pi: torch.Tensor
) -> torch.Tensor:
    """Calculates predefined quantiles for the mixture dist."""
    # Add a dimension to broadcast with the different quantiles. Each parameter
    # tensor shape is batch_size X sequence_length X num_kernels X 1.
    mu_exp = torch.unsqueeze(mu, dim=3)
    b_exp = torch.unsqueeze(b, dim=3)
    tau_exp = torch.unsqueeze(tau, dim=3)
    pi_exp = torch.unsqueeze(pi, dim=3)

    # Returns a tensor shaped batch_size x seq_length x len(quantiles).
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles_tensor = torch.reshape(torch.tensor(quantiles), [1, 1, 1, -1])
    return _calc_mixture_quantile(quantiles_tensor, mu_exp, b_exp, tau_exp, pi_exp)
