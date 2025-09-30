"""Generates representative point predictions from CMAL head parameters.

This method is a deterministic alternative to random sampling CMAL, where there's
memory constraints on GPU and/or CPU. It takes 10 representative points from the
predictive dist, and 32 binary search iterations for quantiles. 10 points are
9 quantiles from 0.1 to 0.9 and a statistical mean of mixture dist.

When n_samples is low, this algorithm should serve as a better approximation.
"""

import torch


@torch.compile()
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
    tau_c = 1 - tau
    means = mu + b * tau * tau_c * (1 / tau**2 - 1 / tau_c**2)
    mean = torch.unsqueeze(torch.sum(pi * means, dim=-1), dim=-1)
    quantiles = _mixture_params_to_quantiles(mu, b, tau, pi)
    # Returned tensor, in last dimension, has the distribution mean followed by
    # the calculated quantiles.
    return torch.concat([mean, quantiles], dim=-1)


def _cdf(
    x: torch.Tensor, mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Computes the Cumulative Distribution Function (CDF) at x for the dists."""
    tau_c = 1 - tau
    return torch.where(
        x <= mu,
        tau * torch.exp(tau_c * (x - mu) / b),
        1 - tau_c * torch.exp(-tau * (x - mu) / b),
    )


def _pdf(
    x: torch.Tensor, mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Computes the Probability Density Function (PDF) at x, the derivative of CDF."""
    tau_c = 1.0 - tau
    indicator = (x > mu).float()
    main_term = tau * tau_c / b  # scaled exp func
    exp_term = torch.exp(
        -indicator * tau * (x - mu) / b - (1.0 - indicator) * tau_c * (mu - x) / b
    )
    return main_term * exp_term


def _mixture_pdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    b: torch.Tensor,
    tau: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """Calculates the PDF of the mixture distribution."""
    return torch.sum(_pdf(x, mu, b, tau) * pi, dim=2, keepdim=True)


def _ppf(
    quantile: torch.Tensor, mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Computes the Percent Point Function (PPF) = the quantile value.

    PPF is inverse of CDF.
    """
    tau_c = 1 - tau
    return torch.where(
        quantile <= tau,
        mu + (b / tau_c) * torch.log(quantile / tau),
        mu - (b / tau) * torch.log((1 - quantile) / tau_c),
    )


def _mixture_cdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    b: torch.Tensor,
    tau: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """Returns the CDF of the mixture dist.

    The CDF is the weighted sum of the CDFs of the components.
    """
    return torch.sum(_cdf(x, mu, b, tau) * pi, dim=2, keepdim=True)


def _search_quantile(
    quantile: torch.Tensor,
    mu: torch.Tensor,
    b: torch.Tensor,
    tau: torch.Tensor,
    pi: torch.Tensor,
    iterations: int = 8,
) -> torch.Tensor:
    """Search for the quantile of a mixture dist via newton-raphson (NR).

    NR works by: x_{n+1} = x_n - f(x_n) / f'(x_n)
    So f(x)  = mixture_cdf(x) - quantile
       f'(x) = CDF(x) dx = PDF(x)
    """
    k = torch.mean(_ppf(quantile, mu, b, tau), dim=2, keepdim=True)
    epsilon = 1e-6  # to avoid zero values

    for _ in range(iterations):
        cdf_val = _mixture_cdf(k, mu, b, tau, pi)
        pdf_val = _mixture_pdf(k, mu, b, tau, pi)
        k = k - (cdf_val - quantile) / (pdf_val + epsilon)

    return torch.squeeze(k, dim=2)


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
    quantiles = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        device=mu.device,
        dtype=mu.dtype,
    )
    return _search_quantile(quantiles.view(1, 1, 1, -1), mu_exp, b_exp, tau_exp, pi_exp)
