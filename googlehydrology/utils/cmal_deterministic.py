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

"""Generates representative point predictions from CMAL head parameters.

This method is a deterministic alternative to random sampling CMAL, where there's
memory constraints on GPU and/or CPU. It takes 10 representative points from the
predictive dist, and searches for quantiles. 10 points are
9 quantiles from 0.1 to 0.9 and a statistical mean of mixture dist.

When n_samples is low, this algorithm should serve as a better approximation.
"""

import torch
import torch.cuda


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
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        # https://www.tandfonline.com/doi/abs/10.1080/03610920500199018 (modified)
        quantiles = _mixture_params_to_quantiles(mu, b, tau, pi)

        mean = mixture_mean(mu, b, tau, pi)
        # Returned tensor, in last dimension, has the distribution mean followed by
        # the calculated quantiles.
        return torch.concat([mean, quantiles], dim=-1)


class _SaturatingUpcast(torch.autograd.Function):
    """Upcast to float32 whose backward saturates instead of overflowing.

    The forward is ``x.float()``. The backward casts the float32 gradient back
    to the input dtype, but first clamps it to ``±finfo(dtype).max`` so that
    gradients beyond the half-precision range become finite, correctly-signed
    values rather than ``±inf``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        ctx.in_dtype = x.dtype
        return x.float()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        limit = torch.finfo(ctx.in_dtype).max
        return grad.clamp(min=-limit, max=limit).to(ctx.in_dtype)


def mixture_mean(
    mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, pi: torch.Tensor
) -> torch.Tensor:
    """Compute the exact mean of a CMAL mixture.

    Unlike the quantiles in `generate_predictions`, the mean has a closed form,
    so it is cheap and differentiable without the Newton-Raphson quantile
    search (e.g. for gradient-based data assimilation).

    Args:
        mu: location parameter
        b: scale parameter
        tau: asymmetry parameter
        pi: mixture weights

    Returns:
        Mixture mean with the last (mixture) dimension reduced to size 1.

    Note:
        Half precision inputs (float16/bfloat16, e.g. under autocast) are
        upcast to float32 for the computation and the result is cast back: in
        half precision the backward pass of ``1 / (tau * (1 - tau))``
        overflows for ``tau`` well inside ``(0, 1)`` (e.g. 0.002), and the
        resulting ``-inf + inf`` in the chain rule yields NaN gradients.
        ``tau`` is clamped to ``[1e-6, 1 - 1e-6]`` *after* the upcast (the
        epsilon is derived from the float32 computation dtype, not the input
        dtype), so the forward value in half precision is exactly the float32
        result cast down and small ``tau`` keep a real gradient. In float32
        the computation is unchanged (no upcast), so it stays bitwise
        identical to `generate_predictions`. The upcast is a custom autograd
        function whose
        backward saturates: gradients whose true magnitude (roughly
        ``b / tau**2``) exceeds the half-precision range are clamped to
        ``±finfo(dtype).max`` instead of overflowing to ``±inf``. This is a
        deliberate bias: a finite, correctly-signed gradient is more useful to
        a (clipped) optimizer than ``inf``.
    """
    out_dtype = tau.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        mu, b, tau, pi = (_SaturatingUpcast.apply(t) for t in (mu, b, tau, pi))
    # Epsilon of the computation dtype (float32 after the upcast): 1e-6.
    eps = max(1e-6, float(torch.finfo(tau.dtype).eps))
    tau = torch.clamp(tau, min=eps, max=1.0 - eps)
    means = mu + b * (1 - 2 * tau) / (tau * (1 - tau))
    return torch.sum(pi * means, dim=-1, keepdim=True).to(out_dtype)


def _cdf_and_pdf(
    x: torch.Tensor, mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the CDF and PDF (CDF') at x for the dists."""
    tau_c = 1.0 - tau
    indicator = x > mu
    z = (x - mu) / b

    cdf = torch.where(
        indicator, 1 - tau_c * torch.exp(-tau * z), tau * torch.exp(z * tau_c)
    )

    indicator = indicator.float()
    pdf = (tau * tau_c / b) * torch.exp(
        -indicator * tau * z - (1.0 - indicator) * tau_c * (-z)
    )

    return cdf, pdf


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


def _mixture_cdf_and_pdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    b: torch.Tensor,
    tau: torch.Tensor,
    pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the CDF and PDF of the mixture dist.

    The CDF is the weighted sum of the CDFs and PDFs of the components.
    """
    cdf, pdf = _cdf_and_pdf(x, mu, b, tau)
    mixture_cdf = torch.sum(cdf * pi, dim=2, keepdim=True)
    mixture_pdf = torch.sum(pdf * pi, dim=2, keepdim=True)
    return mixture_cdf, mixture_pdf


def _search_quantile(
    quantile: torch.Tensor,
    mu: torch.Tensor,
    b: torch.Tensor,
    tau: torch.Tensor,
    pi: torch.Tensor,
    iterations: int = 10,
    epsilon: float = 1e-6,  # to avoid zero values
    frac_confine: float = 0.8,  # to avoid overshooting
) -> torch.Tensor:
    """Search for the quantile of a mixture dist via newton-raphson (NR).

    NR works by: x_{n+1} = x_n - f(x_n) / f'(x_n)
    Need to find a root x for mixture_cdf(x) - quantile = 0
    So f(x)  = mixture_cdf(x) - quantile
       f'(x) = CDF(x) dx = PDF(x)
    """
    ppfs = _ppf(quantile, mu, b, tau)
    low = frac_confine * torch.min(ppfs, dim=2, keepdim=True).values
    high = frac_confine * torch.max(ppfs, dim=2, keepdim=True).values

    k = torch.mean(ppfs, dim=2, keepdim=True)
    for _ in range(iterations):
        cdf_val, pdf_val = _mixture_cdf_and_pdf(k, mu, b, tau, pi)
        k = k - (cdf_val - quantile) / (pdf_val + epsilon)
        k = torch.clamp(k, low, high)

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
    return _search_quantile(
        quantiles.view(1, 1, 1, -1), mu_exp, b_exp, tau_exp, pi_exp
    )
