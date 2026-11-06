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

"""Unit tests for googlehydrology.utils.cmal_deterministic."""

import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from scipy.stats import laplace_asymmetric

from googlehydrology.utils import cmal_deterministic


@pytest.fixture
def sample_cmal_params():
    batch_size = 2
    seq_len = 5
    n_kernels = 3

    # Define location, scale, asymmetry, and mixture weights
    mu = torch.zeros(batch_size, seq_len, n_kernels)
    b = torch.ones(batch_size, seq_len, n_kernels) * 2.0
    tau = torch.full((batch_size, seq_len, n_kernels), 0.5)
    pi = torch.full((batch_size, seq_len, n_kernels), 1.0 / n_kernels)

    return mu, b, tau, pi


@pytest.mark.unit
def test_cdf_and_pdf_properties():
    mu = torch.tensor([0.0])
    b = torch.tensor([1.0])
    tau = torch.tensor([0.5])

    # CDF at median (x = mu when tau = 0.5) should be 0.5
    cdf_median, pdf_median = cmal_deterministic._cdf_and_pdf(mu, mu, b, tau)
    assert np.isclose(cdf_median.item(), 0.5, atol=1e-5)
    assert pdf_median.item() > 0.0

    # CDF at large negative value should be close to 0
    cdf_low, _ = cmal_deterministic._cdf_and_pdf(
        torch.tensor([-10.0]), mu, b, tau
    )
    assert cdf_low.item() < 0.01

    # CDF at large positive value should be close to 1
    cdf_high, _ = cmal_deterministic._cdf_and_pdf(
        torch.tensor([10.0]), mu, b, tau
    )
    assert cdf_high.item() > 0.99

    # Monotonicity check
    xs = torch.linspace(-5, 5, 20)
    cdfs, _ = cmal_deterministic._cdf_and_pdf(xs, mu, b, tau)
    assert torch.all(cdfs[1:] >= cdfs[:-1])


@pytest.mark.unit
def test_ppf_inverse_cdf():
    mu = torch.tensor([1.5])
    b = torch.tensor([0.8])
    tau = torch.tensor([0.4])

    quantiles = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    xs = cmal_deterministic._ppf(quantiles, mu, b, tau)

    # Passing xs back into CDF should retrieve original quantiles
    recovered_cdfs, _ = cmal_deterministic._cdf_and_pdf(xs, mu, b, tau)
    assert torch.allclose(recovered_cdfs, quantiles, atol=1e-4)


@pytest.mark.unit
def test_mixture_cdf_and_pdf(sample_cmal_params):
    mu, b, tau, pi = sample_cmal_params
    x = torch.zeros(2, 5, 1)

    cdf_mix, pdf_mix = cmal_deterministic._mixture_cdf_and_pdf(
        x, mu, b, tau, pi
    )
    assert cdf_mix.shape == (2, 5, 1)
    assert pdf_mix.shape == (2, 5, 1)
    # Since mu=0, tau=0.5 for all components, CDF at 0 should be 0.5
    assert torch.allclose(cdf_mix, torch.tensor(0.5), atol=1e-5)


@pytest.mark.unit
def test_search_quantile(sample_cmal_params):
    mu, b, tau, pi = sample_cmal_params
    # Expand dims for search quantile
    mu_exp = torch.unsqueeze(mu, dim=3)
    b_exp = torch.unsqueeze(b, dim=3)
    tau_exp = torch.unsqueeze(tau, dim=3)
    pi_exp = torch.unsqueeze(pi, dim=3)

    q = torch.tensor([0.5]).view(1, 1, 1, 1)
    median = cmal_deterministic._search_quantile(
        q, mu_exp, b_exp, tau_exp, pi_exp
    )
    # Median of symmetric mixture centered at 0 should be ~0
    assert torch.allclose(median, torch.tensor(0.0), atol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('dtype', 'n_kernels'), [(torch.float32, 1), (torch.float64, 3)]
)
@pytest.mark.parametrize('shift', [-10.0, 10.0])
def test_search_quantile_identical_components(
    dtype: torch.dtype, n_kernels: int, shift: float
) -> None:
    """Single and identical components retain their analytic quantiles."""
    mu = torch.full((2, 3, n_kernels, 1), shift, dtype=dtype)
    b = torch.full_like(mu, 0.8)
    tau = torch.full_like(mu, 0.3)
    pi = torch.full_like(mu, 1.0 / n_kernels)
    q = torch.tensor([0.1, 0.5, 0.9], dtype=dtype).view(1, 1, 1, -1)

    actual = cmal_deterministic._search_quantile(q, mu, b, tau, pi)  # noqa: SLF001
    # SciPy uses kappa and scale, with left/right decay rates kappa^-1/kappa.
    reference = laplace_asymmetric(
        np.sqrt(0.3 / 0.7), loc=shift, scale=0.8 / np.sqrt(0.3 * 0.7)
    ).ppf(q.numpy().ravel())
    expected = torch.tensor(reference, dtype=dtype).expand(2, 3, -1)
    assert actual.shape == (2, 3, 3)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)


@pytest.mark.unit
@pytest.mark.parametrize('shift', [-10.0, 10.0])
def test_search_quantile_mixture_translation(shift: float) -> None:
    """Distinct components match an independent CDF inversion after shifting."""
    mu, b, tau, pi = (
        torch.tensor(values, dtype=torch.float64).view(1, 1, -1, 1)
        for values in (
            [-1.0, 0.2, 1.4],
            [0.7, 1.2, 0.9],
            [0.3, 0.55, 0.7],
            [0.2, 0.5, 0.3],
        )
    )
    q = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=mu.dtype)
    components = laplace_asymmetric(
        np.sqrt(tau.numpy().ravel() / (1 - tau.numpy().ravel())),
        loc=mu.numpy().ravel(),
        scale=b.numpy().ravel()
        / np.sqrt(tau.numpy().ravel() * (1 - tau.numpy().ravel())),
    )

    def cdf(x: float | np.ndarray) -> np.ndarray:
        return np.sum(components.cdf(x) * pi.numpy().ravel(), axis=-1)

    reference = torch.tensor(
        [
            brentq(lambda x, level=level: cdf(x) - level, -100, 100)
            for level in q.numpy()
        ],
        dtype=mu.dtype,
    ).view(1, 1, -1)
    base = cmal_deterministic._search_quantile(  # noqa: SLF001
        q.view(1, 1, 1, -1), mu, b, tau, pi
    )
    shifted = cmal_deterministic._search_quantile(  # noqa: SLF001
        q.view(1, 1, 1, -1), mu + shift, b, tau, pi
    )
    torch.testing.assert_close(base, reference, rtol=0, atol=1e-6)
    torch.testing.assert_close(shifted, reference + shift, rtol=0, atol=1e-6)
    torch.testing.assert_close(shifted, base + shift, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        cdf((shifted.numpy().ravel() - shift)[:, None]),
        q.numpy(),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.unit
def test_search_quantile_frac_confine_compatibility() -> None:
    """Legacy positional and keyword arguments cannot shrink the bracket."""
    mu = torch.full((1, 1, 1, 1), 10.0)
    b = torch.ones_like(mu)
    tau = torch.full_like(mu, 0.5)
    pi = torch.ones_like(mu)
    q = torch.tensor([0.9]).view(1, 1, 1, 1)
    expected = torch.distributions.Laplace(mu, 2 * b).icdf(q).squeeze(2)
    keyword = cmal_deterministic._search_quantile(  # noqa: SLF001
        q, mu, b, tau, pi, frac_confine=0.8
    )
    positional = cmal_deterministic._search_quantile(  # noqa: SLF001
        q, mu, b, tau, pi, 10, 1e-6, 1.0
    )
    torch.testing.assert_close(keyword, expected)
    torch.testing.assert_close(positional, expected)


@pytest.mark.unit
def test_mixture_params_to_quantiles(sample_cmal_params):
    mu, b, tau, pi = sample_cmal_params
    quantiles = cmal_deterministic._mixture_params_to_quantiles(mu, b, tau, pi)
    assert quantiles.shape == (2, 5, 9)
    # Check quantiles are monotonically increasing along quantile dimension
    assert torch.all(quantiles[..., 1:] >= quantiles[..., :-1])


@pytest.mark.unit
def test_generate_predictions(sample_cmal_params):
    mu, b, tau, pi = sample_cmal_params
    mu = mu + torch.tensor([-10.0, 10.0]).view(2, 1, 1)
    # Exercise the real compiled entry point and fail on compiler errors.
    with torch._dynamo.config.patch(suppress_errors=False):  # noqa: SLF001
        preds = cmal_deterministic.generate_predictions(mu, b, tau, pi)
    # Shape should be [batch_size, seq_len, 10] (1 mean + 9 quantiles)
    assert preds.shape == (2, 5, 10)
    assert preds.dtype == mu.dtype
    assert torch.all(torch.isfinite(preds))
    torch.testing.assert_close(preds[..., :1], mu[..., :1])
    # tau=0.5 is an ordinary Laplace distribution with scale 2*b.
    distribution = torch.distributions.Laplace(mu[..., :1], 2 * b[..., :1])
    q = torch.arange(1, 10, dtype=mu.dtype) / 10
    expected = distribution.icdf(q)
    torch.testing.assert_close(preds[..., 1:], expected, rtol=0, atol=2e-6)
