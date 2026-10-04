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
def test_mixture_params_to_quantiles(sample_cmal_params):
    mu, b, tau, pi = sample_cmal_params
    quantiles = cmal_deterministic._mixture_params_to_quantiles(mu, b, tau, pi)
    assert quantiles.shape == (2, 5, 9)
    # Check quantiles are monotonically increasing along quantile dimension
    assert torch.all(quantiles[..., 1:] >= quantiles[..., :-1])


@pytest.mark.unit
def test_generate_predictions(sample_cmal_params):
    mu, b, tau, pi = sample_cmal_params
    preds = cmal_deterministic.generate_predictions(mu, b, tau, pi)
    # Shape should be [batch_size, seq_len, 10] (1 mean + 9 quantiles)
    assert preds.shape == (2, 5, 10)
    assert torch.all(torch.isfinite(preds))


@pytest.mark.unit
@pytest.mark.parametrize('mu_val', [10.0, 0.5, -5.0, -10.0])
@pytest.mark.parametrize('tau_val', [0.25, 0.5, 0.75])
@pytest.mark.parametrize('q_val', [0.05, 0.1, 0.5, 0.9, 0.95])
def test_search_quantile_identical_kernels_unbiased(
    mu_val: float, tau_val: float, q_val: float
):
    """Degenerate (identical-component) mixtures must return exact PPF without bias."""
    batch_size, seq_len, n_kernels = 2, 3, 4
    mu = torch.full((batch_size, seq_len, n_kernels, 1), mu_val)
    b = torch.full((batch_size, seq_len, n_kernels, 1), 1.5)
    tau = torch.full((batch_size, seq_len, n_kernels, 1), tau_val)
    pi = torch.full((batch_size, seq_len, n_kernels, 1), 1.0 / n_kernels)

    q = torch.tensor([q_val]).view(1, 1, 1, 1)
    expected = cmal_deterministic._ppf(q, mu[:, :, :1, :], b[:, :, :1, :], tau[:, :, :1, :]).squeeze(2)
    actual = cmal_deterministic._search_quantile(q, mu, b, tau, pi)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_search_quantile_negative_and_asymmetric_mixtures_invert_cdf():
    """Verify Newton-Raphson converges across negative and asymmetric mixtures and stays within [min_ppf, max_ppf]."""
    mu = torch.tensor([[[-12.0, -6.0, -2.0], [3.0, 8.0, 15.0]]]).unsqueeze(-1)
    b = torch.tensor([[[0.8, 1.5, 2.2], [1.0, 2.0, 1.2]]]).unsqueeze(-1)
    tau = torch.tensor([[[0.2, 0.5, 0.8], [0.3, 0.6, 0.7]]]).unsqueeze(-1)
    pi = torch.tensor([[[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]]).unsqueeze(-1)

    quantiles = torch.tensor([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).view(1, 1, 1, -1)
    q_hat = cmal_deterministic._search_quantile(
        quantiles, mu, b, tau, pi, iterations=25
    )

    ppfs = cmal_deterministic._ppf(quantiles, mu, b, tau)
    min_ppf = torch.min(ppfs, dim=2).values
    max_ppf = torch.max(ppfs, dim=2).values
    assert torch.all(q_hat >= min_ppf - 1e-6)
    assert torch.all(q_hat <= max_ppf + 1e-6)

    # Evaluating the mixture CDF at each estimated quantile should recover `quantiles`
    recovered_cdf, _ = cmal_deterministic._mixture_cdf_and_pdf(
        q_hat.unsqueeze(2), mu, b, tau, pi
    )
    torch.testing.assert_close(
        recovered_cdf.squeeze(2),
        quantiles.view(1, 1, -1).expand_as(recovered_cdf.squeeze(2)),
        rtol=1e-3,
        atol=1e-3,
    )

