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

"""Tests for `cmal_deterministic.mixture_mean`."""

import pytest
import torch

from googlehydrology.utils import cmal_deterministic


def _params(seed=0, b_scale=1.0, mu_scale=1.0, shape=(3, 4, 5)):
    gen = torch.Generator().manual_seed(seed)
    mu = mu_scale * torch.randn(*shape, generator=gen)
    b = b_scale * (torch.rand(*shape, generator=gen) + 0.1)
    tau = torch.rand(*shape, generator=gen) * 0.98 + 0.01
    pi = torch.softmax(torch.randn(*shape, generator=gen), dim=-1)
    return mu, b, tau, pi


@pytest.mark.unit
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_mixture_mean_matches_generate_predictions(seed):
    mu, b, tau, pi = _params(seed)
    mean = cmal_deterministic.mixture_mean(mu, b, tau, pi)
    assert mean.shape == (*mu.shape[:-1], 1)

    # Bitwise identical to the eager (uncompiled) summary.
    eager = cmal_deterministic.generate_predictions.__wrapped__
    assert torch.equal(mean, eager(mu, b, tau, pi)[..., 0:1])

    # torch.compile may fuse ops differently, so allow last-ulp differences.
    summary = cmal_deterministic.generate_predictions(mu, b, tau, pi)
    torch.testing.assert_close(mean, summary[..., 0:1], rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_mixture_mean_gradient_finite_on_sharp_mixtures():
    mu, b, tau, pi = _params(seed=3, b_scale=1e-2, mu_scale=100.0)
    b = torch.full_like(b, 1e-2)
    for t in (mu, b, tau, pi):
        t.requires_grad_(True)

    cmal_deterministic.mixture_mean(mu, b, tau, pi).sum().backward()

    for t in (mu, b, tau, pi):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()
