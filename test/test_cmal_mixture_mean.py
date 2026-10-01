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


def _params(
    seed: int = 0,
    b_scale: float = 1.0,
    mu_scale: float = 1.0,
    shape: tuple[int, ...] = (3, 4, 5),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random CMAL parameters with tau well inside (0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    mu = mu_scale * torch.randn(*shape, generator=gen)
    b = b_scale * (torch.rand(*shape, generator=gen) + 0.1)
    tau = torch.rand(*shape, generator=gen) * 0.98 + 0.01
    pi = torch.softmax(torch.randn(*shape, generator=gen), dim=-1)
    return mu, b, tau, pi


@pytest.mark.unit
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_mixture_mean_matches_generate_predictions(seed: int) -> None:
    """The closed-form mean is bitwise identical to the eager summary mean."""
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
def test_mixture_mean_gradient_finite_on_sharp_mixtures() -> None:
    """Gradients stay finite for very narrow, widely separated components."""
    mu, b, tau, pi = _params(seed=3, b_scale=1e-2, mu_scale=100.0)
    b = torch.full_like(b, 1e-2)
    for t in (mu, b, tau, pi):
        t.requires_grad_()

    cmal_deterministic.mixture_mean(mu, b, tau, pi).sum().backward()

    for t in (mu, b, tau, pi):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize(
    'tau_value', [0.0, 1.0, 1e-7, 1 - 1e-7, 0.002, 1 - 0.002]
)
def test_mixture_mean_finite_at_tau_boundaries(
    tau_value: float, dtype: torch.dtype
) -> None:
    """Forward and backward are finite for tau at or beyond the clamp bounds.

    ``b`` is kept small so that the true ``tau`` gradient (about ``b / tau**2``)
    is representable in float16 for the interior values.
    """
    mu, b, _, pi = _params(seed=4, b_scale=1e-2)
    mu, b, pi = (t.to(dtype).requires_grad_() for t in (mu, b, pi))
    tau = torch.full_like(mu, tau_value, requires_grad=True)

    mean = cmal_deterministic.mixture_mean(mu, b, tau, pi)
    assert mean.dtype == dtype
    assert torch.isfinite(mean).all()

    mean.sum().backward()
    for t in (mu, b, tau, pi):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_mixture_mean_half_precision_large_gradient_is_finite(
    dtype: torch.dtype,
) -> None:
    """Reviewer's case tau=0.002, b=1: forward and all gradients are finite.

    The true tau gradient (about ``b / tau**2 = 2.5e5``) exceeds the float16
    range; the saturating upcast clamps it instead of producing ``inf``.
    """
    mu, _, _, pi = _params(seed=6)
    mu, pi = (t.to(dtype).requires_grad_() for t in (mu, pi))
    b = torch.ones_like(mu, requires_grad=True)
    tau = torch.full_like(mu, 0.002, requires_grad=True)

    mean = cmal_deterministic.mixture_mean(mu, b, tau, pi)
    assert mean.dtype == dtype
    assert torch.isfinite(mean).all()

    mean.sum().backward()
    for t in (mu, b, tau, pi):
        assert t.grad is not None
        assert t.grad.dtype == dtype
        assert torch.isfinite(t.grad).all()
    # The saturated gradient keeps the sign of the true gradient (negative)
    # and is non-zero: the clamp epsilon is the float32 one, not the input
    # dtype's (bfloat16 eps ~ 7.8e-3 would have clamped tau = 0.002 away).
    limit = torch.finfo(dtype).max
    assert (tau.grad.abs() <= limit).all()
    assert (tau.grad.float() < 0).all()
    assert (tau.grad != 0).all()
    if dtype == torch.float16:
        # ~1e5 exceeds the float16 range (65504) and saturates; bfloat16 has
        # the float32 exponent range, so nothing saturates there.
        assert (tau.grad.abs() == limit).any()

    # The forward value is the float32 result cast down.
    expected = cmal_deterministic.mixture_mean(
        *(t.detach().float() for t in (mu, b, tau, pi))
    ).to(dtype)
    torch.testing.assert_close(mean, expected)


@pytest.mark.unit
def test_saturating_upcast_clamps_gradient_to_finfo_max() -> None:
    """A float32 gradient of 1e6 becomes finfo(float16).max in float16."""
    x = torch.tensor([1.0, -1.0], dtype=torch.float16, requires_grad=True)
    y = cmal_deterministic._SaturatingUpcast.apply(x)  # noqa: SLF001
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, x.detach().float())

    y.backward(torch.tensor([1e6, -1e6], dtype=torch.float32))
    limit = torch.finfo(torch.float16).max
    assert x.grad.dtype == torch.float16
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(
        x.grad, torch.tensor([limit, -limit], dtype=torch.float16)
    )

    # Gradients inside the range are passed through unchanged.
    x2 = torch.tensor([2.0], dtype=torch.bfloat16, requires_grad=True)
    cmal_deterministic._SaturatingUpcast.apply(x2).backward(  # noqa: SLF001
        torch.tensor([3.0])
    )
    torch.testing.assert_close(
        x2.grad, torch.tensor([3.0], dtype=torch.bfloat16)
    )


@pytest.mark.unit
def test_float32_path_has_no_upcast_and_matches_eager() -> None:
    """float32 inputs do not go through the autograd function."""
    mu, b, tau, pi = _params(seed=10)
    tau = torch.full_like(tau, 0.002)
    mean = cmal_deterministic.mixture_mean(mu, b, tau, pi)
    eager = cmal_deterministic.generate_predictions.__wrapped__
    assert torch.equal(mean, eager(mu, b, tau, pi)[..., 0:1])


@pytest.mark.unit
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_mixture_mean_two_targets_match_generate_predictions_per_target(
    dtype: torch.dtype,
) -> None:
    """Per-target slices of a [B, T, 2*K] head output match the summary mean."""
    n_targets, k = 2, 4
    mu, b, tau, pi = _params(seed=12, shape=(3, 5, n_targets * k))
    # The CMAL head normalises pi over the full last dimension; the trainer
    # and sampler slice target n as [:, :, n*K:(n+1)*K] (see
    # `samplingutils._subset_target`). Renormalise per target so each slice
    # is a proper mixture.
    pi = torch.cat(
        [torch.softmax(pi[..., n * k : (n + 1) * k], -1) for n in range(2)],
        dim=-1,
    )
    mu, b, tau, pi = (t.to(dtype) for t in (mu, b, tau, pi))
    eager = cmal_deterministic.generate_predictions.__wrapped__
    for n in range(n_targets):
        sl = slice(n * k, (n + 1) * k)
        parts = (mu[..., sl], b[..., sl], tau[..., sl], pi[..., sl])
        mean = cmal_deterministic.mixture_mean(*parts)
        assert mean.shape == (3, 5, 1)
        assert mean.dtype == dtype
        reference = eager(*parts)[..., 0:1]
        if dtype == torch.float32:
            assert torch.equal(mean, reference)
        else:
            # bfloat16: the summary tensor is promoted to float32 by the
            # concat with the float32 quantiles; its mean column is the
            # same bfloat16 value, so casting back compares exactly.
            torch.testing.assert_close(mean, reference.to(dtype))
            fp32 = cmal_deterministic.mixture_mean(
                *(t.float() for t in parts)
            ).to(dtype)
            torch.testing.assert_close(mean, fp32)
    # The two targets are genuinely different mixtures.
    first = cmal_deterministic.mixture_mean(
        mu[..., :k], b[..., :k], tau[..., :k], pi[..., :k]
    )
    second = cmal_deterministic.mixture_mean(
        mu[..., k:], b[..., k:], tau[..., k:], pi[..., k:]
    )
    assert not torch.allclose(first, second)
