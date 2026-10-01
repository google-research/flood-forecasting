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

"""Tests for `BackgroundEmbeddingRegularization`."""

from collections.abc import Callable

import pytest
import torch

from googlehydrology.training import get_regularization_obj
from googlehydrology.training.regularization import (
    BackgroundEmbeddingRegularization,
)
from googlehydrology.utils.config import Config

_EPS = BackgroundEmbeddingRegularization._EPS  # noqa: SLF001


@pytest.fixture
def cfg(get_config: Callable[[str], Config]) -> Config:
    """Real run configuration with the background regularizer enabled."""
    config = get_config('forecast')
    config.update_config({'regularization': ['bg_embedding']})
    return config


def _manual(
    opt: torch.Tensor, base: torch.Tensor, w: float = 1.0
) -> torch.Tensor:
    """Masked per-sequence normalised departure by hand (skips empty seqs)."""
    valid = torch.isfinite(opt) & torch.isfinite(base)
    opt0 = torch.where(valid, opt, 0.0)
    base0 = torch.where(valid, base, 0.0)
    n = valid.flatten(1).sum(1)
    departure = ((opt0 - base0) ** 2).flatten(1).sum(1) / n.clamp(min=1)
    scale = (base0**2).flatten(1).sum(1) / n.clamp(min=1) + _EPS
    return w * (departure / scale)[n > 0].mean()


def _components(seed: int = 0) -> tuple[dict, dict]:
    """Random optimized/baseline pairs with shapes [B, E] and [B, W, E]."""
    gen = torch.Generator().manual_seed(seed)
    opt = {
        'static': torch.randn(4, 8, generator=gen, requires_grad=True),
        'dynamic': torch.randn(4, 5, 8, generator=gen, requires_grad=True),
    }
    base = {
        'static': torch.randn(4, 8, generator=gen),
        'dynamic': torch.randn(4, 5, 8, generator=gen),
    }
    return opt, base


def _other(opt: dict, base: dict, weights: dict | None = None) -> dict:
    """Assemble the ``other_model_data`` dict."""
    other = {'optimized_components': opt, 'baseline_components': base}
    if weights is not None:
        other['component_weights'] = weights
    return other


def _run(
    cfg: Config, opt: torch.Tensor, base: torch.Tensor, w: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward and backward on a fresh leaf; returns (value, grad)."""
    leaf = opt.detach().clone().requires_grad_()
    out = BackgroundEmbeddingRegularization(cfg)(
        {}, {}, _other({'c': leaf}, {'c': base}, {'c': w})
    )
    out.backward()
    return out.detach(), leaf.grad


@pytest.mark.unit
@pytest.mark.parametrize('component', ['static', 'dynamic'])
def test_matches_manual_normalised_formula(cfg: Config, component: str) -> None:
    """The term equals the per-sequence normalised departure for 2-D and 3-D."""
    opt, base = _components()
    reg = BackgroundEmbeddingRegularization(cfg)
    out = reg(
        {},
        {},
        _other({component: opt[component]}, {component: base[component]}),
    )
    torch.testing.assert_close(out, _manual(opt[component], base[component]))
    assert reg.name == 'bg_embedding'


@pytest.mark.unit
def test_component_weights_honoured_and_default_to_one(cfg: Config) -> None:
    """Per-component weights scale each term; missing weights default to 1."""
    opt, base = _components()
    reg = BackgroundEmbeddingRegularization(cfg)
    weighted = reg({}, {}, _other(opt, base, {'static': 1e-3, 'dynamic': 0.5}))
    expected = _manual(opt['static'], base['static'], 1e-3) + _manual(
        opt['dynamic'], base['dynamic'], 0.5
    )
    torch.testing.assert_close(weighted, expected)

    partial = reg({}, {}, _other(opt, base, {'dynamic': 0.0}))
    torch.testing.assert_close(partial, _manual(opt['static'], base['static']))


@pytest.mark.unit
def test_scale_invariance(cfg: Config) -> None:
    """Scaling optimized and baseline by 10 leaves the term unchanged."""
    opt, base = _components()
    reg = BackgroundEmbeddingRegularization(cfg)
    small = reg({}, {}, _other(opt, base))
    scaled_opt = {k: v * 10 for k, v in opt.items()}
    scaled_base = {k: v * 10 for k, v in base.items()}
    large = reg({}, {}, _other(scaled_opt, scaled_base))
    # Only _EPS (1e-3 on a unit-norm background) breaks exact invariance.
    torch.testing.assert_close(large, small, rtol=2 * _EPS, atol=0)


@pytest.mark.unit
def test_baseline_is_detached(cfg: Config) -> None:
    """Gradients reach the optimized tensors but never the baselines."""
    opt, base = _components()
    base = {k: v.clone().requires_grad_() for k, v in base.items()}
    out = BackgroundEmbeddingRegularization(cfg)({}, {}, _other(opt, base))
    out.backward()
    for name in opt:
        assert torch.isfinite(opt[name].grad).all()
        assert base[name].grad is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ('other', 'match'),
    [
        ({}, 'requires non-empty'),
        (
            {'optimized_components': {}, 'baseline_components': {}},
            'requires non-empty',
        ),
        (
            {'optimized_components': {'a': torch.ones(2, 3)}},
            'requires non-empty',
        ),
        (
            {'baseline_components': {'a': torch.ones(2, 3)}},
            'requires non-empty',
        ),
        (
            _other({'a': torch.ones(2, 3)}, {}),
            'requires non-empty',
        ),
        (
            _other({}, {'a': torch.ones(2, 3)}),
            'requires non-empty',
        ),
        (
            _other({'a': torch.ones(2, 3)}, {'b': torch.ones(2, 3)}),
            "No baseline for component 'a'",
        ),
        (
            _other({'a': torch.ones(8)}, {'a': torch.ones(8)}),
            'batch dimension',
        ),
        (
            _other({'a': torch.ones(4, 1, 8)}, {'a': torch.ones(4, 5, 8)}),
            r'optimized shape \(4, 1, 8\)',
        ),
    ],
)
def test_invalid_inputs_raise(cfg: Config, other: dict, match: str) -> None:
    """Missing/empty dicts, missing baselines, 1-D and mismatched shapes."""
    with pytest.raises(ValueError, match=match):
        BackgroundEmbeddingRegularization(cfg)({}, {}, other)


@pytest.mark.unit
def test_zero_weight_component_is_skipped_even_if_nan(cfg: Config) -> None:
    """A zero-weight component is skipped before its tensors are touched."""
    opt = {'bad': torch.randn(4, 8, requires_grad=True)}
    base = {'bad': torch.full((4, 8), float('nan'))}
    out = BackgroundEmbeddingRegularization(cfg)(
        {}, {}, _other(opt, base, {'bad': 0.0})
    )
    torch.testing.assert_close(out, torch.tensor(0.0))

    # Zero weight also bypasses the validation of that component.
    out = BackgroundEmbeddingRegularization(cfg)(
        {},
        {},
        _other({'no_baseline': torch.randn(4)}, base, {'no_baseline': 0.0}),
    )
    torch.testing.assert_close(out, torch.tensor(0.0))


@pytest.mark.unit
@pytest.mark.parametrize('nan_in', ['base', 'opt', 'both'])
@pytest.mark.parametrize('component', ['static', 'dynamic'])
def test_nonfinite_baseline_with_positive_weight_is_masked(
    cfg: Config, nan_in: str, component: str
) -> None:
    """With w > 0, NaN entries are masked and other sequences are unaffected."""
    w = 1.5
    opt, base = _components(seed=7)
    opt, base = opt[component].detach(), base[component].clone()
    clean_value, clean_grad = _run(cfg, opt, base, w)

    opt_nan = opt.clone()
    nan_mask = torch.zeros_like(base, dtype=torch.bool)
    nan_mask[0, ..., :3] = True
    nan_mask[2, ..., 5] = True
    if nan_in in ('base', 'both'):
        base[nan_mask] = float('nan')
    if nan_in in ('opt', 'both'):
        opt_nan[nan_mask] = float('nan')

    value, grad = _run(cfg, opt_nan, base, w)
    assert torch.isfinite(value)
    torch.testing.assert_close(value, _manual(opt_nan, base, w))
    assert torch.isfinite(grad).all()
    # Masked positions get exactly zero gradient, the rest is non-zero.
    assert torch.equal(grad[nan_mask], torch.zeros_like(grad[nan_mask]))
    assert (grad[~nan_mask] != 0).all()
    # Sequences without NaN are untouched by the masking of others.
    untouched = [1, 3]
    torch.testing.assert_close(grad[untouched], clean_grad[untouched])
    assert not torch.isclose(value, clean_value)


@pytest.mark.unit
def test_all_nan_sequence_is_skipped(cfg: Config) -> None:
    """A sequence whose background is all NaN does not enter the mean."""
    opt, base = _components(seed=9)
    opt, base = opt['dynamic'].detach(), base['dynamic'].clone()
    base[1] = float('nan')
    value, grad = _run(cfg, opt, base)
    keep = [0, 2, 3]
    torch.testing.assert_close(value, _manual(opt[keep], base[keep]))
    assert torch.isfinite(grad).all()
    assert torch.equal(grad[1], torch.zeros_like(grad[1]))
    # The remaining sequences see a mean over 3 instead of 4 sequences.
    _, clean_grad = _run(cfg, opt, torch.nan_to_num(base))
    torch.testing.assert_close(grad[keep] * 3, clean_grad[keep] * 4)


@pytest.mark.unit
def test_all_sequences_nan_contributes_zero_with_grad_path(
    cfg: Config,
) -> None:
    """If no sequence is valid the term is zero but still differentiable."""
    opt = torch.randn(3, 4, requires_grad=True)
    base = torch.full((3, 4), float('nan'))
    out = BackgroundEmbeddingRegularization(cfg)(
        {}, {}, _other({'e': opt}, {'e': base})
    )
    assert out.requires_grad
    torch.testing.assert_close(out.detach(), torch.tensor(0.0))
    out.backward()
    assert torch.equal(opt.grad, torch.zeros_like(opt))


@pytest.mark.unit
def test_near_zero_baseline_does_not_explode(cfg: Config) -> None:
    """A tiny-norm background is bounded by _EPS instead of amplified."""
    gen = torch.Generator().manual_seed(11)
    direction = torch.randn(4, 8, generator=gen)
    base_unit = torch.randn(4, 8, generator=gen)
    base_tiny = base_unit * 1e-3
    # Same absolute departure from both backgrounds.
    unit_value, unit_grad = _run(cfg, base_unit + 0.1 * direction, base_unit)
    tiny_value, tiny_grad = _run(cfg, base_tiny + 0.1 * direction, base_tiny)
    # Per sequence i with m_i = mean(base_unit_i^2) the scales are
    # (m_i + _EPS) and (1e-6 * m_i + _EPS) >= _EPS, so value and gradient of
    # the tiny background exceed the unit case by at most
    # max_i (m_i + _EPS) / _EPS ~ 1 / _EPS, instead of ~1e6 without _EPS.
    m = (base_unit**2).mean(dim=1)
    bound = ((m + _EPS) / _EPS).max()
    assert torch.isfinite(tiny_value)
    assert torch.isfinite(tiny_grad).all()
    assert tiny_value <= bound * unit_value
    assert tiny_grad.abs().max() <= bound * unit_grad.abs().max()
    torch.testing.assert_close(
        tiny_value, _manual(base_tiny + 0.1 * direction, base_tiny)
    )


@pytest.mark.unit
def test_float16_computed_in_float32(cfg: Config) -> None:
    """float16 inputs give a finite result and gradients matching float32."""
    opt32, base32 = _components(seed=5)
    # Departure / scale ~ 1e3-1e4: fine in float32, lossy natively in float16.
    opt32 = {k: (v.detach() * 3).requires_grad_() for k, v in opt32.items()}
    base32 = {k: v * 0.05 for k, v in base32.items()}
    reg = BackgroundEmbeddingRegularization(cfg)
    ref = reg({}, {}, _other(opt32, base32))
    ref.backward()

    opt16 = {k: v.detach().half().requires_grad_() for k, v in opt32.items()}
    base16 = {k: v.half() for k, v in base32.items()}
    out = reg({}, {}, _other(opt16, base16))
    assert out.dtype == torch.float16
    assert torch.isfinite(out)
    out.backward()

    # Tolerances: the only float16 round-off is in the inputs (relative
    # 2^-11 ~ 5e-4 per entry) and in the final cast of the output and the
    # gradient (another 5e-4). The value and the gradient are ratios of
    # second moments, so input rounding enters roughly twice: ~2e-3
    # relative, well inside rtol=1e-2 / 2e-2. The gradient atol covers
    # entries whose float16 magnitude is below the normal range.
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    for name, tensor16 in opt16.items():
        assert torch.isfinite(tensor16.grad).all()
        torch.testing.assert_close(
            tensor16.grad.float(), opt32[name].grad, rtol=2e-2, atol=1e-3
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ('entry', 'weight'), [('bg_embedding', 1.0), (('bg_embedding', 0.25), 0.25)]
)
def test_linked_in_factory(
    cfg: Config, entry: str | tuple, weight: float
) -> None:
    """`get_regularization_obj` builds the module with the configured weight."""
    cfg.update_config({'regularization': [entry]})
    (reg,) = get_regularization_obj(cfg)
    assert isinstance(reg, BackgroundEmbeddingRegularization)
    assert reg.weight == weight
    opt, base = _components(seed=2)
    out = reg({}, {}, _other(opt, base))
    expected = _manual(opt['static'], base['static']) + _manual(
        opt['dynamic'], base['dynamic']
    )
    # The module returns the raw term; BaseLoss applies `reg.weight`.
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(reg.weight * out, weight * expected)
