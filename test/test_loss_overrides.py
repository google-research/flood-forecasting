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

"""Tests for the per-call overrides of `BaseLoss.forward`."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from googlehydrology.training import get_loss_obj, get_regularization_obj
from googlehydrology.training.loss import (
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
)
from googlehydrology.training.regularization import (
    BackgroundEmbeddingRegularization,
)
from googlehydrology.utils.config import Config

_EPS = BackgroundEmbeddingRegularization._EPS  # noqa: SLF001


@pytest.fixture
def make_cfg(get_config: Callable[[str], Config]) -> Callable[..., Config]:
    """Build real run configurations with per-test overrides."""

    def _make(**updates: object) -> Config:
        config = get_config('forecast')
        config.update_config(updates)
        return config

    return _make


def _data(
    seq_len: int = 6, seed: int = 0, n_targets: int = 1
) -> tuple[dict, dict]:
    """Random single-frequency prediction and ground truth."""
    gen = torch.Generator().manual_seed(seed)
    y_hat = torch.randn(2, seq_len, n_targets, generator=gen)
    y = torch.randn(2, seq_len, n_targets, generator=gen)
    stds = torch.rand(2, 1, n_targets, generator=gen) + 0.5
    return {'y_hat': y_hat}, {'y': y, 'per_basin_target_stds': stds}


_LOSSES = [MaskedMSELoss, MaskedRMSELoss, MaskedNSELoss]


@pytest.mark.unit
@pytest.mark.parametrize('loss_cls', _LOSSES)
@pytest.mark.parametrize('override', [3, np.int64(3), {'1D': 3}, {'': 3}])
def test_predict_last_n_override_matches_config(
    make_cfg: Callable[..., Config], loss_cls: type, override: int | dict
) -> None:
    """A per-call override gives the same loss as configuring that value."""
    prediction, data = _data()
    expected, _ = loss_cls(make_cfg(predict_last_n={'1D': 3}))(prediction, data)
    actual, _ = loss_cls(make_cfg(predict_last_n={'1D': 6}))(
        prediction, data, predict_last_n=override
    )
    assert torch.equal(actual, expected)


@pytest.mark.unit
def test_predict_last_n_override_multi_frequency(
    make_cfg: Callable[..., Config],
) -> None:
    """Dict overrides touch only the given frequencies; unknown ones raise."""
    cfg = make_cfg(predict_last_n={'1D': 2, '1h': 4})
    cfg_expected = make_cfg(predict_last_n={'1D': 1, '1h': 4})
    gen = torch.Generator().manual_seed(3)
    prediction = {
        'y_hat_1D': torch.randn(1, 3, 1, generator=gen),
        'y_hat_1h': torch.randn(1, 5, 1, generator=gen),
    }
    data = {
        'y_1D': torch.randn(1, 3, 1, generator=gen),
        'y_1h': torch.randn(1, 5, 1, generator=gen),
    }
    expected, _ = MaskedMSELoss(cfg_expected)(prediction, data)
    actual, _ = MaskedMSELoss(cfg)(prediction, data, predict_last_n={'1D': 1})
    assert torch.equal(actual, expected)

    with pytest.raises(ValueError, match='unknown frequency'):
        MaskedMSELoss(cfg)(prediction, data, predict_last_n={'3h': 1})


@pytest.mark.unit
@pytest.mark.parametrize(
    ('override', 'match'),
    [
        (True, 'must be integers'),
        (3.0, 'must be integers'),
        (0, 'must be >= 1'),
        (-3, 'must be >= 1'),
        ({'1D': 0}, 'must be >= 1'),
        ({'1D': False}, 'must be integers'),
        ({'3h': 2}, 'unknown frequency'),
        ({'1D': 2, '3h': 2}, 'exactly one entry'),
    ],
)
def test_predict_last_n_override_invalid_raises(
    make_cfg: Callable[..., Config], override: object, match: str
) -> None:
    """Bools, non-ints, values < 1 and unknown frequencies raise ValueError."""
    prediction, data = _data()
    loss_fn = MaskedMSELoss(make_cfg(predict_last_n={'1D': 6}))
    with pytest.raises(ValueError, match=match):
        loss_fn(prediction, data, predict_last_n=override)


@pytest.mark.unit
def test_predict_last_n_override_unnamed_single_frequency(
    make_cfg: Callable[..., Config],
) -> None:
    """Without frequency names in the config only the '' key is accepted."""
    prediction, data = _data()
    loss_fn = MaskedMSELoss(make_cfg(predict_last_n=6))
    expected, _ = MaskedMSELoss(make_cfg(predict_last_n=3))(prediction, data)
    actual, _ = loss_fn(prediction, data, predict_last_n={'': 3})
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match='unknown frequency'):
        loss_fn(prediction, data, predict_last_n={'1D': 3})


@pytest.mark.unit
def test_config_default_still_allows_zero(
    make_cfg: Callable[..., Config],
) -> None:
    """A config-derived 0 for a frequency is skipped, so only 1h counts."""
    cfg = make_cfg(predict_last_n={'1D': 0, '1h': 4})
    gen = torch.Generator().manual_seed(4)
    prediction = {
        'y_hat_1D': torch.randn(1, 3, 1, generator=gen),
        'y_hat_1h': torch.randn(1, 5, 1, generator=gen),
    }
    data = {
        'y_1D': torch.randn(1, 3, 1, generator=gen),
        'y_1h': torch.randn(1, 5, 1, generator=gen),
    }
    loss, _ = MaskedMSELoss(cfg)(prediction, data)
    expected = 0.5 * torch.mean(
        (prediction['y_hat_1h'][:, -4:] - data['y_1h'][:, -4:]) ** 2
    )
    torch.testing.assert_close(loss, expected)


@pytest.mark.unit
@pytest.mark.parametrize('loss_cls', _LOSSES)
def test_all_nan_window_returns_differentiable_zero(
    make_cfg: Callable[..., Config], loss_cls: type
) -> None:
    """An all-NaN target window gives a finite zero that supports backward."""
    prediction, data = _data()
    prediction['y_hat'].requires_grad_()
    data['y'][:] = float('nan')
    loss, _ = loss_cls(make_cfg(predict_last_n=4))(prediction, data)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    loss.backward()
    assert torch.isfinite(prediction['y_hat'].grad).all()


@pytest.mark.unit
@pytest.mark.parametrize('loss_cls', _LOSSES)
def test_all_nan_target_contributes_zero_in_multi_target(
    make_cfg: Callable[..., Config], loss_cls: type
) -> None:
    """With two targets, an all-NaN target leaves only the other's loss."""
    cfg = make_cfg(
        predict_last_n=4,
        target_variables=['a', 'b'],
        target_loss_weights=[1.0, 1.0],
    )
    prediction, data = _data(n_targets=2)
    prediction['y_hat'].requires_grad_()
    data['y'][..., 0] = float('nan')
    loss, _ = loss_cls(cfg)(prediction, data)

    single = loss_cls(make_cfg(predict_last_n=4))
    expected, _ = single(
        {'y_hat': prediction['y_hat'][..., 1:].detach()},
        {k: v[..., 1:] for k, v in data.items()},
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    grad = prediction['y_hat'].grad
    assert torch.isfinite(grad).all()
    assert torch.equal(grad[..., 0], torch.zeros_like(grad[..., 0]))
    assert (grad[:, -4:, 1] != 0).all()


@pytest.mark.unit
def test_default_path_unchanged_vs_explicit_none(
    make_cfg: Callable[..., Config],
) -> None:
    """Passing the overrides as None is identical to omitting them."""
    prediction, data = _data()
    loss_fn = MaskedNSELoss(make_cfg(predict_last_n=4))
    default, default_all = loss_fn(prediction, data)
    explicit, explicit_all = loss_fn(
        prediction, data, predict_last_n=None, other_model_data=None
    )
    assert torch.equal(default, explicit)
    assert dict(default_all).keys() == dict(explicit_all).keys()
    for key, value in default_all.items():
        torch.testing.assert_close(explicit_all[key], value)


@pytest.mark.unit
@pytest.mark.parametrize('loss_name', ['mse', 'nse'])
def test_other_model_data_reaches_regularization_with_gradient(
    make_cfg: Callable[..., Config], loss_name: str
) -> None:
    """The background term is added to the loss and its gradient is analytic."""
    reg_weight, comp_weight = 0.5, 2.0
    cfg = make_cfg(
        loss=loss_name,
        predict_last_n=4,
        regularization=[('bg_embedding', reg_weight)],
    )
    loss_fn = get_loss_obj(cfg)
    loss_fn.set_regularization_terms(get_regularization_obj(cfg))

    prediction, data = _data()
    gen = torch.Generator().manual_seed(1)
    opt = torch.randn(3, 5, 8, generator=gen, requires_grad=True)
    base = torch.randn(3, 5, 8, generator=gen)
    other = {
        'optimized_components': {'emb': opt},
        'baseline_components': {'emb': base},
        'component_weights': {'emb': comp_weight},
    }

    data_loss, _ = get_loss_obj(cfg)(prediction, data)  # no regularizer
    total_loss, all_losses = loss_fn(prediction, data, other_model_data=other)

    scale = (base**2).mean(dim=(1, 2)) + _EPS
    departure = ((opt - base) ** 2).mean(dim=(1, 2))
    reg_expected = comp_weight * (departure / scale).mean()
    torch.testing.assert_close(all_losses['bg_embedding'], reg_expected)
    torch.testing.assert_close(
        total_loss, data_loss + reg_weight * reg_expected
    )

    total_loss.backward()
    batch, n_elem = opt.shape[0], opt[0].numel()
    grad_expected = (
        reg_weight
        * comp_weight
        * 2
        * (opt.detach() - base)
        / (n_elem * scale[:, None, None] * batch)
    )
    torch.testing.assert_close(opt.grad, grad_expected)


@pytest.mark.unit
def test_nan_targets_are_masked_and_regularizer_still_contributes(
    make_cfg: Callable[..., Config],
) -> None:
    """NaN targets are masked out while the background term still applies."""
    cfg = make_cfg(
        loss='mse', predict_last_n=4, regularization=['bg_embedding']
    )
    loss_fn = get_loss_obj(cfg)
    loss_fn.set_regularization_terms(get_regularization_obj(cfg))

    prediction, data = _data()
    data['y'][0, -1, 0] = float('nan')
    opt = torch.ones(2, 4, requires_grad=True)
    other = {
        'optimized_components': {'emb': opt},
        'baseline_components': {'emb': torch.full((2, 4), 2.0)},
    }
    total_loss, all_losses = loss_fn(prediction, data, other_model_data=other)
    total_loss.backward()

    # Data term: masked MSE over the 7 valid entries of the last 4 steps.
    y_hat, y = prediction['y_hat'][:, -4:], data['y'][:, -4:]
    mask = ~torch.isnan(y)
    assert int(mask.sum()) == 7  # noqa: PLR2004
    data_expected = 0.5 * torch.mean((y_hat[mask] - y[mask]) ** 2)
    torch.testing.assert_close(all_losses['loss'], data_expected)
    # Background term: (1 - 2)^2 / (2^2 + eps) for every element/sequence.
    reg_expected = torch.tensor(1.0 / (4.0 + _EPS))
    torch.testing.assert_close(all_losses['bg_embedding'], reg_expected)
    torch.testing.assert_close(total_loss, data_expected + reg_expected)
    assert torch.isfinite(opt.grad).all()
