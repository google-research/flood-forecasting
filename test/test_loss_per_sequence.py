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

"""Tests for the ``per_sequence`` reduction of the masked losses."""

from collections.abc import Callable

import pytest
import torch

from googlehydrology.training import get_loss_obj
from googlehydrology.training.loss import (
    MaskedCMALLoss,
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
)
from googlehydrology.utils.config import Config

_LOSSES = ['mse', 'nse', 'cmal']
_N_DIST = 3
_SEQ_LEN = 10


@pytest.fixture
def make_cfg(get_config: Callable[[str], Config]) -> Callable[..., Config]:
    """Build real run configurations for a given loss name."""

    def _make(loss_name: str) -> Config:
        config = get_config('forecast')
        config.update_config(
            {
                'loss': loss_name,
                'predict_last_n': _SEQ_LEN,
                'n_distributions': _N_DIST,
            }
        )
        return config

    return _make


def _data(n_valid: tuple[int, ...], seed: int = 0) -> tuple[dict, dict]:
    """Predictions and targets where sequence ``i`` has ``n_valid[i]`` obs."""
    gen = torch.Generator().manual_seed(seed)
    batch = len(n_valid)
    shape = (batch, _SEQ_LEN, _N_DIST)
    y = torch.randn(batch, _SEQ_LEN, 1, generator=gen)
    for i, n in enumerate(n_valid):
        y[i, : _SEQ_LEN - n, 0] = float('nan')
    prediction = {
        'y_hat': torch.randn(batch, _SEQ_LEN, 1, generator=gen),
        'mu': torch.randn(*shape, generator=gen),
        'b': torch.rand(*shape, generator=gen) + 0.1,
        'tau': torch.rand(*shape, generator=gen) * 0.8 + 0.1,
        'pi': torch.softmax(torch.randn(*shape, generator=gen), dim=-1),
    }
    data = {
        'y': y,
        'per_basin_target_stds': torch.rand(batch, 1, 1, generator=gen) + 0.5,
    }
    return prediction, data


def _subset(tensors: dict, idx: list[int]) -> dict:
    """Select the given sequences and keep the batch dimension."""
    return {k: v[idx] for k, v in tensors.items()}


def _grad_key(loss_name: str) -> str:
    return 'mu' if loss_name == 'cmal' else 'y_hat'


def _pooled_by_hand(
    loss_name: str, prediction: dict, data: dict
) -> torch.Tensor:
    """Main's pooled formula over all valid observations of the batch."""
    y = data['y']
    mask = ~torch.isnan(y)
    if loss_name == 'mse':
        return 0.5 * torch.mean((prediction['y_hat'][mask] - y[mask]) ** 2)
    if loss_name == 'nse':
        stds = data['per_basin_target_stds'].expand_as(y)[mask]
        return torch.mean(
            (prediction['y_hat'][mask] - y[mask]) ** 2 / (stds + 0.1) ** 2
        )
    m = mask.squeeze(-1)
    yy = y.squeeze(-1)[m].unsqueeze(-1)
    mu, b, t, p = (prediction[k][m] for k in ('mu', 'b', 'tau', 'pi'))
    err = yy - mu
    log_like = (
        torch.log(t)
        + torch.log(1.0 - t)
        - torch.log(b)
        - torch.max(t * err, (t - 1.0) * err) / b
    )
    return -torch.mean(torch.logsumexp(torch.log(p + 1e-8) + log_like, dim=-1))


@pytest.mark.unit
@pytest.mark.parametrize('loss_name', _LOSSES)
def test_per_sequence_is_mean_of_per_sequence_losses(
    make_cfg: Callable[[str], Config], loss_name: str
) -> None:
    """per_sequence == 0.5 * (loss(seq0) + loss(seq1)) for 1 vs 10 obs."""
    cfg = make_cfg(loss_name)
    pooled_fn = get_loss_obj(cfg)
    per_seq_fn = get_loss_obj(cfg, per_sequence=True)
    assert per_seq_fn._per_sequence is True  # noqa: SLF001
    assert pooled_fn._per_sequence is False  # noqa: SLF001

    prediction, data = _data(n_valid=(1, 10))
    per_seq, _ = per_seq_fn(prediction, data)
    seq0, _ = pooled_fn(_subset(prediction, [0]), _subset(data, [0]))
    seq1, _ = pooled_fn(_subset(prediction, [1]), _subset(data, [1]))
    torch.testing.assert_close(per_seq, 0.5 * (seq0 + seq1))

    # The pooled loss weights the single observation of seq0 1/11, not 1/2.
    pooled, _ = pooled_fn(prediction, data)
    assert not torch.isclose(pooled, per_seq)


@pytest.mark.unit
@pytest.mark.parametrize('loss_name', _LOSSES)
def test_pooled_loss_normalises_by_total_valid_count(
    make_cfg: Callable[[str], Config], loss_name: str
) -> None:
    """Default reduction: sum of per-entry terms / total valid count (5)."""
    pooled_fn = get_loss_obj(make_cfg(loss_name))
    prediction, data = _data(n_valid=(1, 4))
    loss, _ = pooled_fn(prediction, data)
    # pooled = (sum_{seq0} e + sum_{seq1} e) / (1 + 4), where e is the
    # per-entry term; equivalently (1 * mean_seq0 + 4 * mean_seq1) / 5.
    seq0, _ = pooled_fn(_subset(prediction, [0]), _subset(data, [0]))
    seq1, _ = pooled_fn(_subset(prediction, [1]), _subset(data, [1]))
    torch.testing.assert_close(loss, (1 * seq0 + 4 * seq1) / 5)
    torch.testing.assert_close(
        loss, _pooled_by_hand(loss_name, prediction, data)
    )


@pytest.mark.unit
@pytest.mark.parametrize('loss_name', _LOSSES)
def test_per_sequence_gradient_independent_of_other_sequences(
    make_cfg: Callable[[str], Config], loss_name: str
) -> None:
    """The gradient on seq0 does not depend on how many obs seq1 has."""
    per_seq_fn = get_loss_obj(make_cfg(loss_name), per_sequence=True)
    key = _grad_key(loss_name)
    grads = []
    for n_valid_seq1 in (10, 3):
        prediction, data = _data(n_valid=(1, n_valid_seq1))
        prediction[key].requires_grad_()
        loss, _ = per_seq_fn(prediction, data)
        loss.backward()
        assert torch.isfinite(prediction[key].grad).all()
        grads.append(prediction[key].grad[0].clone())
    torch.testing.assert_close(grads[0], grads[1])


@pytest.mark.unit
@pytest.mark.parametrize('loss_name', _LOSSES)
def test_per_sequence_skips_all_nan_sequence(
    make_cfg: Callable[[str], Config], loss_name: str
) -> None:
    """A sequence without observations is excluded from the mean."""
    per_seq_fn = get_loss_obj(make_cfg(loss_name), per_sequence=True)
    prediction, data = _data(n_valid=(4, 0, 7))
    key = _grad_key(loss_name)
    prediction[key].requires_grad_()
    with_empty, _ = per_seq_fn(prediction, data)

    keep = [0, 2]
    without_empty, _ = per_seq_fn(
        {k: v[keep].detach() for k, v in prediction.items()},
        _subset(data, keep),
    )
    torch.testing.assert_close(with_empty, without_empty)

    with_empty.backward()
    grad = prediction[key].grad
    assert torch.isfinite(grad).all()
    assert torch.equal(grad[1], torch.zeros_like(grad[1]))


@pytest.mark.unit
@pytest.mark.parametrize('loss_name', _LOSSES)
def test_per_sequence_all_nan_window_returns_differentiable_zero(
    make_cfg: Callable[[str], Config], loss_name: str
) -> None:
    """With no observation at all the per-sequence loss is a graph zero."""
    per_seq_fn = get_loss_obj(make_cfg(loss_name), per_sequence=True)
    prediction, data = _data(n_valid=(0, 0))
    key = _grad_key(loss_name)
    prediction[key].requires_grad_()
    loss, _ = per_seq_fn(prediction, data)
    assert loss.requires_grad
    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    loss.backward()
    assert torch.equal(prediction[key].grad, torch.zeros_like(prediction[key]))


@pytest.mark.unit
def test_per_sequence_constructor_kwarg_is_keyword_only(
    make_cfg: Callable[[str], Config],
) -> None:
    """``per_sequence`` must be passed by keyword on every loss."""
    cfg = make_cfg('mse')
    for cls in (MaskedMSELoss, MaskedNSELoss, MaskedCMALLoss):
        assert cls(cfg, per_sequence=True)._per_sequence is True  # noqa: SLF001
    with pytest.raises(TypeError):
        MaskedMSELoss(cfg, True)  # noqa: FBT003


@pytest.mark.unit
def test_per_sequence_rejected_for_rmse(
    make_cfg: Callable[[str], Config],
) -> None:
    """RMSE is not a DA loss and does not support per-sequence reduction."""
    cfg = make_cfg('rmse')
    assert isinstance(get_loss_obj(cfg), MaskedRMSELoss)
    with pytest.raises(ValueError, match='not supported for the RMSE'):
        get_loss_obj(cfg, per_sequence=True)
