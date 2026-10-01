# Copyright 2026 Google LLC
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

"""Validate device selection without allocating accelerator tensors."""

# ruff: noqa: SLF001 - These unit tests target the device-selection methods.

from unittest.mock import Mock

import pytest
import torch

from googlehydrology.evaluation.tester import BaseTester
from googlehydrology.training.basetrainer import BaseTrainer
from googlehydrology.utils.config import Config


@pytest.fixture(params=[BaseTrainer, BaseTester], ids=['trainer', 'tester'])
def consumer(request: pytest.FixtureRequest) -> BaseTrainer | BaseTester:
    """Exercise the actual selector without dataset or model initialization."""
    instance = object.__new__(request.param)
    instance.cfg = Config({})
    return instance


@pytest.mark.unit
@pytest.mark.parametrize(
    ('device_count', 'index'),
    [(0, 0), (1, 1), (2, 2), (4, 4), (1, 2), (4, 7)],
)
def test_unavailable_cuda_index_is_rejected(
    consumer: BaseTrainer | BaseTester,
    monkeypatch: pytest.MonkeyPatch,
    device_count: int,
    index: int,
) -> None:
    """An index equal to the visible device count is already out of range."""
    consumer.cfg.device = f'cuda:{index}'
    count = Mock(return_value=device_count)
    monkeypatch.setattr(torch.cuda, 'device_count', count)
    with pytest.raises(RuntimeError, match=f'does not have GPU #{index}'):
        consumer._set_device()
    count.assert_called_once_with()
    assert not hasattr(consumer, 'device')


@pytest.mark.unit
@pytest.mark.parametrize(
    ('device_count', 'index'), [(1, 0), (2, 0), (2, 1), (4, 3)]
)
def test_valid_cuda_index_is_preserved(
    consumer: BaseTrainer | BaseTester,
    monkeypatch: pytest.MonkeyPatch,
    device_count: int,
    index: int,
) -> None:
    """Keep every valid explicitly selected CUDA index unchanged."""
    consumer.cfg.device = f'cuda:{index}'
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: device_count)
    consumer._set_device()
    assert consumer.device == torch.device(f'cuda:{index}')


@pytest.mark.unit
def test_explicit_cpu_does_not_probe_cuda(
    consumer: BaseTrainer | BaseTester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CPU choice must not depend on CUDA availability."""
    consumer.cfg.device = 'cpu'
    probe = Mock(side_effect=AssertionError('CUDA must not be queried'))
    monkeypatch.setattr(torch.cuda, 'device_count', probe)
    monkeypatch.setattr(torch.cuda, 'is_available', probe)
    consumer._set_device()
    assert consumer.device == torch.device('cpu')
    probe.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    ('cuda_available', 'mps_available', 'expected'),
    [
        (True, True, 'cuda:0'),
        (True, False, 'cuda:0'),
        (False, True, 'mps'),
        (False, False, 'cpu'),
    ],
)
def test_automatic_device_priority_is_unchanged(
    consumer: BaseTrainer | BaseTester,
    monkeypatch: pytest.MonkeyPatch,
    expected: str,
    *,
    cuda_available: bool,
    mps_available: bool,
) -> None:
    """Keep automatic selection in CUDA, MPS, CPU priority order."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(
        torch.backends.mps, 'is_available', lambda: mps_available
    )
    consumer._set_device()
    assert consumer.device == torch.device(expected)


@pytest.mark.unit
@pytest.mark.parametrize('available', [False, True])
def test_explicit_mps_selection_is_unchanged(
    consumer: BaseTrainer | BaseTester,
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
) -> None:
    """Retain the existing MPS availability check."""
    consumer.cfg.device = 'mps'
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: available)
    if available:
        consumer._set_device()
        assert consumer.device == torch.device('mps')
    else:
        with pytest.raises(RuntimeError, match='MPS device is not available'):
            consumer._set_device()
