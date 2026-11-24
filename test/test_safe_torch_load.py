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

"""Unit tests for safe_torch_load (CVE-2025-32434 defense-in-depth)."""

import zipfile
from pathlib import Path

import pytest
import torch

from googlehydrology.evaluation.tester import BaseTester
from googlehydrology.training.basetrainer import BaseTrainer
from googlehydrology.utils import safe_torch_load


def test_safe_torch_load_roundtrip_state_dict(tmp_path: Path) -> None:
    """Legitimate state_dict checkpoints load via Path and str."""
    model = torch.nn.Linear(4, 2)
    ckpt_path = tmp_path / 'model_epoch001.pt'
    torch.save(model.state_dict(), ckpt_path)

    loaded_from_path = safe_torch_load(ckpt_path, map_location='cpu')
    loaded_from_str = safe_torch_load(str(ckpt_path), map_location='cpu')

    for key, tensor in model.state_dict().items():
        assert torch.equal(loaded_from_path[key], tensor)
        assert torch.equal(loaded_from_str[key], tensor)


def test_safe_torch_load_rejects_torchscript_archive(tmp_path: Path) -> None:
    """Real TorchScript archives are rejected before torch.load dispatch."""

    class _DummyModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

    scripted = torch.jit.script(_DummyModule())
    ts_path = tmp_path / 'malicious_torchscript.pt'
    scripted.save(str(ts_path))

    with pytest.raises(ValueError, match='TorchScript archive'):
        safe_torch_load(ts_path, map_location='cpu')


@pytest.mark.parametrize(
    'member_name',
    ['constants.pkl', 'archive/constants.pkl', 'nested/dir/constants.pkl'],
)
def test_safe_torch_load_rejects_zip_with_constants_pkl(
    tmp_path: Path, member_name: str
) -> None:
    """Any zip archive containing constants.pkl is refused."""
    zip_path = tmp_path / 'crafted.pt'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr(member_name, b'dummy')

    with pytest.raises(ValueError, match='CVE-2025-32434'):
        safe_torch_load(zip_path, map_location='cpu')


def test_trainer_and_tester_reject_torchscript_checkpoint(
    tmp_path: Path,
) -> None:
    """BaseTrainer and BaseTester both reject TorchScript checkpoints."""
    ts_path = tmp_path / 'model_epoch001.pt'
    with zipfile.ZipFile(ts_path, 'w') as zf:
        zf.writestr('archive/constants.pkl', b'dummy')

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.device = torch.device('cpu')
    trainer.model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match='TorchScript archive'):
        trainer._load_model_weights(ts_path)  # noqa: SLF001

    tester = BaseTester.__new__(BaseTester)
    tester.device = torch.device('cpu')
    tester.run_dir = tmp_path
    tester.model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match='TorchScript archive'):
        tester._load_weights(epoch=1)  # noqa: SLF001
