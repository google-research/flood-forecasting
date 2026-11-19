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

"""Utility helpers for GoogleHydrology."""

import zipfile
from pathlib import Path
from typing import Any

import torch


def safe_torch_load(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> Any:  # noqa: ANN401
    """Load a PyTorch checkpoint with ``weights_only=True``.

    This wrapper additionally rejects TorchScript archives before invoking
    ``torch.load``. On PyTorch <= 2.5.1, ``torch.load`` dispatches TorchScript
    zips to ``torch.jit.load`` before honouring ``weights_only=True``
    (CVE-2025-32434), allowing attacker-authored ``__setstate__`` code inside an
    untrusted checkpoint to execute. This project only saves plain
    ``state_dict`` checkpoints, so a TorchScript archive on a checkpoint path is
    always rejected.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the checkpoint.
    map_location : str | torch.device | None, optional
        Passed through to ``torch.load``.

    Returns
    -------
    Any
        The loaded checkpoint state dict.

    Raises
    ------
    ValueError
        If ``path`` is a TorchScript archive.
    """
    checkpoint_path = Path(path)
    with checkpoint_path.open('rb') as fp:
        if zipfile.is_zipfile(fp):
            fp.seek(0)
            with zipfile.ZipFile(fp) as zf:
                names = zf.namelist()
            if any(
                name == 'constants.pkl' or name.endswith('/constants.pkl')
                for name in names
            ):
                raise ValueError(
                    f'{checkpoint_path} is a TorchScript archive, not a '
                    'state_dict checkpoint. Refusing to load (CVE-2025-32434).'
                )
        fp.seek(0)
        return torch.load(fp, map_location=map_location, weights_only=True)
