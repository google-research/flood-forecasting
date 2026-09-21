# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Storage utilities for HydroATLAS and static attributes extraction."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Union

from static_extractor.config import (
    GCS_HYDROATLAS_BUCKET,
    GCS_HYDROATLAS_GDB_URI,
    get_default_gdb_path,
)

logger = logging.getLogger(__name__)


def is_gcs_path(path: Union[str, Path]) -> bool:
  """Checks if path is a Google Cloud Storage URI."""
  return str(path).startswith(("gs://", "gcs://"))


def download_hydroatlas_from_gcs(
    target_dir: Optional[Union[str, Path]] = None,
    source_uri: str = GCS_HYDROATLAS_GDB_URI,
) -> Path:
  """Downloads BasinATLAS_v10.gdb from Google Cloud Storage to local cache.

  Args:
    target_dir: Optional local directory where BasinATLAS_v10.gdb will be saved.
      Defaults to ~/.cache/googlehydrology/hydroatlas/BasinATLAS_v10.gdb.
    source_uri: GCS source URI.

  Returns:
    Path to local BasinATLAS_v10.gdb directory.
  """
  if target_dir is None:
    dest_path = get_default_gdb_path()
  else:
    target_path = Path(target_dir)
    dest_path = (
        target_path
        if target_path.name.endswith(".gdb")
        else target_path / "BasinATLAS_v10.gdb"
    )

  if dest_path.exists() and any(dest_path.iterdir()):
    logger.debug("BasinATLAS GDB already exists at: %s", dest_path)
    return dest_path

  dest_path.parent.mkdir(parents=True, exist_ok=True)
  logger.info("Downloading HydroATLAS GDB from %s to %s...", source_uri, dest_path)

  if shutil.which("gcloud"):
    cmd = ["gcloud", "storage", "cp", "-r", source_uri, str(dest_path.parent)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if res.returncode == 0 and dest_path.exists():
      logger.debug("Successfully downloaded BasinATLAS GDB via gcloud storage.")
      return dest_path
    raise RuntimeError(
        f"Failed to download BasinATLAS GDB from {source_uri} to {dest_path}: {res.stderr.strip()}"
    )

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  clean_src = source_uri.replace("gs://", "").rstrip("/")
  if not fs.exists(clean_src):
    raise FileNotFoundError(
        f"BasinATLAS GDB does not exist at {source_uri}."
    )
  dest_path.mkdir(parents=True, exist_ok=True)
  fs.get(clean_src, str(dest_path), recursive=True)
  logger.debug("Successfully downloaded BasinATLAS GDB via gcsfs.")
  return dest_path
