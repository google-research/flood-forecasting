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

"""Shared Google Cloud Storage utilities for MultiMet data workflows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Union

logger = logging.getLogger(__name__)


def is_gcs_path(path: Union[str, Path, None]) -> bool:
  """Returns whether path is a Google Cloud Storage URI."""
  if path is None:
    return False
  return str(path).startswith(("gs://", "gcs://", "gs:/", "gcs:/"))


def normalize_gcs_path(path: Union[str, Path]) -> str:
  """Normalizes a GCS URI, restoring double slashes if Path() stripped one."""
  if path is None:
    raise ValueError("Cannot normalize a None path.")
  raw = str(path).strip()
  if not raw:
    raise ValueError("Cannot normalize an empty path.")
  if raw.startswith("gs:/") and not raw.startswith("gs://"):
    return "gs://" + raw[4:]
  if raw.startswith("gcs:/") and not raw.startswith("gcs://"):
    return "gcs://" + raw[5:]
  return raw


def strip_gcs_prefix(gcs_uri: str) -> str:
  """Strips the gs:// or gcs:// prefix from a GCS URI for gcsfs calls."""
  normalized = normalize_gcs_path(gcs_uri)
  if normalized.startswith("gs://"):
    return normalized[5:]
  if normalized.startswith("gcs://"):
    return normalized[6:]
  return normalized


def gcs_path_exists(gcs_uri: str) -> bool:
  """Checks if a GCS URI exists or contains any objects."""
  normalized = normalize_gcs_path(gcs_uri)
  if shutil.which("gcloud"):
    res = subprocess.run(
        ["gcloud", "storage", "ls", normalized],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  clean_uri = strip_gcs_prefix(normalized).rstrip("/")
  return bool(fs.exists(clean_uri))


def read_bytes_from_gcs(source_uri: str) -> bytes:
  """Streams a remote GCS object directly into memory as bytes."""
  if not source_uri:
    raise ValueError("source_uri must be explicitly provided.")
  import gcsfs

  fs = gcsfs.GCSFileSystem()
  normalized = normalize_gcs_path(source_uri)
  remote_path = strip_gcs_prefix(normalized)
  if not fs.exists(remote_path):
    raise FileNotFoundError(f"Remote GCS object does not exist: {normalized}")
  data = fs.cat_file(remote_path)
  if isinstance(data, str):
    return data.encode("utf-8")
  return bytes(data)


def download_file_from_gcs(
    source_uri: str,
    dest_path: Union[str, Path],
    timeout: int = 120,
) -> Path:
  """Atomically downloads a single file from a GCS URI to a local path.

  Args:
    source_uri: GCS URI of the source file.
    dest_path: Local file path where the downloaded file will be saved.
    timeout: Subprocess timeout in seconds when using gcloud storage.

  Returns:
    Path to the downloaded local file.
  """
  if not source_uri:
    raise ValueError("source_uri must be explicitly provided.")
  if not dest_path:
    raise ValueError("dest_path must be explicitly provided.")

  normalized_src = normalize_gcs_path(source_uri)
  target_file = Path(dest_path)
  target_file.parent.mkdir(parents=True, exist_ok=True)
  tmp_file = target_file.with_name(f".{target_file.name}.tmp.{os.getpid()}")

  if shutil.which("gcloud"):
    cmd = ["gcloud", "storage", "cp", normalized_src, str(tmp_file)]
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if res.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 0:
      os.replace(tmp_file, target_file)
      return target_file
    tmp_file.unlink(missing_ok=True)
    raise FileNotFoundError(
        f"Failed to download {normalized_src} to {target_file}: {res.stderr.strip()}"
    )

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  remote_path = strip_gcs_prefix(normalized_src)
  if not fs.exists(remote_path):
    raise FileNotFoundError(f"Remote GCS file does not exist: {normalized_src}")
  fs.get(remote_path, str(tmp_file))
  if tmp_file.exists() and tmp_file.stat().st_size > 0:
    os.replace(tmp_file, target_file)
    return target_file
  tmp_file.unlink(missing_ok=True)
  raise FileNotFoundError(
      f"Downloaded file from {normalized_src} is empty or missing."
  )


def download_directory_from_gcs(
    source_uri: str,
    dest_dir: Union[str, Path],
    timeout: int = 600,
) -> Path:
  """Downloads a directory from Google Cloud Storage to a local directory.

  Args:
    source_uri: GCS URI of the source directory.
    dest_dir: Local directory path where the directory will be saved.
    timeout: Subprocess timeout in seconds when using gcloud storage.

  Returns:
    Path to the local directory.
  """
  if not source_uri:
    raise ValueError("source_uri must be explicitly provided.")
  if not dest_dir:
    raise ValueError("dest_dir must be explicitly provided.")

  normalized_src = normalize_gcs_path(source_uri)
  dest_path = Path(dest_dir)
  dest_path.parent.mkdir(parents=True, exist_ok=True)

  if shutil.which("gcloud"):
    cmd = ["gcloud", "storage", "cp", "-r", normalized_src, str(dest_path.parent)]
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if res.returncode == 0 and dest_path.exists():
      return dest_path
    raise RuntimeError(
        f"Failed to download {normalized_src} to {dest_path}: {res.stderr.strip()}"
    )

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  clean_src = strip_gcs_prefix(normalized_src).rstrip("/")
  if not fs.exists(clean_src):
    raise FileNotFoundError(f"GCS directory does not exist at {normalized_src}.")
  dest_path.mkdir(parents=True, exist_ok=True)
  fs.get(clean_src, str(dest_path), recursive=True)
  return dest_path


def download_hydroatlas_from_gcs(
    target_dir: Union[str, Path],
    source_uri: str,
) -> Path:
  """Downloads BasinATLAS_v10.gdb from Google Cloud Storage to a specified local path.

  Args:
    target_dir: Local directory path where BasinATLAS_v10.gdb will be saved.
    source_uri: GCS source URI for BasinATLAS_v10.gdb.

  Returns:
    Path to the local BasinATLAS_v10.gdb directory.
  """
  if not target_dir:
    raise ValueError("target_dir must be explicitly provided.")
  if not source_uri:
    raise ValueError("source_uri must be explicitly provided.")

  target_path = Path(target_dir)
  dest_path = (
      target_path
      if target_path.name.endswith(".gdb")
      else target_path / "BasinATLAS_v10.gdb"
  )

  if dest_path.exists() and any(dest_path.iterdir()):
    logger.debug("BasinATLAS GDB already exists at: %s", dest_path)
    return dest_path

  logger.info("Downloading HydroATLAS GDB from %s to %s...", source_uri, dest_path)
  return download_directory_from_gcs(source_uri=source_uri, dest_dir=dest_path)


def sync_gcs_directory(gcs_uri: str, local_dest: Union[str, Path]) -> Path:
  """Syncs a GCS directory to a local directory."""
  if not gcs_uri:
    raise ValueError("gcs_uri must be explicitly provided.")
  if not local_dest:
    raise ValueError("local_dest must be explicitly provided.")

  dest_path = Path(local_dest)
  dest_path.mkdir(parents=True, exist_ok=True)
  normalized_uri = normalize_gcs_path(gcs_uri)
  gcs_uri_clean = normalized_uri if normalized_uri.endswith("/") else normalized_uri + "/"
  logger.debug("Syncing %s to local directory %s...", gcs_uri_clean, dest_path)

  if shutil.which("gcloud"):
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "-r", gcs_uri_clean, str(dest_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
      return dest_path
    raise RuntimeError(
        f"Failed to sync GCS URI {gcs_uri} to {dest_path}: {res.stderr.strip()}"
    )

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  clean_src = strip_gcs_prefix(gcs_uri_clean).rstrip("/")
  if not fs.exists(clean_src):
    raise FileNotFoundError(f"GCS directory does not exist: {gcs_uri}")
  fs.get(clean_src, str(dest_path), recursive=True)
  return dest_path


def upload_to_gcs(local_path: Union[str, Path], gcs_dest_uri: str) -> None:
  """Uploads a local file or directory to a GCS destination."""
  if not local_path:
    raise ValueError("local_path must be explicitly provided.")
  if not gcs_dest_uri:
    raise ValueError("gcs_dest_uri must be explicitly provided.")

  local_p = Path(local_path)
  if not local_p.exists():
    raise FileNotFoundError(f"Local path {local_p} not found for GCS upload.")

  normalized_dest = normalize_gcs_path(gcs_dest_uri)
  gcs_dest_clean = (
      normalized_dest if normalized_dest.endswith("/") else normalized_dest + "/"
  )
  target_uri = (
      f"{gcs_dest_clean}{local_p.name}" if local_p.is_file() else gcs_dest_clean
  )
  logger.debug("Uploading %s to %s...", local_p, target_uri)

  if shutil.which("gcloud"):
    cmd = (
        ["gcloud", "storage", "cp", str(local_p), target_uri]
        if local_p.is_file()
        else ["gcloud", "storage", "rsync", "-r", str(local_p), gcs_dest_clean]
    )
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode == 0:
      logger.debug(
          "Successfully uploaded %s to %s via gcloud storage.",
          local_p.name,
          target_uri,
      )
      return
    raise RuntimeError(
        f"Failed to upload {local_p} to {target_uri}: {res.stderr.strip()}"
    )

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  clean_target = strip_gcs_prefix(target_uri)
  if local_p.is_file():
    fs.put(str(local_p), clean_target)
  else:
    fs.put(str(local_p), clean_target, recursive=True)
  logger.debug(
      "Successfully uploaded %s to %s via gcsfs.", local_p.name, target_uri
  )


def upload_file_to_gcs(local_path: Union[str, Path], gcs_uri: str) -> None:
  """Uploads a single local file to an explicit GCS object URI."""
  if not local_path:
    raise ValueError("local_path must be explicitly provided.")
  if not gcs_uri:
    raise ValueError("gcs_uri must be explicitly provided.")

  local_p = Path(local_path)
  if not local_p.is_file():
    raise FileNotFoundError(f"Local file {local_p} not found for GCS upload.")

  normalized_uri = normalize_gcs_path(gcs_uri)
  if shutil.which("gcloud"):
    res = subprocess.run(
        ["gcloud", "storage", "cp", str(local_p), normalized_uri],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
      return
    raise RuntimeError(
        f"Failed to upload {local_p} to {normalized_uri}: {res.stderr.strip()}"
    )

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  fs.put(str(local_p), strip_gcs_prefix(normalized_uri))
