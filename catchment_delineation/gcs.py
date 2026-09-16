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

"""Google Cloud Storage utilities for DEM flow-direction and elevation tiles."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional, Set, Tuple, Union

from catchment_delineation.config import (
    GCS_TILES_URI,
    get_default_cache_dir,
    get_default_tiles_dir,
)
from catchment_delineation.tiles import (
    get_required_tiles_for_bbox,
    is_tile_in_coverage,
    tile_key_to_filename,
)

logger = logging.getLogger(__name__)


def is_gcs_path(path: Union[str, Path]) -> bool:
  """Checks if path is a Google Cloud Storage URI."""
  return str(path).startswith(("gs://", "gcs://", "gs:/", "gcs:/"))


def normalize_gcs_path(path: Union[str, Path]) -> str:
  """Normalizes a GCS URI ensuring proper gs:// or gcs:// scheme even if Path() stripped a slash."""
  s = str(path).strip()
  if s.startswith("gs:/") and not s.startswith("gs://"):
    return "gs://" + s[4:]
  if s.startswith("gcs:/") and not s.startswith("gcs://"):
    return "gcs://" + s[5:]
  return s


def upload_file_to_gcs(local_path: Union[str, Path], gcs_uri: str) -> None:
  """Uploads a local file to a Google Cloud Storage URI.

  Args:
      local_path: Local file path.
      gcs_uri: Target GCS URI (gs://bucket/path/to/file).
  """
  local_p = Path(local_path)
  if not local_p.exists():
    raise FileNotFoundError(f"Local file {local_p} not found for GCS upload.")

  # 1. Try fsspec stream copy
  try:
    import fsspec

    with open(local_p, "rb") as src, fsspec.open(gcs_uri, "wb") as dst:
      dst.write(src.read())
    return
  except Exception as e:
    logger.debug("fsspec GCS upload failed: %s, trying google.cloud.storage", e)

  # 2. Try google.cloud.storage client
  try:
    from google.cloud import storage

    clean_uri = str(gcs_uri).replace("gs://", "").replace("gcs://", "")
    bucket_name, blob_name = clean_uri.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_p))
    return
  except Exception as e:
    logger.debug("google.cloud.storage upload failed: %s", e)

  # 3. Try gcloud storage CLI
  if shutil.which("gcloud"):
    cmd = ["gcloud", "storage", "cp", str(local_p), str(gcs_uri)]
    res = subprocess.run(cmd, capture_output=True, timeout=120)
    if res.returncode == 0:
      return

  raise RuntimeError(f"Failed to upload {local_p} to {gcs_uri}.")


def download_tile_from_gcs(
    lat_top: int,
    lon_left: int,
    target_dir: Optional[Union[str, Path]] = None,
    source_uri: Optional[str] = None,
) -> Path:
  """Downloads a single 5x5 degree DEM tile from GCS to local directory.

  Args:
      lat_top: Top (northern) latitude of the tile.
      lon_left: Left (western) longitude of the tile.
      target_dir: Local destination directory. Defaults to get_default_tiles_dir().
      source_uri: GCS source directory. Defaults to GCS_TILES_URI.

  Returns:
      Path to the local downloaded .npy file.
  """
  filename = tile_key_to_filename(lat_top, lon_left)
  if not is_tile_in_coverage(lat_top, lon_left):
    raise ValueError(
        f"Tile {filename} is outside the global DEM coverage domain (-56° to 60° latitude)."
    )

  directory = Path(target_dir) if target_dir else get_default_tiles_dir()
  directory.mkdir(parents=True, exist_ok=True)

  dest_file = directory / filename

  if dest_file.exists() and dest_file.stat().st_size > 0:
    return dest_file

  base_uri = (source_uri or GCS_TILES_URI).rstrip("/")
  tile_gcs_uri = f"{base_uri}/{filename}"

  logger.info("Downloading DEM tile from %s to %s...", tile_gcs_uri, dest_file)

  # Download to a process-unique temporary file first, then atomically rename
  tmp_file = directory / f".tmp_{os.getpid()}_{filename}"
  try:
    # 1. Try gcloud storage CLI
    if shutil.which("gcloud"):
      try:
        cmd = ["gcloud", "storage", "cp", tile_gcs_uri, str(tmp_file)]
        res = subprocess.run(cmd, capture_output=True, timeout=120)
        if res.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 0:
          tmp_file.replace(dest_file)
          logger.info("Successfully downloaded tile %s via gcloud storage.", filename)
          return dest_file
      except Exception as e:
        logger.warning("gcloud storage tile download failed: %s", e)

    # 2. Try gcsfs
    try:
      import gcsfs

      fs = gcsfs.GCSFileSystem()
      clean_src = tile_gcs_uri.replace("gs://", "")
      if fs.exists(clean_src):
        fs.get(clean_src, str(tmp_file))
        if tmp_file.exists() and tmp_file.stat().st_size > 0:
          tmp_file.replace(dest_file)
          logger.info("Successfully downloaded tile %s via gcsfs.", filename)
          return dest_file
    except Exception as e:
      logger.warning("gcsfs tile download failed: %s", e)
  finally:
    if tmp_file.exists():
      try:
        tmp_file.unlink()
      except OSError:
        pass

  raise RuntimeError(
      f"Failed to download DEM tile {filename} from {tile_gcs_uri} to {dest_file}. "
      "Please verify GCS bucket accessibility and cloud credentials."
  )


def download_tiles_for_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    target_dir: Optional[Union[str, Path]] = None,
    source_uri: Optional[str] = None,
) -> List[Path]:
  """Downloads all missing tiles covering a bounding box from GCS.

  Args:
      min_lat: Minimum latitude.
      min_lon: Minimum longitude.
      max_lat: Maximum latitude.
      max_lon: Maximum longitude.
      target_dir: Destination directory.
      source_uri: Source GCS URI.

  Returns:
      List of paths to required tile files.
  """
  required_keys = get_required_tiles_for_bbox(min_lat, min_lon, max_lat, max_lon)
  downloaded_paths = []
  for lat_top, lon_left in sorted(required_keys):
    p = download_tile_from_gcs(
        lat_top=lat_top,
        lon_left=lon_left,
        target_dir=target_dir,
        source_uri=source_uri,
    )
    downloaded_paths.append(p)
  return downloaded_paths
