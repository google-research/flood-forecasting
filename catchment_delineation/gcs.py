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

"""Google Cloud Storage utilities for DEM flow-direction and elevation tiles."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import fsspec

from catchment_delineation.tiles import (
    get_required_tiles_for_bbox,
    is_tile_in_coverage,
    tile_key_to_filename,
)

logger = logging.getLogger(__name__)


def is_gcs_path(path: str | Path | None) -> bool:
    """Return whether path is a Google Cloud Storage URI."""
    if path is None:
        return False
    return str(path).startswith(('gs://', 'gcs://', 'gs:/', 'gcs:/'))


def normalize_gcs_path(path: str | Path) -> str:
    """Normalize a GCS URI even if Path() stripped a slash."""
    if path is None:
        raise ValueError('Cannot normalize a None path.')
    raw = str(path).strip()
    if not raw:
        raise ValueError('Cannot normalize an empty path.')
    if raw.startswith('gs:/') and not raw.startswith('gs://'):
        return 'gs://' + raw[4:]
    if raw.startswith('gcs:/') and not raw.startswith('gcs://'):
        return 'gcs://' + raw[5:]
    return raw


def upload_file_to_gcs(local_path: str | Path, gcs_uri: str) -> None:
    """Upload a local file to an explicit Google Cloud Storage URI."""
    local_p = Path(local_path)
    if not local_p.is_file():
        raise FileNotFoundError(
            f'Local file {local_p} not found for GCS upload.'
        )
    normalized_uri = normalize_gcs_path(gcs_uri)
    with local_p.open('rb') as src, fsspec.open(normalized_uri, 'wb') as dst:
        dst.write(src.read())


def download_tile_from_gcs(
    lat_top: int,
    lon_left: int,
    target_dir: str | Path,
    source_uri: str,
    *,
    created_files: set[Path] | None = None,
) -> Path:
    """Download a single 5x5 degree DEM tile from an explicit GCS URI."""
    if not target_dir:
        raise ValueError('An explicit target_dir must be provided.')
    if not source_uri:
        raise ValueError('An explicit source_uri must be provided.')

    filename = tile_key_to_filename(lat_top, lon_left)
    if not is_tile_in_coverage(lat_top, lon_left):
        raise ValueError(
            f'Tile {filename} is outside the global DEM coverage domain.'
        )

    directory = Path(target_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    dest_file = directory / filename

    if dest_file.is_file() and dest_file.stat().st_size > 0:
        return dest_file

    base_uri = normalize_gcs_path(source_uri).rstrip('/')
    tile_gcs_uri = f'{base_uri}/{filename}'
    logger.info(
        'Downloading DEM tile from %s to %s...', tile_gcs_uri, dest_file
    )

    tmp_file = directory / f'.tmp_{os.getpid()}_{filename}'
    try:
        with (
            fsspec.open(tile_gcs_uri, 'rb') as src,
            tmp_file.open('wb') as dst,
        ):
            dst.write(src.read())
        if not tmp_file.is_file() or tmp_file.stat().st_size == 0:
            raise RuntimeError(
                f'Downloaded empty DEM tile {filename} from {tile_gcs_uri}.'
            )
        tmp_file.replace(dest_file)
        if created_files is not None:
            created_files.add(dest_file)
        return dest_file
    finally:
        if tmp_file.exists():
            with contextlib.suppress(OSError):
                tmp_file.unlink()


def download_tiles_for_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    target_dir: str | Path,
    source_uri: str,
    *,
    created_files: set[Path] | None = None,
) -> list[Path]:
    """Download all missing tiles covering a bounding box from GCS."""
    required_keys = get_required_tiles_for_bbox(
        min_lat, min_lon, max_lat, max_lon
    )
    downloaded_paths: list[Path] = []
    for lat_top, lon_left in sorted(required_keys):
        path = download_tile_from_gcs(
            lat_top=lat_top,
            lon_left=lon_left,
            target_dir=target_dir,
            source_uri=source_uri,
            created_files=created_files,
        )
        downloaded_paths.append(path)
    return downloaded_paths
