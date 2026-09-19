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

"""Tile key and coverage utilities for 5x5 degree DEM flow-direction grids."""

import math
from pathlib import Path

from catchment_delineation.config import (
    DEM_MAX_LAT,
    DEM_MAX_LON,
    DEM_MIN_LAT,
    DEM_MIN_LON,
    TILE_DEG,
)

MIN_TILE_LAT_TOP: int = -55


def is_coord_in_coverage(lat: float, lon: float) -> bool:
    """Return whether (lat, lon) lies within the global DEM coverage domain."""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    return (
        DEM_MIN_LAT <= lat <= DEM_MAX_LAT and DEM_MIN_LON <= lon <= DEM_MAX_LON
    )


def is_tile_in_coverage(lat_top: int, lon_left: int) -> bool:
    """Return whether a 5x5 degree tile key lies within DEM coverage."""
    max_lon_left = int(DEM_MAX_LON - TILE_DEG)
    return (
        MIN_TILE_LAT_TOP <= lat_top <= int(DEM_MAX_LAT)
        and int(DEM_MIN_LON) <= lon_left <= max_lon_left
    )


def latlon_to_tile_key(lat: float, lon: float) -> tuple[int, int]:
    """Compute (lat_top, lon_left) 5x5 degree tile key for (lat, lon)."""
    lat_top = int(round(math.ceil(lat / TILE_DEG) * TILE_DEG))
    lon_left = int(round(math.floor(lon / TILE_DEG) * TILE_DEG))
    return lat_top, lon_left


def tile_key_to_filename(lat_top: int, lon_left: int) -> str:
    """Return the .npy tile filename for a tile key (e.g., n40w090.npy)."""
    lat_str = f'n{lat_top:02d}' if lat_top >= 0 else f's{abs(lat_top):02d}'
    lon_str = f'w{abs(lon_left):03d}' if lon_left < 0 else f'e{lon_left:03d}'
    return f'{lat_str}{lon_str}.npy'


def get_required_tiles_for_bbox(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> set[tuple[int, int]]:
    """Calculate all 5x5 degree tile keys spanning a geographic bounding box."""
    tiles: set[tuple[int, int]] = set()
    curr_lat = min_lat
    while curr_lat <= max_lat + TILE_DEG:
        curr_lon = min_lon
        while curr_lon <= max_lon + TILE_DEG:
            tiles.add(latlon_to_tile_key(curr_lat, curr_lon))
            curr_lon += TILE_DEG
        curr_lat += TILE_DEG
    return tiles


def is_tile_available(
    lat_top: int, lon_left: int, tiles_dir: str | Path
) -> bool:
    """Check whether the required tile file exists in the user-supplied dir."""
    directory = Path(tiles_dir).expanduser()
    tile_path = directory / tile_key_to_filename(lat_top, lon_left)
    return tile_path.is_file()


def list_available_tiles(tiles_dir: str | Path) -> list[str]:
    """List all .npy tiles in the user-supplied directory."""
    directory = Path(tiles_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(
            f'DEM tiles directory does not exist: {directory}'
        )
    return sorted(f.name for f in directory.glob('*.npy'))
