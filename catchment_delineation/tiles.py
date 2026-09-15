"""Tile management and verification utilities for 5x5 degree DEM flow-direction grids."""

import math
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from catchment_delineation.config import TILE_DEG, get_default_tiles_dir


def latlon_to_tile_key(lat: float, lon: float) -> Tuple[int, int]:
  """Computes (lat_top, lon_left) 5x5 degree tile coordinates for a given (lat, lon)."""
  lat_top = int(round(math.ceil(lat / TILE_DEG) * TILE_DEG))
  lon_left = int(round(math.floor(lon / TILE_DEG) * TILE_DEG))
  return lat_top, lon_left


def tile_key_to_filename(lat_top: int, lon_left: int) -> str:
  """Returns the .npy tile filename for a tile key (e.g., n40w090.npy)."""
  lat_str = f"n{lat_top:02d}" if lat_top >= 0 else f"s{abs(lat_top):02d}"
  lon_str = f"w{abs(lon_left):03d}" if lon_left < 0 else f"e{lon_left:03d}"
  return f"{lat_str}{lon_str}.npy"


def get_required_tiles_for_bbox(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> Set[Tuple[int, int]]:
  """Calculates all 5x5 degree tile keys spanning a geographic bounding box."""
  tiles = set()
  curr_lat = min_lat
  while curr_lat <= max_lat + TILE_DEG:
    curr_lon = min_lon
    while curr_lon <= max_lon + TILE_DEG:
      tiles.add(latlon_to_tile_key(curr_lat, curr_lon))
      curr_lon += TILE_DEG
    curr_lat += TILE_DEG
  return tiles


def is_tile_available(
    lat_top: int, lon_left: int, tiles_dir: Optional[Union[str, Path]] = None
) -> bool:
  """Checks if the required tile file exists on disk."""
  directory = Path(tiles_dir) if tiles_dir else get_default_tiles_dir()
  tile_path = directory / tile_key_to_filename(lat_top, lon_left)
  return tile_path.exists()


def list_available_tiles(
    tiles_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
  """Lists all available .npy tiles in the specified or default tiles directory."""
  directory = Path(tiles_dir) if tiles_dir else get_default_tiles_dir()
  if not directory.exists():
    return []
  return sorted([f.name for f in directory.glob("*.npy")])
