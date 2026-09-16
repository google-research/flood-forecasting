"""Tile management and verification utilities for 5x5 degree DEM flow-direction grids."""

import math
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from catchment_delineation.config import (
    DEM_MAX_LAT,
    DEM_MIN_LAT,
    TILE_DEG,
    get_default_cache_dir,
)


def is_coord_in_coverage(lat: float, lon: float) -> bool:
  """Checks whether given coordinates fall within the global DEM coverage domain (-56° to 60° latitude)."""
  return DEM_MIN_LAT <= lat <= DEM_MAX_LAT


def is_tile_in_coverage(lat_top: int, lon_left: int) -> bool:
  """Checks whether a 5x5 degree tile falls within the global DEM coverage domain.

  HydroSHEDS tiles span latitudes from 56°S (-56) to 60°N (60).
  A tile with lat_top covers [lat_top - 5, lat_top], so valid lat_top values
  range from -50 (covers -55 to -50) down to -55 (covers -60 to -55) and up to 60 (covers 55 to 60).
  """
  return -55 <= lat_top <= int(DEM_MAX_LAT)


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
  """Checks if the required tile file exists locally."""
  directory = Path(tiles_dir).expanduser() if tiles_dir else get_default_cache_dir()
  tile_path = directory / tile_key_to_filename(lat_top, lon_left)
  return tile_path.exists()


def list_available_tiles(
    tiles_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
  """Lists all available .npy tiles in the specified or local cache directory."""
  directory = Path(tiles_dir).expanduser() if tiles_dir else get_default_cache_dir()
  if not directory.exists():
    return []
  return sorted([f.name for f in directory.glob("*.npy")])

