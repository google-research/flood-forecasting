"""Configuration and constants for DEM catchment delineation."""

from pathlib import Path
import os
from typing import List, Tuple

# ESRI D8 Flow Direction reverse inflow mapping:
# (row_offset, col_offset, required_d8_value_in_neighbor)
INFLOW_MAP: List[Tuple[int, int, int]] = [
    (-1, 0, 4),    # North neighbor flows South (4)
    (-1, 1, 8),    # Northeast neighbor flows Southwest (8)
    (0, 1, 16),    # East neighbor flows West (16)
    (1, 1, 32),    # Southeast neighbor flows Northwest (32)
    (1, 0, 64),    # South neighbor flows North (64)
    (1, -1, 128),  # Southwest neighbor flows Northeast (128)
    (0, -1, 1),    # West neighbor flows East (1)
    (-1, -1, 2),   # Northwest neighbor flows Southeast (2)
]

# Resolution & Tile Grid Constants (HydroSHEDS / MERIT 3 arc-second ~90m)
RES_DEG: float = 1.0 / 1200.0  # 3 arc-seconds (~90 meters at equator)
TILE_DEG: float = 5.0          # 5x5 degrees per tile
TILE_CELLS: int = 6000         # 5 deg * 1200 cells/deg = 6000 cells


def get_default_tiles_dir() -> Path:
  """Resolves the default directory containing 5x5 degree DEM flow-direction tiles (.npy)."""
  env_dir = os.environ.get("DEM_TILES_DIR")
  if env_dir:
    p = Path(env_dir)
    if p.exists():
      return p

  candidate_paths = [
      Path.home() / ".cache" / "googlehydrology" / "hydrosheds_dem" / "tiles_5deg",
      Path.home() / ".cache" / "earthkit_hydro" / "data" / "hydrosheds_dem" / "tiles_5deg",
      Path(__file__).resolve().parent.parent / "data" / "dem" / "tiles_5deg",
  ]

  for p in candidate_paths:
    if p.exists():
      return p

  # Default fallback
  return candidate_paths[0]
