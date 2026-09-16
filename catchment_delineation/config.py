"""Configuration and constants for DEM catchment delineation."""

from pathlib import Path
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

# Geographic DEM Coverage Bounds (HydroSHEDS 3 arc-second SRTM global domain)
DEM_MIN_LAT: float = -56.0
DEM_MAX_LAT: float = 60.0


# Cloud Storage Source URIs
GCS_DEM_BUCKET_URI: str = "gs://open-multimet/data/DEMs"
GCS_TILES_URI: str = f"{GCS_DEM_BUCKET_URI}/tiles_5deg"
GCS_ELEVATION_TILES_URI: str = f"{GCS_DEM_BUCKET_URI}/elevation_tiles_5deg"
GCS_DATA_URI: str = "gs://open-multimet/data"
GCS_CARAVAN_COORDINATES_URI: str = f"{GCS_DATA_URI}/caravan/coordinates.csv"
GCS_CATCHMENT_POLYGONS_URI: str = f"{GCS_DATA_URI}/catchment_polygons"

# Default local cache directory for DEM tiles downloaded from the gs bucket
DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "googlehydrology" / "dem"


def get_default_cache_dir() -> Path:
  """Returns the local cache directory for DEM tiles downloaded from the gs bucket."""
  return DEFAULT_CACHE_DIR


# Backwards compatibility alias
get_default_tiles_dir = get_default_cache_dir


