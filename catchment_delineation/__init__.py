"""Catchment Delineation Package.

Pure DEM flow-direction watershed delineation module supporting high-resolution
multi-tile D8 flow direction rasters (HydroSHEDS 90m / MERIT) with seamless
cross-tile boundary routing.
"""

from catchment_delineation.config import (
    GCS_DEM_BUCKET_URI,
    GCS_ELEVATION_TILES_URI,
    GCS_TILES_URI,
    INFLOW_MAP,
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
    get_default_tiles_dir,
)
from catchment_delineation.delineator import (
    DemDelineator,
    delineate_catchment,
    delineate_coordinates,
    delineate_dem,
)
from catchment_delineation.gcs import (
    download_tile_from_gcs,
    download_tiles_for_bbox,
    is_gcs_path,
)
from catchment_delineation.tiles import (
    is_tile_available,
    latlon_to_tile_key,
    list_available_tiles,
    tile_key_to_filename,
)

__all__ = [
    "DemDelineator",
    "delineate_dem",
    "delineate_catchment",
    "delineate_coordinates",
    "get_default_tiles_dir",
    "latlon_to_tile_key",
    "tile_key_to_filename",
    "is_tile_available",
    "list_available_tiles",
    "download_tile_from_gcs",
    "download_tiles_for_bbox",
    "is_gcs_path",
    "GCS_DEM_BUCKET_URI",
    "GCS_TILES_URI",
    "GCS_ELEVATION_TILES_URI",
    "INFLOW_MAP",
    "RES_DEG",
    "TILE_DEG",
    "TILE_CELLS",
]

