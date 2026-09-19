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

"""Catchment Delineation Package.

Pure DEM flow-direction watershed delineation module supporting high-resolution
multi-tile D8 flow-direction rasters (HydroSHEDS 90m / MERIT) with seamless
cross-tile boundary routing.
"""

from catchment_delineation.config import (
    DEM_MAX_LAT,
    DEM_MAX_LON,
    DEM_MIN_LAT,
    DEM_MIN_LON,
    INFLOW_MAP,
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
)
from catchment_delineation.delineator import (
    CatchmentAreaMismatchError,
    CatchmentCoverageError,
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
    'DEM_MAX_LAT',
    'DEM_MAX_LON',
    'DEM_MIN_LAT',
    'DEM_MIN_LON',
    'INFLOW_MAP',
    'RES_DEG',
    'TILE_CELLS',
    'TILE_DEG',
    'CatchmentAreaMismatchError',
    'CatchmentCoverageError',
    'DemDelineator',
    'delineate_catchment',
    'delineate_coordinates',
    'delineate_dem',
    'download_tile_from_gcs',
    'download_tiles_for_bbox',
    'is_gcs_path',
    'is_tile_available',
    'latlon_to_tile_key',
    'list_available_tiles',
    'tile_key_to_filename',
]
