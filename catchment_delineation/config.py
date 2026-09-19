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

"""Grid and D8 flow-direction constants for DEM catchment delineation."""

# ESRI D8 Flow Direction reverse inflow mapping:
# (row_offset, col_offset, required_d8_value_in_neighbor)
INFLOW_MAP: list[tuple[int, int, int]] = [
    (-1, 0, 4),  # North neighbor flows South (4)
    (-1, 1, 8),  # Northeast neighbor flows Southwest (8)
    (0, 1, 16),  # East neighbor flows West (16)
    (1, 1, 32),  # Southeast neighbor flows Northwest (32)
    (1, 0, 64),  # South neighbor flows North (64)
    (1, -1, 128),  # Southwest neighbor flows Northeast (128)
    (0, -1, 1),  # West neighbor flows East (1)
    (-1, -1, 2),  # Northwest neighbor flows Southeast (2)
]

# Resolution & Tile Grid Constants (HydroSHEDS / MERIT 3 arc-second ~90m)
RES_DEG: float = 1.0 / 1200.0  # 3 arc-seconds (~90 meters at equator)
TILE_DEG: float = 5.0  # 5x5 degrees per tile
TILE_CELLS: int = 6000  # 5 deg * 1200 cells/deg = 6000 cells

# Geographic DEM Coverage Bounds (HydroSHEDS 3 arc-second SRTM global domain)
DEM_MIN_LAT: float = -56.0
DEM_MAX_LAT: float = 60.0
DEM_MIN_LON: float = -180.0
DEM_MAX_LON: float = 180.0
