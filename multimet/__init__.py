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

"""MultiMet forcing extraction tools for generating Caravan-MultiMet Zarr stores."""

from __future__ import annotations

import importlib
from typing import Any

from multimet.config import Product
from multimet.config import ProductType
from multimet.geometry import get_bounding_box
from multimet.geometry import load_basin_geometries

_PKG = "multimet"

_LAZY_MODULE_MAPPING = {
    "BaseExtractor": f"{_PKG}.base",
    "CPCExtractor": f"{_PKG}.cpc",
    "ERA5LandExtractor": f"{_PKG}.era5_land",
    "GraphCastExtractor": f"{_PKG}.graphcast",
    "HRESExtractor": f"{_PKG}.hres",
    "IMERGExtractor": f"{_PKG}.imerg",
    "MultiMetZarrWriter": f"{_PKG}.zarr_writer",
    "ZonalWeightCalculator": f"{_PKG}.zonal",
    "ZonalWeightMatrix": f"{_PKG}.zonal",
    "calculate_fao56_penman_monteith_pet": f"{_PKG}.pet",
    "extract_multimet_serial": f"{_PKG}.runner",
}

__all__ = [
    "Product",
    "ProductType",
    "BaseExtractor",
    "CPCExtractor",
    "ERA5LandExtractor",
    "GraphCastExtractor",
    "HRESExtractor",
    "IMERGExtractor",
    "get_bounding_box",
    "load_basin_geometries",
    "MultiMetZarrWriter",
    "ZonalWeightCalculator",
    "ZonalWeightMatrix",
    "calculate_fao56_penman_monteith_pet",
    "extract_multimet_serial",
]


def __getattr__(name: str) -> Any:
  if name in _LAZY_MODULE_MAPPING:
    mod = importlib.import_module(_LAZY_MODULE_MAPPING[name])
    return getattr(mod, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
