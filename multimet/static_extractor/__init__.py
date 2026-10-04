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

"""Caravan Static Attributes Extractor.

Computes Caravan-compatible static watershed attributes from HydroATLAS Level 12
and ERA5-Land datasets for user-supplied watershed polygons.
"""

from multimet.static_extractor.climate import (
    ERA5ClimateLoader,
    ERA5GriddedExtractor,
    calculate_fao_pm_pet,
    calculate_knoben_moisture_and_seasonality,
    compute_caravan_climate_metrics,
)
from multimet.static_extractor.config import (
    ADDITIONAL_PROPERTIES,
    ATTRIBUTE_DEFINITIONS,
    CONTINENT_BBOXES,
    CONTINENT_MAP,
    IGNORE_PROPERTIES,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
    UPSTREAM_PROPERTIES,
)
from multimet.static_extractor.extractor import (
    StaticAttributesExtractor,
    compute_pour_point_properties,
)
from multimet.utils.gcs import (
    download_hydroatlas_from_gcs,
)

__all__ = [
    "StaticAttributesExtractor",
    "discover_datasets",
    "run_batch_extraction",
    "ERA5ClimateLoader",
    "ERA5GriddedExtractor",
    "compute_caravan_climate_metrics",
    "calculate_fao_pm_pet",
    "calculate_knoben_moisture_and_seasonality",
    "compute_pour_point_properties",
    "download_hydroatlas_from_gcs",
    "ATTRIBUTE_DEFINITIONS",
    "MAJORITY_PROPERTIES",
    "POUR_POINT_PROPERTIES",
    "IGNORE_PROPERTIES",
    "ADDITIONAL_PROPERTIES",
    "UPSTREAM_PROPERTIES",
    "CONTINENT_MAP",
    "CONTINENT_BBOXES",
]


def __getattr__(name: str):
  if name in ("discover_datasets", "run_batch_extraction"):
    from multimet.static_extractor import batch_runner
    return getattr(batch_runner, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
