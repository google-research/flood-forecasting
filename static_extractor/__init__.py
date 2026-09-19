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

from static_extractor.climate import (
    ERA5ClimateLoader,
    ERA5GriddedExtractor,
    calculate_fao_pm_pet,
    calculate_knoben_moisture_and_seasonality,
    compute_caravan_climate_metrics,
)
from static_extractor.config import (
    ADDITIONAL_PROPERTIES,
    ATTRIBUTE_DEFINITIONS,
    CONTINENT_MAP,
    GCS_ERA5_CLIMATE_URI,
    GCS_ERA5_GRIDDED_ZARR_URI,
    GCS_HYDROATLAS_BUCKET,
    GCS_HYDROATLAS_GDB_URI,
    IGNORE_PROPERTIES,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
    UPSTREAM_PROPERTIES,
    get_default_era5_cache_dir,
    get_default_gdb_path,
)
from static_extractor.extractor import (
    StaticAttributesExtractor,
    compute_pour_point_properties,
)
from static_extractor.gcs import (
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
    "get_default_gdb_path",
    "get_default_era5_cache_dir",
    "ATTRIBUTE_DEFINITIONS",
    "MAJORITY_PROPERTIES",
    "POUR_POINT_PROPERTIES",
    "IGNORE_PROPERTIES",
    "ADDITIONAL_PROPERTIES",
    "UPSTREAM_PROPERTIES",
    "CONTINENT_MAP",
    "GCS_HYDROATLAS_BUCKET",
    "GCS_HYDROATLAS_GDB_URI",
    "GCS_ERA5_CLIMATE_URI",
    "GCS_ERA5_GRIDDED_ZARR_URI",
]


def __getattr__(name: str):
  if name in ("discover_datasets", "run_batch_extraction"):
    from static_extractor import batch_runner
    return getattr(batch_runner, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

