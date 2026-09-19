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

"""Open-MultiMet gridded archives and catchment timeseries extractors.

This package has two halves that share a single on-disk contract.

Archive builders
----------------
ETL pipelines that assemble the unified, analysis-ready *gridded* archives:

* :mod:`multimet.build_cpc_archive` - NOAA CPC Global Unified daily gauge-based
  precipitation (0.5 degree, 1979 to present).
* :mod:`multimet.build_hres_archive` - ECMWF IFS HRES daily surface forecasts
  at lead days 1..10 (0.25 degree, 2016 to present).
* :mod:`multimet.build_imerg_archive` - NASA GPM IMERG Early V07 daily
  precipitation (0.1 degree, 2000 to present).

Each is independently runnable and exposes a console script
(``build-cpc-archive`` / ``build-hres-archive`` / ``build-imerg-archive``).

Catchment timeseries extractors
-------------------------------
Reduce gridded meteorology to per-basin daily timeseries in the Caravan
MultiMet schema. Every extractor reads either directly from its upstream
third-party provider or -- when the caller supplies an archive store URI --
from one of the gridded archives above. See :mod:`multimet.gridded_archive`.

Submodules and symbols are imported lazily so that ``import multimet`` stays
cheap and does not require the optional cloud, GRIB or geospatial
dependencies to be installed.
"""

from __future__ import annotations

import importlib
from typing import Any

# Submodules reachable as ``multimet.<name>``.
_SUBMODULES = frozenset({
    "base",
    "build_cpc_archive",
    "build_hres_archive",
    "build_imerg_archive",
    "config",
    "cpc",
    "dask_runner",
    "dynamical",
    "era5_land",
    "gcp",
    "geometry",
    "graphcast",
    "gridded_archive",
    "hres",
    "imerg",
    "pet",
    "runner",
    "spatial",
    "storage",
    "zarr_writer",
    "zonal",
})

# Public symbols re-exported from their defining submodule.
_LAZY_SYMBOLS = {
    "Product": "multimet.config",
    "ProductType": "multimet.config",
    "BaseExtractor": "multimet.base",
    "CPCExtractor": "multimet.cpc",
    "ERA5LandExtractor": "multimet.era5_land",
    "GraphCastExtractor": "multimet.graphcast",
    "HRESExtractor": "multimet.hres",
    "IMERGExtractor": "multimet.imerg",
    "get_bounding_box": "multimet.geometry",
    "load_basin_geometries": "multimet.geometry",
    "MultiMetZarrWriter": "multimet.zarr_writer",
    "ZonalWeightCalculator": "multimet.zonal",
    "ZonalWeightMatrix": "multimet.zonal",
    "calculate_fao56_penman_monteith_pet": "multimet.pet",
    "extract_multimet_serial": "multimet.runner",
    "extract_multimet_dask": "multimet.dask_runner",
    "extract_product_dask": "multimet.dask_runner",
    "DynamicalDataLoader": "multimet.dynamical",
    "DynamicalExtractor": "multimet.dynamical",
    "DynamicalIMERGExtractor": "multimet.dynamical",
    "AIFSExtractor": "multimet.dynamical",
    "load_dynamical": "multimet.dynamical",
    "BoundingBox": "multimet.spatial",
    "find_lat_lon_dims": "multimet.spatial",
    "slice_coordinates_by_bounds": "multimet.spatial",
    "slice_dataset_by_bounds": "multimet.spatial",
    # Gridded archive source.
    "ArchiveBandSpec": "multimet.gridded_archive",
    "GriddedArchiveSpec": "multimet.gridded_archive",
    "GriddedArchiveError": "multimet.gridded_archive",
    "extract_from_archive": "multimet.gridded_archive",
    "extract_forecast_from_archive": "multimet.gridded_archive",
    "extract_nowcast_from_archive": "multimet.gridded_archive",
    "get_archive_spec": "multimet.gridded_archive",
    "open_gridded_archive": "multimet.gridded_archive",
}

__all__ = sorted(_SUBMODULES | set(_LAZY_SYMBOLS))


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module objects are untyped.
  if name in _SUBMODULES:
    return importlib.import_module(f"{__name__}.{name}")
  if name in _LAZY_SYMBOLS:
    module = importlib.import_module(_LAZY_SYMBOLS[name])
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return sorted(set(globals()) | set(__all__))
