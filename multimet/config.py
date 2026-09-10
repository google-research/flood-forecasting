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

"""Configuration and schema definitions for the MultiMet forcing extractor."""

from __future__ import annotations

import enum
from typing import Dict, List, Mapping, Optional, Tuple


class ProductType(enum.Enum):
  NOWCAST = "nowcast"
  FORECAST = "forecast"


class Product(enum.Enum):
  ERA5_LAND = "ERA5_LAND"
  CPC = "CPC"
  IMERG = "IMERG"
  CHIRPS = "CHIRPS"
  CHIRPS_GEFS = "CHIRPS_GEFS"
  HRES = "HRES"
  GRAPHCAST = "GRAPHCAST"
  AIFS = "AIFS"
  DYNAMICAL_IMERG = "DYNAMICAL_IMERG"


PRODUCT_TYPES: Mapping[Product, ProductType] = {
    Product.ERA5_LAND: ProductType.NOWCAST,
    Product.CPC: ProductType.NOWCAST,
    Product.IMERG: ProductType.NOWCAST,
    Product.CHIRPS: ProductType.NOWCAST,
    Product.CHIRPS_GEFS: ProductType.FORECAST,
    Product.HRES: ProductType.FORECAST,
    Product.GRAPHCAST: ProductType.FORECAST,
    Product.AIFS: ProductType.FORECAST,
    Product.DYNAMICAL_IMERG: ProductType.NOWCAST,
}

FORECAST_LEAD_DAYS: Mapping[Product, int] = {
    Product.CHIRPS_GEFS: 16,
    Product.HRES: 10,
    Product.GRAPHCAST: 10,
    Product.AIFS: 10,
}

# Target bands / data variable names per product in Caravan-MultiMet.
# Aligned with canonical Caravan v1.5 specification.
PRODUCT_BANDS: Mapping[Product, Tuple[str, ...]] = {
    Product.ERA5_LAND: (
        "era5land_dewpoint_temperature_2m",
        "era5land_potential_evaporation_DEPRECATED",
        "era5land_potential_evaporation_FAO_PENMAN_MONTEITH",
        "era5land_snow_depth_water_equivalent",
        "era5land_surface_net_solar_radiation",
        "era5land_surface_net_thermal_radiation",
        "era5land_surface_pressure",
        "era5land_temperature_2m",
        "era5land_total_precipitation",
        "era5land_u_component_of_wind_10m",
        "era5land_v_component_of_wind_10m",
        "era5land_volumetric_soil_water_layer_1",
        "era5land_volumetric_soil_water_layer_2",
        "era5land_volumetric_soil_water_layer_3",
        "era5land_volumetric_soil_water_layer_4",
    ),
    Product.CPC: ("cpc_precipitation",),
    Product.IMERG: ("imerg_precipitation",),
    Product.CHIRPS: ("chirps_precipitation",),
    Product.CHIRPS_GEFS: ("chirpsgefs_precipitation",),
    Product.HRES: (
        "hres_surface_net_solar_radiation",
        "hres_surface_net_thermal_radiation",
        "hres_surface_pressure",
        "hres_temperature_2m",
        "hres_total_precipitation",
    ),
    Product.GRAPHCAST: (
        "graphcast_temperature_2m",
        "graphcast_total_precipitation",
        "graphcast_u_component_of_wind_10m",
        "graphcast_v_component_of_wind_10m",
    ),
    Product.AIFS: (
        "aifs_temperature_2m",
        "aifs_total_precipitation",
        "aifs_u_component_of_wind_10m",
        "aifs_v_component_of_wind_10m",
    ),
    Product.DYNAMICAL_IMERG: ("imerg_precipitation",),
}

# Canonical dataset global attributes matching Caravan MultiMet v1.1
PRODUCT_METADATA_ATTRS: Mapping[Product, Mapping[str, str]] = {
    Product.CPC: {
        "Citation": (
            "(Interpolation algorithm) Xie_et_al_2007_JHM_EAG.pdf Xie, P.,"
            " A. Yatagai, M. Chen, T. Hayasaka, Y. Fukushima, C. Liu, and"
            " S. Yang (2007), A gauge-based analysis of daily precipitation over"
            " East Asia, J. Hydrometeorol., 8, 607. 626.\n(Gauge Algorithm"
            " Evaluation) Chen_et_al_2008_JGR_Gauge_Algo.pdf Chen, M., W. Shi,"
            " P. Xie, V. B. S. Silva, V E. Kousky, R. Wayne Higgins, and"
            " J. E. Janowiak (2008), Assessing objective techniques for"
            " gauge-based analyses of global daily precipitation, J. Geophys."
            " Res., 113, D04110, doi:10.1029/2007JD009132.\n\n"
        ),
        "License": (
            "Usage Restrictions: None.\nSee"
            " https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html"
        ),
        "Product": "CPC",
        "Released": "2024-11-18",
        "Sources": (
            "CPC Global Unified Gauge-Based Analysis of Daily Precipitation"
            " data provided by the NOAA PSL, from their website at"
            " https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html"
        ),
        "Units": "precipitation [mm]",
        "Version": "1.1",
    },
    Product.IMERG: {
        "Citation": (
            "Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, Jackson"
            " Tan (2024), GPM IMERG Early Precipitation L3 Half Hourly 0.1"
            " degree x 0.1 degree V07, Greenbelt, MD, Goddard Earth Sciences"
            " Data and Information Services Center (GES DISC), Accessed:"
            " [November 2024], 10.5067/GPM/IMERG/3B-HH-E/07"
        ),
        "License": (
            "GPM and TRMM data are freely available at all levels for which the"
            " particular sensor or sensor combination has been processed by"
            " GPM. For the GPM Core Observatory this is for Levels 0 through 3"
            " products (as applicable).  For the partner satellites in the GPM"
            " constellation this is Levels 1c through 3 (as applicable).\nSee"
            " https://gpm.nasa.gov/data/policy"
        ),
        "Product": "IMERG v07 Early",
        "Released": "2024-11-18",
        "Sources": (
            "IMERG (Integrated Multi-satellitE Retrievals for GPM) by NASA."
            " This data is based on IMERG-Early v07"
            " https://gpm.nasa.gov/data/imerg"
        ),
        "Units": "precipitation [mm]",
        "Version": "1.1",
    },
    Product.ERA5_LAND: {
        "Citation": (
            "Muñoz Sabater, J. (2019): ERA5-Land hourly data from 1950 to"
            " present. Copernicus Climate Change Service (C3S) Climate Data"
            " Store (CDS). DOI: 10.24381/cds.e2161bac"
        ),
        "License": (
            "https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf"
        ),
        "Product": "ERA5-Land",
        "Released": "2024-11-18",
        "Sources": (
            "All forcing and state variables are derived from ERA5-Land hourly"
            " by ECMWF."
            " https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land"
        ),
        "Units": (
            "dewpoint_temperature_2m: Dew point temperature [°C]\n"
            "potential_evaporation_DEPRECATED: Potential Evaporation [mm]"
            " (original potential_evaporation from ERA5-Land)\n"
            "potential_evaporation_FAO_PENMAN_MONTEITH: Potential Evaporation"
            " [mm] (FAO Penman-Monteith computed from ERA5-Land inputs)\n"
            "snow_depth_water_equivalent: Snow-Water-Equivalent [mm]\n"
            "surface_net_solar_radiation: Surface net solar radiation [W/m2]\n"
            "surface_net_thermal_radiation: Surface net thermal radiation"
            " [W/m2]\n"
            "surface_pressure: Surface pressure [kPa]\n"
            "temperature_2m: 2m air temperature [°C]\n"
            "u_component_of_wind_10m: U-component of wind at 10m [m/s]\n"
            "v_component_of_wind_10m: V-component of wind at 10m [m/s]\n"
            "volumetric_soil_water_layer_1: Volumetric soil water layer 1"
            " (0-7cm) [m3/m3]\n"
            "volumetric_soil_water_layer_2: Volumetric soil water layer 2"
            " (7-28cm) [m3/m3]\n"
            "volumetric_soil_water_layer_3: Volumetric soil water layer 3"
            " (28-100cm) [m3/m3]\n"
            "volumetric_soil_water_layer_4: Volumetric soil water layer 4"
            " (100-289cm) [m3/m3]\n"
            "total_precipitation: Total precipitation [mm]"
        ),
        "Version": "1.1",
    },
    Product.HRES: {
        "Citation": "",
        "License": (
            "https://apps.ecmwf.int/datasets/licences/general/\nSource"
            " www.ecmwf.int\nCopyright © 2024 European Centre for Medium-Range"
            " Weather Forecasts (ECMWF).\nThis data is published under a"
            " Creative Commons Attribution 4.0 International (CC BY 4.0)."
            " https://creativecommons.org/licenses/by/4.0/\nECMWF does not"
            " accept any liability whatsoever for any error or omission in the"
            " data, their availability, or for any loss or damage arising from"
            " their use."
        ),
        "Product": "ECMWF-IFS-HRES",
        "Released": "2024-11-18",
        "Sources": (
            "HRES forecast from IFS by ECMWF:"
            " https://www.ecmwf.int/en/forecasts/documentation-and-support/medium-range-forecasts "
        ),
        "Units": (
            "surface_pressure: Surface pressure [kPa]\nsurface_net_solar_radiation:"
            " Surface net solar radiation [unavailable in WeatherBench 2 HRES archive]"
            "\nsurface_net_thermal_radiation: Surface net thermal"
            " radiation [unavailable in WeatherBench 2 HRES archive]\ntemperature_2m: 2m air temperature"
            " [°C]\ntotal_precipitation: Total precipitation [mm]"
        ),
        "Unavailable_Bands": (
            "hres_surface_net_solar_radiation, hres_surface_net_thermal_radiation"
            " (not archived in WeatherBench 2 HRES dataset)"
        ),
        "Version": "1.1",
    },
    Product.GRAPHCAST: {
        "Citation": (
            "R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M."
            " Fortunato, F. Alet, S. Ravuri, T. Ewalds, Z. Eaton-Rosen, W. Hu,"
            " et al. Learning skillful medium-range global weather forecasting."
            " Science, page eadi2336, 2023"
        ),
        "License": "There are no limitations on using this data.",
        "Product": "GraphCast",
        "Released": "2024-11-18",
        "Sources": (
            "The data was provided directly from GraphCast authors."
            " https://github.com/google-deepmind/graphcast\nThis version of"
            " GraphCast has been generated by finetuning the model to HRES, and"
            " using HRES data as input, as opposed to ERA5.\nThis means that"
            " this is similar to the quality the GraphCast model can generate in"
            " real-time."
        ),
        "Units": (
            "temperature_2m: 2m air temperature [°C]\ntotal_precipitation:"
            " Total precipitation [mm]\nu_component_of_wind_10m: U-component of"
            " wind at 10m [m/s]\nv_component_of_wind_10m: V-component of wind"
            " at 10m [m/s]"
        ),
        "Version": "1.1",
    },
    Product.AIFS: {
        "Citation": (
            "Lang, S., et al. (2024), AIFS - ECMWF's machine-learning data"
            " assimilation and forecasting system. arXiv:2406.01465."
        ),
        "License": "CC-BY-4.0",
        "Product": "ECMWF-AIFS",
        "Released": "2024-04-01",
        "Sources": (
            "AIFS single-forecast dataset provided by ECMWF via dynamical.org."
            " https://dynamical.org/catalog/ecmwf-aifs-single-forecast"
        ),
        "Units": (
            "temperature_2m: 2m air temperature [°C]\n"
            "total_precipitation: Total precipitation [mm]\n"
            "u_component_of_wind_10m: U-component of wind at 10m [m/s]\n"
            "v_component_of_wind_10m: V-component of wind at 10m [m/s]"
        ),
        "Version": "1.0",
    },
    Product.DYNAMICAL_IMERG: {
        "Citation": (
            "Huffman, G.J., et al. (2024), GPM IMERG Early Precipitation L3"
            " Half Hourly 0.1 degree x 0.1 degree V07 via dynamical.org."
        ),
        "License": "https://gpm.nasa.gov/data/policy",
        "Product": "IMERG v07 Early (dynamical.org)",
        "Released": "2024-11-18",
        "Sources": (
            "IMERG-Early v07 from NASA GPM, accessed via dynamical.org Icechunk"
            " catalog. https://dynamical.org/catalog/nasa-imerg-analysis-early"
        ),
        "Units": "precipitation [mm]",
        "Version": "1.1",
    },
}

# Storage path templates / defaults (aligned with flood-forecasting team).
DEFAULT_STORAGE_PATHS: Mapping[Product, Mapping[str, str]] = {
    Product.ERA5_LAND: {
        "wb2_s2s_zarr": (
            "gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr"
        ),
        "ee_image_collection": "ECMWF/ERA5_LAND/HOURLY",
    },
    Product.CPC: {
        "psl_netcdf": (
            "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/"
        ),
        "ee_image_collection": "NOAA/CPC/GLOBAL_PRECIP",
    },
    Product.IMERG: {
        "gesdisc_url": (
            "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07/"
        ),
        "ee_image_collection": "NASA/GPM_L3/IMERG_V07",
    },
    Product.CHIRPS: {
        "chc_netcdf": (
            "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
        ),
        "chc_tifs": (
            "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/"
        ),
        "ee_image_collection": "UCSB-CHG/CHIRPS/DAILY",
    },
    Product.CHIRPS_GEFS: {
        "chc_forecast_v2": (
            "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v2/daily/global/"
        ),
        "chc_forecast_v3": (
            "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/daily/global/"
        ),
    },
    Product.HRES: {
        "wb2_zarr": (
            "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
        ),
        # ECMWF Open Data public archive
        "ecmwf_open_data": "https://data.ecmwf.int/forecasts/",
    },
    Product.GRAPHCAST: {
        "wb2_zarr": (
            "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr"
        ),
    },
    Product.AIFS: {
        "dynamical_id": "ecmwf-aifs-single-forecast",
    },
    Product.DYNAMICAL_IMERG: {
        "dynamical_id": "nasa-imerg-analysis-early",
    },
}

# Zarr default chunking for Map-Only Direct Chunk Writing:
# 1 chunk along date dimension allows independent lock-free worker writes.
DEFAULT_CHUNKS_NOWCAST = {"basin": -1, "date": 1}
DEFAULT_CHUNKS_FORECAST = {"basin": -1, "date": 1, "lead_time": -1}
