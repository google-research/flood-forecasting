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
    Product.CPC: (
        "cpc_precipitation",
        "cpc_num_stations",
    ),
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

# Companion audit variable recording the area-weighted fraction [0.0, 1.0] of
# missing (NaN) pixels within each catchment polygon at each timestep, matching
# Google's internal flood-forecasting CookieCutterResult.missing_values field.
MISSING_FRACTION_VAR: Mapping[Product, str] = {
    Product.ERA5_LAND: "era5land_missing_fraction",
    Product.CPC: "cpc_missing_fraction",
    Product.IMERG: "imerg_missing_fraction",
    Product.CHIRPS: "chirps_missing_fraction",
    Product.CHIRPS_GEFS: "chirpsgefs_missing_fraction",
    Product.HRES: "hres_missing_fraction",
    Product.GRAPHCAST: "graphcast_missing_fraction",
    Product.AIFS: "aifs_missing_fraction",
    Product.DYNAMICAL_IMERG: "imerg_missing_fraction",
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
        "Units": (
            "cpc_precipitation: precipitation [mm/day]\n"
            "cpc_num_stations: reporting rain gauge stations [count]"
        ),
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
            "temperature_2m: Daily mean 2m air temperature [°C]\n"
            "temperature_2m_max: Daily maximum 2m air temperature [°C]\n"
            "temperature_2m_min: Daily minimum 2m air temperature [°C]\n"
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
        "Citation": (
            "ECMWF (2024): IFS High-Resolution (HRES) Operational Atmospheric"
            " Model Forecasts. European Centre for Medium-Range Weather"
            " Forecasts."
        ),
        "License": (
            "Creative Commons Attribution 4.0 International (CC-BY-4.0)."
            " https://www.ecmwf.int/en/forecasts/datasets/open-data"
        ),
        "Product": "ECMWF IFS HRES (10-day forecast)",
        "Released": "2024-11-18",
        "Sources": "ECMWF Operational High-Resolution Forecasts (0.1 / 0.25 deg).",
        "Units": (
            "surface_net_solar_radiation: Surface net solar radiation [W/m2]\n"
            "surface_net_thermal_radiation: Surface net thermal radiation"
            " [W/m2]\n"
            "surface_pressure: Surface pressure [kPa]\n"
            "temperature_2m: 2m air temperature [°C]\n"
            "total_precipitation: Total precipitation [mm]"
        ),
        "Version": "1.1",
    },
    Product.GRAPHCAST: {
        "Citation": (
            "Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P.,"
            " Fortunato, M., Alet, F., ... & Battaglia, P. (2023). Learning"
            " skillful medium-range global weather forecasting. Science,"
            " 382(6677), 1416-1421."
        ),
        "License": (
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
            " International (CC-BY-NC-SA 4.0). Google DeepMind."
        ),
        "Product": "GraphCast Operational Forecast (10-day)",
        "Released": "2024-11-18",
        "Sources": "Google DeepMind GraphCast medium-range weather forecast.",
        "Units": (
            "temperature_2m: 2m air temperature [°C]\n"
            "total_precipitation: Total precipitation [mm]\n"
            "u_component_of_wind_10m: U-component of wind at 10m [m/s]\n"
            "v_component_of_wind_10m: V-component of wind at 10m [m/s]"
        ),
        "Version": "1.1",
    },
    Product.AIFS: {
        "Citation": (
            "Lang, S., Alexe, M., Chantry, M., Dramsch, J., Dueben, P.,"
            " Lessig, C., ... & Nipen, T. (2024). AIFS - ECMWF's data-driven"
            " forecasting system. arXiv:2406.01465."
        ),
        "License": (
            "Creative Commons Attribution 4.0 International (CC-BY-4.0)."
            " ECMWF Open Data / dynamical.org."
        ),
        "Product": "ECMWF AIFS Single Forecast (10-day)",
        "Released": "2024-11-18",
        "Sources": (
            "ECMWF Artificial Intelligence Forecasting System (AIFS) accessed"
            " via dynamical.org Icechunk catalog."
            " https://dynamical.org/catalog/ecmwf-aifs-single-forecast"
        ),
        "Units": (
            "temperature_2m: 2m air temperature [°C]\n"
            "total_precipitation: Total precipitation [mm]\n"
            "u_component_of_wind_10m: U-component of wind at 10m [m/s]\n"
            "v_component_of_wind_10m: V-component of wind at 10m [m/s]"
        ),
        "Version": "1.1",
    },
    Product.CHIRPS: {
        "Citation": (
            "Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J.,"
            " Shukla, S., Husak, G., Rowland, J., Harrison, L., Hoell, A. and"
            " Michaelsen, J. (2015), The climate hazards infrared precipitation"
            " with stations—a new environmental record for monitoring extremes."
            " Scientific Data 2, 150066. doi:10.1038/sdata.2015.66"
        ),
        "License": (
            "This datasets are in the public domain. To the extent possible"
            " under law, Pete Peterson has waived all copyright and related or"
            " neighboring rights to Climate Hazards Group Infrared"
            " Precipitation with Stations (CHIRPS).\nSee"
            " https://chc.ucsb.edu/data/chirps"
        ),
        "Product": "CHIRPS v2.0",
        "Released": "2024-11-18",
        "Sources": (
            "CHIRPS v2.0 daily global precipitation by Climate Hazards Center"
            " (CHC), UC Santa Barbara. https://chc.ucsb.edu/data/chirps"
        ),
        "Units": "precipitation [mm]",
        "Version": "1.1",
    },
    Product.CHIRPS_GEFS: {
        "Citation": (
            "Harrison, L., Landsfeld, M., Husak, G., Davenport, F., Shukla,"
            " S., Turner, W., Peterson, P., & Funk, C. (2022). Advancing early"
            " warning capabilities with CHIRPS-compatible NCEP GEFS"
            " precipitation forecasts. Scientific Data, 9(1), 355."
        ),
        "License": (
            "Public domain. Climate Hazards Center, UC Santa Barbara."
            " https://chc.ucsb.edu/data/chirps-gefs"
        ),
        "Product": "CHIRPS-GEFS (16-day forecast)",
        "Released": "2024-11-18",
        "Sources": (
            "CHIRPS-GEFS bias-corrected NCEP GEFS precipitation forecasts by"
            " UC Santa Barbara Climate Hazards Center."
        ),
        "Units": "precipitation [mm]",
        "Version": "1.1",
    },
    Product.DYNAMICAL_IMERG: {
        "Citation": (
            "Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, Jackson"
            " Tan (2024), GPM IMERG Early Precipitation L3 Half Hourly 0.1"
            " degree x 0.1 degree V07, Greenbelt, MD, Goddard Earth Sciences"
            " Data and Information Services Center (GES DISC)."
        ),
        "License": "NASA GPM Open Data Policy. https://gpm.nasa.gov/data/policy",
        "Product": "IMERG v07 Early (dynamical.org catalog)",
        "Released": "2024-11-18",
        "Sources": (
            "IMERG-Early v07 from NASA GPM, accessed via dynamical.org Icechunk"
            " catalog. https://dynamical.org/catalog/nasa-imerg-analysis-early"
        ),
        "Units": "precipitation [mm]",
        "Version": "1.1",
    },
}

# Upstream agency HTTP endpoints and catalog identifiers for direct third-party
# downloading. Note: NO gs:// bucket paths are hardcoded here — any Zarr or
# gridded archive path must be explicitly supplied by the user.
DEFAULT_STORAGE_PATHS: Mapping[Product, Mapping[str, str]] = {
    Product.ERA5_LAND: {},
    Product.CPC: {
        "psl_netcdf": (
            "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/"
        ),
    },
    Product.IMERG: {
        "gesdisc_url": (
            "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07/"
        ),
    },
    Product.CHIRPS: {
        "chc_netcdf": (
            "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
        ),
        "chc_tifs": (
            "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/"
        ),
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
        # ECMWF Open Data public HTTP archive
        "ecmwf_open_data": "https://data.ecmwf.int/forecasts/",
    },

    Product.GRAPHCAST: {},
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
