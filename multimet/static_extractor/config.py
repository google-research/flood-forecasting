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

"""Configuration and schema definitions for the Caravan static attributes extractor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

# -------------------------------------------------------------------------
# Caravan HydroATLAS Property Definitions & Classifications
# Reference: https://github.com/kratzert/Caravan/blob/main/code/Caravan_part1_Earth_Engine.ipynb
# -------------------------------------------------------------------------

# 1. Discrete Categorical Properties (Aggregated by Area-Weighted Majority Vote)
MAJORITY_PROPERTIES: List[str] = [
    "clz_cl_smj",  # Climate zones (18 classes)
    "cls_cl_smj",  # Climate strata (125 classes)
    "glc_cl_smj",  # Land cover (22 classes)
    "pnv_cl_smj",  # Potential natural vegetation (15 classes)
    "wet_cl_smj",  # Wetland (12 classes)
    "tbi_cl_smj",  # Terrestrial biomes (14 classes)
    "tec_cl_smj",  # Terrestrial Ecoregions (846 classes)
    "fmh_cl_smj",  # Freshwater Major Habitat Types (13 classes)
    "fec_cl_smj",  # Freshwater Ecoregions (426 classes)
    "lit_cl_smj",  # Lithological classes (16 classes)
]

# 2. Pour-Point / Upstream Terminal Properties
POUR_POINT_PROPERTIES: List[str] = [
    "dis_m3_pmn",
    "dis_m3_pmx",
    "dis_m3_pyr",
    "lkv_mc_usu",
    "rev_mc_usu",
    "ria_ha_usu",
    "riv_tc_usu",
    "pop_ct_usu",
    "dor_pc_pva",
]

# 3. HydroSHEDS System / Topological Properties (Ignored or processed separately)
IGNORE_PROPERTIES: List[str] = [
    "system:index",
    "COAST",
    "DIST_MAIN",
    "DIST_SINK",
    "ENDO",
    "MAIN_BAS",
    "NEXT_SINK",
    "ORDER_",
    "PFAF_ID",
    "SORT",
    "Shape_Length",
    "Shape_Area",
    "Shape_Leng",
]

# 4. Topological Navigation Attributes (Used for subbasin accounting)
ADDITIONAL_PROPERTIES: List[str] = ["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA"]

# 5. Upstream Attributes Ignored (Per-subbasin '_s' areal properties used instead)
UPSTREAM_PROPERTIES: List[str] = [
    "aet_mm_uyr", "ari_ix_uav", "cly_pc_uav", "cmi_ix_uyr", "crp_pc_use", "ele_mt_uav", "ero_kh_uav",
    "for_pc_use", "gdp_ud_usu", "gla_pc_use", "glc_pc_u01", "glc_pc_u02", "glc_pc_u03", "glc_pc_u04",
    "glc_pc_u05", "glc_pc_u06", "glc_pc_u07", "glc_pc_u08", "glc_pc_u09", "glc_pc_u10", "glc_pc_u11",
    "glc_pc_u12", "glc_pc_u13", "glc_pc_u14", "glc_pc_u15", "glc_pc_u16", "glc_pc_u17", "glc_pc_u18",
    "glc_pc_u19", "glc_pc_u20", "glc_pc_u21", "glc_pc_u22", "hft_ix_u09", "hft_ix_u93", "inu_pc_ult",
    "inu_pc_umn", "inu_pc_umx", "ire_pc_use", "kar_pc_use", "lka_pc_use", "nli_ix_uav", "pac_pc_use",
    "pet_mm_uyr", "pnv_pc_u01", "pnv_pc_u02", "pnv_pc_u03", "pnv_pc_u04", "pnv_pc_u05", "pnv_pc_u06",
    "pnv_pc_u07", "pnv_pc_u08", "pnv_pc_u09", "pnv_pc_u10", "pnv_pc_u11", "pnv_pc_u12", "pnv_pc_u13",
    "pnv_pc_u14", "pnv_pc_u15", "pop_ct_ssu", "ppd_pk_uav", "pre_mm_uyr", "prm_pc_use", "pst_pc_use",
    "ria_ha_ssu", "riv_tc_ssu", "rdd_mk_uav", "slp_dg_uav", "slt_pc_uav", "snd_pc_uav", "snw_pc_uyr",
    "soc_th_uav", "swc_pc_uyr", "tmp_dc_uyr", "urb_pc_use", "wet_pc_u01", "wet_pc_u02", "wet_pc_u03",
    "wet_pc_u04", "wet_pc_u05", "wet_pc_u06", "wet_pc_u07", "wet_pc_u08", "wet_pc_u09", "wet_pc_ug1",
    "wet_pc_ug2", "gad_id_smj",
]

# 6. HydroBASINS Continent ID prefix mapping (First digit of HYBAS_ID)
CONTINENT_MAP: Mapping[int, str] = {
    1: "af",  # Africa
    2: "eu",  # Europe
    3: "si",  # Siberia
    4: "as",  # Central / South / East Asia
    5: "au",  # Australia / Oceania
    6: "sa",  # South America
    7: "na",  # North America
    8: "ar",  # Arctic
    9: "gr",  # Greenland
}

# 7. Cloud Storage Data Stores
GCS_HYDROATLAS_BUCKET: str = "gs://open-multimet/ancillary-data/hydroatlas"
GCS_HYDROATLAS_GDB_URI: str = f"{GCS_HYDROATLAS_BUCKET}/BasinATLAS_v10.gdb"
GCS_ERA5_CLIMATE_URI: str = f"{GCS_HYDROATLAS_BUCKET}/era5_climate"
GCS_ERA5_GRIDDED_ZARR_URI: str = "gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr"
GCS_BENCHMARK_URI: str = "gs://open-multimet/ancillary-data/benchmarks/benchmark_basins_500.parquet"

# Caravan Collection and Subdataset Subdirectory Mapping
CARAVAN_SUBDIR_MAPPING: Dict[str, Tuple[str, str]] = {
    # caravan-original/
    "CAMELS": ("caravan-original", "camels"),
    "CAMELSAUS": ("caravan-original", "camelsaus"),
    "CAMELSBR": ("caravan-original", "camelsbr"),
    "CAMELSCL": ("caravan-original", "camelscl"),
    "CAMELSGB": ("caravan-original", "camelsgb"),
    "HYSETS": ("caravan-original", "hysets"),
    "LAMAH": ("caravan-original", "lamah"),
    # caravan-extensions/
    "CAMELSCH": ("caravan-extensions", "camelsch"),
    "CAMELSCZ": ("caravan-extensions", "camelscz"),
    "CAMELSDE": ("caravan-extensions", "camelsde"),
    "CAMELSDK": ("caravan-extensions", "camelsdk"),
    "CAMELSES": ("caravan-extensions", "camelses"),
    "GRDC": ("caravan-extensions", "grdc"),
    "IL": ("caravan-extensions", "il"),
    "LAMAHICE": ("caravan-extensions", "lamahice"),
    # google-internal/
    "CAMELSCOL": ("google-internal", "camelscol"),
    "CAMELSFR": ("google-internal", "camelsfr"),
    "CAMELSIND": ("google-internal", "camelsind"),
    "CAMELSLUX": ("google-internal", "camelslux"),
    "CAMELSNZ": ("google-internal", "camelsnz"),
    "CAMELSPE": ("google-internal", "camelspe"),
}

# Curated Attribute Definitions with Metadata
ATTRIBUTE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Topography & Physiography
    "basin_area": {"name": "Basin Area", "category": "Topography", "unit": "km²", "scale": 1.0, "desc": "Total catchment drainage area in square kilometers"},
    "ele_mt_sav": {"name": "Mean Elevation", "category": "Topography", "unit": "m", "scale": 1.0, "desc": "Spatial mean catchment elevation"},
    "ele_mt_smn": {"name": "Min Elevation", "category": "Topography", "unit": "m", "scale": 1.0, "desc": "Minimum catchment elevation"},
    "ele_mt_smx": {"name": "Max Elevation", "category": "Topography", "unit": "m", "scale": 1.0, "desc": "Maximum catchment elevation"},
    "slp_dg_sav": {"name": "Mean Slope", "category": "Topography", "unit": "deg", "scale": 0.1, "desc": "Spatial mean terrain slope in degrees"},
    "sgr_dk_sav": {"name": "Stream Gradient", "category": "Topography", "unit": "dm/km", "scale": 1.0, "desc": "Stream gradient of river reaches"},

    # Climate & Seasonality
    "tmp_dc_syr": {"name": "Mean Annual Temperature", "category": "Climate", "unit": "°C", "scale": 0.1, "desc": "Spatial mean annual air temperature (HydroATLAS)"},
    "pre_mm_syr": {"name": "Mean Annual Precipitation", "category": "Climate", "unit": "mm/yr", "scale": 1.0, "desc": "Spatial mean annual precipitation total (HydroATLAS)"},
    "pet_mm_syr": {"name": "Potential Evapotranspiration", "category": "Climate", "unit": "mm/yr", "scale": 1.0, "desc": "Spatial mean annual potential evapotranspiration (HydroATLAS)"},
    "aet_mm_syr": {"name": "Actual Evapotranspiration", "category": "Climate", "unit": "mm/yr", "scale": 1.0, "desc": "Spatial mean annual actual evapotranspiration (AET)"},
    "ari_ix_sav": {"name": "Global Aridity Index", "category": "Climate", "unit": "index", "scale": 0.01, "desc": "Ratio of mean annual precipitation to PET (HydroATLAS)"},
    "cmi_ix_syr": {"name": "Climate Moisture Index", "category": "Climate", "unit": "index", "scale": 1.0, "desc": "Indicator of water availability vs evaporative demand"},
    "snw_pc_syr": {"name": "Snow Cover Extent", "category": "Climate", "unit": "%", "scale": 1.0, "desc": "Average annual snow cover extent percentage"},
    "run_mm_syr": {"name": "Annual Runoff", "category": "Climate", "unit": "mm/yr", "scale": 1.0, "desc": "Spatial mean annual natural surface runoff"},

    # ERA5-Land Caravan Climate Indices (1981-2020)
    "p_mean": {"name": "Mean Daily Precip (ERA5)", "category": "Climate", "unit": "mm/day", "scale": 1.0, "desc": "Long-term daily mean precipitation from ERA5-Land (1981-2020)"},
    "pet_mean": {"name": "Potential Evapotranspiration (FAO PM)", "category": "Climate", "unit": "mm/day", "scale": 1.0, "desc": "FAO-56 Penman-Monteith potential evapotranspiration (1981-2020)"},
    "pet_mean_FAO_PM": {"name": "Potential Evapotranspiration (FAO PM)", "category": "Climate", "unit": "mm/day", "scale": 1.0, "desc": "FAO-56 Penman-Monteith potential evapotranspiration (1981-2020)"},
    "pet_mean_ERA5_LAND": {"name": "Potential Evaporation (ERA5-Land Native)", "category": "Climate", "unit": "mm/day", "scale": 1.0, "desc": "Long-term daily mean potential evaporation from ERA5-Land (1981-2020)"},
    "aridity": {"name": "Aridity Index (FAO PM)", "category": "Climate", "unit": "ratio", "scale": 1.0, "desc": "Ratio of FAO Penman-Monteith PET to precipitation (1981-2020)"},
    "aridity_FAO_PM": {"name": "Aridity Index (FAO PM)", "category": "Climate", "unit": "ratio", "scale": 1.0, "desc": "Ratio of FAO Penman-Monteith PET to precipitation (1981-2020)"},
    "aridity_ERA5_LAND": {"name": "Aridity Index (ERA5-Land Native)", "category": "Climate", "unit": "ratio", "scale": 1.0, "desc": "Ratio of ERA5-Land native potential evaporation to precipitation (1981-2020)"},
    "frac_snow": {"name": "Snow Fraction (ERA5)", "category": "Climate", "unit": "%", "scale": 100.0, "desc": "Fraction of precipitation falling when mean temperature < 0°C (1981-2020)"},
    "moisture_index": {"name": "Moisture Index (FAO PM)", "category": "Climate", "unit": "index", "scale": 1.0, "desc": "Annual moisture index following Knoben et al. 2018 with FAO PM PET (1981-2020)"},
    "moisture_index_FAO_PM": {"name": "Moisture Index (FAO PM)", "category": "Climate", "unit": "index", "scale": 1.0, "desc": "Annual moisture index following Knoben et al. 2018 with FAO PM PET (1981-2020)"},
    "moisture_index_ERA5_LAND": {"name": "Moisture Index (ERA5-Land Native)", "category": "Climate", "unit": "index", "scale": 1.0, "desc": "Annual moisture index following Knoben et al. 2018 with ERA5-Land native PEV (1981-2020)"},
    "seasonality": {"name": "Seasonality Index (FAO PM)", "category": "Climate", "unit": "index", "scale": 1.0, "desc": "Precipitation and FAO PM PET seasonality following Knoben et al. 2018 (1981-2020)"},
    "seasonality_FAO_PM": {"name": "Seasonality Index (FAO PM)", "category": "Climate", "unit": "index", "scale": 1.0, "desc": "Precipitation and FAO PM PET seasonality following Knoben et al. 2018 (1981-2020)"},
    "seasonality_ERA5_LAND": {"name": "Seasonality Index (ERA5-Land Native)", "category": "Climate", "unit": "index", "scale": 1.0, "desc": "Precipitation and ERA5-Land native PEV seasonality following Knoben et al. 2018 (1981-2020)"},
    "high_prec_freq": {"name": "High Precip Frequency", "category": "Climate", "unit": "days/yr", "scale": 365.25, "desc": "Annual frequency of extreme precipitation days (>= 5x mean daily precip)"},
    "high_prec_dur": {"name": "High Precip Duration", "category": "Climate", "unit": "days", "scale": 1.0, "desc": "Mean duration of consecutive extreme precipitation days (1981-2020)"},
    "low_prec_freq": {"name": "Low Precip Frequency", "category": "Climate", "unit": "days/yr", "scale": 365.25, "desc": "Annual frequency of dry days (< 1 mm/day) (1981-2020)"},
    "low_prec_dur": {"name": "Low Precip Duration", "category": "Climate", "unit": "days", "scale": 1.0, "desc": "Mean duration of consecutive dry spells (< 1 mm/day) (1981-2020)"},

    # Soils & Geology
    "cly_pc_sav": {"name": "Clay Content", "category": "Soils", "unit": "%", "scale": 1.0, "desc": "Volumetric fraction of clay in topsoil"},
    "slt_pc_sav": {"name": "Silt Content", "category": "Soils", "unit": "%", "scale": 1.0, "desc": "Volumetric fraction of silt in topsoil"},
    "snd_pc_sav": {"name": "Sand Content", "category": "Soils", "unit": "%", "scale": 1.0, "desc": "Volumetric fraction of sand in topsoil"},
    "soc_th_sav": {"name": "Soil Organic Carbon", "category": "Soils", "unit": "t/ha", "scale": 1.0, "desc": "Mass density of organic carbon in soil column"},
    "swc_pc_syr": {"name": "Soil Water Content", "category": "Soils", "unit": "%", "scale": 1.0, "desc": "Annual average volumetric soil water content"},
    "gwt_cm_sav": {"name": "Water Table Depth", "category": "Soils", "unit": "cm", "scale": 1.0, "desc": "Mean depth from land surface to groundwater table"},
    "kar_pc_sse": {"name": "Karst Area Fraction", "category": "Soils", "unit": "%", "scale": 1.0, "desc": "Percentage of catchment underlain by karst formations"},
    "ero_kh_sav": {"name": "Soil Erodibility", "category": "Soils", "unit": "K-factor", "scale": 0.01, "desc": "USLE soil erodibility factor"},

    # Land Cover & Vegetation
    "for_pc_sse": {"name": "Forest Fraction", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Total forest canopy cover percentage"},
    "crp_pc_sse": {"name": "Cropland Fraction", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Agricultural cropland percentage"},
    "pst_pc_sse": {"name": "Pasture Fraction", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Managed grazing / pasture land percentage"},
    "ire_pc_sse": {"name": "Irrigated Land", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Equipped for irrigation agricultural land"},
    "urb_pc_sse": {"name": "Urban Extent", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Artificial impervious urban land percentage"},
    "gla_pc_sse": {"name": "Glacier Fraction", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Perennial glacier and ice sheet coverage"},
    "prm_pc_sse": {"name": "Permafrost Extent", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Catchment area under continuous or discontinuous permafrost"},
    "pac_pc_sse": {"name": "Protected Natural Areas", "category": "Land Cover", "unit": "%", "scale": 1.0, "desc": "Catchment area inside designated nature reserves"},

    # Hydrology & Inundation
    "lka_pc_sse": {"name": "Lake Area Fraction", "category": "Hydrology", "unit": "%", "scale": 1.0, "desc": "Catchment surface area covered by natural lakes"},
    "inu_pc_smn": {"name": "Min Inundation Extent", "category": "Hydrology", "unit": "%", "scale": 1.0, "desc": "Minimum monthly surface water inundation extent"},
    "inu_pc_smx": {"name": "Max Inundation Extent", "category": "Hydrology", "unit": "%", "scale": 1.0, "desc": "Maximum monthly surface water inundation extent"},
    "inu_pc_slt": {"name": "Long-Term Inundation", "category": "Hydrology", "unit": "%", "scale": 1.0, "desc": "Long-term maximum surface water inundation extent"},

    # Anthropogenic & Human Modification
    "ppd_pk_sav": {"name": "Population Density", "category": "Anthropogenic", "unit": "people/km²", "scale": 1.0, "desc": "Spatial mean human population density"},
    "nli_ix_sav": {"name": "Nighttime Lights Index", "category": "Anthropogenic", "unit": "index", "scale": 0.1, "desc": "Satellite-derived nighttime luminosity"},
    "rdd_mk_sav": {"name": "Road Network Density", "category": "Anthropogenic", "unit": "m/km²", "scale": 1.0, "desc": "Total road length per unit catchment area"},
    "hft_ix_s09": {"name": "Human Footprint Index", "category": "Anthropogenic", "unit": "index (0-100)", "scale": 0.1, "desc": "Cumulative terrestrial human footprint score"},
    "gdp_ud_sav": {"name": "Gross Domestic Product (GDP)", "category": "Anthropogenic", "unit": "USD/capita", "scale": 1.0, "desc": "Economic output per capita in the basin"},
    "hdi_ix_sav": {"name": "Human Development Index (HDI)", "category": "Anthropogenic", "unit": "index (0-1)", "scale": 0.001, "desc": "Socioeconomic human development index"},
}


def get_default_gdb_path() -> Path:
  """Returns the local runtime staging path for BasinATLAS_v10.gdb.

  The data store is gs://open-multimet/ancillary-data/hydroatlas/BasinATLAS_v10.gdb.
  This local directory serves strictly as a temporary runtime staging cache.
  """
  return (
      Path.home()
      / ".cache"
      / "googlehydrology"
      / "hydroatlas"
      / "BasinATLAS_v10.gdb"
  )


def get_default_era5_cache_dir() -> Path:
  """Returns the local runtime staging directory for continental ERA5 climate index files.

  The data store is gs://open-multimet/ancillary-data/hydroatlas/era5_climate/.
  This local directory serves strictly as a temporary runtime staging cache.
  """
  target = Path.home() / ".cache" / "googlehydrology" / "era5_climate"
  target.mkdir(parents=True, exist_ok=True)
  return target
