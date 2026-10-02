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

"""Static Catchment Attribute Extraction Engine for Caravan & HydroATLAS.

Interfaces with global BasinATLAS (HydroATLAS v1.0 Level 12) local geodatabase
or shapefile to compute exact area-weighted physiographic, hydro-environmental,
soil, land-cover, climatology, and anthropogenic attributes for user-supplied
watershed polygons following the Caravan aggregation methodology:
- Area-weighted majority voting for discrete categorical classes
- Downstream topological outlet tracing via NEXT_DOWN for pour-point metrics
- Area-weighted averaging for continuous physiographic & hydro-climatic properties
- 40-year ERA5 climate indices (1981-2020)
"""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pyproj
import shapely.geometry
from tqdm.auto import tqdm
import xarray as xr

from multimet.static_extractor.climate import (
    ERA5ClimateLoader,
    ERA5GriddedExtractor,
    compute_caravan_climate_metrics,
)
from multimet.static_extractor.config import (
    ADDITIONAL_PROPERTIES,
    ATTRIBUTE_DEFINITIONS,
    IGNORE_PROPERTIES,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
    UPSTREAM_PROPERTIES,
)
from multimet.static_extractor.gcs import download_hydroatlas_from_gcs

shape = shapely.geometry.shape
Point = shapely.geometry.Point
Polygon = shapely.geometry.Polygon
MultiPolygon = shapely.geometry.MultiPolygon
box = shapely.geometry.box
_WGS84_GEOD = pyproj.Geod(ellps="WGS84")

warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth.*")

logger = logging.getLogger(__name__)

_WORKER_EXTRACTOR: Optional[StaticAttributesExtractor] = None
_WORKER_EXTRACTOR_KEY: Optional[Tuple[str, Optional[str], Optional[str]]] = None


def _get_worker_extractor(
    gdb_path: str,
    era5_cache_dir: Optional[str],
    gridded_era5_uri: Optional[str],
) -> StaticAttributesExtractor:
  global _WORKER_EXTRACTOR, _WORKER_EXTRACTOR_KEY
  key = (gdb_path, era5_cache_dir, gridded_era5_uri)
  if _WORKER_EXTRACTOR is None or _WORKER_EXTRACTOR_KEY != key:
    _WORKER_EXTRACTOR = StaticAttributesExtractor(
        gdb_path=gdb_path,
        era5_cache_dir=era5_cache_dir,
        gridded_era5_uri=gridded_era5_uri,
    )
    _WORKER_EXTRACTOR_KEY = key
  return _WORKER_EXTRACTOR


def _worker_extract_polygon(args: tuple) -> Dict[str, Any]:
  (
      geom,
      gid,
      min_overlap_threshold,
      era5_source,
      gdb_path,
      era5_cache_dir,
      gridded_era5_uri,
      skip_climate,
  ) = args
  ext = _get_worker_extractor(gdb_path, era5_cache_dir, gridded_era5_uri)
  return ext.extract_attributes_for_polygon(
      geom,
      catchment_id=gid,
      min_overlap_threshold=min_overlap_threshold,
      era5_source=era5_source,
      _batch_mode=True,
      _skip_climate=skip_climate,
  )


def compute_pour_point_properties(
    basin_data: Dict[str, List[Any]],
    min_overlap_threshold: float = 0.0,
    pour_point_properties: Optional[List[str]] = None,
) -> Dict[str, float]:
  """Computes Caravan pour-point metrics by following NEXT_DOWN to find the outlet sub-basins."""
  props_to_compute = pour_point_properties or POUR_POINT_PROPERTIES
  if not props_to_compute:
    return {}

  weights = np.array(basin_data.get("weights", []))
  sub_areas = np.array(basin_data.get("SUB_AREA", []))
  if len(weights) == 0 or len(sub_areas) == 0:
    return {p: np.nan for p in props_to_compute}

  percentage_overlap = np.where(sub_areas > 0, weights / sub_areas, 0.0)
  if len(percentage_overlap) == 0:
    return {p: np.nan for p in props_to_compute}

  current_basin_pos = int(np.argmax(percentage_overlap))
  next_down_id = basin_data["NEXT_DOWN"][current_basin_pos]

  # Traverse downstream until leaving the polygon or hitting ocean (0)
  while True:
    if next_down_id == 0:
      break
    if next_down_id not in basin_data["HYBAS_ID"]:
      break
    next_down_pos = basin_data["HYBAS_ID"].index(next_down_id)
    if percentage_overlap[next_down_pos] < 0.5:
      break
    next_down_id = basin_data["NEXT_DOWN"][next_down_pos]

  # Find all sub-basins draining into the terminal downstream outlet
  direct_upstream_polygons = []
  for i, next_down in enumerate(basin_data["NEXT_DOWN"]):
    if (next_down == next_down_id) and (
        (basin_data["weights"][i] > min_overlap_threshold)
        or (percentage_overlap[i] > 0.5)
    ):
      direct_upstream_polygons.append(i)

  if not direct_upstream_polygons:
    direct_upstream_polygons = [current_basin_pos]

  aggregated = {}
  for prop in props_to_compute:
    if prop in basin_data:
      aggregated[prop] = float(
          sum(basin_data[prop][i] for i in direct_upstream_polygons)
      )
    else:
      aggregated[prop] = np.nan
  return aggregated


class StaticAttributesExtractor:
  """Extracts and computes Caravan & HydroATLAS static catchment attributes."""

  def __init__(
      self,
      gdb_path: Union[str, Path],
      era5_source: Optional[str] = None,
      era5_cache_dir: Optional[Union[str, Path]] = None,
      gridded_era5_uri: Optional[Union[str, Path]] = None,
      gcs_gdb_uri: Optional[str] = None,
      gcs_era5_climate_uri: Optional[str] = None,
  ):
    """Initializes the StaticAttributesExtractor.

    Args:
      gdb_path: Path to local BasinATLAS_v10.gdb directory or BasinATLAS_v10_lev12.shp.
      era5_source: Optional sourcing mode for ERA5 climate attributes ("hybas"
        or "gridded"). Must be provided either at initialization or when calling
        extraction methods (unless timeseries_df is passed).
      era5_cache_dir: Directory containing continental ERA5 climate index files
        (required when era5_source="hybas").
      gridded_era5_uri: GCS URI or local path to gridded daily ERA5 Zarr store
        (required when era5_source="gridded"; optional when era5_source="hybas"
        to compute *_ERA5_LAND attributes).
      gcs_gdb_uri: Optional GCS URI from which to download BasinATLAS_v10.gdb
        into gdb_path if gdb_path does not yet exist locally.
      gcs_era5_climate_uri: Optional GCS URI from which to download continental
        ERA5 climate tables into era5_cache_dir if not yet present locally.
    """
    if not gdb_path:
      raise ValueError("gdb_path must be explicitly provided.")
    if era5_source is not None and era5_source.lower() not in {"hybas", "gridded"}:
      raise ValueError(
          f"Invalid era5_source {era5_source!r}; must be 'hybas' or 'gridded'."
      )
    self.era5_source = era5_source.lower() if era5_source else None
    self.gdb_path = Path(gdb_path)

    if (
        not self.gdb_path.exists()
        or (self.gdb_path.is_dir() and not any(self.gdb_path.iterdir()))
    ):
      if gcs_gdb_uri:
        logger.info(
            "BasinATLAS GDB not found at %s. Downloading from %s...",
            self.gdb_path,
            gcs_gdb_uri,
        )
        self.gdb_path = download_hydroatlas_from_gcs(
            target_dir=self.gdb_path, source_uri=gcs_gdb_uri
        )
      else:
        raise FileNotFoundError(
            f"BasinATLAS dataset not found at {self.gdb_path}."
        )

    # Determine if target is a FileGDB directory or shapefile
    self.is_shapefile = str(self.gdb_path).endswith(".shp")
    self.layer_name = None if self.is_shapefile else "BasinATLAS_v10_lev12"

    if self.is_shapefile:
      info = pyogrio.read_info(self.gdb_path)
    else:
      info = pyogrio.read_info(self.gdb_path, layer=self.layer_name)
    self.all_gdb_fields = list(info["fields"])

    self.use_properties = [
        p
        for p in self.all_gdb_fields
        if p not in IGNORE_PROPERTIES + UPSTREAM_PROPERTIES
    ]
    self.caravan_feature_names = [
        p for p in self.use_properties if p not in ADDITIONAL_PROPERTIES
    ]

    self.era5_cache_dir = Path(era5_cache_dir) if era5_cache_dir else None
    self.gcs_era5_climate_uri = gcs_era5_climate_uri
    self.era5_loader = (
        ERA5ClimateLoader(
            cache_dir=self.era5_cache_dir,
            gcs_source_uri=self.gcs_era5_climate_uri,
        )
        if self.era5_cache_dir is not None
        else None
    )

    self.gridded_era5_uri = str(gridded_era5_uri) if gridded_era5_uri else None
    self.gridded_extractor = (
        ERA5GriddedExtractor(zarr_uri=self.gridded_era5_uri)
        if self.gridded_era5_uri is not None
        else None
    )

    if self.era5_source == "hybas" and self.era5_loader is None:
      raise ValueError("era5_cache_dir must be provided when era5_source='hybas'.")
    if self.era5_source == "gridded" and self.gridded_extractor is None:
      raise ValueError("gridded_era5_uri must be provided when era5_source='gridded'.")

  def _era5_land_variants_from_gridded(
      self,
      geom,
      baseline_years: Tuple[int, int],
  ) -> Dict[str, float]:
    """Computes the four *_ERA5_LAND climate attributes from the gridded Zarr store."""
    if self.gridded_extractor is None:
      raise ValueError("gridded_era5_uri is not configured on this extractor.")
    keys = (
        "pet_mean_ERA5_LAND",
        "aridity_ERA5_LAND",
        "moisture_index_ERA5_LAND",
        "seasonality_ERA5_LAND",
    )
    gridded = self.gridded_extractor.extract_climate_metrics_for_polygon(
        geom, baseline_years=baseline_years
    )
    return {k: gridded.get(k, np.nan) for k in keys}

  def _read_subbasins_in_bbox(
      self, bbox: Tuple[float, float, float, float]
  ) -> gpd.GeoDataFrame:
    """Reads Level 12 sub-basins within bounding box from GDB or shapefile."""
    if not self.gdb_path.exists():
      raise FileNotFoundError(
          f"BasinATLAS dataset not found at {self.gdb_path}."
      )

    read_kwargs = {"bbox": bbox}
    if not self.is_shapefile:
      read_kwargs["layer"] = self.layer_name

    return pyogrio.read_dataframe(self.gdb_path, **read_kwargs)

  def _resolve_era5_source(self, era5_source: Optional[str]) -> str:
    source = (era5_source or self.era5_source or "").strip().lower()
    if source not in {"hybas", "gridded"}:
      raise ValueError(
          "era5_source must be explicitly specified as either 'hybas' or 'gridded'."
      )
    if source == "hybas" and self.era5_loader is None:
      raise ValueError("era5_cache_dir must be provided when era5_source='hybas'.")
    if source == "gridded" and self.gridded_extractor is None:
      raise ValueError("gridded_era5_uri must be provided when era5_source='gridded'.")
    return source

  def _apply_climate_indices_to_result(
      self, res: Dict[str, Any], era5_indices: Dict[str, float]
  ) -> None:
    """Merges climate indices into a result dictionary."""
    caravan_attributes = res["caravan_attributes"]
    processed_attributes = res["processed_attributes"]
    categories_dict = res["categories"]
    summary = res["summary"]

    for k, v in era5_indices.items():
      caravan_attributes[k] = v

    categories_dict["Climate"] = [
        item for item in categories_dict.get("Climate", []) if item["key"] not in era5_indices
    ]
    for attr_key in era5_indices:
      if attr_key in ATTRIBUTE_DEFINITIONS:
        defn = ATTRIBUTE_DEFINITIONS[attr_key]
        raw_val = caravan_attributes[attr_key]
        scaled_val = np.nan if pd.isna(raw_val) else round(float(raw_val) * defn["scale"], 3)
        processed_attributes[attr_key] = scaled_val
        categories_dict[defn["category"]].append({
            "key": attr_key,
            "name": defn["name"],
            "value": scaled_val,
            "unit": defn["unit"],
            "description": defn["desc"],
            "category": defn["category"],
        })

    summary["era5_p_mean_mm_day"] = processed_attributes.get("p_mean", np.nan)
    summary["era5_pet_mean_mm_day"] = processed_attributes.get("pet_mean_ERA5_LAND", np.nan)
    summary["era5_fao_pet_mean_mm_day"] = processed_attributes.get("pet_mean_FAO_PM", np.nan)
    summary["era5_aridity"] = processed_attributes.get("aridity_ERA5_LAND", np.nan)
    summary["era5_fao_aridity"] = processed_attributes.get("aridity_FAO_PM", np.nan)
    summary["era5_frac_snow_pc"] = processed_attributes.get("frac_snow", np.nan)

  def extract_attributes_for_polygon(
      self,
      polygon_geojson: Union[Dict, Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame],
      catchment_id: Optional[str] = None,
      min_overlap_threshold: float = 0.0,
      baseline_years: Tuple[int, int] = (1981, 2020),
      timeseries_df: Optional[pd.DataFrame] = None,
      era5_source: Optional[str] = None,
      _batch_mode: bool = False,
      _skip_climate: bool = False,
  ) -> Dict[str, Any]:
    """Calculates Caravan HydroATLAS static attributes for a watershed polygon.

    Args:
      polygon_geojson: GeoJSON Feature, Geometry dict, Shapely Polygon/MultiPolygon,
        or single-row GeoDataFrame.
      catchment_id: Catchment identifier string (required unless present as
        'gauge_id' or 'catchment_id' in the input Feature or GeoDataFrame).
      min_overlap_threshold: Minimum area threshold in km2 for filtering small overlap slivers.
      baseline_years: Tuple of start and end years for climate baseline (default 1981-2020).
      timeseries_df: Optional daily timeseries DataFrame containing columns
        (total_precipitation or prcp, temperature or 2m_temperature,
        potential_evaporation or pet) to compute climate indices directly.
      era5_source: Required ERA5 sourcing mode ('hybas' or 'gridded') when timeseries_df is not provided.

    Returns:
      Dictionary containing extracted attributes, summary, categories, and area metadata.
    """
    actual_era5_source = None
    if (timeseries_df is None or timeseries_df.empty) and not _skip_climate:
      actual_era5_source = self._resolve_era5_source(era5_source)

    # 1. Parse Input Geometry and Catchment ID
    if isinstance(polygon_geojson, (gpd.GeoDataFrame, gpd.GeoSeries)):
      if len(polygon_geojson) != 1:
        raise ValueError(
            f"Expected a single-row GeoDataFrame or GeoSeries, got {len(polygon_geojson)} rows."
        )
      geom = (
          polygon_geojson.geometry.iloc[0]
          if hasattr(polygon_geojson, "geometry")
          else polygon_geojson.iloc[0]
      )
      if catchment_id is None and hasattr(polygon_geojson, "columns"):
        for id_col in ("gauge_id", "catchment_id"):
          if id_col in polygon_geojson.columns:
            catchment_id = str(polygon_geojson[id_col].iloc[0])
            break
    elif isinstance(polygon_geojson, dict):
      if polygon_geojson.get("type") == "Feature":
        geom_dict = polygon_geojson["geometry"]
        props = polygon_geojson.get("properties") or {}
        if catchment_id is None:
          catchment_id = props.get("gauge_id") or props.get("catchment_id")
      else:
        geom_dict = polygon_geojson
      geom = shape(geom_dict)
    else:
      geom = polygon_geojson

    if not catchment_id:
      raise ValueError(
          "catchment_id must be provided explicitly or present as 'gauge_id' / 'catchment_id' in the input feature."
      )
    catchment_id = str(catchment_id)

    if geom is None or geom.is_empty or not geom.is_valid:
      raise ValueError(
          f"Target polygon geometry for catchment {catchment_id!r} is empty or topologically invalid."
      )

    if geom.area <= 0:
      raise ValueError("Target polygon area must be greater than 0.")

    minx, miny, maxx, maxy = geom.bounds

    # 2. Read BasinATLAS Level 12 Sub-basins within Bounding Box
    bbox = (minx - 0.02, miny - 0.02, maxx + 0.02, maxy + 0.02)
    gdf_subbasins = self._read_subbasins_in_bbox(bbox)

    # 3. Calculate exact geometric intersections and WGS84 geodesic area weights in km²
    if len(gdf_subbasins) > 0:
      intersections = gdf_subbasins.geometry.intersection(geom)
      valid_mask = ~intersections.is_empty
      gdf_matched = gdf_subbasins[valid_mask].copy()
      gdf_matched["intersect_geom"] = intersections[valid_mask]
      gdf_matched["intersect_area_km2"] = [
          float(abs(_WGS84_GEOD.geometry_area_perimeter(g)[0]) / 1e6)
          for g in gdf_matched["intersect_geom"]
      ]
    else:
      gdf_matched = gpd.GeoDataFrame()

    # 4. Collect Sub-basin Data with Caravan Overlap Rules
    basin_data = defaultdict(list)
    if len(gdf_matched) == 0:
      logger.warning(
          "No BasinATLAS Level 12 units intersect catchment '%s' (bounds=%s); "
          "setting HydroATLAS attributes to NaN.",
          catchment_id,
          bbox,
      )
    else:
      for _, row in gdf_matched.iterrows():
        int_area = float(row["intersect_area_km2"])
        if "SUB_AREA" not in row or pd.isna(row["SUB_AREA"]) or float(row["SUB_AREA"]) <= 0:
          raise ValueError(
              "HydroATLAS sub-basin is missing a valid positive 'SUB_AREA' attribute."
          )
        sub_area = float(row["SUB_AREA"])

        # Caravan filtering threshold: either > min_overlap_threshold or >50% of sub-basin
        if (int_area > min_overlap_threshold) or (int_area / sub_area > 0.5):
          for prop in self.use_properties:
            if prop in row:
              basin_data[prop].append(row[prop])
          basin_data["weights"].append(int_area)

        basin_data["area_fragments"].append(int_area)

      if not basin_data["weights"]:
        logger.warning(
            "All intersecting sub-basins for catchment '%s' fell below "
            "min_overlap_threshold=%.3f km²; setting HydroATLAS attributes to NaN.",
            catchment_id,
            min_overlap_threshold,
        )

    weights = np.array(basin_data["weights"], dtype=float)
    mask = weights > min_overlap_threshold
    masked_weights = weights[mask]

    # 5. Aggregate Caravan Properties
    caravan_attributes: Dict[str, Any] = {}
    skip_props = {
        "weights",
        "UP_AREA",
        "area_fragments",
        "HYBAS_ID",
        "NEXT_DOWN",
        "SUB_AREA",
        "geometry",
        "geom",
        "Shape",
    }

    for key in self.use_properties:
      if key in skip_props or key in POUR_POINT_PROPERTIES:
        continue
      if key not in basin_data or len(masked_weights) == 0:
        caravan_attributes[key] = np.nan
        continue

      val = np.array(basin_data[key], dtype=float)
      masked_val = val[mask]

      # Caravan rule for wetland classes: no wetland (-999 / -9999 / <0) is mapped to class 13
      if key == "wet_cl_smj":
        masked_val = np.where(
            (masked_val == -999) | (masked_val == -9999) | (masked_val < 0),
            13,
            masked_val,
        )

      valid_idx = (masked_val > -900) & (~np.isnan(masked_val))

      if not np.any(valid_idx):
        caravan_attributes[key] = np.nan
      else:
        if key in MAJORITY_PROPERTIES:
          # Area-Weighted Majority Vote
          valid_vals = masked_val[valid_idx].astype(int)
          valid_w = masked_weights[valid_idx]
          val_counts = np.bincount(valid_vals, weights=valid_w)
          caravan_attributes[key] = int(val_counts.argmax())
        else:
          # Area-Weighted Average
          caravan_attributes[key] = float(
              np.average(
                  masked_val[valid_idx], weights=masked_weights[valid_idx]
              )
          )

    # 5b. Downstream Outlet Pour-Point Properties
    pour_point_attrs = compute_pour_point_properties(
        basin_data,
        min_overlap_threshold=min_overlap_threshold,
        pour_point_properties=POUR_POINT_PROPERTIES,
    )
    for k, v in pour_point_attrs.items():
      caravan_attributes[k] = v

    # 6. Extract / Compute ERA5-Land Climate Attributes (1981-2020)
    era5_indices = {}
    if not _skip_climate:
      if timeseries_df is not None and not timeseries_df.empty:
        p_col = next((c for c in ["total_precipitation", "prcp", "precip", "tp"] if c in timeseries_df.columns), None)
        t_col = next((c for c in ["temperature", "2m_temperature", "temp", "t2m"] if c in timeseries_df.columns), None)
        pet_era5_col = next((c for c in ["potential_evaporation", "pet_era5", "pev"] if c in timeseries_df.columns), None)
        pet_fao_col = next((c for c in ["pet_fao", "pet_mean_FAO_PM", "fao_pet"] if c in timeseries_df.columns), None)

        if not p_col or not t_col:
          raise ValueError(
              f"timeseries_df is missing required precipitation/temperature columns (found {list(timeseries_df.columns)})."
          )
        p_series = timeseries_df[p_col]
        t_series = timeseries_df[t_col]
        pet_era5_series = timeseries_df[pet_era5_col] if pet_era5_col else None
        pet_fao_series = timeseries_df[pet_fao_col] if pet_fao_col else None
        era5_indices = compute_caravan_climate_metrics(
            precipitation=p_series,
            temperature=t_series,
            pet_era5=pet_era5_series,
            pet_fao=pet_fao_series,
        )
      elif actual_era5_source == "gridded":
        era5_indices = self.gridded_extractor.extract_climate_metrics_for_polygon(
            geom, baseline_years=baseline_years
        )
      else:
        hybas_ids = (
            [int(hid) for hid in gdf_matched["HYBAS_ID"].values]
            if len(gdf_matched) > 0
            else []
        )
        intersect_weights = (
            [float(w) for w in gdf_matched["intersect_area_km2"].values]
            if len(gdf_matched) > 0
            else []
        )
        era5_indices = self.era5_loader.get_indices_for_subbasins(
            hybas_ids, intersect_weights
        )
        if self.gridded_extractor is not None:
          era5_indices.update(
              self._era5_land_variants_from_gridded(geom, baseline_years)
          )

      for k, v in era5_indices.items():
        caravan_attributes[k] = v

    # 7. Drainage Area & Aggregation Fraction
    total_frag_area = (
        float(sum(basin_data["area_fragments"]))
        if basin_data["area_fragments"]
        else float(abs(_WGS84_GEOD.geometry_area_perimeter(geom)[0]) / 1e6)
    )
    caravan_attributes["area"] = total_frag_area
    caravan_attributes["basin_area"] = total_frag_area
    caravan_attributes["area_fraction_used_for_aggregation"] = (
        float(sum(masked_weights) / total_frag_area)
        if total_frag_area > 0 and len(masked_weights) > 0
        else 0.0
    )

    # 8. Curated UI Schema Formatting
    processed_attributes: Dict[str, Any] = {}
    categories_dict: Dict[str, List[Dict[str, Any]]] = {
        "Topography": [],
        "Climate": [],
        "Soils": [],
        "Land Cover": [],
        "Hydrology": [],
        "Anthropogenic": [],
    }

    for attr_key, defn in ATTRIBUTE_DEFINITIONS.items():
      if attr_key in caravan_attributes:
        raw_val = caravan_attributes[attr_key]
        if pd.isna(raw_val):
          scaled_val = np.nan
        else:
          scaled_val = round(float(raw_val) * defn["scale"], 3)
          if defn["unit"] in ["m", "mm/yr", "people", "M m³"]:
            scaled_val = (
                round(scaled_val, 1)
                if defn["unit"] != "people"
                else int(round(scaled_val))
            )
          elif defn["unit"].startswith("class"):
            scaled_val = int(scaled_val)

        item = {
            "key": attr_key,
            "name": defn["name"],
            "value": scaled_val,
            "unit": defn["unit"],
            "description": defn["desc"],
            "category": defn["category"],
        }
        processed_attributes[attr_key] = scaled_val
        categories_dict[defn["category"]].append(item)

    # 9. Summary Metrics
    summary = {
        "catchment_id": catchment_id,
        "elevation_mean_m": processed_attributes.get("ele_mt_sav", np.nan),
        "slope_mean_deg": processed_attributes.get("slp_dg_sav", np.nan),
        "annual_precip_mm": processed_attributes.get("pre_mm_syr", np.nan),
        "annual_temp_c": processed_attributes.get("tmp_dc_syr", np.nan),
        "aridity_index": processed_attributes.get("ari_ix_sav", np.nan),
        "era5_p_mean_mm_day": processed_attributes.get("p_mean", np.nan),
        "era5_pet_mean_mm_day": processed_attributes.get(
            "pet_mean_ERA5_LAND", np.nan
        ),
        "era5_fao_pet_mean_mm_day": processed_attributes.get(
            "pet_mean_FAO_PM", np.nan
        ),
        "era5_aridity": processed_attributes.get("aridity_ERA5_LAND", np.nan),
        "era5_fao_aridity": processed_attributes.get("aridity_FAO_PM", np.nan),
        "era5_frac_snow_pc": processed_attributes.get("frac_snow", np.nan),
        "forest_fraction_pc": processed_attributes.get("for_pc_sse", np.nan),
        "cropland_fraction_pc": processed_attributes.get("crp_pc_sse", np.nan),
        "urban_fraction_pc": processed_attributes.get("urb_pc_sse", np.nan),
        "dominant_land_cover_class": caravan_attributes.get("glc_cl_smj", np.nan),
        "soil_clay_pc": processed_attributes.get("cly_pc_sav", np.nan),
        "soil_sand_pc": processed_attributes.get("snd_pc_sav", np.nan),
        "soil_silt_pc": processed_attributes.get("slt_pc_sav", np.nan),
        "groundwater_table_depth_cm": processed_attributes.get(
            "gwt_cm_sav", np.nan
        ),
        "soil_water_content_pc": processed_attributes.get("swc_pc_syr", np.nan),
        "inundation_max_pc": processed_attributes.get("inu_pc_smx", np.nan),
        "total_area_km2": round(total_frag_area, 2),
        "intersected_subbasins": len(gdf_matched),
        "subbasin_ids": (
            [int(hid) for hid in gdf_matched["HYBAS_ID"].values]
            if len(gdf_matched) > 0
            else []
        ),
    }

    return {
        "catchment_id": catchment_id,
        "caravan_attributes": caravan_attributes,
        "raw_attributes": caravan_attributes,
        "summary": summary,
        "categories": categories_dict,
        "processed_attributes": processed_attributes,
        "intersected_subbasins_count": len(gdf_matched),
        "total_area_km2": round(total_frag_area, 2),
    }

  def _populate_gridded_climate_batch(
      self,
      results: List[Dict[str, Any]],
      poly_tasks: List[Tuple[Any, str]],
      baseline_years: Tuple[int, int] = (1981, 2020),
  ) -> None:
    """Computes gridded ERA5 climate metrics in a single pass over Zarr chunks for all polygons."""
    if self.gridded_extractor is None:
      raise ValueError("gridded_era5_uri must be provided when era5_source='gridded'.")
    climate_map = self.gridded_extractor.extract_climate_metrics_for_polygons_batch(
        poly_tasks, baseline_years=baseline_years
    )
    for r in results:
      if r:
        cid = r["catchment_id"]
        self._apply_climate_indices_to_result(r, climate_map[cid])

  def extract_attributes_batch(
      self,
      features: List[Dict[str, Any]],
      min_overlap_threshold: float = 0.0,
      era5_source: Optional[str] = None,
  ) -> List[Dict[str, Any]]:
    """Extracts Caravan attributes for a list of GeoJSON Feature dicts."""
    actual_era5_source = self._resolve_era5_source(era5_source)
    skip_climate = actual_era5_source == "gridded"

    results = []
    poly_tasks = []
    for feat in features:
      if not isinstance(feat, dict):
        raise ValueError(
            "Each item in features must be a GeoJSON Feature dictionary containing 'properties' with 'gauge_id' or 'catchment_id'."
        )
      props = feat.get("properties") or {}
      c_id = props.get("gauge_id") or props.get("catchment_id")
      if not c_id:
        raise ValueError(
            "Each feature in extract_attributes_batch must specify 'gauge_id' or 'catchment_id' in its 'properties'."
        )
      c_id = str(c_id)
      geom = shape(feat["geometry"] if feat.get("type") == "Feature" else feat)
      poly_tasks.append((geom, c_id))
      res = self.extract_attributes_for_polygon(
          feat,
          catchment_id=c_id,
          min_overlap_threshold=min_overlap_threshold,
          era5_source=actual_era5_source,
          _skip_climate=skip_climate,
      )
      results.append(res)

    if skip_climate:
      self._populate_gridded_climate_batch(results, poly_tasks)

    return results

  def extract_attributes_from_file(
      self,
      input_path: Union[str, Path],
      output_csv_path: Optional[Union[str, Path]] = None,
      id_column: Optional[str] = None,
      min_overlap_threshold: float = 0.0,
      era5_source: Optional[str] = None,
      workers: int = 1,
      show_progress: bool = True,
      dataset_name: Optional[str] = None,
  ) -> pd.DataFrame:
    """Extracts Caravan attributes for all features in a vector file (Shapefile, GeoJSON, GeoPackage, Parquet).

    Args:
      input_path: Path to vector polygon file.
      output_csv_path: Optional path to save extracted attributes CSV.
      id_column: Name of column to use for basin / gauge ID (defaults to 'gauge_id').
      min_overlap_threshold: Minimum area threshold in km2.
      era5_source: Required ERA5 sourcing mode ('hybas' or 'gridded') if not set on extractor.
      workers: Number of parallel processes to use (default 1).
      show_progress: Whether to show an interactive tqdm progress bar.
      dataset_name: Optional dataset label to display in the progress bar.

    Returns:
      Pandas DataFrame with extracted attributes, indexed by gauge_id.
    """
    actual_era5_source = self._resolve_era5_source(era5_source)
    skip_climate = actual_era5_source == "gridded"

    in_path = Path(input_path)
    if not in_path.exists():
      raise FileNotFoundError(f"Input vector file does not exist: {in_path}")

    if str(in_path).endswith((".parquet", ".geoparquet")):
      gdf = gpd.read_parquet(in_path)
    else:
      gdf = gpd.read_file(in_path)

    if gdf.crs is None:
      raise ValueError(
          f"Input file {in_path} has no coordinate reference system (CRS) defined."
      )
    if not gdf.crs.is_geographic:
      gdf = gdf.to_crs(epsg=4326)

    target_id_col = id_column if id_column is not None else "gauge_id"
    if target_id_col not in gdf.columns:
      raise ValueError(
          f"ID column {target_id_col!r} not found in {in_path} (available columns: {list(gdf.columns)}). "
          "Pass id_column explicitly."
      )

    if gdf[target_id_col].isna().any():
      raise ValueError(
          f"ID column {target_id_col!r} in {in_path} contains null values."
      )
    id_series = gdf[target_id_col].astype(str)
    if id_series.duplicated().any():
      dup_ids = id_series[id_series.duplicated()].unique().tolist()
      raise ValueError(
          f"ID column {target_id_col!r} in {in_path} contains duplicate IDs: {dup_ids[:5]}"
      )

    tasks = []
    for _, row in gdf.iterrows():
      gid = str(row[target_id_col])
      tasks.append((row.geometry, gid))

    ds_label = dataset_name or in_path.stem.replace("_basin_shapes", "").replace("_basins", "")

    if workers > 1 and len(tasks) > 1:
      import concurrent.futures
      import multiprocessing as mp

      worker_args = [
          (
              geom,
              gid,
              min_overlap_threshold,
              actual_era5_source,
              str(self.gdb_path),
              str(self.era5_cache_dir) if self.era5_cache_dir else None,
              self.gridded_era5_uri,
              skip_climate,
          )
          for geom, gid in tasks
      ]
      logger.debug(
          "Processing %d catchments in parallel with %d workers...",
          len(tasks),
          workers,
      )
      ctx = mp.get_context("spawn")
      results = [None] * len(tasks)
      with concurrent.futures.ProcessPoolExecutor(
          max_workers=workers, mp_context=ctx
      ) as executor:
        future_to_idx = {
            executor.submit(_worker_extract_polygon, arg): i
            for i, arg in enumerate(worker_args)
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(tasks),
            desc=f"  ↳ {ds_label}",
            unit="basin",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        ):
          idx = future_to_idx[future]
          results[idx] = future.result()
    else:
      results = []
      for geom, gid in tqdm(
          tasks,
          desc=f"  ↳ {ds_label}",
          unit="basin",
          leave=False,
          dynamic_ncols=True,
          disable=not show_progress,
      ):
        res = self.extract_attributes_for_polygon(
            geom,
            catchment_id=gid,
            min_overlap_threshold=min_overlap_threshold,
            era5_source=actual_era5_source,
            _batch_mode=True,
            _skip_climate=skip_climate,
        )
        results.append(res)

    if skip_climate:
      self._populate_gridded_climate_batch(results, tasks)

    df = self.export_caravan_csv(
        results, output_csv_path=output_csv_path if output_csv_path else None
    )
    return df

  def export_caravan_csv(
      self,
      results: Union[List[Dict[str, Any]], pd.DataFrame],
      output_csv_path: Optional[Union[str, Path]] = None,
  ) -> pd.DataFrame:
    """Formats and exports Caravan attributes to standard CSV."""
    if isinstance(results, pd.DataFrame):
      df = results
    else:
      rows = []
      gauge_ids = []
      for r in results:
        gid = r["catchment_id"]
        gauge_ids.append(gid)
        rows.append(r["caravan_attributes"])
      df = pd.DataFrame(rows, index=gauge_ids)
      df.index.name = "gauge_id"

    # Sort columns alphabetically, ensuring basin_area is first
    sorted_cols = sorted(df.columns)
    if "basin_area" in sorted_cols:
      sorted_cols.remove("basin_area")
      sorted_cols = ["basin_area"] + sorted_cols
    df = df[sorted_cols].sort_index(axis=0)

    if output_csv_path:
      p = Path(output_csv_path)
      p.parent.mkdir(parents=True, exist_ok=True)
      df.to_csv(p)
      logger.debug("Saved Caravan static attributes to %s (shape: %s)", p, df.shape)

    return df

  def append_attributes_to_zarr(
      self,
      master_zarr_path: Union[str, Path],
      basin_id: str,
      attributes_dict: Optional[Dict[str, Any]] = None,
      *,
      attributes: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Appends static Caravan & HydroATLAS attributes to a master Zarr store along the 'basin' dimension."""
    master_path = Path(master_zarr_path)
    if not master_path.exists():
      raise FileNotFoundError(f"Zarr store does not exist at {master_path}.")

    payload = attributes if attributes is not None else attributes_dict
    if not payload:
      raise ValueError("No attributes dictionary provided to append_attributes_to_zarr.")

    caravan_attrs = (
        payload["caravan_attributes"]
        if "caravan_attributes" in payload
        else payload
    )
    if not caravan_attrs:
      raise ValueError("Provided attributes dictionary is empty.")

    ds = xr.open_zarr(str(master_path)).load()
    if "basin" not in ds.dims:
      raise KeyError(
          f"Dimension 'basin' not found in Zarr store {master_path} (dims: {list(ds.dims)})."
      )

    basin_list = [str(b) for b in ds["basin"].values]
    if str(basin_id) not in basin_list:
      raise KeyError(
          f"Basin ID {basin_id!r} not found in 'basin' coordinate of {master_path}."
      )

    basin_idx = basin_list.index(str(basin_id))

    for key, val in caravan_attrs.items():
      if isinstance(val, (int, float, np.integer, np.floating)):
        var_name = f"caravan_{key}"
        if var_name not in ds:
          arr = np.full((len(basin_list),), np.nan, dtype=np.float32)
          arr[basin_idx] = float(val)
          ds[var_name] = (["basin"], arr)
        else:
          ds[var_name].values[basin_idx] = float(val)

    ds.to_zarr(str(master_path), mode="w", consolidated=True)
    logger.info(
        "Appended Caravan static attributes for %s to %s", basin_id, master_path
    )
