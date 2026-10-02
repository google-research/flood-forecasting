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

"""Unit and integration tests for Caravan Static Attributes Extractor."""

import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import zarr

from multimet.static_extractor import (
    ATTRIBUTE_DEFINITIONS,
    MAJORITY_PROPERTIES,
    StaticAttributesExtractor,
    calculate_fao_pm_pet,
    calculate_knoben_moisture_and_seasonality,
    compute_caravan_climate_metrics,
    compute_pour_point_properties,
)
from multimet.static_extractor.cli import main as cli_main, parse_args
from multimet.static_extractor.extractor import _worker_extract_polygon


def test_schema_definitions():
  """Verifies core schema definitions and attribute categories."""
  assert len(MAJORITY_PROPERTIES) == 10
  assert "clz_cl_smj" in MAJORITY_PROPERTIES
  assert "glc_cl_smj" in MAJORITY_PROPERTIES
  assert "lit_cl_smj" in MAJORITY_PROPERTIES

  assert "basin_area" in ATTRIBUTE_DEFINITIONS
  assert "ele_mt_sav" in ATTRIBUTE_DEFINITIONS
  assert "p_mean" in ATTRIBUTE_DEFINITIONS
  assert ATTRIBUTE_DEFINITIONS["ele_mt_sav"]["category"] == "Topography"
  assert ATTRIBUTE_DEFINITIONS["p_mean"]["category"] == "Climate"


def test_fao_pm_pet_calculation():
  """Tests FAO-56 Penman-Monteith daily reference evapotranspiration."""
  dates = pd.date_range("2020-01-01", periods=5, freq="D")
  sp = pd.Series([101.3] * 5, index=dates)
  t2m = pd.Series([20.0, 25.0, 15.0, 30.0, 10.0], index=dates)
  d2m = pd.Series([15.0, 18.0, 10.0, 20.0, 5.0], index=dates)
  u10 = pd.Series([2.0] * 5, index=dates)
  v10 = pd.Series([1.5] * 5, index=dates)
  ssr = pd.Series([800000.0] * 5, index=dates)
  str_s = pd.Series([200000.0] * 5, index=dates)

  pet = calculate_fao_pm_pet(
      surface_pressure_kpa=sp,
      temperature_2m_c=t2m,
      dewpoint_temperature_2m_c=d2m,
      u_component_of_wind_10m=u10,
      v_component_of_wind_10m=v10,
      surface_net_solar_radiation_mean=ssr,
      surface_net_thermal_radiation_mean=str_s,
  )

  assert len(pet) == 5
  assert (pet >= 0.0).all()
  assert 2.0 <= pet.mean() <= 8.0


def test_knoben_moisture_and_seasonality():
  """Tests Knoben et al. (2018) annual moisture and seasonality indices."""
  dates = pd.date_range("2020-01-01", periods=365, freq="D")
  day_of_year = dates.dayofyear
  p_vals = 3.0 + 2.0 * np.cos(2 * np.pi * day_of_year / 365.0)
  pet_vals = 3.0 - 2.0 * np.cos(2 * np.pi * day_of_year / 365.0)

  p = pd.Series(p_vals, index=dates)
  pet = pd.Series(pet_vals, index=dates)

  mi, seasonality = calculate_knoben_moisture_and_seasonality(p, pet)
  assert -1.0 <= mi <= 1.0
  assert seasonality >= 0.0


def test_caravan_climate_metrics_extremes():
  """Tests extreme precipitation indices (high/low prec freq and duration)."""
  dates = pd.date_range("2020-01-01", periods=100, freq="D")
  p_vals = np.ones(100) * 2.0
  p_vals[10:13] = 12.0
  p_vals[20:25] = 0.5

  p = pd.Series(p_vals, index=dates)
  t = pd.Series(np.ones(100) * 15.0, index=dates)
  pet = pd.Series(np.ones(100) * 3.0, index=dates)

  metrics = compute_caravan_climate_metrics(p, t, pet)

  assert metrics["p_mean"] > 0
  assert metrics["high_prec_freq"] > 0
  assert metrics["high_prec_dur"] == 3.0
  assert metrics["low_prec_freq"] > 0
  assert metrics["low_prec_dur"] == 5.0
  assert metrics["frac_snow"] == 0.0


def test_pour_point_properties():
  """Tests downstream topological outlet tracing via NEXT_DOWN."""
  basin_data = {
      "HYBAS_ID": [712000001, 712000002, 712000003],
      "NEXT_DOWN": [712000002, 712000003, 0],
      "SUB_AREA": [100.0, 100.0, 100.0],
      "weights": [80.0, 90.0, 95.0],
      "dis_m3_pyr": [10.0, 20.0, 50.0],
  }
  res = compute_pour_point_properties(
      basin_data, pour_point_properties=["dis_m3_pyr"]
  )
  assert res["dis_m3_pyr"] == 50.0

  empty_res = compute_pour_point_properties(
      {"weights": [], "SUB_AREA": []}, pour_point_properties=["dis_m3_pyr"]
  )
  assert np.isnan(empty_res["dis_m3_pyr"])


def _build_synthetic_hydroatlas_env(tmp_path: Path, include_native_pet: bool = True):
  """Creates a self-contained synthetic BasinATLAS shapefile, ERA5 table, and Zarr store."""
  poly1 = shapely.geometry.box(-87.0, 40.0, -86.0, 41.0)
  poly2 = shapely.geometry.box(-86.0, 40.0, -85.0, 41.0)

  gdf = gpd.GeoDataFrame(
      {
          "HYBAS_ID": [712000001, 712000002],
          "NEXT_DOWN": [712000002, 0],
          "SUB_AREA": [5000.0, 5000.0],
          "UP_AREA": [5000.0, 10000.0],
          "ele_mt_sav": [200.0, 400.0],
          "slp_dg_sav": [10.0, 30.0],
          "pre_mm_syr": [800.0, 1200.0],
          "tmp_dc_syr": [100.0, 200.0],
          "ari_ix_sav": [50.0, 150.0],
          "cly_pc_sav": [20.0, 40.0],
          "snd_pc_sav": [50.0, 30.0],
          "slt_pc_sav": [30.0, 30.0],
          "for_pc_sse": [60.0, 20.0],
          "crp_pc_sse": [10.0, 50.0],
          "urb_pc_sse": [5.0, 15.0],
          "gwt_cm_sav": [100.0, 300.0],
          "swc_pc_syr": [40.0, 60.0],
          "inu_pc_smx": [2.0, 8.0],
          "glc_cl_smj": [4, 12],
          "clz_cl_smj": [7, 9],
          "lit_cl_smj": [1, 3],
          "dis_m3_pyr": [15.0, 45.0],
          "run_mm_syr": [300.0, 500.0],
      },
      geometry=[poly1, poly2],
      crs="EPSG:4326",
  )
  shp_path = tmp_path / "synthetic_hydroatlas.shp"
  gdf.to_file(shp_path)

  era5_cache = tmp_path / "era5_cache"
  era5_cache.mkdir(parents=True, exist_ok=True)
  rec1 = {
      "gauge_id": "hybas_712000001",
      "p_mean": 3.0,
      "pet_mean": 1.5,
      "aridity": 0.5,
      "frac_snow": 0.1,
      "moisture_index": 0.4,
      "seasonality": 0.2,
      "high_prec_freq": 0.05,
      "high_prec_dur": 1.2,
      "low_prec_freq": 0.6,
      "low_prec_dur": 3.5,
  }
  rec2 = {
      "gauge_id": "hybas_712000002",
      "p_mean": 5.0,
      "pet_mean": 2.5,
      "aridity": 0.5,
      "frac_snow": 0.3,
      "moisture_index": 0.6,
      "seasonality": 0.4,
      "high_prec_freq": 0.09,
      "high_prec_dur": 1.6,
      "low_prec_freq": 0.4,
      "low_prec_dur": 2.5,
  }
  (era5_cache / "na_climate_indices.txt").write_text(
      json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n", encoding="utf-8"
  )

  zarr_dir = tmp_path / "synthetic_era5.zarr"
  root = zarr.open_group(str(zarr_dir), mode="w")
  n_times = 60
  lats = np.linspace(39.8, 41.2, 15, dtype=np.float32)
  lons = np.linspace(-87.2, -84.8, 25, dtype=np.float32)
  root.create_array("latitude", data=lats)
  root.create_array("longitude", data=lons)
  time_arr = root.create_array("time", data=np.arange(n_times, dtype=np.int64))
  time_arr.attrs["units"] = "days since 2000-01-01"

  shape = (n_times, len(lats), len(lons))
  root.create_array("era5land_total_precipitation", data=np.full(shape, 4.0, dtype=np.float32))
  root.create_array("era5land_temperature_2m", data=np.full(shape, 15.0, dtype=np.float32))
  root.create_array(
      "era5land_potential_evaporation_FAO_PENMAN_MONTEITH",
      data=np.full(shape, 2.0, dtype=np.float32),
  )
  if include_native_pet:
    root.create_array(
        "potential_evaporation",
        data=np.full(shape, 4.0, dtype=np.float32),
    )

  return shp_path, era5_cache, zarr_dir


def test_extract_attributes_for_polygon_end_to_end(tmp_path):
  """Tests extract_attributes_for_polygon across continuous, categorical, pour-point, and climate modes."""
  shp_path, era5_cache, zarr_dir = _build_synthetic_hydroatlas_env(tmp_path, include_native_pet=True)

  extractor = StaticAttributesExtractor(
      gdb_path=shp_path,
      era5_cache_dir=era5_cache,
      gridded_era5_uri=str(zarr_dir),
  )

  query_poly = shapely.geometry.box(-86.25, 40.1, -85.25, 40.9)
  feature = {
      "type": "Feature",
      "properties": {"gauge_id": "test_basin_01"},
      "geometry": shapely.geometry.mapping(query_poly),
  }

  res = extractor.extract_attributes_for_polygon(feature, era5_source="hybas")
  assert res["catchment_id"] == "test_basin_01"
  assert res["intersected_subbasins_count"] == 2

  attrs = res["caravan_attributes"]
  assert np.isclose(attrs["ele_mt_sav"], 350.0, rtol=1e-3)
  assert np.isclose(attrs["slp_dg_sav"], 25.0, rtol=1e-3)
  assert np.isclose(attrs["area_fraction_used_for_aggregation"], 1.0)

  assert attrs["glc_cl_smj"] == 12
  assert attrs["clz_cl_smj"] == 9
  assert attrs["lit_cl_smj"] == 3

  assert np.isclose(attrs["dis_m3_pyr"], 45.0)

  assert np.isclose(attrs["p_mean"], 4.5, rtol=1e-3)
  assert np.isclose(attrs["pet_mean"], 2.25, rtol=1e-3)
  assert np.isclose(attrs["pet_mean_FAO_PM"], 2.25, rtol=1e-3)
  assert np.isclose(attrs["pet_mean_ERA5_LAND"], 4.0, rtol=1e-3)
  assert np.isclose(attrs["aridity_ERA5_LAND"], 1.0, rtol=1e-3)

  # Gridded mode: both FAO-PM (2.0) and ERA5-Land (4.0) come from the Zarr store
  res_gridded = extractor.extract_attributes_for_polygon(query_poly, catchment_id="b_grid", era5_source="gridded")
  g_attrs = res_gridded["caravan_attributes"]
  assert np.isclose(g_attrs["p_mean"], 4.0)
  assert np.isclose(g_attrs["pet_mean"], 2.0)
  assert np.isclose(g_attrs["pet_mean_FAO_PM"], 2.0)
  assert np.isclose(g_attrs["pet_mean_ERA5_LAND"], 4.0)
  assert np.isclose(g_attrs["aridity_FAO_PM"], 0.5)
  assert np.isclose(g_attrs["aridity_ERA5_LAND"], 1.0)

  # Direct timeseries_df mode with only FAO-PM PET column provided -> ERA5-Land stays NaN
  dates = pd.date_range("2000-01-01", periods=40, freq="D")
  ts_df = pd.DataFrame(
      {
          "total_precipitation": np.full(40, 5.0),
          "temperature": np.full(40, 12.0),
          "pet_fao": np.full(40, 2.5),
      },
      index=dates,
  )
  res_ts = extractor.extract_attributes_for_polygon(query_poly, catchment_id="b_ts", timeseries_df=ts_df)
  ts_attrs = res_ts["caravan_attributes"]
  assert np.isclose(ts_attrs["pet_mean"], 2.5)
  assert np.isclose(ts_attrs["pet_mean_FAO_PM"], 2.5)
  assert np.isnan(ts_attrs["pet_mean_ERA5_LAND"])
  assert np.isnan(ts_attrs["aridity_ERA5_LAND"])

  # Non-intersecting polygon returns NaN (never substitutes nearest sub-basin)
  non_inter_poly = shapely.geometry.box(-89.0, 40.1, -88.5, 40.5)
  res_no_inter = extractor.extract_attributes_for_polygon(non_inter_poly, catchment_id="offshore", era5_source="hybas")
  assert res_no_inter["intersected_subbasins_count"] == 0
  assert res_no_inter["caravan_attributes"]["area_fraction_used_for_aggregation"] == 0.0
  assert np.isnan(res_no_inter["caravan_attributes"]["ele_mt_sav"])
  assert np.isnan(res_no_inter["caravan_attributes"]["dis_m3_pyr"])

  # High min_overlap_threshold filtering all slivers returns NaN
  res_filtered = extractor.extract_attributes_for_polygon(
      query_poly, catchment_id="filtered", min_overlap_threshold=1e6, era5_source="hybas"
  )
  assert res_filtered["caravan_attributes"]["area_fraction_used_for_aggregation"] == 0.0
  assert np.isnan(res_filtered["caravan_attributes"]["ele_mt_sav"])

  # Omitting era5_source when timeseries_df is not provided must raise ValueError
  with pytest.raises(ValueError, match="era5_source must be explicitly specified"):
    extractor.extract_attributes_for_polygon(query_poly, catchment_id="b_err")

  # Omitting catchment_id when input is a bare geometry must raise ValueError
  with pytest.raises(ValueError, match="catchment_id must be provided"):
    extractor.extract_attributes_for_polygon(query_poly, era5_source="hybas")


def test_extract_attributes_batch_and_file_io(tmp_path):
  """Tests extract_attributes_batch and extract_attributes_from_file with GeoJSON and Parquet inputs."""
  shp_path, era5_cache, zarr_dir = _build_synthetic_hydroatlas_env(tmp_path, include_native_pet=False)
  extractor = StaticAttributesExtractor(
      gdb_path=shp_path,
      era5_cache_dir=era5_cache,
      gridded_era5_uri=str(zarr_dir),
      era5_source="hybas",
  )

  b1 = shapely.geometry.box(-86.8, 40.2, -86.2, 40.8)
  b2 = shapely.geometry.box(-85.8, 40.2, -85.2, 40.8)
  batch_res = extractor.extract_attributes_batch(
      [
          {"type": "Feature", "properties": {"gauge_id": "g1"}, "geometry": shapely.geometry.mapping(b1)},
          {"type": "Feature", "properties": {"gauge_id": "g2"}, "geometry": shapely.geometry.mapping(b2)},
      ]
  )
  assert len(batch_res) == 2
  assert np.isclose(batch_res[0]["caravan_attributes"]["ele_mt_sav"], 200.0)
  assert np.isclose(batch_res[1]["caravan_attributes"]["ele_mt_sav"], 400.0)
  assert np.isnan(batch_res[0]["caravan_attributes"]["pet_mean_ERA5_LAND"])

  basins_gdf = gpd.GeoDataFrame({"gauge_id": ["g1", "g2"]}, geometry=[b1, b2], crs="EPSG:4326")
  geojson_path = tmp_path / "input_basins.geojson"
  out_csv = tmp_path / "extracted_attrs.csv"
  basins_gdf.to_file(geojson_path, driver="GeoJSON")

  df = extractor.extract_attributes_from_file(geojson_path, output_csv_path=out_csv, workers=1, show_progress=False)
  assert list(df.index) == ["g1", "g2"]
  assert out_csv.exists()
  assert np.isclose(df.loc["g1", "ele_mt_sav"], 200.0)
  assert np.isclose(df.loc["g2", "ele_mt_sav"], 400.0)
  assert np.isclose(df.loc["g1", "pet_mean_FAO_PM"], 1.5)
  assert np.isnan(df.loc["g1", "pet_mean_ERA5_LAND"])

  # Test single-pass gridded batch extraction via extract_attributes_from_file(era5_source="gridded")
  df_gridded = extractor.extract_attributes_from_file(
      geojson_path, era5_source="gridded", workers=1, show_progress=False
  )
  assert list(df_gridded.index) == ["g1", "g2"]
  assert np.isclose(df_gridded.loc["g1", "p_mean"], 4.0)
  assert np.isclose(df_gridded.loc["g1", "pet_mean_FAO_PM"], 2.0)
  assert np.isnan(df_gridded.loc["g1", "pet_mean_ERA5_LAND"])

  # GeoDataFrame and raw geometry dict inputs
  res_gdf = extractor.extract_attributes_for_polygon(basins_gdf.iloc[[0]])
  assert res_gdf["catchment_id"] == "g1"
  res_geom_dict = extractor.extract_attributes_for_polygon(
      shapely.geometry.mapping(b1), catchment_id="explicit_id"
  )
  assert res_geom_dict["catchment_id"] == "explicit_id"

  # Worker function and export_caravan_csv with DataFrame
  w_res = _worker_extract_polygon((b1, "w1", 0.0, "hybas", str(shp_path), str(era5_cache), str(zarr_dir), False))
  assert w_res["catchment_id"] == "w1"
  assert extractor.export_caravan_csv(df).shape == df.shape

  # Missing/unreachable Zarr store or GDB path must raise an error instead of returning silent NaNs
  with pytest.raises(FileNotFoundError):
    StaticAttributesExtractor(gdb_path=tmp_path / "nonexistent.shp")

  ext_missing_zarr = StaticAttributesExtractor(
      gdb_path=shp_path,
      era5_cache_dir=era5_cache,
      gridded_era5_uri=str(tmp_path / "does_not_exist.zarr"),
  )
  with pytest.raises(Exception):
    ext_missing_zarr.extract_attributes_for_polygon(b1, catchment_id="b1", era5_source="gridded")

  # Missing ID column in input file must raise ValueError instead of inventing basin_1, basin_2
  with pytest.raises(ValueError, match="ID column 'nonexistent_col' not found"):
    extractor.extract_attributes_from_file(geojson_path, id_column="nonexistent_col", show_progress=False)

  # CLI main end-to-end execution
  cli_out_csv = tmp_path / "cli_out.csv"
  cli_main([
      "--input", str(geojson_path),
      "--output", str(cli_out_csv),
      "--era5-source", "hybas",
      "--gdb-path", str(shp_path),
      "--era5-cache-dir", str(era5_cache),
      "--gridded-era5-uri", str(zarr_dir),
  ])
  assert cli_out_csv.exists()


def test_cli_parsing():
  """Tests CLI argument parsing and verifies required path flags."""
  # Missing --gdb-path or --era5-source must fail
  with pytest.raises(SystemExit):
    parse_args(["--input", "basins.geojson", "--output", "attrs.csv"])

  # Missing --era5-cache-dir when --era5-source=hybas must fail
  with pytest.raises(SystemExit):
    parse_args([
        "--input", "basins.geojson",
        "--output", "attrs.csv",
        "--gdb-path", "/data/BasinATLAS_v10.gdb",
        "--era5-source", "hybas",
    ])

  # Missing --gridded-era5-uri when --era5-source=gridded must fail
  with pytest.raises(SystemExit):
    parse_args([
        "--input", "basins.geojson",
        "--output", "attrs.csv",
        "--gdb-path", "/data/BasinATLAS_v10.gdb",
        "--era5-source", "gridded",
    ])

  args = parse_args([
      "--input", "basins.geojson",
      "--output", "attrs.csv",
      "--gdb-path", "/data/BasinATLAS_v10.gdb",
      "--era5-source", "hybas",
      "--era5-cache-dir", "/data/era5_climate",
  ])
  assert args.input == "basins.geojson"
  assert args.output == "attrs.csv"
  assert args.gdb_path == "/data/BasinATLAS_v10.gdb"
  assert args.era5_cache_dir == "/data/era5_climate"
  assert args.min_overlap_threshold == 0.0
  assert args.era5_source == "hybas"

  args_gridded = parse_args([
      "--input", "basins.geojson",
      "--output", "attrs.csv",
      "--gdb-path", "/data/BasinATLAS_v10.gdb",
      "--era5-source", "gridded",
      "--gridded-era5-uri", "gs://my-bucket/era5.zarr",
      "--workers", "16",
  ])
  assert args_gridded.era5_source == "gridded"
  assert args_gridded.gridded_era5_uri == "gs://my-bucket/era5.zarr"
  assert args_gridded.workers == 16


def test_era5_gridded_extractor_synthetic(tmp_path):
  """Tests ERA5GriddedExtractor with a synthetic local Zarr dataset."""
  from multimet.static_extractor.climate import ERA5GriddedExtractor

  zarr_dir = tmp_path / "synthetic_era5.zarr"
  root = zarr.open_group(str(zarr_dir), mode="w")

  n_times = 50
  lats = np.linspace(40.0, 41.0, 11, dtype=np.float32)
  lons = np.linspace(-87.0, -86.0, 11, dtype=np.float32)

  root.create_array("latitude", data=lats)
  root.create_array("longitude", data=lons)
  time_arr = root.create_array("time", data=np.arange(n_times, dtype=np.int64))
  time_arr.attrs["units"] = "days since 2000-01-01"

  p_data = np.full((n_times, len(lats), len(lons)), 4.0, dtype=np.float32)
  t_data = np.full((n_times, len(lats), len(lons)), 18.0, dtype=np.float32)
  pet_data = np.full((n_times, len(lats), len(lons)), 2.0, dtype=np.float32)

  p_data[:, :4, :4] = np.nan
  t_data[:, :4, :4] = np.nan
  pet_data[:, :4, :4] = np.nan

  root.create_array("era5land_total_precipitation", data=p_data)
  root.create_array("era5land_temperature_2m", data=t_data)
  root.create_array("era5land_potential_evaporation_FAO_PENMAN_MONTEITH", data=pet_data)

  extractor = ERA5GriddedExtractor(zarr_uri=str(zarr_dir))

  poly = shapely.geometry.box(-86.8, 40.2, -86.2, 40.8)
  metrics = extractor.extract_climate_metrics_for_polygon(poly, baseline_years=None)

  assert np.isclose(metrics["p_mean"], 4.0)
  assert np.isclose(metrics["pet_mean"], 2.0)
  assert np.isclose(metrics["pet_mean_FAO_PM"], 2.0)
  assert np.isclose(metrics["aridity"], 0.5)
  assert np.isclose(metrics["aridity_FAO_PM"], 0.5)
  assert np.isnan(metrics["pet_mean_ERA5_LAND"])
  assert np.isnan(metrics["aridity_ERA5_LAND"])
  assert np.isnan(metrics["moisture_index_ERA5_LAND"])
  assert np.isnan(metrics["seasonality_ERA5_LAND"])
  assert metrics["frac_snow"] == 0.0

  offshore_poly = shapely.geometry.box(-120.0, 10.0, -119.5, 10.5)
  offshore_metrics = extractor.extract_climate_metrics_for_polygon(offshore_poly, baseline_years=None)
  assert np.isnan(offshore_metrics["p_mean"])
  assert np.isnan(offshore_metrics["pet_mean_FAO_PM"])

  zarr_units_dir = tmp_path / "synthetic_era5_units.zarr"
  root_u = zarr.open_group(str(zarr_units_dir), mode="w")
  root_u.create_array("latitude", data=lats)
  root_u.create_array("longitude", data=lons)
  t_arr = root_u.create_array("time", data=np.arange(n_times, dtype=np.int64) * 24)
  t_arr.attrs["units"] = "hours since 2000-01-01 00:00:00"
  p_arr = root_u.create_array("era5land_total_precipitation", data=np.full((n_times, len(lats), len(lons)), 0.004, dtype=np.float32))
  p_arr.attrs["units"] = "m"
  temp_arr = root_u.create_array("era5land_temperature_2m", data=np.full((n_times, len(lats), len(lons)), 288.15, dtype=np.float32))
  temp_arr.attrs["units"] = "K"
  pet_arr = root_u.create_array("era5land_potential_evaporation_FAO_PENMAN_MONTEITH", data=np.full((n_times, len(lats), len(lons)), -0.002, dtype=np.float32))
  pet_arr.attrs["units"] = "m"

  ext_u = ERA5GriddedExtractor(zarr_uri=str(zarr_units_dir))
  m_u = ext_u.extract_climate_metrics_for_polygon(poly, baseline_years=None)
  assert np.isclose(m_u["p_mean"], 4.0)
  assert np.isclose(m_u["pet_mean_FAO_PM"], 2.0)
  assert m_u["frac_snow"] == 0.0


def test_batch_runner_discovery(tmp_path):
  """Tests discover_datasets in batch_runner across multiple dataset folders."""
  from multimet.static_extractor.batch_runner import discover_datasets

  parent = tmp_path / "caravan_root"
  ds1 = parent / "camels"
  ds2 = parent / "hysets"
  ds1.mkdir(parents=True)
  ds2.mkdir(parents=True)

  (ds1 / "camels_basin_shapes.shp").touch()
  (ds2 / "hysets.geojson").touch()

  datasets = discover_datasets(parent_dirs=[str(parent)])
  assert "camels" in datasets
  assert "hysets" in datasets
  assert datasets["camels"].name == "camels_basin_shapes.shp"
  assert datasets["hysets"].name == "hysets.geojson"

  nested_parent = tmp_path / "staged" / "caravan" / "caravan"
  ds_nested = nested_parent / "lamah"
  ds_nested.mkdir(parents=True)
  (ds_nested / "lamah_basin_shapes.shp").touch()

  datasets_nested = discover_datasets(parent_dirs=[str(tmp_path / "staged" / "caravan")])
  assert "lamah" in datasets_nested
  assert datasets_nested["lamah"].name == "lamah_basin_shapes.shp"

  # Nonexistent input file must raise FileNotFoundError
  with pytest.raises(FileNotFoundError):
    discover_datasets(input_files=[str(tmp_path / "missing.shp")])


def test_benchmark_metrics_continuous():
  """Tests continuous statistical validation metrics calculation."""
  from multimet.static_extractor.benchmark import compute_continuous_metrics

  y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
  y_pred = np.array([10.1, 19.9, 30.2, 39.8, 50.1])

  res = compute_continuous_metrics(y_true, y_pred)
  assert res["n"] == 5
  assert res["pearson_r"] > 0.999
  assert res["spearman_rho"] == 1.0
  assert res["r2"] > 0.999
  assert res["mae"] < 0.2
  assert res["rmse"] < 0.2
  assert res["max_abs_error"] == 0.2
  assert res["med_rel_error_pct"] < 1.0
  assert res["max_rel_error_pct"] < 1.5


def test_benchmark_metrics_categorical():
  """Tests categorical classification accuracy calculation."""
  from multimet.static_extractor.benchmark import compute_categorical_metrics

  y_true = np.array([1, 2, 3, 4, 5, 2, 1, 3])
  y_pred = np.array([1, 2, 3, 4, 5, 2, 1, 4])

  res = compute_categorical_metrics(y_true, y_pred)
  assert res["n"] == 8
  assert res["accuracy_pct"] == 87.5
  assert res["classes_count"] == 5


def test_benchmark_attribute_categorization():
  """Tests categorization of all standard attribute names."""
  from multimet.static_extractor.benchmark import get_attribute_category

  assert get_attribute_category("ele_mt_sav") == "Topography"
  assert get_attribute_category("tmp_dc_syr") == "Climate (HydroATLAS)"
  assert get_attribute_category("p_mean") == "Caravan ERA5 Climate"
  assert get_attribute_category("aridity_ERA5_LAND") == "Caravan ERA5 Climate"
  assert get_attribute_category("run_mm_syr") == "Hydrology"
  assert get_attribute_category("cly_pc_sav") == "Soils & Geology"
  assert get_attribute_category("for_pc_sse") == "Land Cover"
  assert get_attribute_category("ppd_pk_sav") == "Anthropogenic"
  assert get_attribute_category("glc_cl_smj") == "Land Cover"
  assert get_attribute_category("wet_cl_smj") == "Hydrology"


def test_batch_runner_gcs_output_and_args(tmp_path, monkeypatch):
  """Verifies GCS output handling and multi parent-dir argument parsing."""
  from unittest.mock import MagicMock
  from multimet.static_extractor.batch_runner import parse_args, run_batch_extraction

  args = parse_args([
      "-p", "gs://open-multimet/data/caravan_shapefiles/caravan/",
      "-p", "gs://open-multimet/data/caravan_shapefiles/caravan_extensions/",
      "-o", "gs://open-multimet/data/caravan_static_attributes/",
      "--gdb-path", "/data/BasinATLAS_v10.gdb",
      "--era5-source", "hybas",
      "--era5-cache-dir", "/data/era5_climate",
      "--staging-dir", "/data/staged",
      "--workers", "14",
      "--combine",
  ])
  assert len(args.parent_dirs) == 2
  assert args.output_dir == "gs://open-multimet/data/caravan_static_attributes/"
  assert args.workers == 14
  assert args.combine is True

  args_multi = parse_args([
      "-p", "dir1", "dir2", "dir3",
      "-o", "/tmp/out",
      "--gdb-path", "/data/BasinATLAS_v10.gdb",
      "--era5-source", "gridded",
      "--gridded-era5-uri", "gs://my-bucket/era5.zarr",
  ])
  assert len(args_multi.parent_dirs) == 1
  assert len(args_multi.parent_dirs[0]) == 3

  mock_upload = MagicMock()
  monkeypatch.setattr("multimet.static_extractor.batch_runner.upload_to_gcs", mock_upload)
  monkeypatch.setattr("multimet.static_extractor.batch_runner.gcs_path_exists", lambda uri: False)

  dummy_df = pd.DataFrame({"basin_id": ["b1"], "ele_mt_sav": [100.0]})
  dummy_shp = tmp_path / "test.shp"
  dummy_shp.write_text("dummy")

  mock_extractor = MagicMock()
  mock_extractor.extract_attributes_from_file.return_value = dummy_df
  monkeypatch.setattr(
      "multimet.static_extractor.batch_runner.StaticAttributesExtractor",
      lambda **kwargs: mock_extractor,
  )

  results = run_batch_extraction(
      dataset_map={"test_ds": dummy_shp},
      output_dir="gs://open-multimet/data/caravan_static_attributes/",
      gdb_path=tmp_path / "BasinATLAS_v10.gdb",
      era5_source="hybas",
      era5_cache_dir=tmp_path / "era5",
      workers=1,
      staging_cache_dir=tmp_path / "staged",
      combine=True,
      resume=False,
  )

  assert "test_ds" in results
  assert mock_upload.call_count == 2


def test_batch_runner_progress_and_quiet_logging(tmp_path):
  """Verifies that --verbose and --no-progress flags are properly parsed and setup_logging configures root logger."""
  import logging
  from multimet.static_extractor.batch_runner import parse_args, setup_logging

  args = parse_args([
      "-o", str(tmp_path / "out"),
      "--gdb-path", "/data/BasinATLAS_v10.gdb",
      "--era5-source", "hybas",
      "--era5-cache-dir", "/data/era5_climate",
  ])
  assert args.verbose is False
  assert args.show_progress is True

  args_v = parse_args([
      "-o", str(tmp_path / "out"),
      "--gdb-path", "/data/BasinATLAS_v10.gdb",
      "--era5-source", "hybas",
      "--era5-cache-dir", "/data/era5_climate",
      "-v",
      "--no-progress",
  ])
  assert args_v.verbose is True
  assert args_v.show_progress is False

  setup_logging(verbose=False)
  assert logging.getLogger("static_extractor").level == logging.WARNING

  setup_logging(verbose=True)
  assert logging.getLogger().level == logging.DEBUG


def test_export_subdataset_partitioned_files(tmp_path):
  """Tests partitioning of extracted attributes into HydroATLAS, Caravan, and Parquet tables."""
  from multimet.static_extractor.batch_runner import export_subdataset_partitioned_files

  df = pd.DataFrame(
      {
          "basin_area": [100.0, 200.0],
          "ele_mt_sav": [500.0, 600.0],
          "p_mean": [3.5, 4.0],
          "pet_mean": [2.5, 2.8],
          "aridity": [0.7, 0.8],
      },
      index=["camels_01", "camels_02"],
  )
  df.index.name = "gauge_id"

  out_sub = tmp_path / "out" / "camels"
  res = export_subdataset_partitioned_files(
      df=df,
      ds_name="camels",
      output_sub_dir=out_sub,
  )

  assert res["hydroatlas"].exists()
  assert res["caravan"].exists()
  assert res["parquet"].exists()

  df_hydro = pd.read_csv(res["hydroatlas"], index_col=0)
  assert "basin_area" in df_hydro.columns
  assert "ele_mt_sav" in df_hydro.columns
  assert "p_mean" not in df_hydro.columns

  df_caravan = pd.read_csv(res["caravan"], index_col=0)
  assert "p_mean" in df_caravan.columns
  assert "pet_mean" in df_caravan.columns
  assert "aridity" in df_caravan.columns

  df_parquet = pd.read_parquet(res["parquet"])
  assert len(df_parquet) == 2
  assert "basin_area" in df_parquet.columns
  assert "p_mean" in df_parquet.columns


def test_batch_runner_partition_outputs(tmp_path, monkeypatch):
  """Verifies that partition_outputs=True partitions output per dataset and supports resume."""
  from unittest.mock import MagicMock
  from multimet.static_extractor.batch_runner import run_batch_extraction

  uploaded_uris = []
  def mock_upload(local_file, gcs_dest):
    uploaded_uris.append((Path(local_file).name, gcs_dest))

  monkeypatch.setattr("multimet.static_extractor.batch_runner.upload_to_gcs", mock_upload)
  monkeypatch.setattr("multimet.static_extractor.batch_runner.gcs_path_exists", lambda uri: False)

  ds_dir = tmp_path / "camels"
  ds_dir.mkdir()
  dummy_shp = ds_dir / "camels_basin_shapes.shp"
  dummy_shp.write_text("dummy")

  dummy_df = pd.DataFrame(
      {"basin_area": [150.0], "ele_mt_sav": [300.0], "p_mean": [3.2]},
      index=["camels_01"],
  )
  dummy_df.index.name = "gauge_id"

  mock_extractor = MagicMock()
  mock_extractor.extract_attributes_from_file.return_value = dummy_df
  monkeypatch.setattr(
      "multimet.static_extractor.batch_runner.StaticAttributesExtractor",
      lambda **kwargs: mock_extractor,
  )

  results = run_batch_extraction(
      dataset_map={"camels": dummy_shp},
      output_dir="gs://my-bucket/attributes/",
      gdb_path=tmp_path / "BasinATLAS_v10.gdb",
      era5_source="hybas",
      era5_cache_dir=tmp_path / "era5",
      workers=1,
      staging_cache_dir=tmp_path / "staged",
      partition_outputs=True,
      resume=True,
  )

  assert "camels" in results
  uploaded_filenames = [u[0] for u in uploaded_uris]
  assert "attributes_hydroatlas_camels.csv" in uploaded_filenames
  assert "attributes_caravan_camels.csv" in uploaded_filenames
  assert "attributes_camels.parquet" in uploaded_filenames
  for _, dest in uploaded_uris:
    assert dest == "gs://my-bucket/attributes/camels/"

  # Re-running with resume=True should skip camels
  prev_upload_count = len(uploaded_uris)
  results_resume = run_batch_extraction(
      dataset_map={"camels": dummy_shp},
      output_dir="gs://my-bucket/attributes/",
      gdb_path=tmp_path / "BasinATLAS_v10.gdb",
      era5_source="hybas",
      era5_cache_dir=tmp_path / "era5",
      workers=1,
      staging_cache_dir=tmp_path / "staged",
      partition_outputs=True,
      resume=True,
  )
  assert "camels" in results_resume
  assert len(uploaded_uris) == prev_upload_count


def test_append_attributes_to_zarr_validation(tmp_path):
  """Verifies append_attributes_to_zarr writes attributes and raises on missing store or basin."""
  import xarray as xr

  shp_path, era5_cache, _ = _build_synthetic_hydroatlas_env(tmp_path, include_native_pet=False)
  extractor = StaticAttributesExtractor(
      gdb_path=shp_path,
      era5_cache_dir=era5_cache,
      era5_source="hybas",
  )

  # Nonexistent Zarr path must raise FileNotFoundError
  with pytest.raises(FileNotFoundError):
    extractor.append_attributes_to_zarr(
        master_zarr_path=tmp_path / "missing.zarr",
        basin_id="b1",
        attributes={"ele_mt_sav": 250.0},
    )

  # Create valid Zarr store with 'basin' dimension
  zarr_path = tmp_path / "master.zarr"
  ds = xr.Dataset(coords={"basin": ["b1", "b2"]})
  ds.to_zarr(str(zarr_path), mode="w")

  # Unknown basin_id must raise KeyError
  with pytest.raises(KeyError, match="not found in 'basin' coordinate"):
    extractor.append_attributes_to_zarr(
        master_zarr_path=zarr_path,
        basin_id="unknown_basin",
        attributes={"ele_mt_sav": 250.0},
    )

  # Valid flat attributes dict writes directly to Zarr store
  extractor.append_attributes_to_zarr(
      master_zarr_path=zarr_path,
      basin_id="b1",
      attributes={"ele_mt_sav": 250.0},
  )
  ds_reloaded = xr.open_zarr(str(zarr_path))
  assert "caravan_ele_mt_sav" in ds_reloaded
  assert np.isclose(float(ds_reloaded["caravan_ele_mt_sav"].values[0]), 250.0)
  assert np.isnan(float(ds_reloaded["caravan_ele_mt_sav"].values[1]))


def test_no_silent_fallbacks_or_masked_errors(tmp_path):
  """Ensures out-of-bounds polygons, missing baseline years, and malformed inputs are not silently masked."""
  from multimet.static_extractor.climate import ERA5GriddedExtractor

  zarr_path = tmp_path / "synthetic_era5.zarr"
  root = zarr.open(str(zarr_path), mode="w")
  lats = np.array([10.0, 11.0], dtype=np.float64)
  lons = np.array([20.0, 21.0], dtype=np.float64)
  times = np.arange(10, dtype=np.int64)
  lat_arr = root.create_array("latitude", shape=lats.shape, dtype=lats.dtype)
  lat_arr[:] = lats
  lon_arr = root.create_array("longitude", shape=lons.shape, dtype=lons.dtype)
  lon_arr[:] = lons
  t_arr = root.create_array("time", shape=times.shape, dtype=times.dtype)
  t_arr[:] = times
  t_arr.attrs["units"] = "days since 2022-01-01"
  p_arr = root.create_array("total_precipitation", shape=(10, 2, 2), dtype=np.float32)
  p_arr[:] = 2.0
  p_arr.attrs["units"] = "mm"
  temp_arr = root.create_array("temperature_2m", shape=(10, 2, 2), dtype=np.float32)
  temp_arr[:] = 15.0
  temp_arr.attrs["units"] = "degC"

  gridded = ERA5GriddedExtractor(zarr_uri=str(zarr_path))

  # 1. Polygon outside coordinate domain -> must return NaN (no centroid snapping)
  out_of_bounds_poly = shapely.geometry.box(50.0, 50.0, 51.0, 51.0)
  res_oob = gridded.extract_climate_metrics_for_polygon(out_of_bounds_poly, baseline_years=(2022, 2022))
  assert np.isnan(res_oob["p_mean"])

  # 2. Polygon inside domain, but no dates in baseline_years -> must return NaN (no silent date fallback)
  in_bounds_poly = shapely.geometry.box(20.0, 10.0, 21.0, 11.0)
  res_no_dates = gridded.extract_climate_metrics_for_polygon(in_bounds_poly, baseline_years=(1981, 2020))
  assert np.isnan(res_no_dates["p_mean"])


def test_no_download_in_memory_cloud_streaming(tmp_path, monkeypatch):
  """Verifies --no-download / no_download=True streams HydroATLAS and ERA5 in memory without local disk downloads."""
  import io
  import gcsfs

  shp_path, era5_cache, _ = _build_synthetic_hydroatlas_env(tmp_path, include_native_pet=False)
  gdf_full = gpd.read_file(shp_path)

  # Build synthetic cloud byte payloads in memory (never written to local cache dir)
  single_gpq_buf = io.BytesIO()
  gdf_full.to_parquet(single_gpq_buf)
  single_gpq_bytes = single_gpq_buf.getvalue()

  subpoly_gdf = gdf_full[["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA", "geometry"]].copy()
  subpoly_buf = io.BytesIO()
  subpoly_gdf.to_parquet(subpoly_buf)
  subpoly_bytes = subpoly_buf.getvalue()

  attr_df = pd.DataFrame(gdf_full.drop(columns="geometry"))
  attr_df["HYBAS_ID"] = [f"hybas_{int(h)}" for h in attr_df["HYBAS_ID"]]
  attr_buf = io.BytesIO()
  attr_df.to_parquet(attr_buf, index=False)
  attr_bytes = attr_buf.getvalue()

  climate_bytes = (era5_cache / "na_climate_indices.txt").read_bytes()

  fake_objects = {
      "test-bucket/hydroatlas/hydro_atlas_lev12.parquet": attr_bytes,
      "test-bucket/hydroatlas/subpolygons/hybas_na_lev12_v1c.geoparquet": subpoly_bytes,
      "test-bucket/hydroatlas/era5_climate/na_climate_indices.txt": climate_bytes,
      "test-bucket/single/hydroatlas.geoparquet": single_gpq_bytes,
  }

  class FakeGCSFileSystem:
    def exists(self, path: str) -> bool:
      clean = path.replace("gs://", "").rstrip("/")
      if clean in fake_objects:
        return True
      prefix = clean + "/"
      return any(k.startswith(prefix) for k in fake_objects)

    def ls(self, path: str):
      clean = path.replace("gs://", "").rstrip("/") + "/"
      return sorted([k for k in fake_objects if k.startswith(clean)])

    def cat_file(self, path: str) -> bytes:
      clean = path.replace("gs://", "").rstrip("/")
      if clean not in fake_objects:
        raise FileNotFoundError(clean)
      return fake_objects[clean]

    def open(self, path: str, mode: str = "rb", **kwargs):
      data = self.cat_file(path)
      if "r" in mode and "b" not in mode:
        return io.StringIO(data.decode(kwargs.get("encoding", "utf-8")))
      return io.BytesIO(data)

  monkeypatch.setattr(gcsfs, "GCSFileSystem", FakeGCSFileSystem)

  # Ensure no local cache directories are created when no_download=True
  uncreated_gdb_dir = tmp_path / "should_not_be_created_gdb"
  uncreated_era5_dir = tmp_path / "should_not_be_created_era5"

  ext_cloud = StaticAttributesExtractor(
      gdb_path=uncreated_gdb_dir,
      era5_cache_dir=uncreated_era5_dir,
      gcs_gdb_uri="gs://test-bucket/hydroatlas/BasinATLAS_v10.gdb",
      gcs_era5_climate_uri="gs://test-bucket/hydroatlas/era5_climate",
      era5_source="hybas",
      no_download=True,
  )
  assert not uncreated_gdb_dir.exists()
  assert not uncreated_era5_dir.exists()

  query_poly = shapely.geometry.box(-86.25, 40.1, -85.25, 40.9)
  res_cloud = ext_cloud.extract_attributes_for_polygon(query_poly, catchment_id="cloud_01")
  attrs = res_cloud["caravan_attributes"]
  assert np.isclose(attrs["ele_mt_sav"], 350.0, rtol=1e-3)
  assert np.isclose(attrs["slp_dg_sav"], 25.0, rtol=1e-3)
  assert attrs["glc_cl_smj"] == 12
  assert np.isclose(attrs["dis_m3_pyr"], 45.0)
  assert np.isclose(attrs["p_mean"], 4.5, rtol=1e-3)
  assert np.isclose(attrs["pet_mean_FAO_PM"], 2.25, rtol=1e-3)
  assert not uncreated_gdb_dir.exists()
  assert not uncreated_era5_dir.exists()

  # Also test single GeoParquet file on GCS with no local paths passed at all
  ext_single_gpq = StaticAttributesExtractor(
      gcs_gdb_uri="gs://test-bucket/single/hydroatlas.geoparquet",
      gcs_era5_climate_uri="gs://test-bucket/hydroatlas/era5_climate",
      era5_source="hybas",
      no_download=True,
  )
  res_single = ext_single_gpq.extract_attributes_for_polygon(query_poly, catchment_id="cloud_02")
  assert np.isclose(res_single["caravan_attributes"]["ele_mt_sav"], 350.0, rtol=1e-3)

  # Test CLI with --no-download and no --gdb-path or --era5-cache-dir
  in_geojson = tmp_path / "query.geojson"
  out_csv = tmp_path / "cloud_out.csv"
  gpd.GeoDataFrame({"gauge_id": ["cloud_cli"]}, geometry=[query_poly], crs="EPSG:4326").to_file(
      in_geojson, driver="GeoJSON"
  )
  cli_main([
      "--input", str(in_geojson),
      "--output", str(out_csv),
      "--era5-source", "hybas",
      "--gcs-gdb-uri", "gs://test-bucket/hydroatlas",
      "--gcs-era5-climate-uri", "gs://test-bucket/hydroatlas/era5_climate",
      "--no-download",
  ])
  assert out_csv.exists()
  df_out = pd.read_csv(out_csv, index_col=0)
  assert np.isclose(df_out.loc["cloud_cli", "ele_mt_sav"], 350.0, rtol=1e-3)
  assert np.isclose(df_out.loc["cloud_cli", "p_mean"], 4.5, rtol=1e-3)

