#!/usr/bin/env python3
"""Builds a globally balanced 1,000+ basin benchmark dataset for catchment delineation."""

import math
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import shapely.wkt

def compute_geodesic_area(geom) -> float:
  """Approximates geodesic area in km2 using latitude scaling."""
  centroid = geom.centroid
  lat = centroid.y
  lat_scale = 111.0
  lon_scale = 111.0 * math.cos(math.radians(lat))
  return float(geom.area * lat_scale * lon_scale)

def get_quadrant(lat: float, lon: float) -> str:
  lat_str = "N" if lat >= 0 else "S"
  lon_str = "E" if lon >= 0 else "W"
  return f"{lat_str}{lon_str}"

def get_size_tier(area_km2: float) -> str:
  if area_km2 < 100:
    return "1_micro"
  elif area_km2 < 500:
    return "2_small"
  elif area_km2 < 2500:
    return "3_medium"
  elif area_km2 < 10000:
    return "4_large"
  else:
    return "5_macro"

def main():
  data_dir = Path("/usr/local/google/home/gsnearing/data")
  out_file = Path("/usr/local/google/home/gsnearing/Projects/catchment-delineation/catchment_delineation/data/benchmark_basins_1000.parquet")
  out_file.parent.mkdir(parents=True, exist_ok=True)

  world = gpd.read_file(data_dir / "input/naturalearth_lowres.geojson")

  # 1. Load GRDC basins
  grdc_attr = pd.read_csv(data_dir / "input/attributes/grdc_attributes.csv", index_col=0).reset_index().rename(columns={"index": "gauge_id"})
  grdc_attr = grdc_attr[["gauge_id", "latitude", "longitude", "calculated_drain_area"]].dropna()
  grdc_shapes = gpd.read_file(data_dir / "caravan_shapefiles/caravan_extensions/grdc/grdc_basin_shapes.shp")
  grdc_merged = grdc_shapes.merge(grdc_attr, on="gauge_id")

  # Spatial join with world continent
  pts = [Point(xy) for xy in zip(grdc_merged["longitude"], grdc_merged["latitude"])]
  grdc_pts_gdf = gpd.GeoDataFrame(grdc_merged[["gauge_id"]], geometry=pts, crs="EPSG:4326")
  grdc_joined = gpd.sjoin(grdc_pts_gdf, world[["continent", "geometry"]], how="left", predicate="within")
  # Dedup any multi-matches
  grdc_joined = grdc_joined.drop_duplicates(subset=["gauge_id"])
  grdc_merged = grdc_merged.merge(grdc_joined[["gauge_id", "continent"]], on="gauge_id")
  grdc_merged = grdc_merged.dropna(subset=["continent"])

  # 2. Load CAMELS-IND (Asia)
  camelsind_shapes = gpd.read_file(data_dir / "caravan_shapefiles/caravan_google_internal_extensions/camelsind/camelsind_basin_shapes.shp")
  caravan_coords = pd.read_csv(data_dir / "input/attributes/caravan_coordinates.csv")
  caravan_coords["gauge_id_short"] = caravan_coords["gauge_id"].str.lower().str.replace("caravan_", "")
  camelsind_merged = camelsind_shapes.merge(
      caravan_coords[["gauge_id_short", "CARAVAN:gauge_lat", "CARAVAN:gauge_lon"]],
      left_on="gauge_id",
      right_on="gauge_id_short"
  )
  camelsind_merged = camelsind_merged.rename(columns={"CARAVAN:gauge_lat": "latitude", "CARAVAN:gauge_lon": "longitude"})
  camelsind_merged["continent"] = "Asia"
  camelsind_merged["calculated_drain_area"] = camelsind_merged["geometry"].apply(compute_geodesic_area)

  # Combine candidate pools
  candidates = []
  
  # Select per continent (target 200 per continent)
  continents = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
  
  for cont in continents:
    if cont == "Asia":
      pool = pd.concat([
          camelsind_merged[["gauge_id", "latitude", "longitude", "calculated_drain_area", "continent", "geometry"]],
          grdc_merged[grdc_merged["continent"] == "Asia"][["gauge_id", "latitude", "longitude", "calculated_drain_area", "continent", "geometry"]]
      ], ignore_index=True)
    else:
      pool = grdc_merged[grdc_merged["continent"] == cont][["gauge_id", "latitude", "longitude", "calculated_drain_area", "continent", "geometry"]].copy()

    pool["size_tier"] = pool["calculated_drain_area"].apply(get_size_tier)
    
    # Stratified sample across size tiers (target ~40 per tier, or proportional if small tiers have fewer)
    sampled_cont = []
    tiers = sorted(pool["size_tier"].unique())
    for t in tiers:
      sub = pool[pool["size_tier"] == t]
      n = min(len(sub), 45)
      sampled_cont.append(sub.sample(n=n, random_state=42))
    
    cont_df = pd.concat(sampled_cont, ignore_index=True)
    if len(cont_df) > 200:
      cont_df = cont_df.sample(n=200, random_state=42)
    elif len(cont_df) < 200 and len(pool) > len(cont_df):
      rem = pool[~pool["gauge_id"].isin(cont_df["gauge_id"])]
      needed = min(200 - len(cont_df), len(rem))
      cont_df = pd.concat([cont_df, rem.sample(n=needed, random_state=42)], ignore_index=True)

    candidates.append(cont_df)
    print(f"Sampled {cont}: {len(cont_df)} basins")

  final_df = pd.concat(candidates, ignore_index=True)
  final_df["hemisphere"] = final_df.apply(lambda r: get_quadrant(r["latitude"], r["longitude"]), axis=1)
  final_df["size_tier"] = final_df["calculated_drain_area"].apply(get_size_tier)
  
  # Ensure geometry is stored as WKT for clean parquet portability
  final_df["geometry_wkt"] = final_df["geometry"].apply(lambda g: g.wkt)
  final_df["reference_area_km2"] = final_df["calculated_drain_area"].round(2)
  
  export_cols = ["gauge_id", "continent", "hemisphere", "size_tier", "latitude", "longitude", "reference_area_km2", "geometry_wkt"]
  out_df = final_df[export_cols].copy()
  out_df.to_parquet(out_file, index=False)

  print(f"\n==========================================")
  print(f"Benchmark dataset created: {len(out_df)} basins")
  print(f"Saved to {out_file}")
  print(f"Per Continent:\n{out_df['continent'].value_counts()}")
  print(f"Per Hemisphere Quadrant:\n{out_df['hemisphere'].value_counts()}")
  print(f"Per Size Tier:\n{out_df['size_tier'].value_counts().sort_index()}")
  print(f"==========================================")

if __name__ == "__main__":
  main()
