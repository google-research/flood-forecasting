# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a stratified multi-continent reference watershed benchmark dataset."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

_KM_PER_DEGREE: float = 111.0
_TIER_MICRO_MAX_KM2: float = 100.0
_TIER_SMALL_MAX_KM2: float = 500.0
_TIER_MEDIUM_MAX_KM2: float = 2500.0
_TIER_LARGE_MAX_KM2: float = 10000.0
_MAX_SAMPLES_PER_TIER: int = 45
_TARGET_PER_CONTINENT: int = 200


def compute_geodesic_area(geom: BaseGeometry) -> float:
    """Compute approximate spherical area in km2 for a WGS84 geometry."""
    centroid_lat = float(geom.centroid.y)
    scale = (
        _KM_PER_DEGREE * _KM_PER_DEGREE * math.cos(math.radians(centroid_lat))
    )
    return float(geom.area * scale)


def get_quadrant(lat: float, lon: float) -> str:
    """Return hemisphere quadrant label (NE, NW, SE, SW)."""
    lat_str = 'N' if lat >= 0 else 'S'
    lon_str = 'E' if lon >= 0 else 'W'
    return f'{lat_str}{lon_str}'


def get_size_tier(area_km2: float) -> str:
    """Classify drainage area into 5 standard size tiers."""
    if area_km2 < _TIER_MICRO_MAX_KM2:
        return '1_micro'
    if area_km2 < _TIER_SMALL_MAX_KM2:
        return '2_small'
    if area_km2 < _TIER_MEDIUM_MAX_KM2:
        return '3_medium'
    if area_km2 < _TIER_LARGE_MAX_KM2:
        return '4_large'
    return '5_macro'


def main(argv: list[str] | None = None) -> int:
    """Build benchmark parquet from user-supplied input data directory."""
    parser = argparse.ArgumentParser(
        description='Build stratified reference watershed benchmark dataset.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Explicit root directory containing source reference shapefiles.',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Explicit output file path (.parquet).',
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_file = Path(args.output).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Input data_dir does not exist: {data_dir}')
    out_file.parent.mkdir(parents=True, exist_ok=True)

    world = gpd.read_file(data_dir / 'input/naturalearth_lowres.geojson')
    grdc_attr = (
        pd.read_csv(
            data_dir / 'input/attributes/grdc_attributes.csv', index_col=0
        )
        .reset_index()
        .rename(columns={'index': 'gauge_id'})
    )
    grdc_attr = grdc_attr[
        ['gauge_id', 'latitude', 'longitude', 'calculated_drain_area']
    ].dropna()
    grdc_shapes = gpd.read_file(
        data_dir
        / 'caravan_shapefiles/caravan_extensions/grdc/grdc_basin_shapes.shp'
    )
    grdc_merged = grdc_shapes.merge(grdc_attr, on='gauge_id')

    pts = [
        Point(xy)
        for xy in zip(
            grdc_merged['longitude'], grdc_merged['latitude'], strict=True
        )
    ]
    grdc_pts_gdf = gpd.GeoDataFrame(
        grdc_merged[['gauge_id']], geometry=pts, crs='EPSG:4326'
    )
    grdc_joined = gpd.sjoin(
        grdc_pts_gdf,
        world[['continent', 'geometry']],
        how='left',
        predicate='within',
    ).drop_duplicates(subset=['gauge_id'])
    grdc_merged = grdc_merged.merge(
        grdc_joined[['gauge_id', 'continent']], on='gauge_id'
    ).dropna(subset=['continent'])

    camelsind_shapes = gpd.read_file(
        data_dir / 'caravan_shapefiles/caravan_google_internal_extensions/'
        'camelsind/camelsind_basin_shapes.shp'
    )
    caravan_coords = pd.read_csv(
        data_dir / 'input/attributes/caravan_coordinates.csv'
    )
    caravan_coords['gauge_id_short'] = (
        caravan_coords['gauge_id'].str.lower().str.replace('caravan_', '')
    )
    camelsind_merged = camelsind_shapes.merge(
        caravan_coords[
            ['gauge_id_short', 'CARAVAN:gauge_lat', 'CARAVAN:gauge_lon']
        ],
        left_on='gauge_id',
        right_on='gauge_id_short',
    ).rename(
        columns={
            'CARAVAN:gauge_lat': 'latitude',
            'CARAVAN:gauge_lon': 'longitude',
        }
    )
    camelsind_merged['continent'] = 'Asia'
    camelsind_merged['calculated_drain_area'] = camelsind_merged[
        'geometry'
    ].apply(compute_geodesic_area)

    continents = [
        'Africa',
        'Asia',
        'Europe',
        'North America',
        'South America',
        'Oceania',
    ]
    cols = [
        'gauge_id',
        'latitude',
        'longitude',
        'calculated_drain_area',
        'continent',
        'geometry',
    ]
    candidates = []

    for cont in continents:
        if cont == 'Asia':
            pool = pd.concat(
                [
                    camelsind_merged[cols],
                    grdc_merged[grdc_merged['continent'] == 'Asia'][cols],
                ],
                ignore_index=True,
            )
        else:
            pool = grdc_merged[grdc_merged['continent'] == cont][cols].copy()

        pool['size_tier'] = pool['calculated_drain_area'].apply(get_size_tier)
        sampled_cont = []
        for tier in sorted(pool['size_tier'].unique()):
            sub = pool[pool['size_tier'] == tier]
            n_take = min(len(sub), _MAX_SAMPLES_PER_TIER)
            sampled_cont.append(sub.sample(n=n_take, random_state=42))

        cont_df = pd.concat(sampled_cont, ignore_index=True)
        if len(cont_df) > _TARGET_PER_CONTINENT:
            cont_df = cont_df.sample(n=_TARGET_PER_CONTINENT, random_state=42)
        elif len(cont_df) < _TARGET_PER_CONTINENT and len(pool) > len(cont_df):
            rem = pool[~pool['gauge_id'].isin(cont_df['gauge_id'])]
            needed = min(_TARGET_PER_CONTINENT - len(cont_df), len(rem))
            cont_df = pd.concat(
                [cont_df, rem.sample(n=needed, random_state=42)],
                ignore_index=True,
            )
        candidates.append(cont_df)

    final_df = pd.concat(candidates, ignore_index=True)
    final_df['hemisphere'] = final_df.apply(
        lambda r: get_quadrant(r['latitude'], r['longitude']), axis=1
    )
    final_df['size_tier'] = final_df['calculated_drain_area'].apply(
        get_size_tier
    )
    final_df['geometry_wkt'] = final_df['geometry'].apply(lambda g: g.wkt)
    final_df['reference_area_km2'] = final_df['calculated_drain_area'].round(2)

    export_cols = [
        'gauge_id',
        'continent',
        'hemisphere',
        'size_tier',
        'latitude',
        'longitude',
        'reference_area_km2',
        'geometry_wkt',
    ]
    out_df = final_df[export_cols].copy()
    out_df.to_parquet(out_file, index=False)
    sys.stdout.write(
        f'Benchmark dataset created: {len(out_df)} basins -> {out_file}\n'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
