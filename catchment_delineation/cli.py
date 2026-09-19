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

"""Command-Line Interface for DEM Catchment Delineation.

Delineates upstream drainage catchment boundaries from a DEM by supplying
a single coordinate pair, a list of coordinates, or a CSV/Parquet file, with
all input and output paths explicitly supplied by the user.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import shutil
import sys
import tempfile
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

import fsspec
import geopandas as gpd
import pandas as pd

from catchment_delineation.delineator import (
    CatchmentCoverageError,
    DemDelineator,
    build_missing_feature,
)
from catchment_delineation.gcs import (
    download_tile_from_gcs,
    is_gcs_path,
    normalize_gcs_path,
    upload_file_to_gcs,
)
from catchment_delineation.tiles import (
    is_coord_in_coverage,
    is_tile_in_coverage,
    latlon_to_tile_key,
    list_available_tiles,
    tile_key_to_filename,
)

logger = logging.getLogger('catchment_delineation.cli')

_LAT_COLUMN_ALLOWLIST: tuple[str, ...] = (
    'latitude',
    'lat',
    'gauge_lat',
    'caravan:gauge_lat',
    'outlet_lat',
    'pour_point_lat',
)
_LON_COLUMN_ALLOWLIST: tuple[str, ...] = (
    'longitude',
    'lon',
    'long',
    'lng',
    'gauge_lon',
    'caravan:gauge_lon',
    'outlet_lon',
    'pour_point_lon',
)
_ID_COLUMN_ALLOWLIST: tuple[str, ...] = (
    'gauge_id',
    'catchment_id',
    'station_id',
    'hybas_id',
    'id',
    'caravan:gauge_id',
)
_EXPECTED_COORD_PARTS: int = 2
_MAX_PRECACHE_THREADS: int = 32
_PROGRESS_INTERVAL_DIVISOR: int = 20


def _delineate_worker(
    task: tuple[
        float,
        float,
        str | None,
        str | None,
        str | None,
        str | None,
        int,
        int | None,
        float | None,
        float,
    ],
) -> tuple[dict[str, Any], list[str]]:
    """Run single-basin delineation in a worker process."""
    (
        lat,
        lon,
        cid,
        tiles_dir,
        gcs_uri,
        cache_dir,
        snap_window,
        max_cells,
        expected_area,
        area_tolerance,
    ) = task
    delineator = DemDelineator(
        tiles_dir=Path(tiles_dir) if tiles_dir else None,
        gcs_uri=gcs_uri,
        cache_dir=Path(cache_dir) if cache_dir else None,
        cache_tiles=True,
    )
    try:
        feature = delineator.delineate(
            lat=lat,
            lon=lon,
            catchment_id=cid,
            snap_window_cells=snap_window,
            max_cells=max_cells,
            expected_area_km2=expected_area,
            area_tolerance=area_tolerance,
        )
    except CatchmentCoverageError as err:
        logger.warning(
            'Catchment coverage abort for %s (%.4f, %.4f): %s',
            cid,
            lat,
            lon,
            err,
        )
        feature = build_missing_feature(lat, lon, cid, str(err))
    created = [str(p) for p in delineator.created_cache_files]
    return feature, created


def parse_coord_str(coord_str: str) -> tuple[float, float]:
    """Parse a coordinate string like '39.6828,-88.7729' into (lat, lon)."""
    parts = coord_str.strip().split(',')
    if len(parts) != _EXPECTED_COORD_PARTS:
        parts = coord_str.strip().split()
    if len(parts) != _EXPECTED_COORD_PARTS:
        raise ValueError(
            f"Invalid coordinate format '{coord_str}'. Expected 'lat,lon'."
        )
    return float(parts[0]), float(parts[1])


def _resolve_column(
    fieldnames: list[str],
    explicit_col: str | None,
    allowlist: tuple[str, ...],
    role: str,
    *,
    required: bool,
) -> str | None:
    """Resolve a column name strictly without substring guessing."""
    if explicit_col is not None:
        if explicit_col not in fieldnames:
            raise ValueError(
                f"Specified {role} column '{explicit_col}' not found in "
                f'columns: {fieldnames}'
            )
        return explicit_col

    lower_to_actual: dict[str, list[str]] = {}
    for col in fieldnames:
        lower_to_actual.setdefault(col.strip().lower(), []).append(col)

    matched: list[str] = []
    for candidate in allowlist:
        if candidate in lower_to_actual:
            matched.extend(lower_to_actual[candidate])

    if len(matched) > 1:
        raise ValueError(
            f'Ambiguous {role} columns {matched} in file. Please specify '
            f'--{role}-col explicitly.'
        )
    if len(matched) == 1:
        return matched[0]
    if required:
        raise ValueError(
            f'Coordinate file must contain an explicit {role} column '
            f'(one of {allowlist}). Found columns: {fieldnames}'
        )
    return None


def _read_single_coord_table(path_str: str) -> pd.DataFrame:
    """Read a single CSV or Parquet coordinate table without path fallback."""
    if not is_gcs_path(path_str):
        local_p = Path(path_str).expanduser()
        if not local_p.exists():
            raise FileNotFoundError(
                f"Coordinate input path does not exist: '{path_str}'"
            )
        if local_p.is_dir():
            caravan_files = sorted(local_p.rglob('attributes_other_*.csv'))
            if not caravan_files:
                raise FileNotFoundError(
                    'No Caravan attribute coordinate files '
                    f"('attributes_other_*.csv') found under '{local_p}'."
                )
            frames = [pd.read_csv(f) for f in caravan_files]
            return pd.concat(frames, ignore_index=True)
        path_str = str(local_p)

    try:
        if path_str.endswith(('.parquet', '.geoparquet')):
            return pd.read_parquet(path_str)
        return pd.read_csv(path_str)
    except pd.errors.EmptyDataError as err:
        raise ValueError(
            f"Coordinate file '{path_str}' is empty or has no header."
        ) from err


def load_coords_from_file(
    file_path: str | Path,
    lat_col_arg: str | None = None,
    lon_col_arg: str | None = None,
    id_col_arg: str | None = None,
) -> tuple[list[tuple[float, float]], list[str | None]]:
    """Parse lat/lon coordinates and optional IDs from an explicit path."""
    path_str = (
        normalize_gcs_path(file_path)
        if is_gcs_path(file_path)
        else str(file_path)
    )
    df = _read_single_coord_table(path_str)

    if df.empty and len(df.columns) == 0:
        raise ValueError(
            f"Coordinate file '{file_path}' is empty or has no header."
        )

    fieldnames = [str(c) for c in df.columns]
    lat_col = _resolve_column(
        fieldnames, lat_col_arg, _LAT_COLUMN_ALLOWLIST, 'lat', required=True
    )
    lon_col = _resolve_column(
        fieldnames, lon_col_arg, _LON_COLUMN_ALLOWLIST, 'lon', required=True
    )
    id_col = _resolve_column(
        fieldnames, id_col_arg, _ID_COLUMN_ALLOWLIST, 'id', required=False
    )
    assert lat_col is not None
    assert lon_col is not None

    coords = list(
        zip(
            pd.to_numeric(df[lat_col], errors='raise').astype(float),
            pd.to_numeric(df[lon_col], errors='raise').astype(float),
            strict=True,
        )
    )
    if id_col is not None:
        ids: list[str | None] = [
            str(v).strip() if pd.notna(v) and str(v).strip() != '' else None
            for v in df[id_col]
        ]
    else:
        ids = [None] * len(coords)

    return coords, ids


def load_coords_from_csv(
    csv_path: str | Path,
    lat_col_arg: str | None = None,
    lon_col_arg: str | None = None,
    id_col_arg: str | None = None,
) -> tuple[list[tuple[float, float]], list[str | None]]:
    """Alias for load_coords_from_file."""
    return load_coords_from_file(csv_path, lat_col_arg, lon_col_arg, id_col_arg)


def _sanitize_feature_for_export(feat: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a Feature without substituting 0.0 defaults."""
    props = dict(feat.get('properties') or {})
    gid = props.get('gauge_id')
    if gid is None:
        gid = props.get('catchment_id')
    if gid is None or str(gid).strip() == '':
        raise KeyError(
            'Feature is missing required gauge_id / catchment_id property.'
        )
    gid_str = str(gid).strip()

    geom = feat.get('geometry')
    status = str(props.get('status', 'SUCCESS'))
    if geom is None or status.startswith('MISSING_DATA'):
        nan_val = float('nan')
        return {
            'type': 'Feature',
            'properties': {
                'gauge_id': gid_str,
                'area': nan_val,
                'gauge_lat': nan_val,
                'gauge_lon': nan_val,
                'catchment_id': gid_str,
                'area_km2': nan_val,
                'status': status,
            },
            'geometry': None,
        }

    raw_area = (
        props['area']
        if 'area' in props and props['area'] is not None
        else props.get('area_km2')
    )
    if raw_area is None:
        raise KeyError(
            f"Feature '{gid_str}' is missing required area / area_km2."
        )
    area_val = float(raw_area)

    outlet_val = props.get('outlet')
    if isinstance(outlet_val, dict):
        if (
            'latitude' not in outlet_val
            or outlet_val['latitude'] is None
            or 'longitude' not in outlet_val
            or outlet_val['longitude'] is None
        ):
            raise KeyError(
                f"Feature '{gid_str}' outlet is missing latitude/longitude."
            )
        snapped_lat = float(outlet_val['latitude'])
        snapped_lon = float(outlet_val['longitude'])
    else:
        if props.get('gauge_lat') is None or props.get('gauge_lon') is None:
            raise KeyError(
                f"Feature '{gid_str}' is missing outlet coordinates."
            )
        snapped_lat = float(props['gauge_lat'])
        snapped_lon = float(props['gauge_lon'])

    return {
        'type': 'Feature',
        'properties': {
            'gauge_id': gid_str,
            'area': round(area_val, 4),
            'gauge_lat': round(snapped_lat, 6),
            'gauge_lon': round(snapped_lon, 6),
            'catchment_id': gid_str,
            'area_km2': round(area_val, 4),
            'status': status,
        },
        'geometry': geom,
    }


def _extract_caravan_subdataset(gauge_id: str) -> str:
    """Extract standard Caravan subdataset prefix from '<subdataset>_<id>'."""
    parts = gauge_id.strip().split('_')
    if len(parts) < _EXPECTED_COORD_PARTS or not parts[0]:
        raise ValueError(
            f"Gauge ID '{gauge_id}' does not follow the Caravan "
            "'<subdataset>_<id>' naming convention required by "
            '--preserve-caravan-dirs.'
        )
    if parts[0].upper() == 'CARAVAN' and len(parts) >= 3 and parts[1]:
        return parts[1].lower()
    return parts[0].lower()


def _features_to_geodataframe(
    sanitized_features: list[dict[str, Any]],
) -> gpd.GeoDataFrame:
    """Convert sanitized Feature dicts (including None geometries) to a GDF."""
    if not sanitized_features:
        return gpd.GeoDataFrame(
            columns=[
                'gauge_id',
                'area',
                'gauge_lat',
                'gauge_lon',
                'catchment_id',
                'area_km2',
                'status',
                'geometry',
            ],
            geometry='geometry',
            crs='EPSG:4326',
        )
    gdf = gpd.GeoDataFrame.from_features(sanitized_features, crs='EPSG:4326')
    contract_cols = [
        c
        for c in ('gauge_id', 'area', 'gauge_lat', 'gauge_lon', 'geometry')
        if c in gdf.columns
    ]
    other_cols = [c for c in gdf.columns if c not in contract_cols]
    return gdf[contract_cols + other_cols]


def _write_partition_outputs(
    target_folder: str,
    subdataset: str,
    sanitized_features: list[dict[str, Any]],
    formats: list[str],
) -> None:
    """Write GeoParquet, GeoJSON, and/or Shapefile outputs for a partition."""
    if not is_gcs_path(target_folder):
        Path(target_folder).mkdir(parents=True, exist_ok=True)

    gdf = _features_to_geodataframe(sanitized_features)

    if 'geoparquet' in formats or 'parquet' in formats:
        out_gpq = f'{target_folder}/{subdataset}_basin_shapes.geoparquet'
        gdf.to_parquet(out_gpq)

    if 'geojson' in formats:
        out_geojson = f'{target_folder}/{subdataset}_basin_shapes.geojson'
        fc = {'type': 'FeatureCollection', 'features': sanitized_features}
        if is_gcs_path(out_geojson):
            with fsspec.open(out_geojson, 'w', encoding='utf-8') as handle:
                json.dump(fc, handle)
        else:
            Path(out_geojson).write_text(json.dumps(fc), encoding='utf-8')

    if 'shp' in formats:
        shp_cols = ['gauge_id', 'area', 'gauge_lat', 'gauge_lon', 'geometry']
        shp_gdf = gdf[[c for c in shp_cols if c in gdf.columns]]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_shp = Path(tmpdir) / f'{subdataset}_basin_shapes.shp'
            shp_gdf.to_file(tmp_shp, encoding='utf-8')
            tmp_cpg = Path(tmpdir) / f'{subdataset}_basin_shapes.cpg'
            if not tmp_cpg.exists():
                tmp_cpg.write_text('UTF-8\n', encoding='utf-8')

            if is_gcs_path(target_folder):
                for part in Path(tmpdir).iterdir():
                    upload_file_to_gcs(part, f'{target_folder}/{part.name}')
            else:
                for part in Path(tmpdir).iterdir():
                    shutil.copy(part, Path(target_folder) / part.name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'DEM Watershed Catchment Delineation Tool.\n'
            'All input/output and tile source paths must be explicitly '
            'provided by the caller.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    coord_group = parser.add_argument_group('Coordinate Input Options')
    coord_group.add_argument(
        '--lat', type=float, help='Latitude of outlet pour point.'
    )
    coord_group.add_argument(
        '--lon', type=float, help='Longitude of outlet pour point.'
    )
    coord_group.add_argument(
        '--coords',
        nargs='+',
        help="One or more coordinate pairs as 'lat,lon' strings.",
    )
    coord_group.add_argument(
        '--csv',
        type=str,
        help='Explicit path (local or gs://) to CSV/Parquet coordinate file.',
    )
    coord_group.add_argument(
        '--id',
        type=str,
        default=None,
        help='Custom catchment ID (for single-coordinate delineation).',
    )
    coord_group.add_argument(
        '--workers',
        '-w',
        type=int,
        default=1,
        help='Number of parallel worker processes for batch delineation.',
    )
    coord_group.add_argument(
        '--lat-col',
        type=str,
        default=None,
        help='Explicit latitude column name in CSV/Parquet file.',
    )
    coord_group.add_argument(
        '--lon-col',
        type=str,
        default=None,
        help='Explicit longitude column name in CSV/Parquet file.',
    )
    coord_group.add_argument(
        '--id-col',
        type=str,
        default=None,
        help='Explicit catchment/gauge ID column name in CSV/Parquet file.',
    )

    config_group = parser.add_argument_group('Algorithm & Tile Settings')
    config_group.add_argument(
        '--tiles-dir',
        type=str,
        default=None,
        help=(
            'Directory containing 5x5 degree DEM flow-direction .npy tiles '
            '(local directory path or gs:// URI).'
        ),
    )
    config_group.add_argument(
        '--gcs-uri',
        type=str,
        default=None,
        help='Explicit GCS URI containing 5x5 degree DEM .npy tiles.',
    )
    config_group.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Explicit local cache directory when downloading from --gcs-uri.',
    )
    config_group.add_argument(
        '--snap-window',
        type=int,
        default=12,
        help='Snap search window half-width in cells (default: 12).',
    )
    config_group.add_argument(
        '--max-cells',
        type=int,
        default=None,
        help='Optional upstream cell limit (default: None, no truncation).',
    )
    config_group.add_argument(
        '--expected-area',
        type=float,
        default=None,
        help='Optional expected drainage area in km2 for single pour point.',
    )
    config_group.add_argument(
        '--area-col',
        type=str,
        default=None,
        help='Optional column name in CSV/Parquet with expected area in km2.',
    )
    config_group.add_argument(
        '--area-tolerance',
        type=float,
        default=0.50,
        help='Relative tolerance around expected area (default: 0.50).',
    )

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help='Output file path (.geojson, .geoparquet, .parquet, .shp).',
    )
    output_group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output root directory for partitioned Caravan outputs.',
    )
    output_group.add_argument(
        '--preserve-caravan-dirs',
        action='store_true',
        help=(
            'Write partitioned outputs into standard Caravan '
            'shapefiles/<subdataset>/<subdataset>_basin_shapes.* layout.'
        ),
    )
    output_group.add_argument(
        '--format',
        choices=['all', 'geoparquet', 'parquet', 'geojson', 'shp'],
        default='all',
        help='Output format(s) for --output-dir (default: all).',
    )
    output_group.add_argument(
        '--clean-cache',
        action='store_true',
        help='Delete only the tile files downloaded during this run.',
    )
    output_group.add_argument(
        '--pretty',
        action='store_true',
        help='Format output JSON with indentation.',
    )
    output_group.add_argument(
        '--list-tiles',
        action='store_true',
        help='List available DEM tiles in --tiles-dir and exit.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DEM catchment delineation CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_tiles:
        if not args.tiles_dir:
            sys.stderr.write(
                'Error: --tiles-dir must be provided with --list-tiles.\n'
            )
            return 1
        tiles = list_available_tiles(args.tiles_dir)
        sys.stdout.write(
            f'DEM Tiles Directory: {args.tiles_dir} '
            f'(Total: {len(tiles)} tiles)\n'
        )
        for tile_name in tiles:
            sys.stdout.write(f'  {tile_name}\n')
        return 0

    if args.preserve_caravan_dirs and not (args.output_dir or args.output):
        sys.stderr.write(
            'Error: --preserve-caravan-dirs requires --output-dir or -o.\n'
        )
        return 1

    coords_to_process: list[tuple[float, float]] = []
    ids_to_process: list[str | None] = []
    areas_to_process: list[float | None] = []

    if args.lat is not None and args.lon is not None:
        coords_to_process.append((args.lat, args.lon))
        ids_to_process.append(args.id)
        areas_to_process.append(args.expected_area)

    if args.coords:
        for c_str in args.coords:
            coords_to_process.append(parse_coord_str(c_str))
            ids_to_process.append(None)
            areas_to_process.append(args.expected_area)

    if args.csv:
        csv_coords, csv_ids = load_coords_from_file(
            args.csv,
            lat_col_arg=args.lat_col,
            lon_col_arg=args.lon_col,
            id_col_arg=args.id_col,
        )
        coords_to_process.extend(csv_coords)
        ids_to_process.extend(csv_ids)
        if args.area_col:
            df_area = _read_single_coord_table(
                normalize_gcs_path(args.csv)
                if is_gcs_path(args.csv)
                else str(args.csv)
            )
            if args.area_col not in df_area.columns:
                sys.stderr.write(
                    f"Error: --area-col '{args.area_col}' not found in "
                    f'columns {list(df_area.columns)}.\n'
                )
                return 1
            areas_to_process.extend(
                float(v) if pd.notna(v) else None
                for v in pd.to_numeric(df_area[args.area_col], errors='raise')
            )
        else:
            areas_to_process.extend([args.expected_area] * len(csv_coords))

    if not coords_to_process:
        parser.print_help(sys.stderr)
        sys.stderr.write(
            '\nError: No coordinates provided. Specify --lat and --lon, '
            '--coords, or --csv.\n'
        )
        return 1

    try:
        delineator = DemDelineator(
            tiles_dir=args.tiles_dir,
            gcs_uri=args.gcs_uri,
            cache_dir=args.cache_dir,
        )
    except ValueError as err:
        sys.stderr.write(f'Error: {err}\n')
        return 1

    created_cache_files: set[Path] = set()
    try:
        if len(coords_to_process) == 1 and not args.coords and not args.csv:
            lat, lon = coords_to_process[0]
            cid = ids_to_process[0]
            exp_area = areas_to_process[0]
            result = delineator.delineate(
                lat=lat,
                lon=lon,
                snap_window_cells=args.snap_window,
                max_cells=args.max_cells,
                catchment_id=cid,
                expected_area_km2=exp_area,
                area_tolerance=args.area_tolerance,
            )
            created_cache_files.update(delineator.created_cache_files)
        elif args.workers > 1 and len(coords_to_process) > 1:
            if delineator.gcs_uri is not None:
                assert delineator.cache_dir is not None
                delineator.cache_dir.mkdir(parents=True, exist_ok=True)
                needed_tile_keys: set[tuple[int, int]] = set()
                for lat, lon in coords_to_process:
                    if is_coord_in_coverage(lat, lon):
                        tk = latlon_to_tile_key(lat, lon)
                        if is_tile_in_coverage(tk[0], tk[1]):
                            needed_tile_keys.add(tk)
                missing_tiles = [
                    tk
                    for tk in sorted(needed_tile_keys)
                    if not (
                        delineator.cache_dir
                        / tile_key_to_filename(tk[0], tk[1])
                    ).is_file()
                ]
                if missing_tiles:
                    gcs_uri_str = delineator.gcs_uri
                    cache_dir_path = delineator.cache_dir

                    def _dl(tk: tuple[int, int]) -> None:
                        download_tile_from_gcs(
                            lat_top=tk[0],
                            lon_left=tk[1],
                            target_dir=cache_dir_path,
                            source_uri=gcs_uri_str,
                            created_files=created_cache_files,
                        )

                    max_threads = min(_MAX_PRECACHE_THREADS, len(missing_tiles))
                    with ThreadPoolExecutor(max_workers=max_threads) as pool:
                        list(pool.map(_dl, missing_tiles))

            tasks = [
                (
                    lat,
                    lon,
                    cid,
                    str(delineator.tiles_dir) if delineator.tiles_dir else None,
                    delineator.gcs_uri,
                    str(delineator.cache_dir) if delineator.cache_dir else None,
                    args.snap_window,
                    args.max_cells,
                    exp_area,
                    args.area_tolerance,
                )
                for (lat, lon), cid, exp_area in zip(
                    coords_to_process,
                    ids_to_process,
                    areas_to_process,
                    strict=True,
                )
            ]
            indexed_features: list[tuple[int, dict[str, Any]]] = []
            total = len(tasks)
            mp_ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp_ctx
            ) as pool:
                futures = {
                    pool.submit(_delineate_worker, t): idx
                    for idx, t in enumerate(tasks)
                }
                completed = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    completed += 1
                    step = max(1, total // _PROGRESS_INTERVAL_DIVISOR)
                    if completed % step == 0 or completed == total:
                        pct = (completed / total) * 100.0
                        sys.stderr.write(
                            f'Progress: [{completed}/{total}] catchments '
                            f'evaluated ({pct:.1f}%)\n'
                        )
                    feat_item, worker_created = fut.result()
                    created_cache_files.update(Path(p) for p in worker_created)
                    indexed_features.append((idx, feat_item))

            indexed_features.sort(key=lambda pair: pair[0])
            result = {
                'type': 'FeatureCollection',
                'features': [feat for _, feat in indexed_features],
            }
        else:
            result = delineator.delineate_batch(
                coords=coords_to_process,
                ids=ids_to_process,
                snap_window_cells=args.snap_window,
                max_cells=args.max_cells,
                expected_areas_km2=areas_to_process,
                area_tolerance=args.area_tolerance,
            )
            created_cache_files.update(delineator.created_cache_files)
    except CatchmentCoverageError as err:
        sys.stderr.write(f'\nCatchment Delineation Aborted: {err}\n')
        return 1
    finally:
        if args.clean_cache:
            delineator.created_cache_files.update(created_cache_files)
            delineator.clean_created_cache()

    _write_cli_outputs(args, result)
    return 0


def _write_cli_outputs(
    args: argparse.Namespace, result: dict[str, Any]
) -> None:
    """Write single-file, Caravan-partitioned, or stdout CLI output."""
    if args.preserve_caravan_dirs or args.output_dir:
        raw_target = args.output_dir or args.output
        if not raw_target:
            raise ValueError('An explicit --output-dir or -o path is required.')
        base_out = (
            normalize_gcs_path(raw_target).rstrip('/')
            if is_gcs_path(raw_target)
            else str(Path(raw_target).expanduser()).rstrip('/')
        )
        feats = (
            result['features']
            if result.get('type') == 'FeatureCollection'
            else [result]
        )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for feat in feats:
            sanitized = _sanitize_feature_for_export(feat)
            gid = sanitized['properties']['gauge_id']
            subdataset = (
                _extract_caravan_subdataset(gid)
                if args.preserve_caravan_dirs
                else 'catchments'
            )
            grouped.setdefault(subdataset, []).append(sanitized)

        formats = (
            ['geoparquet', 'geojson', 'shp']
            if args.format == 'all'
            else [args.format]
        )
        for subdataset, group_feats in sorted(grouped.items()):
            if args.preserve_caravan_dirs:
                if base_out.endswith('/shapefiles') or base_out == 'shapefiles':
                    target_folder = f'{base_out}/{subdataset}'
                else:
                    target_folder = f'{base_out}/shapefiles/{subdataset}'
            else:
                target_folder = base_out

            _write_partition_outputs(
                target_folder, subdataset, group_feats, formats
            )
        return

    if args.output and args.output != '-':
        out_str = (
            normalize_gcs_path(args.output)
            if is_gcs_path(args.output)
            else str(Path(args.output).expanduser())
        )
        feats = (
            result['features']
            if result.get('type') == 'FeatureCollection'
            else [result]
        )
        sanitized_features = [_sanitize_feature_for_export(f) for f in feats]
        gdf = _features_to_geodataframe(sanitized_features)

        if is_gcs_path(out_str):
            if out_str.endswith(('.parquet', '.geoparquet')):
                gdf.to_parquet(out_str)
            elif out_str.endswith('.shp'):
                shp_cols = [
                    'gauge_id',
                    'area',
                    'gauge_lat',
                    'gauge_lon',
                    'geometry',
                ]
                shp_gdf = gdf[[c for c in shp_cols if c in gdf.columns]]
                with tempfile.TemporaryDirectory() as tmpdir:
                    shp_name = Path(out_str).name
                    tmp_shp = Path(tmpdir) / shp_name
                    shp_gdf.to_file(tmp_shp, encoding='utf-8')
                    tmp_cpg = Path(tmpdir) / f'{Path(out_str).stem}.cpg'
                    if not tmp_cpg.exists():
                        tmp_cpg.write_text('UTF-8\n', encoding='utf-8')
                    gcs_parent = out_str.rsplit('/', 1)[0]
                    for shp_part in Path(tmpdir).iterdir():
                        upload_file_to_gcs(
                            shp_part, f'{gcs_parent}/{shp_part.name}'
                        )
            else:
                indent = 2 if args.pretty else None
                fc = {
                    'type': 'FeatureCollection',
                    'features': sanitized_features,
                }
                with fsspec.open(out_str, 'w', encoding='utf-8') as handle:
                    json.dump(fc, handle, indent=indent)
                    handle.write('\n')
        else:
            out_path = Path(out_str)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.suffix.lower() in ('.parquet', '.geoparquet'):
                gdf.to_parquet(out_path)
            elif out_path.suffix.lower() == '.shp':
                shp_cols = [
                    'gauge_id',
                    'area',
                    'gauge_lat',
                    'gauge_lon',
                    'geometry',
                ]
                shp_gdf = gdf[[c for c in shp_cols if c in gdf.columns]]
                shp_gdf.to_file(out_path, encoding='utf-8')
                cpg_path = out_path.with_suffix('.cpg')
                if not cpg_path.exists():
                    cpg_path.write_text('UTF-8\n', encoding='utf-8')
            else:
                indent = 2 if args.pretty else None
                fc = {
                    'type': 'FeatureCollection',
                    'features': sanitized_features,
                }
                out_path.write_text(
                    json.dumps(fc, indent=indent) + '\n', encoding='utf-8'
                )

        n_valid = sum(
            1 for f in sanitized_features if f.get('geometry') is not None
        )
        n_missing = len(sanitized_features) - n_valid
        sys.stderr.write(
            f'Delineated {n_valid}/{len(sanitized_features)} catchment(s) '
            f'({n_missing} missing/out-of-coverage) to {out_str}\n'
        )
        return

    indent = 2 if args.pretty else None
    sys.stdout.write(json.dumps(result, indent=indent) + '\n')


if __name__ == '__main__':
    sys.exit(main())
