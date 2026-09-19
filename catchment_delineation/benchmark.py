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

"""Global Catchment Delineation Benchmark Suite.

Evaluates DEM flow-direction catchment delineation against user-supplied
reference watershed polygons across continents, hemisphere quadrants, and
drainage area tiers.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

import pandas as pd
import shapely.wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from catchment_delineation.delineator import (
    CatchmentCoverageError,
    DemDelineator,
)
from catchment_delineation.gcs import (
    download_tile_from_gcs,
    is_gcs_path,
    normalize_gcs_path,
)
from catchment_delineation.tiles import (
    is_coord_in_coverage,
    is_tile_in_coverage,
    latlon_to_tile_key,
    tile_key_to_filename,
)

logger = logging.getLogger('catchment_delineation.benchmark')

_MAX_PRECACHE_THREADS: int = 16
_PROGRESS_INTERVAL: int = 50
_IOU_THRESHOLD_80: float = 0.80
_IOU_THRESHOLD_90: float = 0.90


def compute_iou_and_metrics(
    del_geom: BaseGeometry | None,
    ref_geom: BaseGeometry | None,
    ref_area_km2: float,
    del_area_km2: float,
) -> tuple[float, float, float, float]:
    """Compute spatial overlap and area error metrics without 0.0 fallbacks."""
    nan_val = float('nan')
    if (
        del_geom is None
        or ref_geom is None
        or del_geom.is_empty
        or ref_geom.is_empty
        or not math.isfinite(ref_area_km2)
        or not math.isfinite(del_area_km2)
        or ref_area_km2 <= 0.0
    ):
        return nan_val, nan_val, nan_val, nan_val

    if not del_geom.is_valid:
        del_geom = del_geom.buffer(0)
    if not ref_geom.is_valid:
        ref_geom = ref_geom.buffer(0)

    intersection = float(del_geom.intersection(ref_geom).area)
    union = float(del_geom.union(ref_geom).area)
    denom = float(del_geom.area + ref_geom.area)
    if union <= 0.0 or denom <= 0.0:
        return nan_val, nan_val, nan_val, nan_val

    iou = float(intersection / union)
    dice = float((2.0 * intersection) / denom)
    area_rel_diff_pct = float(
        (del_area_km2 - ref_area_km2) / ref_area_km2 * 100.0
    )
    abs_area_err_pct = abs(area_rel_diff_pct)
    return iou, dice, area_rel_diff_pct, abs_area_err_pct


def _evaluate_single_basin(
    row_dict: dict[str, Any],
    tiles_dir: str | None = None,
    gcs_uri: str | None = None,
    cache_dir: str | None = None,
    snap_window_cells: int = 12,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate a single reference basin in a worker process."""
    gauge_id = str(row_dict['gauge_id'])
    lat = float(row_dict['latitude'])
    lon = float(row_dict['longitude'])
    ref_area_km2 = float(row_dict['reference_area_km2'])
    ref_wkt = str(row_dict['geometry_wkt'])
    continent = row_dict['continent']
    hemisphere = row_dict['hemisphere']
    size_tier = row_dict['size_tier']

    delineator = DemDelineator(
        tiles_dir=tiles_dir,
        gcs_uri=gcs_uri,
        cache_dir=cache_dir,
        cache_tiles=True,
    )
    t0 = time.time()
    try:
        res = delineator.delineate(
            lat=lat,
            lon=lon,
            catchment_id=gauge_id,
            snap_window_cells=snap_window_cells,
        )
        elapsed = time.time() - t0
        props = res['properties']
        del_area_km2 = float(props['area_km2'])
        tiles_spanned = float(props['tiles_spanned_count'])
        snap_dist_m = float(props['outlet']['snap_distance_m'])

        del_geom = shape(res['geometry'])
        ref_geom = shapely.wkt.loads(ref_wkt)
        iou, dice, area_rel_diff, abs_area_err = compute_iou_and_metrics(
            del_geom, ref_geom, ref_area_km2, del_area_km2
        )
        record = {
            'gauge_id': gauge_id,
            'continent': continent,
            'hemisphere': hemisphere,
            'size_tier': size_tier,
            'latitude': lat,
            'longitude': lon,
            'ref_area_km2': ref_area_km2,
            'del_area_km2': del_area_km2,
            'iou': round(iou, 4) if math.isfinite(iou) else float('nan'),
            'dice': round(dice, 4) if math.isfinite(dice) else float('nan'),
            'area_bias_pct': (
                round(area_rel_diff, 2)
                if math.isfinite(area_rel_diff)
                else float('nan')
            ),
            'abs_area_err_pct': (
                round(abs_area_err, 2)
                if math.isfinite(abs_area_err)
                else float('nan')
            ),
            'snap_dist_m': round(snap_dist_m, 1),
            'tiles_spanned': tiles_spanned,
            'elapsed_sec': round(elapsed, 3),
            'status': 'SUCCESS',
        }
    except CatchmentCoverageError as err:
        elapsed = time.time() - t0
        nan_val = float('nan')
        record = {
            'gauge_id': gauge_id,
            'continent': continent,
            'hemisphere': hemisphere,
            'size_tier': size_tier,
            'latitude': lat,
            'longitude': lon,
            'ref_area_km2': ref_area_km2,
            'del_area_km2': nan_val,
            'iou': nan_val,
            'dice': nan_val,
            'area_bias_pct': nan_val,
            'abs_area_err_pct': nan_val,
            'snap_dist_m': nan_val,
            'tiles_spanned': nan_val,
            'elapsed_sec': round(elapsed, 3),
            'status': f'OUT_OF_COVERAGE: {err}',
        }

    created = [str(p) for p in delineator.created_cache_files]
    return record, created


def print_summary_table(
    df: pd.DataFrame, group_title: str, group_col: str
) -> None:
    """Print formatted summary metrics grouped by a column."""
    if group_col not in df.columns or df.empty:
        return
    sys.stdout.write(f'\n{group_title}\n' + '-' * 80 + '\n')
    for group_val, grp_df in df.groupby(group_col):
        valid_iou = grp_df['iou'].dropna()
        med_iou = (
            float(valid_iou.median()) if not valid_iou.empty else float('nan')
        )
        pct_80 = (
            float((valid_iou >= _IOU_THRESHOLD_80).mean() * 100.0)
            if not valid_iou.empty
            else float('nan')
        )
        sys.stdout.write(
            f'  {group_val!s:20s} | n={len(grp_df):4d} '
            f'(valid={len(valid_iou):4d}) | '
            f'median IoU={med_iou:.3f} | IoU>=0.80: {pct_80:.1f}%\n'
        )


def _load_benchmark_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load the user-supplied benchmark dataset without fallback paths."""
    if not dataset_path:
        raise ValueError(
            'An explicit dataset_path (.parquet or gs:// URI) is required.'
        )
    if is_gcs_path(dataset_path):
        return pd.read_parquet(normalize_gcs_path(dataset_path))

    local_p = Path(dataset_path).expanduser().resolve()
    if not local_p.is_file():
        raise FileNotFoundError(
            f'Benchmark dataset file does not exist: {local_p}'
        )
    return pd.read_parquet(local_p)


def run_benchmark(
    dataset_path: str | Path,
    tiles_dir: str | Path | None = None,
    *,
    gcs_uri: str | None = None,
    cache_dir: str | Path | None = None,
    samples: int | None = None,
    continents: list[str] | None = None,
    size_tiers: list[str] | None = None,
    workers: int = 8,
    output_path: str | Path | None = None,
    snap_window_cells: int = 12,
    clean_cache: bool = False,
) -> pd.DataFrame:
    """Execute catchment delineation benchmark on an explicit dataset."""
    delineator = DemDelineator(
        tiles_dir=tiles_dir, gcs_uri=gcs_uri, cache_dir=cache_dir
    )
    df = _load_benchmark_dataset(dataset_path)

    if continents:
        df = df[df['continent'].isin(continents)]
    if size_tiers:
        df = df[df['size_tier'].isin(size_tiers)]

    if samples and samples < len(df):
        sampled_dfs = []
        for _, grp in df.groupby(['continent', 'size_tier']):
            n_take = max(1, int(len(grp) * samples / len(df)))
            sampled_dfs.append(
                grp.sample(n=min(len(grp), n_take), random_state=42)
            )
        df = pd.concat(sampled_dfs, ignore_index=True)
        if len(df) > samples:
            df = df.sample(n=samples, random_state=42)

    created_cache_files: set[Path] = set()
    try:
        if delineator.gcs_uri is not None:
            assert delineator.cache_dir is not None
            delineator.cache_dir.mkdir(parents=True, exist_ok=True)
            needed_tile_keys: set[tuple[int, int]] = set()
            for row in df.itertuples():
                lat = float(row.latitude)
                lon = float(row.longitude)
                if is_coord_in_coverage(lat, lon):
                    tk = latlon_to_tile_key(lat, lon)
                    if is_tile_in_coverage(tk[0], tk[1]):
                        needed_tile_keys.add(tk)

            missing_tiles = [
                tk
                for tk in sorted(needed_tile_keys)
                if not (
                    delineator.cache_dir / tile_key_to_filename(tk[0], tk[1])
                ).is_file()
            ]
            if missing_tiles:
                cache_dir_path = delineator.cache_dir
                gcs_uri_str = delineator.gcs_uri

                def _download_one(tk: tuple[int, int]) -> None:
                    download_tile_from_gcs(
                        lat_top=tk[0],
                        lon_left=tk[1],
                        target_dir=cache_dir_path,
                        source_uri=gcs_uri_str,
                        created_files=created_cache_files,
                    )

                max_threads = min(_MAX_PRECACHE_THREADS, len(missing_tiles))
                with ThreadPoolExecutor(max_workers=max_threads) as pool:
                    list(pool.map(_download_one, missing_tiles))

        rows = df.to_dict(orient='records')
        indexed_results: list[tuple[int, dict[str, Any]]] = []
        t_start = time.time()
        mp_ctx = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp_ctx
        ) as executor:
            futures = {
                executor.submit(
                    _evaluate_single_basin,
                    row,
                    tiles_dir=(
                        str(delineator.tiles_dir)
                        if delineator.tiles_dir
                        else None
                    ),
                    gcs_uri=delineator.gcs_uri,
                    cache_dir=(
                        str(delineator.cache_dir)
                        if delineator.cache_dir
                        else None
                    ),
                    snap_window_cells=snap_window_cells,
                ): idx
                for idx, row in enumerate(rows)
            }
            done_count = 0
            total = len(futures)
            for fut in as_completed(futures):
                idx = futures[fut]
                res_item, worker_created = fut.result()
                created_cache_files.update(Path(p) for p in worker_created)
                indexed_results.append((idx, res_item))
                done_count += 1
                if done_count % _PROGRESS_INTERVAL == 0 or done_count == total:
                    pct = (done_count / total) * 100.0
                    sys.stdout.write(
                        f'Progress: [{done_count}/{total}] basins evaluated '
                        f'({pct:.1f}%)\n'
                    )

        indexed_results.sort(key=lambda pair: pair[0])
        res_df = pd.DataFrame([r for _, r in indexed_results])
        total_time = time.time() - t_start

        valid_df = res_df[res_df['status'] == 'SUCCESS']
        out_of_coverage = int(
            res_df['status'].str.startswith('OUT_OF_COVERAGE').sum()
        )
        sys.stdout.write('\n' + '=' * 80 + '\n')
        sys.stdout.write('GLOBAL CATCHMENT DELINEATION BENCHMARK RESULTS\n')
        sys.stdout.write('=' * 80 + '\n')
        sys.stdout.write(f'Total Basins Evaluated : {len(res_df)}\n')
        sys.stdout.write(f'Total Wall-Clock Time  : {total_time:.1f}s\n')
        sys.stdout.write(
            f'Successful Delineations: {len(valid_df)} / {len(res_df)}\n'
        )
        if out_of_coverage > 0:
            sys.stdout.write(
                f'Out-of-Coverage Basins : {out_of_coverage} / {len(res_df)}\n'
            )
        if not valid_df.empty:
            sys.stdout.write(
                f'Median IoU (valid)     : {valid_df["iou"].median():.3f}\n'
            )
            sys.stdout.write(
                f'Median Dice (valid)    : {valid_df["dice"].median():.3f}\n'
            )
            pct80 = (valid_df['iou'] >= _IOU_THRESHOLD_80).mean() * 100.0
            pct90 = (valid_df['iou'] >= _IOU_THRESHOLD_90).mean() * 100.0
            sys.stdout.write(f'Basins IoU >= 0.80     : {pct80:.1f}%\n')
            sys.stdout.write(f'Basins IoU >= 0.90     : {pct90:.1f}%\n')
            sys.stdout.write(
                'Median Abs Area Err    : '
                f'{valid_df["abs_area_err_pct"].median():.1f}%\n'
            )
        print_summary_table(res_df, 'PERFORMANCE BY CONTINENT', 'continent')
        print_summary_table(
            res_df, 'PERFORMANCE BY HEMISPHERE QUADRANT', 'hemisphere'
        )
        print_summary_table(
            res_df, 'PERFORMANCE BY BASIN SIZE TIER', 'size_tier'
        )
        sys.stdout.write('=' * 80 + '\n')

        if output_path:
            out_p = Path(output_path).expanduser().resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if str(out_p).endswith('.parquet'):
                res_df.to_parquet(out_p, index=False)
            else:
                res_df.to_csv(out_p, index=False)

        return res_df
    finally:
        if clean_cache:
            delineator.created_cache_files.update(created_cache_files)
            delineator.clean_created_cache()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the catchment delineation benchmark."""
    os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(
        description=(
            'Run global catchment delineation benchmark. '
            'All dataset and tile paths must be supplied explicitly.'
        )
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Explicit path to benchmark reference dataset (.parquet or gs://).',
    )
    parser.add_argument(
        '--tiles-dir',
        type=str,
        default=None,
        help='Explicit local directory containing DEM .npy tiles (or gs://).',
    )
    parser.add_argument(
        '--gcs-uri',
        type=str,
        default=None,
        help='Explicit GCS URI containing DEM .npy tiles.',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Explicit local cache directory when --gcs-uri is used.',
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Optional stratified subsample size.',
    )
    parser.add_argument(
        '--continents',
        nargs='+',
        default=None,
        help='Filter by specific continents.',
    )
    parser.add_argument(
        '--size-tiers',
        nargs='+',
        default=None,
        help='Filter by specific size tiers.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker processes.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output file path (.csv or .parquet) for results.',
    )
    parser.add_argument(
        '--snap-window',
        type=int,
        default=12,
        help='Snap search window half-width in cells.',
    )
    parser.add_argument(
        '--clean-cache',
        action='store_true',
        help='Delete only the tile files downloaded during this benchmark run.',
    )
    args = parser.parse_args(argv)
    run_benchmark(
        dataset_path=args.dataset,
        tiles_dir=args.tiles_dir,
        gcs_uri=args.gcs_uri,
        cache_dir=args.cache_dir,
        samples=args.samples,
        continents=args.continents,
        size_tiers=args.size_tiers,
        workers=args.workers,
        output_path=args.output,
        snap_window_cells=args.snap_window,
        clean_cache=args.clean_cache,
    )
    return 0


if __name__ == '__main__':
    with contextlib.suppress(SystemExit):
        sys.exit(main())
