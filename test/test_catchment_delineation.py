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

"""Unit and integration tests for DEM catchment delineation module."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from catchment_delineation import (
    RES_DEG,
    TILE_CELLS,
    CatchmentCoverageError,
    DemDelineator,
    is_tile_available,
    latlon_to_tile_key,
    list_available_tiles,
    tile_key_to_filename,
)
from catchment_delineation.benchmark import (
    compute_iou_and_metrics,
    run_benchmark,
)
from catchment_delineation.cli import (
    _sanitize_feature_for_export,
    load_coords_from_csv,
    load_coords_from_file,
    main,
    parse_coord_str,
)
from catchment_delineation.gcs import is_gcs_path, normalize_gcs_path

_SOUTH_D8: int = 4
_WEST_D8: int = 16
_EAST_D8: int = 1


def _write_synthetic_tile(
    directory: Path, filename: str = 'n40w090.npy'
) -> Path:
    """Create a synthetic 5x5 degree tile in an isolated tmp directory."""
    directory.mkdir(parents=True, exist_ok=True)
    tile_file = directory / filename
    arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    for r_c in (381, 1000, 2000):
        for c_c in (1000, 1473, 2000):
            arr[r_c - 1, c_c] = _SOUTH_D8
            arr[r_c - 2, c_c] = _SOUTH_D8
            arr[r_c - 1, c_c + 1] = _WEST_D8
            arr[r_c, c_c - 1] = _EAST_D8
    np.save(tile_file, arr)
    return tile_file


@pytest.mark.unit
def test_tile_key_and_filename() -> None:
    lat_top, lon_left = latlon_to_tile_key(39.6828, -88.7729)
    assert lat_top == 40
    assert lon_left == -90
    assert tile_key_to_filename(lat_top, lon_left) == 'n40w090.npy'

    lat_s, lon_e = latlon_to_tile_key(-12.4, 25.6)
    assert lat_s == -10
    assert lon_e == 25
    assert tile_key_to_filename(lat_s, lon_e) == 's10e025.npy'


@pytest.mark.unit
def test_parse_coord_str() -> None:
    lat, lon = parse_coord_str('39.6828,-88.7729')
    assert pytest.approx(lat, abs=1e-4) == 39.6828
    assert pytest.approx(lon, abs=1e-4) == -88.7729

    lat2, lon2 = parse_coord_str('40.5   -86.2')
    assert pytest.approx(lat2, abs=1e-4) == 40.5
    assert pytest.approx(lon2, abs=1e-4) == -86.2

    with pytest.raises(ValueError, match='Invalid coordinate format'):
        parse_coord_str('invalid_coord')


@pytest.mark.unit
def test_explicit_paths_required() -> None:
    with pytest.raises(ValueError, match='explicit tile source is required'):
        DemDelineator()

    with pytest.raises(ValueError, match='explicit cache_dir is required'):
        DemDelineator(gcs_uri='gs://test-bucket/tiles')


@pytest.mark.unit
def test_load_coords_from_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / 'test_coords.csv'
    with csv_file.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['gauge_id', 'latitude', 'longitude'])
        writer.writerow(['G1', '39.6828', '-88.7729'])
        writer.writerow(['G2', '40.4172', '-86.8858'])

    coords, ids = load_coords_from_csv(csv_file)
    assert len(coords) == 2
    assert ids == ['G1', 'G2']
    assert pytest.approx(coords[0][0], abs=1e-4) == 39.6828
    assert pytest.approx(coords[0][1], abs=1e-4) == -88.7729


@pytest.mark.unit
def test_missing_coordinate_file_raises_not_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing local coordinate files must raise FileNotFoundError."""
    monkeypatch.chdir(tmp_path)
    for missing_name in ('coordinates.csv', 'caravan', 'caravan_coordinates'):
        with pytest.raises(FileNotFoundError, match='does not exist'):
            load_coords_from_file(missing_name)


@pytest.mark.unit
def test_strict_column_resolution(tmp_path: Path) -> None:
    """Column detection must not match substrings like 'y' in survey_year."""
    csv_file = tmp_path / 'survey.csv'
    pd.DataFrame(
        {
            'survey_year': [1999.0],
            'max_flow': [500.0],
            'latitude': [39.6828],
            'longitude': [-88.7729],
        }
    ).to_csv(csv_file, index=False)

    coords, _ = load_coords_from_file(csv_file)
    assert pytest.approx(coords[0][0], abs=1e-4) == 39.6828
    assert pytest.approx(coords[0][1], abs=1e-4) == -88.7729

    bad_csv = tmp_path / 'bbox.csv'
    pd.DataFrame({'xmin': [1.0], 'ymin': [2.0]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match='explicit lat column'):
        load_coords_from_file(bad_csv)

    ambig_csv = tmp_path / 'ambig.csv'
    pd.DataFrame(
        {'lat': [10.0], 'latitude': [39.6828], 'longitude': [-88.7729]}
    ).to_csv(ambig_csv, index=False)
    with pytest.raises(ValueError, match='Ambiguous lat columns'):
        load_coords_from_file(ambig_csv)


@pytest.mark.unit
def test_load_coords_crawls_caravan_attributes_dir(tmp_path: Path) -> None:
    """When given a directory, crawls standard Caravan attributes_other_*.csv."""
    attr_dir = tmp_path / 'attributes' / 'camels'
    attr_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            'gauge_id': ['camels_01013500'],
            'gauge_lat': [39.6828],
            'gauge_lon': [-88.7729],
        }
    ).to_csv(attr_dir / 'attributes_other_camels.csv', index=False)

    coords, ids = load_coords_from_file(tmp_path)
    assert ids == ['camels_01013500']
    assert pytest.approx(coords[0][0], abs=1e-4) == 39.6828


@pytest.mark.unit
def test_synthetic_dem_delineator(tmp_path: Path) -> None:
    tile_arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    tile_arr[99, 100] = _SOUTH_D8
    tile_arr[98, 100] = _SOUTH_D8
    tile_arr[99, 101] = _WEST_D8
    np.save(tmp_path / 'n40w090.npy', tile_arr)

    delineator = DemDelineator(tiles_dir=tmp_path)
    out_lat = 40.0 - 100.0 * RES_DEG
    out_lon = -90.0 + 100.0 * RES_DEG

    res = delineator.delineate(lat=out_lat, lon=out_lon, snap_window_cells=1)
    assert res['type'] == 'Feature'
    props = res['properties']
    assert props['upstream_cells_count'] == 4
    assert props['area_km2'] > 0
    assert res['geometry']['type'] in ('Polygon', 'MultiPolygon')
    assert is_tile_available(40, -90, tmp_path)
    assert list_available_tiles(tmp_path) == ['n40w090.npy']


@pytest.mark.unit
def test_cross_tile_delineation_watertight(tmp_path: Path) -> None:
    """Verify seamless watershed traversal across a 5x5 degree tile seam."""
    south = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    north = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    col = 1200
    for row in (0, 1, 2):
        south[row, col] = _SOUTH_D8
    for row in (TILE_CELLS - 1, TILE_CELLS - 2, TILE_CELLS - 3):
        north[row, col] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', south)
    np.save(tmp_path / 'n45w090.npy', north)

    delineator = DemDelineator(tiles_dir=tmp_path)
    lat = 40.0 - 3 * RES_DEG
    lon = -90.0 + col * RES_DEG
    feat = delineator.delineate(lat=lat, lon=lon, snap_window_cells=0)
    assert feat['properties']['upstream_cells_count'] == 7
    assert feat['properties']['tiles_spanned_count'] == 2


@pytest.mark.unit
def test_missing_tile_aborts_in_both_local_and_gcs_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing or failed neighbor tiles must abort in BOTH local and GCS modes."""
    import catchment_delineation.delineator as cd_del

    south = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    col = 1200
    for row in (0, 1, 2):
        south[row, col] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', south)

    lat = 40.0 - 3 * RES_DEG
    lon = -90.0 + col * RES_DEG

    local_delin = DemDelineator(tiles_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match='n45w090.npy'):
        local_delin.delineate(lat=lat, lon=lon, snap_window_cells=0)

    def _fail_download(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError('Simulated GCS 503 failure')

    monkeypatch.setattr(cd_del, 'download_tile_from_gcs', _fail_download)
    gcs_delin = DemDelineator(
        gcs_uri='gs://test-bucket/tiles', cache_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match='Simulated GCS 503 failure'):
        gcs_delin.delineate(lat=lat, lon=lon, snap_window_cells=0)


@pytest.mark.unit
def test_max_cells_raises_never_truncates(tmp_path: Path) -> None:
    """Exceeding an explicit max_cells cap must raise CatchmentCoverageError."""
    tile_arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    for row in range(90, 100):
        tile_arr[row, 100] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', tile_arr)

    delineator = DemDelineator(tiles_dir=tmp_path)
    lat = 40.0 - 100 * RES_DEG
    lon = -90.0 + 100 * RES_DEG

    full_feat = delineator.delineate(lat=lat, lon=lon, snap_window_cells=0)
    assert full_feat['properties']['upstream_cells_count'] == 11

    with pytest.raises(CatchmentCoverageError, match='exceeded max_cells=5'):
        delineator.delineate(lat=lat, lon=lon, snap_window_cells=0, max_cells=5)


@pytest.mark.unit
def test_delineate_batch_missing_in_means_nan_out(tmp_path: Path) -> None:
    """Out-of-coverage or NaN coordinates in a batch produce NaN/None records."""
    _write_synthetic_tile(tmp_path)
    delineator = DemDelineator(tiles_dir=tmp_path)
    coords = [(39.6828, -88.7729), (65.0, -150.0), (float('nan'), -88.0)]
    ids = ['valid_1', 'ooc_north', 'nan_coord']
    fc = delineator.delineate_batch(coords=coords, ids=ids)

    assert len(fc['features']) == 3
    assert fc['features'][0]['geometry'] is not None
    assert fc['features'][0]['properties']['status'] == 'SUCCESS'

    for missing_feat in fc['features'][1:]:
        assert missing_feat['geometry'] is None
        assert math.isnan(missing_feat['properties']['area_km2'])
        assert missing_feat['properties']['status'].startswith('MISSING_DATA')


@pytest.mark.unit
def test_sanitize_feature_rejects_missing_fields_no_zero_fill() -> None:
    """Missing area or outlet coordinates must raise KeyError, never fill 0.0."""
    incomplete_feat = {
        'type': 'Feature',
        'properties': {'catchment_id': 'g1'},
        'geometry': {'type': 'Polygon', 'coordinates': []},
    }
    with pytest.raises(KeyError, match='missing required area'):
        _sanitize_feature_for_export(incomplete_feat)


@pytest.mark.unit
def test_clean_cache_only_removes_newly_downloaded_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--clean-cache must never delete user tiles_dir or pre-existing files."""
    import catchment_delineation.delineator as cd_del

    user_tiles_dir = tmp_path / 'user_tiles'
    _write_synthetic_tile(user_tiles_dir)
    user_readme = user_tiles_dir / 'USER_NOTES.txt'
    user_readme.write_text('do not delete', encoding='utf-8')

    out_json = tmp_path / 'out_local.geojson'
    rc = main(
        [
            '--lat',
            '39.6828',
            '--lon',
            '-88.7729',
            '--tiles-dir',
            str(user_tiles_dir),
            '--clean-cache',
            '-o',
            str(out_json),
        ]
    )
    assert rc == 0
    assert user_tiles_dir.is_dir()
    assert user_readme.is_file()
    assert (user_tiles_dir / 'n40w090.npy').is_file()

    cache_dir = tmp_path / 'gcs_cache'
    cache_dir.mkdir()
    preexisting = cache_dir / 'keep_me.txt'
    preexisting.write_text('keep', encoding='utf-8')

    def _mock_dl(
        lat_top: int,
        lon_left: int,
        target_dir: str | Path,
        source_uri: str,
        *,
        created_files: set[Path] | None = None,
    ) -> Path:
        del source_uri
        fname = tile_key_to_filename(lat_top, lon_left)
        path = _write_synthetic_tile(Path(target_dir), fname)
        if created_files is not None:
            created_files.add(path)
        return path

    monkeypatch.setattr(cd_del, 'download_tile_from_gcs', _mock_dl)
    rc2 = main(
        [
            '--lat',
            '39.6828',
            '--lon',
            '-88.7729',
            '--gcs-uri',
            'gs://test-bucket/tiles',
            '--cache-dir',
            str(cache_dir),
            '--clean-cache',
            '-o',
            str(tmp_path / 'out_gcs.geojson'),
        ]
    )
    assert rc2 == 0
    assert preexisting.is_file()
    assert not (cache_dir / 'n40w090.npy').exists()


@pytest.mark.unit
def test_cli_preserve_caravan_dirs(tmp_path: Path) -> None:
    """Verify --preserve-caravan-dirs outputs standard shapefiles/<subdataset>."""
    tiles_dir = tmp_path / 'tiles'
    _write_synthetic_tile(tiles_dir)

    csv_file = tmp_path / 'caravan_multi.csv'
    pd.DataFrame(
        {
            'gauge_id': ['camels_01013500', 'grdc_6340110'],
            'gauge_lat': [39.6828, 38.3333],
            'gauge_lon': [-88.7729, -86.8858],
        }
    ).to_csv(csv_file, index=False)

    out_dir = tmp_path / 'caravan_root'
    rc = main(
        [
            '--csv',
            str(csv_file),
            '--tiles-dir',
            str(tiles_dir),
            '--output-dir',
            str(out_dir),
            '--preserve-caravan-dirs',
            '--format',
            'geoparquet',
        ]
    )
    assert rc == 0
    camels_pq = (
        out_dir / 'shapefiles' / 'camels' / 'camels_basin_shapes.geoparquet'
    )
    grdc_pq = out_dir / 'shapefiles' / 'grdc' / 'grdc_basin_shapes.geoparquet'
    assert camels_pq.is_file()
    assert grdc_pq.is_file()
    gdf = gpd.read_parquet(camels_pq)
    assert list(gdf['gauge_id']) == ['camels_01013500']


@pytest.mark.unit
def test_benchmark_metrics_return_nan_not_zero(tmp_path: Path) -> None:
    """Metric calculation failures or zero ref areas must return NaN, not 0.0."""
    poly = box(0, 0, 1, 1)
    iou, dice, bias, abs_err = compute_iou_and_metrics(poly, poly, 0.0, 100.0)
    assert math.isnan(iou)
    assert math.isnan(dice)
    assert math.isnan(bias)
    assert math.isnan(abs_err)

    tiles_dir = tmp_path / 'tiles'
    _write_synthetic_tile(tiles_dir)
    bench_pq = tmp_path / 'bench.parquet'
    pd.DataFrame(
        {
            'gauge_id': ['valid_1', 'ooc_1'],
            'continent': ['North America', 'North America'],
            'hemisphere': ['NW', 'NW'],
            'size_tier': ['1_micro', '1_micro'],
            'latitude': [39.6828, 65.0],
            'longitude': [-88.7729, -150.0],
            'reference_area_km2': [0.03, 100.0],
            'geometry_wkt': [
                box(-88.78, 39.68, -88.77, 39.69).wkt,
                box(-150.1, 64.9, -149.9, 65.1).wkt,
            ],
        }
    ).to_parquet(bench_pq, index=False)

    res_df = run_benchmark(
        dataset_path=bench_pq, tiles_dir=tiles_dir, workers=1
    )
    ooc_row = res_df[res_df['gauge_id'] == 'ooc_1'].iloc[0]
    assert math.isnan(ooc_row['iou'])
    assert math.isnan(ooc_row['area_bias_pct'])


@pytest.mark.unit
def test_gcs_helpers() -> None:
    assert is_gcs_path('gs://bucket/path')
    assert is_gcs_path('gcs://bucket/path')
    assert not is_gcs_path('/local/path')
    assert normalize_gcs_path('gs:/bucket/path') == 'gs://bucket/path'
    with pytest.raises(ValueError, match='Cannot normalize'):
        normalize_gcs_path('')
