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

"""Unit and end-to-end tests for NetCDF input loading."""

import warnings
from pathlib import Path, PureWindowsPath
from typing import Any

import dask
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from googlehydrology.datasetzoo import caravan
from googlehydrology.datasetzoo.multimet import (
    Multimet,
    MultimetDataLoader,
    _is_zarr_store,
)
from googlehydrology.datautils.netcdf import (
    _is_cloud_path,
    find_product_netcdf_dir,
    has_netcdf_files,
    load_netcdf_attributes,
    load_netcdf_timeseries,
)
from googlehydrology.run import eval_run, start_run
from googlehydrology.utils.config import Config

_EXPECTED_UNION_DATES = 7
_EXPECTED_LEADS = 2
_EXPECTED_COMBINED_META_DATES = 6


def _make_synthetic_dataset(
    basins: list[str],
    dates: pd.DatetimeIndex,
    lead_days: list[int],
    *,
    dtype: type = np.float64,
    seed: int = 123,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    n_b = len(basins)
    n_d = len(dates)
    n_l = len(lead_days)
    leads = pd.to_timedelta(lead_days, unit='D')
    return xr.Dataset(
        {
            'area': (('basin',), rng.uniform(50.0, 500.0, n_b).astype(dtype)),
            'p_mean': (('basin',), rng.uniform(1.0, 5.0, n_b).astype(dtype)),
            'streamflow': (
                ('basin', 'date'),
                rng.uniform(0.5, 20.0, (n_b, n_d)).astype(dtype),
            ),
            'era5land_precip': (
                ('basin', 'date'),
                rng.uniform(0.0, 15.0, (n_b, n_d)).astype(dtype),
            ),
            'era5land_temp': (
                ('basin', 'date'),
                rng.uniform(-5.0, 30.0, (n_b, n_d)).astype(dtype),
            ),
            'hres_precip': (
                ('basin', 'date', 'lead_time'),
                rng.uniform(0.0, 15.0, (n_b, n_d, n_l)).astype(dtype),
            ),
        },
        coords={
            'basin': basins,
            'date': dates,
            'lead_time': leads,
        },
    )


def _write_zarr_layout(root: Path, ds: xr.Dataset) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    statics_path = root / 'attributes.zarr'
    targets_path = root / 'streamflow.zarr'
    dynamics_root = root / 'dynamics'
    era5_path = dynamics_root / 'ERA5_LAND' / 'timeseries.zarr'
    hres_path = dynamics_root / 'HRES' / 'timeseries.zarr'
    era5_path.parent.mkdir(parents=True, exist_ok=True)
    hres_path.parent.mkdir(parents=True, exist_ok=True)

    ds32 = ds.astype(np.float32)
    ds32[['area', 'p_mean']].to_zarr(statics_path, consolidated=False)
    ds32[['streamflow']].to_zarr(targets_path, consolidated=False)
    ds32[['era5land_precip', 'era5land_temp']].to_zarr(
        era5_path, consolidated=False
    )
    ds32[['hres_precip']].to_zarr(hres_path, consolidated=False)
    return {
        'statics': statics_path,
        'targets': targets_path,
        'dynamics': dynamics_root,
    }


def _write_multibasin_netcdf_layout(
    root: Path, ds: xr.Dataset, *, product_subdirs: bool
) -> dict[str, Path]:
    statics_dir = root / 'statics'
    targets_dir = root / 'targets'
    dynamics_dir = root / 'dynamics'
    statics_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    dynamics_dir.mkdir(parents=True, exist_ok=True)

    mid_b = max(1, ds.sizes['basin'] // 2)
    ds[['area', 'p_mean']].isel(basin=slice(0, mid_b)).to_netcdf(
        statics_dir / 'attrs_part1.nc'
    )
    ds[['area', 'p_mean']].isel(basin=slice(mid_b, None)).to_netcdf(
        statics_dir / 'attrs_part2.nc'
    )

    mid_d = ds.sizes['date'] // 2
    ds[['streamflow']].isel(date=slice(0, mid_d)).to_netcdf(
        targets_dir / 'streamflow_part1.nc'
    )
    ds[['streamflow']].isel(date=slice(mid_d, None)).to_netcdf(
        targets_dir / 'streamflow_part2.nc'
    )

    if product_subdirs:
        era5_dir = dynamics_dir / 'ERA5_LAND'
        hres_dir = dynamics_dir / 'HRES'
        era5_dir.mkdir(parents=True, exist_ok=True)
        hres_dir.mkdir(parents=True, exist_ok=True)
        ds[['era5land_precip', 'era5land_temp']].isel(
            date=slice(0, mid_d)
        ).to_netcdf(era5_dir / 'era5_part1.nc')
        ds[['era5land_precip', 'era5land_temp']].isel(
            date=slice(mid_d, None)
        ).to_netcdf(era5_dir / 'era5_part2.nc')
        ds[['hres_precip']].isel(date=slice(0, mid_d)).to_netcdf(
            hres_dir / 'hres_part1.nc'
        )
        ds[['hres_precip']].isel(date=slice(mid_d, None)).to_netcdf(
            hres_dir / 'hres_part2.nc'
        )
    else:
        ds[['era5land_precip', 'era5land_temp']].isel(
            date=slice(0, mid_d)
        ).to_netcdf(dynamics_dir / 'era5_part1.nc')
        ds[['era5land_precip', 'era5land_temp']].isel(
            date=slice(mid_d, None)
        ).to_netcdf(dynamics_dir / 'era5_part2.nc')
        ds[['hres_precip']].isel(date=slice(0, mid_d)).to_netcdf(
            dynamics_dir / 'hres_part1.nc'
        )
        ds[['hres_precip']].isel(date=slice(mid_d, None)).to_netcdf(
            dynamics_dir / 'hres_part2.nc'
        )

    return {
        'statics': statics_dir,
        'targets': targets_dir,
        'dynamics': dynamics_dir,
    }


def _write_per_basin_netcdf_layout(
    root: Path, ds: xr.Dataset, *, product_subdirs: bool
) -> dict[str, Path]:
    statics_dir = root / 'statics'
    targets_dir = root / 'targets'
    dynamics_dir = root / 'dynamics'
    statics_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    dynamics_dir.mkdir(parents=True, exist_ok=True)

    ds[['area', 'p_mean']].to_netcdf(statics_dir / 'attributes.nc')

    for basin in ds.coords['basin'].to_numpy().tolist():
        b_ds = ds.sel(basin=basin, drop=True)
        b_ds[['streamflow']].to_netcdf(targets_dir / f'{basin}.nc')
        if product_subdirs:
            era5_dir = dynamics_dir / 'ERA5_LAND'
            hres_dir = dynamics_dir / 'HRES'
            era5_dir.mkdir(parents=True, exist_ok=True)
            hres_dir.mkdir(parents=True, exist_ok=True)
            b_ds[['era5land_precip', 'era5land_temp']].to_netcdf(
                era5_dir / f'{basin}.nc'
            )
            b_ds[['hres_precip']].to_netcdf(hres_dir / f'{basin}.nc')
        else:
            b_ds[['era5land_precip', 'era5land_temp', 'hres_precip']].to_netcdf(
                dynamics_dir / f'{basin}.nc'
            )

    return {
        'statics': statics_dir,
        'targets': targets_dir,
        'dynamics': dynamics_dir,
    }


def _make_multimet_config(
    tmp_path: Path,
    name: str,
    paths: dict[str, Path],
    basins: list[str],
    overrides: dict[str, Any] | None = None,
) -> Config:
    basin_file = tmp_path / f'{name}_basins.txt'
    basin_file.write_text('\n'.join(basins) + '\n', encoding='utf-8')
    run_dir = tmp_path / f'{name}_run'
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict: dict[str, Any] = {
        'experiment_name': name,
        'run_dir': str(run_dir),
        'dataset': 'multimet',
        'train_basin_file': str(basin_file),
        'validation_basin_file': str(basin_file),
        'test_basin_file': str(basin_file),
        'statics_data_dir': str(paths['statics']),
        'targets_data_dir': str(paths['targets']),
        'dynamics_data_dir': str(paths['dynamics']),
        'static_attributes': ['area', 'p_mean'],
        'target_variables': ['streamflow'],
        'hindcast_inputs': ['era5land_precip', 'era5land_temp'],
        'forecast_inputs': ['hres_precip'],
        'train_start_date': '10/01/2000',
        'train_end_date': '20/01/2000',
        'validation_start_date': '10/01/2000',
        'validation_end_date': '20/01/2000',
        'test_start_date': '10/01/2000',
        'test_end_date': '20/01/2000',
        'seq_length': 4,
        'lead_time': 2,
        'forecast_overlap': 1,
        'predict_last_n': 2,
        'nan_handling_method': 'none',
        'timestep_counter': False,
        'loss': 'MSE',
        'lazy_load': False,
        'device': 'cpu',
    }
    if overrides:
        cfg_dict.update(overrides)
    return Config(cfg_dict)


@pytest.mark.unit
def test_discovery_helpers(tmp_path: Path) -> None:
    """Verify NetCDF directory and product subdirectory discovery."""
    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    assert not has_netcdf_files(empty_dir)
    assert not has_netcdf_files('gs://bucket/path')

    nc_dir = tmp_path / 'ERA5_LAND'
    nc_dir.mkdir()
    xr.Dataset({'v': (('basin',), [1.0])}, coords={'basin': ['b1']}).to_netcdf(
        nc_dir / 'data.nc4'
    )
    assert has_netcdf_files(nc_dir)
    assert find_product_netcdf_dir(tmp_path, 'ERA5_LAND') == nc_dir
    assert find_product_netcdf_dir(tmp_path, 'era5land') == nc_dir
    assert find_product_netcdf_dir(tmp_path, 'MISSING') is None


@pytest.mark.unit
def test_load_netcdf_attributes_multibasin_and_float64_cast(
    tmp_path: Path,
) -> None:
    """Load basin-split static attributes and lazily cast float64 to float32."""
    attr_dir = tmp_path / 'attrs'
    attr_dir.mkdir()
    xr.Dataset(
        {
            'area': (('basin',), np.array([10.5, 20.5], dtype=np.float64)),
            'p_mean': (('basin',), np.array([2.1, 3.2], dtype=np.float64)),
        },
        coords={'basin': ['b1', 'b2']},
    ).to_netcdf(attr_dir / 'part1.nc')
    xr.Dataset(
        {
            'area': (('basin',), np.array([30.5], dtype=np.float64)),
            'p_mean': (('basin',), np.array([4.3], dtype=np.float64)),
        },
        coords={'basin': ['b3']},
    ).to_netcdf(attr_dir / 'part2.nc4')

    ds = caravan.load_caravan_attributes(
        data_dir=attr_dir, basins=['b3', 'b1'], features=['area']
    )
    try:
        assert list(ds.data_vars) == ['area']
        assert ds.coords['basin'].to_numpy().tolist() == ['b3', 'b1']
        assert ds['area'].dtype == np.float32
        np.testing.assert_allclose(
            ds['area'].to_numpy(), np.array([30.5, 10.5], dtype=np.float32)
        )
    finally:
        ds.close()


@pytest.mark.unit
def test_load_netcdf_unsorted_and_interleaved_basin_splits(
    tmp_path: Path,
) -> None:
    """Combine non-monotonic/interleaved basin-split statics and 2D splits."""
    attr_dir = tmp_path / 'attrs_interleaved'
    attr_dir.mkdir()
    xr.Dataset(
        {'area': (('basin',), np.array([5.0, 1.0, 4.0], dtype=np.float32))},
        coords={'basin': ['e', 'a', 'd']},
    ).to_netcdf(attr_dir / 'part_0.nc')
    xr.Dataset(
        {'area': (('basin',), np.array([6.0, 2.0, 3.0], dtype=np.float32))},
        coords={'basin': ['f', 'b', 'c']},
    ).to_netcdf(attr_dir / 'part_1.nc')

    ds_attr = caravan.load_caravan_attributes(
        data_dir=attr_dir,
        basins=['a', 'b', 'c', 'd', 'e', 'f'],
        features=['area'],
    )
    try:
        np.testing.assert_array_equal(
            ds_attr['area'].to_numpy(),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        )
    finally:
        ds_attr.close()

    ts_dir = tmp_path / 'ts_2d_interleaved'
    ts_dir.mkdir()
    d1 = pd.date_range('2020-01-01', periods=2, freq='D')
    d2 = pd.date_range('2020-01-03', periods=2, freq='D')
    xr.Dataset(
        {
            'streamflow': (
                ('basin', 'date'),
                np.array([[1.0, 2.0], [3.0, 4.0], [17.0, 18.0]], np.float32),
            )
        },
        coords={'basin': ['d', 'b', 'e'], 'date': d1},
    ).to_netcdf(ts_dir / 'west_2020_a.nc')
    xr.Dataset(
        {
            'streamflow': (
                ('basin', 'date'),
                np.array([[5.0, 6.0], [7.0, 8.0], [19.0, 20.0]], np.float32),
            )
        },
        coords={'basin': ['b', 'd', 'e'], 'date': d2},
    ).to_netcdf(ts_dir / 'west_2020_b.nc')
    xr.Dataset(
        {
            'streamflow': (
                ('basin', 'date'),
                np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
            )
        },
        coords={'basin': ['c', 'a'], 'date': d1},
    ).to_netcdf(ts_dir / 'east_2020_a.nc')
    xr.Dataset(
        {
            'streamflow': (
                ('basin', 'date'),
                np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32),
            )
        },
        coords={'basin': ['c', 'a'], 'date': d2},
    ).to_netcdf(ts_dir / 'east_2020_b.nc')

    ds_ts = load_netcdf_timeseries(
        data_dir=ts_dir,
        basins=['a', 'b', 'c', 'd', 'e'],
        features=['streamflow'],
    )
    try:
        expected = np.array(
            [
                [11.0, 12.0, 15.0, 16.0],
                [3.0, 4.0, 5.0, 6.0],
                [9.0, 10.0, 13.0, 14.0],
                [1.0, 2.0, 7.0, 8.0],
                [17.0, 18.0, 19.0, 20.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(ds_ts['streamflow'].to_numpy(), expected)
    finally:
        ds_ts.close()


@pytest.mark.unit
def test_load_netcdf_attributes_bytes_basin_coord_netcdf3(
    tmp_path: Path,
) -> None:
    """Decode byte-string basin coordinates from NetCDF3 classic files."""
    attr_dir = tmp_path / 'attrs_nc3'
    attr_dir.mkdir()
    xr.Dataset(
        {'area': (('basin',), np.array([15.0, 25.0], dtype=np.float64))},
        coords={'basin': np.array([b'b1', b'b2'])},
    ).to_netcdf(attr_dir / 'attrs.nc', engine='scipy')

    ds = load_netcdf_attributes(attr_dir, basins=['b2'], features=['area'])
    try:
        assert ds.coords['basin'].to_numpy().tolist() == ['b2']
        assert ds['area'].dtype == np.float32
        np.testing.assert_allclose(ds['area'].to_numpy(), [25.0])
    finally:
        ds.close()


@pytest.mark.unit
def test_load_netcdf_timeseries_per_basin_varying_dates_and_batching(
    tmp_path: Path,
) -> None:
    """Open only requested per-basin files and align differing date ranges."""
    ts_dir = tmp_path / 'per_basin'
    ts_dir.mkdir()
    d1 = pd.date_range('2020-01-01', periods=5, freq='D')
    d2 = pd.date_range('2020-01-03', periods=5, freq='D')
    leads = pd.to_timedelta([1, 2], unit='D')

    # Save with 0D scalar 'basin' and per-basin scalar 'lat' coords (default
    # xarray `.sel(basin=b).to_netcdf(...)` without `drop=True`).
    xr.Dataset(
        {
            'hres_p': (
                ('date', 'lead_time'),
                np.ones((5, 2), dtype=np.float64) * 1.5,
            )
        },
        coords={'date': d1, 'lead_time': leads, 'basin': 'b1', 'lat': 10.0},
    ).to_netcdf(ts_dir / 'b1.nc')
    xr.Dataset(
        {
            'hres_p': (
                ('date', 'lead_time'),
                np.ones((5, 2), dtype=np.float64) * 2.5,
            )
        },
        coords={'date': d2, 'lead_time': leads, 'basin': 'b2', 'lat': 20.0},
    ).to_netcdf(ts_dir / 'b2.NC4')
    # Alphabetically earlier corrupt file and macOS AppleDouble dotfile must
    # not be opened or break per-basin loading.
    (ts_dir / 'a_corrupt.nc').write_bytes(b'not a valid netcdf file')
    (ts_dir / '._b1.nc').write_bytes(b'macos appledouble sidecar')

    ds = load_netcdf_timeseries(
        ts_dir,
        basins=['b1', 'b2'],
        features=['hres_p'],
        require_lead_time=True,
        batch_size=1,
    )
    try:
        assert ds['hres_p'].dtype == np.float32
        assert ds.coords['basin'].to_numpy().tolist() == ['b1', 'b2']
        assert 'lat' not in ds.coords
        assert ds.sizes['date'] == _EXPECTED_UNION_DATES
        assert ds.sizes['lead_time'] == _EXPECTED_LEADS
        assert np.isnan(ds['hres_p'].sel(basin='b2', date='2020-01-01')).all()
        assert np.isnan(ds['hres_p'].sel(basin='b1', date='2020-01-07')).all()
        np.testing.assert_array_equal(
            ds['hres_p'].sel(basin='b1', date='2020-01-03').to_numpy(),
            np.array([1.5, 1.5], dtype=np.float32),
        )
    finally:
        ds.close()


@pytest.mark.unit
def test_netcdf_loader_fail_loud_errors(tmp_path: Path) -> None:
    """Reject missing files, variables, coordinates, units, or mixed layouts."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match='No NetCDF'):
        load_netcdf_attributes(empty)
    with pytest.raises(FileNotFoundError, match='No NetCDF'):
        load_netcdf_timeseries(empty, basins=['b1'])

    wrong_date = tmp_path / 'wrong_date'
    wrong_date.mkdir()
    xr.Dataset(
        {'streamflow': (('basin', 'time'), np.ones((1, 3), dtype=np.float32))},
        coords={
            'basin': ['b1'],
            'time': pd.date_range('2020-01-01', periods=3),
        },
    ).to_netcdf(wrong_date / 'sf.nc')
    with pytest.raises(ValueError, match="Required 'date'"):
        load_netcdf_timeseries(
            wrong_date, basins=['b1'], features=['streamflow']
        )

    valid_mb = tmp_path / 'valid_mb'
    valid_mb.mkdir()
    xr.Dataset(
        {'streamflow': (('basin', 'date'), np.ones((1, 3), dtype=np.float32))},
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-01', periods=3),
        },
    ).to_netcdf(valid_mb / 'sf.nc')
    with pytest.raises(ValueError, match='nonexistent'):
        load_netcdf_timeseries(
            valid_mb, basins=['b1'], features=['nonexistent']
        )
    with pytest.raises(ValueError, match='b_missing'):
        load_netcdf_timeseries(
            valid_mb, basins=['b1', 'b_missing'], features=['streamflow']
        )

    with pytest.raises(ValueError, match='lead_time'):
        load_netcdf_timeseries(
            valid_mb,
            basins=['b1'],
            features=['streamflow'],
            require_lead_time=True,
        )

    bad_lt = tmp_path / 'bad_lt'
    bad_lt.mkdir()
    xr.Dataset(
        {
            'hres_p': (
                ('basin', 'date', 'lead_time'),
                np.ones((1, 3, 2), dtype=np.float32),
            )
        },
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-01', periods=3),
            'lead_time': [1, 2],
        },
    ).to_netcdf(bad_lt / 'fc.nc')
    with pytest.raises(ValueError, match='timedelta64'):
        load_netcdf_timeseries(
            bad_lt,
            basins=['b1'],
            features=['hres_p'],
            require_lead_time=True,
        )

    mixed_fc = tmp_path / 'mixed_fc'
    mixed_fc.mkdir()
    xr.Dataset(
        {
            'hres_p': (
                ('basin', 'date', 'lead_time'),
                np.ones((1, 3, 2), dtype=np.float32),
            ),
            'era5_p': (('basin', 'date'), np.ones((1, 3), dtype=np.float32)),
        },
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-01', periods=3),
            'lead_time': pd.to_timedelta([1, 2], unit='D'),
        },
    ).to_netcdf(mixed_fc / 'fc.nc')
    with pytest.raises(ValueError, match='era5_p'):
        load_netcdf_timeseries(
            mixed_fc,
            basins=['b1'],
            features=['hres_p', 'era5_p'],
            require_lead_time=True,
        )

    mixed_layout = tmp_path / 'mixed_layout'
    mixed_layout.mkdir()
    xr.Dataset(
        {'streamflow': (('basin', 'date'), np.ones((1, 3), dtype=np.float32))},
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-01', periods=3),
        },
    ).to_netcdf(mixed_layout / 'a_multi.nc')
    xr.Dataset(
        {'streamflow': (('date',), np.ones(3, dtype=np.float32))},
        coords={'date': pd.date_range('2020-01-01', periods=3)},
    ).to_netcdf(mixed_layout / 'b2.nc')
    with pytest.raises(
        ValueError, match='Cannot mix multi-basin and per-basin'
    ):
        load_netcdf_timeseries(
            mixed_layout, basins=['b1'], features=['streamflow']
        )

    pb_only = tmp_path / 'pb_only'
    pb_only.mkdir()
    xr.Dataset(
        {'streamflow': (('date',), np.ones(3, dtype=np.float32))},
        coords={'date': pd.date_range('2020-01-01', periods=3)},
    ).to_netcdf(pb_only / 'b1.nc')
    with pytest.raises(ValueError, match='multi-basin layout'):
        load_netcdf_attributes(pb_only, basins=['b1'], features=['streamflow'])

    with pytest.raises(FileNotFoundError, match='b_missing'):
        load_netcdf_timeseries(
            pb_only, basins=['b1', 'b_missing'], features=['streamflow']
        )
    xr.Dataset(
        {'streamflow': (('date',), np.ones(3, dtype=np.float32))},
        coords={'date': pd.date_range('2020-01-01', periods=3)},
    ).to_netcdf(pb_only / 'b1.nc4')
    with pytest.raises(ValueError, match='Ambiguous per-basin'):
        load_netcdf_timeseries(pb_only, basins=['b1'], features=['streamflow'])


@pytest.mark.unit
@pytest.mark.parametrize(
    'layout_kind',
    [
        'multibasin_product_subdirs',
        'multibasin_flat',
        'per_basin_product_subdirs',
        'per_basin_flat',
    ],
)
@pytest.mark.parametrize('lazy_load', [False, True])
def test_multimet_netcdf_matches_zarr_bit_for_bit(
    tmp_path: Path,
    layout_kind: str,
    lazy_load: bool,  # noqa: FBT001
) -> None:
    """Confirm NetCDF layouts produce identical batches to Zarr in Multimet."""
    basins = ['basin_01', 'basin_02', 'basin_03']
    dates = pd.date_range('2000-01-01', periods=30, freq='D')
    ds = _make_synthetic_dataset(
        basins, dates, lead_days=[1, 2, 3], dtype=np.float64
    )

    zarr_paths = _write_zarr_layout(tmp_path / 'zarr_ref', ds)
    if layout_kind == 'multibasin_product_subdirs':
        nc_paths = _write_multibasin_netcdf_layout(
            tmp_path / 'nc', ds, product_subdirs=True
        )
    elif layout_kind == 'multibasin_flat':
        nc_paths = _write_multibasin_netcdf_layout(
            tmp_path / 'nc', ds, product_subdirs=False
        )
    elif layout_kind == 'per_basin_product_subdirs':
        nc_paths = _write_per_basin_netcdf_layout(
            tmp_path / 'nc', ds, product_subdirs=True
        )
    else:
        nc_paths = _write_per_basin_netcdf_layout(
            tmp_path / 'nc', ds, product_subdirs=False
        )

    zarr_cfg = _make_multimet_config(tmp_path, 'zarr', zarr_paths, basins)
    nc_cfg = _make_multimet_config(
        tmp_path,
        f'nc_{layout_kind}_{lazy_load}',
        nc_paths,
        basins,
        overrides={'lazy_load': lazy_load},
    )

    with dask.config.set(scheduler='threads'):
        zarr_ds = Multimet(cfg=zarr_cfg, is_train=True, period='train')
        nc_ds = Multimet(cfg=nc_cfg, is_train=True, period='train')

        assert len(nc_ds) == len(zarr_ds)
        assert len(nc_ds) > 0

        zarr_loader = MultimetDataLoader(
            zarr_ds,
            batch_size=8,
            shuffle=False,
            lazy_load=False,
            logging_level=30,
            collate_fn=Multimet.collate_fn,
        )
        nc_loader = MultimetDataLoader(
            nc_ds,
            batch_size=8,
            shuffle=False,
            lazy_load=lazy_load,
            logging_level=30,
            collate_fn=Multimet.collate_fn,
        )

        zarr_batches = list(zarr_loader)
        nc_batches = list(nc_loader)
        assert len(nc_batches) == len(zarr_batches)
        for zb, nb in zip(zarr_batches, nc_batches, strict=True):
            np.testing.assert_array_equal(nb['date'], zb['date'])
            np.testing.assert_array_equal(
                nb['basin_index'].numpy(), zb['basin_index'].numpy()
            )
            assert nb['x_s'].dtype == torch.float32
            assert nb['y'].dtype == torch.float32
            np.testing.assert_array_equal(nb['x_s'].numpy(), zb['x_s'].numpy())
            np.testing.assert_array_equal(nb['y'].numpy(), zb['y'].numpy())
            for k in zb['x_d_hindcast']:
                assert nb['x_d_hindcast'][k].dtype == torch.float32
                np.testing.assert_array_equal(
                    nb['x_d_hindcast'][k].numpy(),
                    zb['x_d_hindcast'][k].numpy(),
                )
            for k in zb['x_d_forecast']:
                assert nb['x_d_forecast'][k].dtype == torch.float32
                np.testing.assert_array_equal(
                    nb['x_d_forecast'][k].numpy(),
                    zb['x_d_forecast'][k].numpy(),
                )


@pytest.mark.unit
def test_multimet_netcdf_hindcast_only_shared_dir_and_union_mapping(
    tmp_path: Path,
) -> None:
    """Support shared data_dir, hindcast-only configs, and union_mapping."""
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('2000-01-01', periods=30, freq='D')
    ds = _make_synthetic_dataset(
        basins, dates, lead_days=[1, 2], dtype=np.float64
    )
    # Fill cpc_precip NaNs from era5land_precip via union_mapping.
    cpc_vals = ds['era5land_precip'].to_numpy().copy()
    cpc_vals[:, :15] = np.nan
    ds['cpc_precip'] = (('basin', 'date'), cpc_vals)

    shared_dir = tmp_path / 'shared_nc'
    shared_dir.mkdir()
    ds[['area', 'p_mean']].to_netcdf(shared_dir / 'attributes.nc')
    ds[['streamflow']].to_netcdf(shared_dir / 'streamflow.nc')
    ds[['era5land_precip', 'era5land_temp', 'cpc_precip']].to_netcdf(
        shared_dir / 'hindcast.nc'
    )

    paths = {
        'statics': shared_dir,
        'targets': shared_dir,
        'dynamics': shared_dir,
    }
    cfg = _make_multimet_config(
        tmp_path,
        'hindcast_only',
        paths,
        basins,
        overrides={
            'hindcast_inputs': ['cpc_precip', 'era5land_temp'],
            'forecast_inputs': [],
            'union_mapping': {'cpc_precip': 'era5land_precip'},
            'lead_time': 0,
            'forecast_overlap': 0,
        },
    )
    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    assert len(dataset) > 0
    sample = dataset[0]
    assert 'x_d' in sample
    assert set(sample['x_d'].keys()) == {'cpc_precip', 'era5land_temp'}
    assert not np.isnan(sample['x_d']['cpc_precip']).any()


@pytest.mark.integration
@pytest.mark.parametrize('lazy_load', [False, True])
def test_end_to_end_train_evaluate_infer_with_netcdf(
    tmp_path: Path,
    lazy_load: bool,  # noqa: FBT001
) -> None:
    """Run full train, evaluate, and infer pipeline purely from NetCDF files."""
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('2000-01-01', periods=45, freq='D')
    ds = _make_synthetic_dataset(
        basins, dates, lead_days=[1, 2, 3], dtype=np.float64, seed=7
    )
    nc_paths = _write_multibasin_netcdf_layout(
        tmp_path / f'nc_{lazy_load}', ds, product_subdirs=True
    )

    basin_file = tmp_path / f'basins_{lazy_load}.txt'
    basin_file.write_text('\n'.join(basins) + '\n', encoding='utf-8')
    run_dir = tmp_path / f'run_{lazy_load}'

    cfg = Config(
        {
            'experiment_name': f'nc_e2e_{lazy_load}',
            'run_dir': str(run_dir),
            'dataset': 'multimet',
            'model': 'handoff_forecast_lstm',
            'head': 'regression',
            'hidden_size': 8,
            'state_handoff_network': {
                'hiddens': [8],
                'activation': ['tanh'],
                'dropout': 0.0,
            },
            'statics_embedding': {
                'hiddens': [4],
                'activation': ['tanh'],
                'dropout': 0.0,
            },
            'train_basin_file': str(basin_file),
            'validation_basin_file': str(basin_file),
            'test_basin_file': str(basin_file),
            'statics_data_dir': str(nc_paths['statics']),
            'targets_data_dir': str(nc_paths['targets']),
            'dynamics_data_dir': str(nc_paths['dynamics']),
            'static_attributes': ['area', 'p_mean'],
            'target_variables': ['streamflow'],
            'hindcast_inputs': ['era5land_precip', 'era5land_temp'],
            'forecast_inputs': ['hres_precip'],
            'train_start_date': '10/01/2000',
            'train_end_date': '31/01/2000',
            'validation_start_date': '10/01/2000',
            'validation_end_date': '31/01/2000',
            'test_start_date': '10/01/2000',
            'test_end_date': '31/01/2000',
            'seq_length': 5,
            'lead_time': 2,
            'forecast_overlap': 1,
            'predict_last_n': 2,
            'nan_handling_method': 'none',
            'timestep_counter': True,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'initial_learning_rate': 0.01,
            'epochs': 1,
            'batch_size': 8,
            'metrics': ['NSE'],
            'validate_every': 1,
            'validate_n_random_basins': 2,
            'log_tensorboard': False,
            'log_n_figures': 0,
            'save_git_diff': False,
            'lazy_load': lazy_load,
            'seed': 42,
            'device': 'cpu',
        }
    )

    with dask.config.set(scheduler='threads'):
        start_run(cfg, gpu=-1)
        actual_run_dir = cfg.run_dir
        assert (actual_run_dir / 'model_epoch001.pt').is_file()
        assert (actual_run_dir / 'scaler.zarr').exists()

        eval_cfg = Config(actual_run_dir / 'config.yml')
        eval_run(
            eval_cfg, run_dir=actual_run_dir, period='test', epoch=1, gpu=-1
        )
        metrics_csv = (
            actual_run_dir / 'test' / 'model_epoch001' / 'test_metrics.csv'
        )
        assert metrics_csv.is_file()
        metrics_df = pd.read_csv(metrics_csv, index_col=0)
        assert sorted(metrics_df.index.tolist()) == sorted(basins)
        assert np.isfinite(metrics_df['NSE'].to_numpy()).all()

        infer_cfg = Config(actual_run_dir / 'config.yml')
        infer_cfg.inference_mode = True
        infer_cfg.tester_skip_obs_all_nan = False
        eval_run(
            infer_cfg, run_dir=actual_run_dir, period='test', epoch=1, gpu=-1
        )
        results_zarr = (
            actual_run_dir / 'test' / 'model_epoch001' / 'test_results.zarr'
        )
        assert results_zarr.exists()
        with xr.open_zarr(results_zarr) as res_ds:
            assert sorted(res_ds.coords['basin'].to_numpy().tolist()) == sorted(
                basins
            )
            assert np.isfinite(res_ds['streamflow_sim'].to_numpy()).all()


@pytest.mark.unit
def test_stray_nc_does_not_regress_zarr_or_legacy_caravan(
    tmp_path: Path,
) -> None:
    """Stray root .nc files must not override Zarr stores or legacy Caravan."""
    basins = ['camels_01', 'camels_02']
    dates = pd.date_range('2000-01-01', periods=30, freq='D')
    ds = _make_synthetic_dataset(basins, dates, lead_days=[1, 2])

    # 1. Zarr layout with stray .nc at dynamics root -> Zarr must still win.
    zarr_paths = _write_zarr_layout(tmp_path / 'zarr_with_stray', ds)
    xr.Dataset(
        {'stray': (('date',), np.ones(5))}, coords={'date': dates[:5]}
    ).to_netcdf(zarr_paths['dynamics'] / 'stray.nc')
    cfg = _make_multimet_config(tmp_path, 'zarr_stray', zarr_paths, basins)
    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    assert len(dataset) > 0

    # 2. Legacy Caravan root with stray root .nc -> legacy Caravan wins.
    legacy_root = tmp_path / 'legacy_caravan'
    attr_sub = legacy_root / 'attributes' / 'camels'
    ts_sub = legacy_root / 'timeseries' / 'netcdf' / 'camels'
    attr_sub.mkdir(parents=True)
    ts_sub.mkdir(parents=True)
    pd.DataFrame({'gauge_id': basins, 'area': [10.0, 20.0]}).to_csv(
        attr_sub / 'attributes_caravan_camels.csv', index=False
    )
    for idx, basin in enumerate(basins):
        xr.Dataset(
            {'streamflow': (('date',), np.ones(10, np.float32) * (idx + 1))},
            coords={'date': dates[:10]},
        ).to_netcdf(ts_sub / f'{basin}.nc')
    xr.Dataset(
        {'noise': (('date',), np.zeros(5))}, coords={'date': dates[:5]}
    ).to_netcdf(legacy_root / 'stray_sample.nc')

    loaded_attr = caravan.load_caravan_attributes(
        legacy_root, basins=basins, features=['area']
    )
    np.testing.assert_array_equal(
        loaded_attr['area'].to_numpy(), np.array([10.0, 20.0], dtype=np.float32)
    )
    loaded_ts = caravan.load_caravan_timeseries(
        legacy_root, basins=basins, target_features=['streamflow']
    )
    assert loaded_ts.sizes['basin'] == len(basins)


@pytest.mark.unit
def test_shared_data_dir_with_product_subdirs_and_no_2d_attr_leak(
    tmp_path: Path,
) -> None:
    """Shared data_dir with root statics/targets + product subdirs works."""
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('2000-01-01', periods=30, freq='D')
    ds = _make_synthetic_dataset(basins, dates, lead_days=[1, 2])

    shared = tmp_path / 'shared_root'
    (shared / 'era5-land').mkdir(parents=True)
    (shared / 'HRES').mkdir(parents=True)
    ds[['area', 'p_mean']].to_netcdf(shared / 'attributes.nc')
    ds[['streamflow']].to_netcdf(shared / 'streamflow.nc')
    ds[['era5land_precip', 'era5land_temp']].to_netcdf(
        shared / 'era5-land' / 'era5.nc'
    )
    ds[['hres_precip']].to_netcdf(shared / 'HRES' / 'hres.nc')

    # Hyphenated product dir resolves cleanly.
    assert find_product_netcdf_dir(shared, 'ERA5_LAND') == shared / 'era5-land'

    # load_caravan_attributes(features=None) returns only 1D static attributes.
    attrs_all = caravan.load_caravan_attributes(shared, basins=basins)
    try:
        assert sorted(attrs_all.data_vars) == ['area', 'p_mean']
        assert 'date' not in attrs_all.dims
    finally:
        attrs_all.close()

    # Multimet loads root statics/targets + product-subdir hindcast & forecast,
    # including 3D hres_precip used as both hindcast and forecast input.
    paths = {'statics': shared, 'targets': shared, 'dynamics': shared}
    cfg = _make_multimet_config(
        tmp_path,
        'shared_subdirs',
        paths,
        basins,
        overrides={
            'hindcast_inputs': [
                'era5land_precip',
                'era5land_temp',
                'hres_precip',
            ],
            'forecast_inputs': ['hres_precip'],
        },
    )
    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    assert len(dataset) > 0
    assert 'hres_precip' in dataset[0]['x_d_hindcast']
    assert 'hres_precip' in dataset[0]['x_d_forecast']


@pytest.mark.unit
def test_non_midnight_subdaily_nat_and_duplicate_dates_rejected(
    tmp_path: Path,
) -> None:
    """Reject 12:00 timestamps, sub-daily steps, NaT, and duplicate dates."""
    noon_dates = pd.date_range('2020-01-01 12:00:00', periods=4, freq='D')
    noon_dir = tmp_path / 'noon'
    noon_dir.mkdir()
    xr.Dataset(
        {'q': (('basin', 'date'), np.ones((1, 4), dtype=np.float32))},
        coords={'basin': ['b1'], 'date': noon_dates},
    ).to_netcdf(noon_dir / 'daymet.nc')
    with pytest.raises(ValueError, match=r'daymet\.nc.*midnight'):
        load_netcdf_timeseries(noon_dir, basins=['b1'], features=['q'])

    subdaily_dir = tmp_path / 'subdaily'
    subdaily_dir.mkdir()
    xr.Dataset(
        {'q': (('date',), np.ones(8, dtype=np.float32))},
        coords={'date': pd.date_range('2020-01-01', periods=8, freq='6h')},
    ).to_netcdf(subdaily_dir / 'b1.nc')
    with pytest.raises(ValueError, match=r'b1\.nc.*midnight'):
        load_netcdf_timeseries(subdaily_dir, basins=['b1'], features=['q'])

    dup_dir = tmp_path / 'dup_date'
    dup_dir.mkdir()
    dup_dates = pd.DatetimeIndex(['2020-01-01', '2020-01-01', '2020-01-02'])
    xr.Dataset(
        {'q': (('date',), np.ones(3, dtype=np.float32))},
        coords={'date': dup_dates},
    ).to_netcdf(dup_dir / 'b1.nc')
    with pytest.raises(ValueError, match=r'b1\.nc.*duplicate'):
        load_netcdf_timeseries(dup_dir, basins=['b1'], features=['q'])

    nat_dir = tmp_path / 'nat_date'
    nat_dir.mkdir()
    nat_dates = pd.DatetimeIndex(['2020-01-01', pd.NaT, '2020-01-03'])
    xr.Dataset(
        {'q': (('date',), np.ones(3, dtype=np.float32))},
        coords={'date': nat_dates},
    ).to_netcdf(nat_dir / 'b1.nc')
    with pytest.raises(ValueError, match=r'b1\.nc.*NaT'):
        load_netcdf_timeseries(nat_dir, basins=['b1'], features=['q'])


@pytest.mark.unit
def test_integer_and_bool_vars_cast_to_float32_without_overflow(
    tmp_path: Path,
) -> None:
    """Integer/bool static and ragged per-basin target vars cast to float32."""
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('2000-01-01', periods=30, freq='D')
    base = _make_synthetic_dataset(basins, dates, lead_days=[1, 2])

    statics_dir = tmp_path / 'statics_int'
    targets_dir = tmp_path / 'targets_int'
    dynamics_dir = tmp_path / 'dynamics_int'
    statics_dir.mkdir()
    targets_dir.mkdir()
    dynamics_dir.mkdir()

    # uint8 / int16 values whose squares would overflow without float32 cast,
    # plus int64 and bool attributes.
    xr.Dataset(
        {
            'area': (('basin',), np.array([100, 250], dtype=np.int16)),
            'p_mean': (('basin',), np.array([20, 200], dtype=np.uint8)),
            'land_class': (('basin',), np.array([1, 5], dtype=np.int64)),
            'is_snow': (('basin',), np.array([True, False], dtype=np.bool_)),
        },
        coords={'basin': basins},
    ).to_netcdf(statics_dir / 'attrs.nc')

    # Ragged int32 per-basin targets (reindex introduces NaNs -> would become
    # float64 if cast happened only before reindex).
    xr.Dataset(
        {'streamflow': (('date',), np.arange(1, 31, dtype=np.int32))},
        coords={'date': dates},
    ).to_netcdf(targets_dir / 'basin_01.nc')
    xr.Dataset(
        {'streamflow': (('date',), np.arange(5, 30, dtype=np.int32))},
        coords={'date': dates[5:]},
    ).to_netcdf(targets_dir / 'basin_02.nc')

    base[['era5land_precip', 'era5land_temp', 'hres_precip']].to_netcdf(
        dynamics_dir / 'all.nc'
    )

    cfg = _make_multimet_config(
        tmp_path,
        'int_bool_cast',
        {
            'statics': statics_dir,
            'targets': targets_dir,
            'dynamics': dynamics_dir,
        },
        basins,
        overrides={
            'static_attributes': ['area', 'p_mean', 'land_class', 'is_snow']
        },
    )
    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    assert len(dataset) > 0
    loader = MultimetDataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        lazy_load=False,
        logging_level=30,
        collate_fn=Multimet.collate_fn,
    )
    batch = next(iter(loader))
    assert batch['x_s'].dtype == torch.float32
    assert batch['y'].dtype == torch.float32


@pytest.mark.unit
def test_decode_timedelta_only_affects_lead_time(tmp_path: Path) -> None:
    """Non-lead_time vars with CF time units stay numeric float32."""
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('2000-01-01', periods=30, freq='D')
    base = _make_synthetic_dataset(basins, dates, lead_days=[1, 2])

    statics_dir = tmp_path / 'statics_td'
    targets_dir = tmp_path / 'targets_td'
    dynamics_dir = tmp_path / 'dynamics_td'
    statics_dir.mkdir()
    targets_dir.mkdir()
    dynamics_dir.mkdir()

    attr_ds = base[['area', 'p_mean']].copy()
    attr_ds['p_mean'].attrs['units'] = 'days'
    attr_ds.to_netcdf(statics_dir / 'attrs.nc')

    base[['streamflow']].to_netcdf(targets_dir / 'targets.nc')

    dyn_ds = xr.Dataset(
        {
            'era5land_precip': (
                ('basin', 'date'),
                base['era5land_precip'].to_numpy(),
                {'units': 'hours'},
            ),
            'era5land_temp': base['era5land_temp'],
            'hres_precip': (
                ('basin', 'date', 'lead_time'),
                base['hres_precip'].to_numpy(),
            ),
        },
        coords={
            'basin': basins,
            'date': dates,
            'lead_time': ('lead_time', [24, 48], {'units': 'hours'}),
        },
    )
    dyn_ds.to_netcdf(dynamics_dir / 'dyn.nc')

    cfg = _make_multimet_config(
        tmp_path,
        'selective_td',
        {
            'statics': statics_dir,
            'targets': targets_dir,
            'dynamics': dynamics_dir,
        },
        basins,
    )
    dataset = Multimet(cfg=cfg, is_train=True, period='train')
    assert len(dataset) > 0
    assert dataset._dataset['p_mean'].dtype == np.float32  # noqa: SLF001
    assert dataset._dataset['era5land_precip'].dtype == np.float32  # noqa: SLF001


@pytest.mark.unit
def test_overlapping_files_rejected_and_unrequested_metadata_ignored(
    tmp_path: Path,
) -> None:
    """Reject overlapping time/basin splits; ignore conflicting unused vars."""
    # Overlapping boundary day in time-split files.
    overlap_dir = tmp_path / 'overlap_time'
    overlap_dir.mkdir()
    xr.Dataset(
        {'q': (('basin', 'date'), np.ones((1, 5), dtype=np.float32))},
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-01', periods=5, freq='D'),
        },
    ).to_netcdf(overlap_dir / 'part1.nc')
    xr.Dataset(
        {'q': (('basin', 'date'), np.ones((1, 5), dtype=np.float32))},
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-05', periods=5, freq='D'),
        },
    ).to_netcdf(overlap_dir / 'part2.nc')
    with pytest.raises(ValueError, match=r'Overlapping.*part1\.nc.*part2\.nc'):
        load_netcdf_timeseries(overlap_dir, basins=['b1'], features=['q'])

    # Overlapping basin batches in static files.
    overlap_b = tmp_path / 'overlap_basin'
    overlap_b.mkdir()
    xr.Dataset(
        {'area': (('basin',), np.array([1.0, 2.0], np.float32))},
        coords={'basin': ['b1', 'b2']},
    ).to_netcdf(overlap_b / 'a.nc')
    xr.Dataset(
        {'area': (('basin',), np.array([2.0, 3.0], np.float32))},
        coords={'basin': ['b2', 'b3']},
    ).to_netcdf(overlap_b / 'b.nc')
    with pytest.raises(ValueError, match=r'Overlapping.*a\.nc.*b\.nc'):
        load_netcdf_attributes(overlap_b, basins=['b1', 'b2', 'b3'])

    # Conflicting unrequested per-file variable must NOT trigger MergeError.
    meta_dir = tmp_path / 'unrequested_meta'
    meta_dir.mkdir()
    xr.Dataset(
        {
            'q': (('basin', 'date'), np.ones((1, 3), dtype=np.float32)),
            'processing_version': ((), 1),
        },
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-01', periods=3, freq='D'),
        },
    ).to_netcdf(meta_dir / 'y1.nc')
    xr.Dataset(
        {
            'q': (('basin', 'date'), np.ones((1, 3), dtype=np.float32) * 2),
            'processing_version': ((), 2),
        },
        coords={
            'basin': ['b1'],
            'date': pd.date_range('2020-01-04', periods=3, freq='D'),
        },
    ).to_netcdf(meta_dir / 'y2.nc')
    ds_meta = load_netcdf_timeseries(meta_dir, basins=['b1'], features=['q'])
    try:
        assert ds_meta.sizes['date'] == _EXPECTED_COMBINED_META_DATES
    finally:
        ds_meta.close()


@pytest.mark.unit
def test_lead_time_validation_errors(tmp_path: Path) -> None:
    """Reject empty, 0-based, sub-daily, and unsorted lead_time coords."""
    dates = pd.date_range('2020-01-01', periods=3, freq='D')

    # 0-based lead_time (must start at >= 1 day).
    zero_dir = tmp_path / 'lt_zero'
    zero_dir.mkdir()
    xr.Dataset(
        {'f': (('basin', 'date', 'lead_time'), np.ones((1, 3, 3), np.float32))},
        coords={
            'basin': ['b1'],
            'date': dates,
            'lead_time': ('lead_time', [0, 1, 2], {'units': 'days'}),
        },
    ).to_netcdf(zero_dir / 'f.nc')
    with pytest.raises(ValueError, match=r'>= 1 day'):
        load_netcdf_timeseries(
            zero_dir, basins=['b1'], features=['f'], require_lead_time=True
        )

    # Sub-daily lead_time.
    sub_dir = tmp_path / 'lt_subdaily'
    sub_dir.mkdir()
    xr.Dataset(
        {'f': (('basin', 'date', 'lead_time'), np.ones((1, 3, 2), np.float32))},
        coords={
            'basin': ['b1'],
            'date': dates,
            'lead_time': ('lead_time', [24, 36], {'units': 'hours'}),
        },
    ).to_netcdf(sub_dir / 'f.nc')
    with pytest.raises(ValueError, match=r'whole-day'):
        load_netcdf_timeseries(
            sub_dir, basins=['b1'], features=['f'], require_lead_time=True
        )

    # Unsorted lead_time.
    unsorted_dir = tmp_path / 'lt_unsorted'
    unsorted_dir.mkdir()
    xr.Dataset(
        {'f': (('basin', 'date', 'lead_time'), np.ones((1, 3, 3), np.float32))},
        coords={
            'basin': ['b1'],
            'date': dates,
            'lead_time': ('lead_time', [2, 1, 3], {'units': 'days'}),
        },
    ).to_netcdf(unsorted_dir / 'f.nc')
    with pytest.raises(ValueError, match=r'strictly increasing'):
        load_netcdf_timeseries(
            unsorted_dir, basins=['b1'], features=['f'], require_lead_time=True
        )


@pytest.mark.unit
def test_cloud_path_windows_and_validation_edge_cases(tmp_path: Path) -> None:
    """Verify Windows gs:// paths, local gs/ dirs, empty basins, file paths."""
    assert _is_cloud_path('gs://caravan-multimet/v1.1')
    assert _is_cloud_path(PureWindowsPath('gs://caravan-multimet/v1.1'))
    assert _is_cloud_path('gs:\\caravan-multimet\\v1.1')
    assert not _is_cloud_path('gs/local_data')
    assert _is_zarr_store(Path('gs://caravan-multimet/v1.1/HRES'))
    assert not _is_zarr_store(tmp_path / 'missing.zarr')

    # Single .nc file passed directly raises clear error.
    nc_file = tmp_path / 'single.nc'
    xr.Dataset(
        {'area': (('basin',), np.array([1.0], np.float32))},
        coords={'basin': ['b1']},
    ).to_netcdf(nc_file)
    with pytest.raises(ValueError, match=r'Expected a directory'):
        caravan.load_caravan_attributes(nc_file, basins=['b1'])

    # Empty basin list raises ValueError in multi-basin mode too.
    with pytest.raises(ValueError, match=r'non-empty'):
        load_netcdf_attributes(tmp_path, basins=[])
    with pytest.raises(ValueError, match=r'non-empty'):
        load_netcdf_timeseries(tmp_path, basins=[])
    with pytest.raises(ValueError, match=r'batch_size'):
        load_netcdf_timeseries(tmp_path, basins=['b1'], batch_size=0)

    # Numeric basin IDs fail with clear dtype hint.
    num_dir = tmp_path / 'num_basin'
    num_dir.mkdir()
    xr.Dataset(
        {'area': (('basin',), np.array([1.0], np.float32))},
        coords={'basin': np.array([101], dtype=np.int64)},
    ).to_netcdf(num_dir / 'a.nc')
    with pytest.raises(ValueError, match=r'string basin IDs'):
        load_netcdf_attributes(num_dir, basins=['101'])


@pytest.mark.unit
def test_close_hooks_cdf5_and_scipy_mmap_with_file_cache_maxsize(
    tmp_path: Path,
) -> None:
    """Verify ds.close() attaches closers, CDF-5 fallback, and scipy mmap."""
    # Single-file, multi-file, and per-basin datasets all expose a close hook.
    nc3_dir = tmp_path / 'nc3_scipy'
    nc3_dir.mkdir()
    dates = pd.date_range('2020-01-01', periods=6, freq='D')
    xr.Dataset(
        {'q': (('date',), np.arange(6, dtype=np.float32))},
        coords={'date': dates},
    ).to_netcdf(nc3_dir / 'b1.nc', engine='scipy')
    xr.Dataset(
        {'q': (('date',), np.arange(6, dtype=np.float32) * 2)},
        coords={'date': dates},
    ).to_netcdf(nc3_dir / 'b2.nc', engine='scipy')

    with (
        xr.set_options(file_cache_maxsize=1),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always', RuntimeWarning)
        ds = load_netcdf_timeseries(
            nc3_dir, basins=['b1', 'b2'], features=['q']
        )
        assert ds._close is not None  # noqa: SLF001
        vals = ds['q'].compute()
        assert vals.shape == (2, 6)
        ds.close()
        mmap_warns = [w for w in caught if 'mmap=True' in str(w.message)]
        assert not mmap_warns

    # CDF-5 (NETCDF3_64BIT_DATA) exercises the netcdf4 fallback path.
    cdf5_dir = tmp_path / 'cdf5'
    cdf5_dir.mkdir()
    xr.Dataset(
        {'area': (('basin',), np.array([12.5], dtype=np.float64))},
        coords={'basin': ['b1']},
    ).to_netcdf(
        cdf5_dir / 'a.nc', engine='netcdf4', format='NETCDF3_64BIT_DATA'
    )
    ds_cdf5 = load_netcdf_attributes(cdf5_dir, basins=['b1'], features=['area'])
    try:
        assert ds_cdf5._close is not None  # noqa: SLF001
        assert ds_cdf5['area'].dtype == np.float32
        np.testing.assert_array_equal(
            ds_cdf5['area'].to_numpy(), np.array([12.5], dtype=np.float32)
        )
    finally:
        ds_cdf5.close()
