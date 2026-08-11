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

"""Unit tests for googlehydrology.datasetzoo.caravan."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from googlehydrology.datasetzoo import caravan


@pytest.fixture
def mock_caravan_dir(tmp_path):
    """Create synthetic Caravan directory structure with attributes and timeseries."""
    root = tmp_path / 'caravan_data'
    attr_dir = root / 'attributes' / 'camelsus'
    attr_dir.mkdir(parents=True)

    # Attributes CSV
    attr_df = pd.DataFrame({
        'gauge_id': ['camelsus_01022500', 'camelsus_01547700'],
        'area': [100.5, 250.0],
        'p_mean': [3.5, 4.2],
    })
    attr_df.to_csv(attr_dir / 'attributes_other.csv', index=False)

    # Timeseries netCDF
    ts_nc_dir = root / 'timeseries' / 'netcdf' / 'camelsus'
    ts_nc_dir.mkdir(parents=True)

    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    for basin in ['camelsus_01022500', 'camelsus_01547700']:
        ds = xr.Dataset(
            {
                'streamflow': (['date'], np.random.rand(10).astype(np.float32)),
                'precipitation': (['date'], np.random.rand(10).astype(np.float32)),
            },
            coords={'date': dates},
        )
        ds.to_netcdf(ts_nc_dir / f'{basin}.nc')

    # Timeseries CSV
    ts_csv_dir = root / 'timeseries' / 'csv' / 'camelsus'
    ts_csv_dir.mkdir(parents=True)
    for basin in ['camelsus_01022500', 'camelsus_01547700']:
        df = pd.DataFrame({
            'date': dates,
            'streamflow': np.random.rand(10).astype(np.float32),
            'precipitation': np.random.rand(10).astype(np.float32),
        })
        df.to_csv(ts_csv_dir / f'{basin}.csv', index=False)

    return root


@pytest.mark.unit
def test_load_caravan_attributes(mock_caravan_dir):
    # Load all attributes
    ds = caravan.load_caravan_attributes(data_dir=mock_caravan_dir)
    assert 'basin' in ds.coords
    assert len(ds['basin']) == 2
    assert 'area' in ds.data_vars or 'area' in ds.coords

    # Subdataset loading
    ds_sub = caravan.load_caravan_attributes(data_dir=mock_caravan_dir, subdataset='camelsus')
    assert len(ds_sub['basin']) == 2

    # Specific basins
    ds_basin = caravan.load_caravan_attributes(
        data_dir=mock_caravan_dir,
        basins=['camelsus_01022500'],
    )
    assert len(ds_basin['basin']) == 1

    # Missing subdataset error
    with pytest.raises(FileNotFoundError, match='No subdataset non_existent'):
        caravan.load_caravan_attributes(data_dir=mock_caravan_dir, subdataset='non_existent')

    # Missing basin in attribute file error
    with pytest.raises(ValueError, match='missing static attributes'):
        caravan.load_caravan_attributes(
            data_dir=mock_caravan_dir,
            basins=['camelsus_missing_gauge'],
        )


@pytest.mark.unit
def test_load_csvs_as_ds(mock_caravan_dir):
    csv_paths = {
        'camelsus_01022500': mock_caravan_dir / 'timeseries' / 'csv' / 'camelsus' / 'camelsus_01022500.csv',
        'camelsus_01547700': mock_caravan_dir / 'timeseries' / 'csv' / 'camelsus' / 'camelsus_01547700.csv',
    }
    ds = caravan.load_csvs_as_ds(csv_paths)
    assert 'basin' in ds.dims or 'basin' in ds.coords
    assert 'date' in ds.dims or 'date' in ds.coords
    assert 'streamflow' in ds.data_vars


@pytest.mark.unit
def test_load_caravan_timeseries_together_netcdf(mock_caravan_dir):
    basins = ['camelsus_01022500', 'camelsus_01547700']
    ds = caravan.load_caravan_timeseries_together(
        data_dir=mock_caravan_dir,
        basins=basins,
        target_features=['streamflow'],
        csv=False,
    )
    assert 'basin' in ds.coords
    assert 'streamflow' in ds.data_vars
    assert len(ds['basin']) == 2


@pytest.mark.unit
def test_load_caravan_timeseries_together_csv(mock_caravan_dir):
    basins = ['camelsus_01022500']
    ds = caravan.load_caravan_timeseries_together(
        data_dir=mock_caravan_dir,
        basins=basins,
        target_features=['streamflow'],
        csv=True,
    )
    assert 'streamflow' in ds.data_vars


@pytest.mark.unit
def test_load_caravan_timeseries_missing_file_error(mock_caravan_dir):
    with pytest.raises(FileNotFoundError, match='No basin file found'):
        caravan.load_caravan_timeseries_together(
            data_dir=mock_caravan_dir,
            basins=['camelsus_nonexistent'],
            target_features=['streamflow'],
            csv=False,
        )
