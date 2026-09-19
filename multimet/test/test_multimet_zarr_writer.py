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

"""Unit tests for MultiMet Zarr store writer and schema validator."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet.config import (
    PRODUCT_BANDS,
    Product,
)
from multimet.zarr_writer import MultiMetZarrWriter


pytestmark = pytest.mark.unit


@pytest.fixture
def writer(tmp_path) -> MultiMetZarrWriter:
  return MultiMetZarrWriter(tmp_path)


def test_write_and_append_nowcast(writer: MultiMetZarrWriter):
  basins_1 = ["basin_A", "basin_B"]
  dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
  precip_1 = np.random.rand(len(basins_1), len(dates)).astype(np.float32)

  ds1 = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], precip_1),
      },
      coords={
          "basin": basins_1,
          "date": dates.values,
      },
  )

  # 1. Write initial dataset
  store_path = writer.write_or_append(ds1, Product.CPC)
  assert store_path.endswith("CPC/timeseries.zarr")

  read_ds = xr.open_zarr(store_path)
  assert list(read_ds["basin"].values) == basins_1
  assert len(read_ds["date"]) == 10
  assert read_ds["cpc_precipitation"].dtype == np.float32

  # 2. Append new basin_C
  basins_2 = ["basin_C"]
  precip_2 = np.random.rand(len(basins_2), len(dates)).astype(np.float32)
  ds2 = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], precip_2),
      },
      coords={
          "basin": basins_2,
          "date": dates.values,
      },
  )

  writer.write_or_append(ds2, Product.CPC)

  read_ds2 = xr.open_zarr(store_path)
  assert list(read_ds2["basin"].values) == ["basin_A", "basin_B", "basin_C"]
  assert read_ds2["cpc_precipitation"].shape == (3, 10)


def test_write_and_append_forecast(writer: MultiMetZarrWriter):
  basins_1 = ["basin_X"]
  dates = pd.date_range("2021-01-01", "2021-01-05", freq="D")
  leads = pd.to_timedelta(range(1, 11), unit="D")
  shape = (len(basins_1), len(dates), len(leads))

  data_vars = {
      band: (
          ["basin", "date", "lead_time"],
          np.random.rand(*shape).astype(np.float32),
      )
      for band in PRODUCT_BANDS[Product.HRES]
  }

  ds = xr.Dataset(
      data_vars=data_vars,
      coords={
          "basin": basins_1,
          "date": dates.values,
          "lead_time": leads.values,
      },
  )

  store_path = writer.write_or_append(ds, Product.HRES)
  assert store_path.endswith("HRES/timeseries.zarr")

  read_ds = xr.open_zarr(store_path)
  assert "lead_time" in read_ds.coords
  assert len(read_ds["lead_time"]) == 10
  for band in PRODUCT_BANDS[Product.HRES]:
    assert band in read_ds.data_vars


def test_schema_validation_failure(writer: MultiMetZarrWriter):
  # Missing band
  ds_bad = xr.Dataset(
      data_vars={"wrong_var": (["basin", "date"], np.zeros((2, 2), dtype=np.float32))},
      coords={"basin": ["b1", "b2"], "date": pd.date_range("2020-01-01", periods=2)},
  )
  with pytest.raises(ValueError):
    writer.write_or_append(ds_bad, Product.CPC)


def test_append_dates_nowcast(writer: MultiMetZarrWriter):
  basins = ["basin_1", "basin_2"]
  dates_initial = pd.date_range("2020-01-01", periods=5, freq="D")
  ds_init = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((2, 5), dtype=np.float32)),
      },
      coords={"basin": basins, "date": dates_initial.values},
  )
  store_path = writer.write_or_append(ds_init, Product.CPC)

  # Append 3 new dates
  dates_new = pd.date_range("2020-01-06", periods=3, freq="D")
  start_idx, end_idx = writer.append_dates(Product.CPC, dates_new)
  assert start_idx == 5
  assert end_idx == 8

  ds_extended = xr.open_zarr(store_path)
  assert len(ds_extended["date"]) == 8
  assert ds_extended["cpc_precipitation"].shape == (2, 8)
  # Verify original values preserved
  assert np.all(ds_extended["cpc_precipitation"].values[:, :5] == 1.0)
  # Verify new slots are NaN before write
  assert np.all(np.isnan(ds_extended["cpc_precipitation"].values[:, 5:]))

  # Verify direct chunk write into new slots works
  writer.write_direct_chunk(Product.CPC, "cpc_precipitation", 5, np.array([42.0, 43.0], dtype=np.float32))
  writer.consolidate_metadata(Product.CPC)

  ds_written = xr.open_zarr(store_path)
  assert np.all(ds_written["cpc_precipitation"].values[:, 5] == [42.0, 43.0])


def test_append_basins_nowcast(writer: MultiMetZarrWriter):
  basins_initial = ["basin_A", "basin_B"]
  dates = pd.date_range("2020-01-01", periods=5, freq="D")
  ds_init = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((2, 5), dtype=np.float32)),
      },
      coords={"basin": basins_initial, "date": dates.values},
  )
  store_path = writer.write_or_append(ds_init, Product.CPC)

  # Append new basin_C and basin_D over the SAME dates
  basins_new = ["basin_C", "basin_D"]
  ds_new = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.full((2, 5), 88.0, dtype=np.float32)),
      },
      coords={"basin": basins_new, "date": dates.values},
  )
  writer.append_basins(Product.CPC, ds_new)

  ds_result = xr.open_zarr(store_path)
  assert list(ds_result["basin"].values) == ["basin_A", "basin_B", "basin_C", "basin_D"]
  assert ds_result["cpc_precipitation"].shape == (4, 5)
  assert np.all(ds_result["cpc_precipitation"].sel(basin="basin_C").values == 88.0)


def test_reject_simultaneous_2d_expansion(writer: MultiMetZarrWriter):
  basins_initial = ["basin_1", "basin_2"]
  dates_initial = pd.date_range("2020-01-01", periods=5, freq="D")
  ds_init = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((2, 5), dtype=np.float32)),
      },
      coords={"basin": basins_initial, "date": dates_initial.values},
  )
  writer.write_or_append(ds_init, Product.CPC)

  # Attempt to append BOTH new basins AND new dates
  basins_new = ["basin_3"]
  dates_new = pd.date_range("2020-01-06", periods=3, freq="D")
  ds_both = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((1, 3), dtype=np.float32)),
      },
      coords={"basin": basins_new, "date": dates_new.values},
  )

  with pytest.raises(ValueError, match="Cannot add both new basins and new dates"):
    writer.write_or_append(ds_both, Product.CPC)


def test_expand_date_range_prepending_and_overlap(writer: MultiMetZarrWriter):
  basins = ["basin_1"]
  dates = pd.date_range("2020-01-10", periods=5, freq="D")
  ds = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 5), dtype=np.float32))},
      coords={"basin": basins, "date": dates.values},
  )
  writer.write_or_append(ds, Product.CPC)

  # 1. Prepend earlier dates: 2020-01-01 to 2020-01-03
  earlier_dates = pd.date_range("2020-01-01", periods=3, freq="D")
  all_dates, indices = writer.expand_date_range(Product.CPC, earlier_dates)

  assert len(all_dates) == 8
  assert indices == [0, 1, 2]

  store_path = writer.get_store_path(Product.CPC)
  with xr.open_zarr(store_path) as res_ds:
    # First 3 days should be NaN (unwritten skeleton)
    assert np.isnan(res_ds["cpc_precipitation"].values[0, :3]).all()
    # Days 3..8 (the original 5 dates) should be preserved
    np.testing.assert_array_equal(
        res_ds["cpc_precipitation"].values[0, 3:],
        np.ones(5, dtype=np.float32),
    )

  # 2. Prepending via append_dates also seamlessly delegates to expand_date_range
  even_earlier = pd.date_range("2019-12-30", periods=2, freq="D")
  start_idx, end_idx = writer.append_dates(Product.CPC, even_earlier)
  assert start_idx == 0
  assert end_idx == 2
  info = writer.get_store_info(Product.CPC)
  assert len(info["dates"]) == 10
  with xr.open_zarr(store_path) as res_ds2:
    # Original data at 2020-01-10..14 should now be shifted to indices 5..10
    np.testing.assert_array_equal(
        res_ds2["cpc_precipitation"].values[0, 5:],
        np.ones(5, dtype=np.float32),
    )
    # The earlier 5 days should be NaN
    assert np.isnan(res_ds2["cpc_precipitation"].values[0, :5]).all()

  # 3. Partial overlap without expanding date range
  overlap_dates = pd.date_range("2020-01-10", periods=3, freq="D")
  all_dates3, indices3 = writer.expand_date_range(Product.CPC, overlap_dates)
  assert len(all_dates3) == 10
  assert indices3 == [5, 6, 7]


def test_reject_mismatched_dates_on_append_basins(writer: MultiMetZarrWriter):
  basins = ["basin_1"]
  dates = pd.date_range("2020-01-01", periods=5, freq="D")
  ds = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 5), dtype=np.float32))},
      coords={"basin": basins, "date": dates.values},
  )
  writer.write_or_append(ds, Product.CPC)

  ds_mismatched_dates = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 4), dtype=np.float32))},
      coords={"basin": ["basin_2"], "date": pd.date_range("2020-01-01", periods=4).values},
  )
  with pytest.raises(ValueError, match="dates mismatch"):
    writer.append_basins(Product.CPC, ds_mismatched_dates)


def test_write_or_append_prepending_and_rewriting(writer: MultiMetZarrWriter):
  """Verifies write_or_append supports prepending earlier dates and updating overlapping dates."""
  basins = ["basin_1"]
  # Initial dataset: 2020-01-05 to 2020-01-07 (3 days, all 1.0)
  dates_init = pd.date_range("2020-01-05", periods=3, freq="D")
  ds_init = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 3), dtype=np.float32))},
      coords={"basin": basins, "date": dates_init.values},
  )
  writer.write_or_append(ds_init, Product.CPC)

  # Second dataset: 2020-01-03 to 2020-01-05 (prepending 03, 04, overlapping 05 with 2.0)
  dates_update = pd.date_range("2020-01-03", periods=3, freq="D")
  ds_update = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.full((1, 3), 2.0, dtype=np.float32))},
      coords={"basin": basins, "date": dates_update.values},
  )
  writer.write_or_append(ds_update, Product.CPC)

  store_path = writer.get_store_path(Product.CPC)
  with xr.open_zarr(store_path) as res_ds:
    # Dates should now span 2020-01-03 to 2020-01-07 (5 days)
    assert len(res_ds["date"]) == 5
    vals = res_ds["cpc_precipitation"].values[0]
    # 2020-01-03, 04, 05 should be 2.0
    np.testing.assert_array_equal(vals[:3], [2.0, 2.0, 2.0])
    # 2020-01-06, 07 should still be 1.0
    np.testing.assert_array_equal(vals[3:], [1.0, 1.0])


def test_expand_date_range_forecast_product(writer: MultiMetZarrWriter):
  """Verifies date expansion (prepending and postpending) works on 3D forecast stores."""
  basins = ["basin_A"]
  dates = pd.date_range("2020-01-05", periods=2, freq="D")
  leads = pd.to_timedelta(range(1, 11), unit="D")
  shape = (len(basins), len(dates), len(leads))
  data_vars = {
      band: (
          ["basin", "date", "lead_time"],
          np.ones(shape, dtype=np.float32),
      )
      for band in PRODUCT_BANDS[Product.HRES]
  }
  ds = xr.Dataset(
      data_vars=data_vars,
      coords={"basin": basins, "date": dates.values, "lead_time": leads.values},
  )
  writer.write_or_append(ds, Product.HRES)

  # Prepend 2 days and postpend 1 day
  req_dates = pd.date_range("2020-01-03", periods=5, freq="D")
  all_dates, indices = writer.expand_date_range(Product.HRES, req_dates)
  assert len(all_dates) == 5
  assert indices == [0, 1, 2, 3, 4]

  store_path = writer.get_store_path(Product.HRES)
  with xr.open_zarr(store_path) as res_ds:
    vals = res_ds["hres_surface_net_solar_radiation"].values
    assert vals.shape == (1, 5, 10)
    # Days 0, 1 (prepended 03, 04) should be NaN
    assert np.isnan(vals[0, :2, :]).all()
    # Days 2, 3 (original 05, 06) should be 1.0
    np.testing.assert_array_equal(vals[0, 2:4, :], np.ones((2, 10), dtype=np.float32))
    # Day 4 (postpended 07) should be NaN
    assert np.isnan(vals[0, 4, :]).all()

