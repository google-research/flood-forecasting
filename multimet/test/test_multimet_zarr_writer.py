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
