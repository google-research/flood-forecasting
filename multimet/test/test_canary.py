# Copyright 2026 Google LLC
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

"""Canaries for the upstream feeds the gridded archive builders depend on."""

from __future__ import annotations

import datetime
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimet import build_cpc_archive as cpc_module
from multimet import build_hres_archive as hres_module
from multimet import build_imerg_archive as imerg_module

pytestmark = pytest.mark.canary

MIN_FINITE_FRACTION = 0.05

PLAUSIBLE_RANGES = {
    "temperature_2m": (180.0, 340.0),
    "surface_pressure": (30000.0, 110000.0),
    "total_precipitation": (0.0, 3.0),
}

requires_gcsfs = pytest.mark.skipif(
    hres_module.gcsfs is None,
    reason="gcsfs is not installed; cloud sources cannot be reached",
)


def _finite_fraction(values: np.ndarray) -> float:
  """Returns the fraction of ``values`` that are finite."""
  return float(np.isfinite(values).mean())


def _assert_field_is_live(name: str, values: np.ndarray) -> None:
  """Asserts an extracted field contains real, physically plausible data."""
  finite = _finite_fraction(values)
  assert finite >= MIN_FINITE_FRACTION, (
      f"{name}: only {finite:.1%} of cells are finite. The upstream source "
      "is probably returning nothing."
  )

  if name in PLAUSIBLE_RANGES:
    low, high = PLAUSIBLE_RANGES[name]
    finite_values = values[np.isfinite(values)]
    observed_low = float(finite_values.min())
    observed_high = float(finite_values.max())
    assert low <= observed_low and observed_high <= high, (
        f"{name}: observed range [{observed_low:.4g}, {observed_high:.4g}] "
        f"falls outside the plausible range [{low:g}, {high:g}]."
    )


class TestNoaaPslCpc:
  """Canaries for the NOAA PSL CPC precipitation feed."""

  def test_yearly_file_is_published(self) -> None:
    year = datetime.date.today().year - 1
    url = cpc_module.NOAA_PSL_URL_TEMPLATE.format(year=year)
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "OpenMultiMet/1.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
      assert response.status == 200
      size = int(response.headers.get("Content-Length", 0))

    assert size > 1024 * 1024, f"{url} is only {size} bytes."

  def test_yearly_netcdf_still_matches_expected_layout(
      self, tmp_path: Path
  ) -> None:
    year = datetime.date.today().year - 1
    path = cpc_module.ensure_psl_cpc_netcdf(year, cache_dir=str(tmp_path))

    dataset = cpc_module.process_cpc_netcdf_to_dataset(
        path,
        target_start_date=pd.Timestamp(f"{year}-06-01"),
        target_end_date=pd.Timestamp(f"{year}-06-03"),
    )
    assert dataset is not None, "no dates survived the filter"

    assert cpc_module.CPC_VARIABLE in dataset
    assert dataset.sizes["latitude"] == len(cpc_module.CPC_LATS)
    assert dataset.sizes["longitude"] == len(cpc_module.CPC_LONS)

    latitudes = dataset["latitude"].values
    longitudes = dataset["longitude"].values
    assert np.all(np.diff(latitudes) > 0), "latitude is no longer ascending"
    assert longitudes.min() >= -180.0 and longitudes.max() < 180.0

    _assert_field_is_live(
        "cpc_precipitation", dataset[cpc_module.CPC_VARIABLE].values
    )


@requires_gcsfs
class TestWeatherBench2:
  """Canaries for the WeatherBench 2 HRES archive (dates <= 2023-01-10)."""

  @pytest.fixture(scope="class")
  def source(self) -> hres_module.WeatherBench2Source:
    return hres_module.WeatherBench2Source()

  def test_archive_is_openable(
      self, source: hres_module.WeatherBench2Source
  ) -> None:
    assert len(source.latitudes) == len(hres_module.HRES_LATS)
    assert len(source.longitudes) == len(hres_module.HRES_LONS)

  def test_date_extraction_returns_live_data(
      self, source: hres_module.WeatherBench2Source
  ) -> None:
    extracted = source.extract_date(pd.Timestamp("2022-06-01"))

    assert extracted is not None

    for name in hres_module.HRES_VARIABLES:
      assert name in extracted
      assert extracted[name].shape == (
          hres_module.NUM_LEAD_DAYS,
          len(hres_module.HRES_LATS),
          len(hres_module.HRES_LONS),
      )

    for name in ("temperature_2m", "surface_pressure", "total_precipitation"):
      _assert_field_is_live(name, extracted[name])


@requires_gcsfs
class TestEcmwfOpenData:
  """Canaries for the 0.25-degree ECMWF open data feed."""

  @pytest.fixture(scope="class")
  def source(self) -> hres_module.ECMWFOpenDataSource:
    return hres_module.ECMWFOpenDataSource()

  def _has_0p25_index(
      self, source: hres_module.ECMWFOpenDataSource, date: pd.Timestamp
  ) -> bool:
    stamp = date.strftime("%Y%m%d")
    prefix = f"ecmwf-open-data/{stamp}/00z/ifs/0p25/oper/{stamp}000000"
    return bool(source.fs.exists(f"{prefix}-24h-oper-fc.index"))

  def test_recent_forecast_is_published(
      self, source: hres_module.ECMWFOpenDataSource
  ) -> None:
    today = pd.Timestamp(datetime.date.today())
    found = [
        (today - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        for back in range(5)
        if self._has_0p25_index(source, today - pd.Timedelta(days=back))
    ]
    assert found, "no 0p25 ECMWF open data forecast found in the last 5 days"


@requires_gcsfs
@pytest.mark.slow
class TestEcmwfOpenDataDecoding:
  """GRIB decoding canaries for 0.25-degree ECMWF Open Data."""

  @pytest.fixture(autouse=True)
  def _require_eccodes(self) -> None:
    pytest.importorskip(
        "eccodes", reason="eccodes is required to decode ECMWF GRIB2"
    )

  def test_grib_decodes_to_live_fields(self) -> None:
    source = hres_module.ECMWFOpenDataSource()
    extracted = source.extract_date(
        pd.Timestamp("2024-06-01"),
        hres_module.HRES_LATS,
        hres_module.HRES_LONS,
    )

    assert extracted is not None

    for name in hres_module.HRES_VARIABLES:
      assert extracted[name].shape == (
          hres_module.NUM_LEAD_DAYS,
          len(hres_module.HRES_LATS),
          len(hres_module.HRES_LONS),
      )

    for name in ("temperature_2m", "surface_pressure", "total_precipitation"):
      _assert_field_is_live(name, extracted[name])

    precipitation = extracted["total_precipitation"]
    finite = precipitation[np.isfinite(precipitation)]
    assert finite.min() >= 0.0


class TestNasaCmrImerg:
  """Canaries for the NASA Earthdata CMR IMERG V07 granule discovery feed."""

  def test_half_hourly_cmr_returns_48_granules(self) -> None:
    urls = imerg_module.query_cmr_granules(
        imerg_module.IMERG_HHR_SHORT_NAME, pd.Timestamp("2024-06-01")
    )
    assert len(urls) == 48
    assert all(url.endswith((".RT-H5", ".HDF5")) for url in urls)

  def test_daily_cmr_returns_granule(self) -> None:
    urls = imerg_module.query_cmr_granules(
        imerg_module.IMERG_DAILY_SHORT_NAME, pd.Timestamp("2024-06-01")
    )
    assert len(urls) == 1
    assert urls[0].endswith(".nc4")
