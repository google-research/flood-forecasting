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
from multimet import hres_schema

pytestmark = pytest.mark.canary

MIN_FINITE_FRACTION = 0.05

# Archive units: degC, kPa, mm/day, W m-2.
PLAUSIBLE_RANGES = {
    "temperature_2m_mean": (-95.0, 60.0),
    "temperature_2m_min": (-95.0, 60.0),
    "temperature_2m_max": (-95.0, 65.0),
    "surface_pressure_mean": (40.0, 110.0),
    "total_precipitation_sum": (0.0, 2000.0),
    "surface_net_solar_radiation_mean": (-1.0, 450.0),
    "surface_net_thermal_radiation_mean": (-300.0, 150.0),
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
class TestEcmwfOpenData:
  """Canaries for the 0.25-degree ECMWF open data feed."""

  def test_recent_forecast_is_published(self) -> None:
    source = hres_module.ECMWFOpenDataSource()
    today = pd.Timestamp(datetime.date.today())
    found = []
    for back in range(5):
      date = today - pd.Timedelta(days=back)
      index = (
          f"{source._run_prefix(date)}-"
          f"{hres_schema.FORECAST_STEPS[-1]}h-oper-fc.index"
      )
      if source.fs.exists(index):
        found.append(date.strftime("%Y-%m-%d"))
    assert found, "no complete 0p25 ECMWF Open Data run in the last 5 days"


@pytest.fixture(scope="module")
def extracted() -> dict[str, np.ndarray]:
  """One real July run date from ECMWF Open Data (about 1 minute)."""
  pytest.importorskip(
      "eccodes", reason="eccodes is required to decode ECMWF GRIB2"
  )
  return hres_module.ECMWFOpenDataSource().extract_date(
      pd.Timestamp("2025-07-15")
  )


@requires_gcsfs
@pytest.mark.slow
class TestEcmwfOpenDataDecoding:
  """End-to-end decoding of one real Open Data run."""

  def test_every_variable_is_complete(
      self, extracted: dict[str, np.ndarray]
  ) -> None:
    assert tuple(extracted) == hres_schema.VARIABLES
    for name, values in extracted.items():
      assert values.shape == (10, 721, 1440), name
      assert np.isfinite(values).all(), name

  def test_values_are_in_archive_units(
      self, extracted: dict[str, np.ndarray]
  ) -> None:
    for name, (low, high) in PLAUSIBLE_RANGES.items():
      values = extracted[name]
      assert low <= float(values.min()) and float(values.max()) <= high, name

  def test_grid_is_not_flipped(self, extracted: dict[str, np.ndarray]) -> None:
    # July: Antarctica (south) is much colder than the Arctic (north).
    t = extracted["temperature_2m_mean"][0]
    lat = hres_schema.LATITUDES
    assert t[lat < -70].mean() < t[lat > 70].mean() - 20.0

  def test_grid_is_not_shifted(self, extracted: dict[str, np.ndarray]) -> None:
    # Tibet (32N, 90E) has low surface pressure; the US Gulf Coast
    # (32N, 90W = 270E) is near sea level.
    sp = extracted["surface_pressure_mean"][0]
    i = int(np.argmin(np.abs(hres_schema.LATITUDES - 32.0)))
    j_tibet = int(np.argmin(np.abs(hres_schema.LONGITUDES - 90.0)))
    j_gulf = int(np.argmin(np.abs(hres_schema.LONGITUDES - 270.0)))
    assert sp[i, j_tibet] < 65.0
    assert sp[i, j_gulf] > 95.0


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
