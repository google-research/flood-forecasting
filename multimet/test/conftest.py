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

"""Shared fixtures for the gridded archive builder tests.

Every fixture here produces *synthetic* data on the local filesystem. The unit
and integration suites never touch NOAA PSL, WeatherBench 2, ECMWF Open Data,
or GCS, so they are hermetic and safe to run in CI.

The one exception is ``test_canary.py``, which deliberately talks to those live
services. Those tests are skipped unless ``--run-canary`` is passed; see
:func:`pytest_collection_modifyitems` below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet import build_hres_archive as hres_module

_CANARY_FLAG = "--run-canary"


def pytest_addoption(parser: pytest.Parser) -> None:
  """Registers the opt-in flag for live upstream canaries."""
  parser.addoption(
      _CANARY_FLAG,
      action="store_true",
      default=False,
      help=(
          "Run canaries against live third-party feeds (NOAA PSL, "
          "WeatherBench 2, ECMWF Open Data). Requires network access."
      ),
  )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
  """Skips canary tests unless they were explicitly requested."""
  if config.getoption(_CANARY_FLAG):
    return

  skip_canary = pytest.mark.skip(
      reason=f"live upstream canary; pass {_CANARY_FLAG} to run"
  )
  for item in items:
    if "canary" in item.keywords:
      item.add_marker(skip_canary)


PSL_LATS = np.linspace(89.75, -89.75, 360, dtype=np.float32)
PSL_LONS = np.linspace(0.25, 359.75, 720, dtype=np.float32)
PSL_MISSING_VALUE = -9.96921e36

FAKE_HRES_LATS = np.linspace(-90.0, 90.0, 4, dtype=np.float32)
FAKE_HRES_LONS = np.linspace(0.0, 315.0, 8, dtype=np.float32)


def make_psl_precip_array(
    dates: pd.DatetimeIndex,
    *,
    fill_value: float = 1.0,
    missing_cells: Iterable[tuple[int, int, int]] = (),
) -> np.ndarray:
  """Builds a synthetic NOAA PSL ``precip`` array."""
  data = np.empty((len(dates), len(PSL_LATS), len(PSL_LONS)), dtype=np.float32)
  for i in range(len(dates)):
    data[i] = fill_value + i
  for index in missing_cells:
    data[index] = PSL_MISSING_VALUE
  return data


def write_psl_netcdf(
    directory: Path,
    year: int,
    dates: pd.DatetimeIndex,
    *,
    fill_value: float = 1.0,
    missing_cells: Iterable[tuple[int, int, int]] = (),
) -> Path:
  """Writes a synthetic ``precip.{year}.nc`` file in NOAA PSL layout."""
  data = make_psl_precip_array(
      dates, fill_value=fill_value, missing_cells=missing_cells
  )
  dataset = xr.Dataset(
      data_vars={"precip": (["time", "lat", "lon"], data)},
      coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
  )
  path = directory / f"precip.{year}.nc"
  dataset.to_netcdf(path)
  dataset.close()
  return path


@pytest.fixture
def psl_cache(tmp_path: Path) -> Path:
  """Directory used as the builder's NetCDF download cache."""
  cache = tmp_path / "psl_cache"
  cache.mkdir()
  return cache


@pytest.fixture
def write_psl_year(psl_cache: Path) -> Callable[..., Path]:
  """Factory that drops a synthetic NOAA PSL year into the download cache."""

  def _write(
      year: int,
      start: str,
      end: str,
      *,
      fill_value: float = 1.0,
      missing_cells: Iterable[tuple[int, int, int]] = (),
  ) -> Path:
    dates = pd.date_range(start, end, freq="D")
    return write_psl_netcdf(
        psl_cache,
        year,
        dates,
        fill_value=fill_value,
        missing_cells=missing_cells,
    )

  return _write


class FakeHRESSource:
  """In-memory stand-in for the HRES upstream sources."""

  def __init__(
      self,
      available: Iterable[str] | None = None,
      offset: float = 0.0,
      latitudes: np.ndarray = FAKE_HRES_LATS,
      longitudes: np.ndarray = FAKE_HRES_LONS,
  ):
    self.available = None if available is None else set(available)
    self.offset = offset
    self.latitudes = latitudes
    self.longitudes = longitudes
    self.requested: list[str] = []

  def value_for(self, date: pd.Timestamp, variable: str) -> float:
    """Deterministic value written for a given date/variable pair."""
    day_of_year = float(pd.Timestamp(date).dayofyear)
    variable_index = float(hres_module.HRES_VARIABLES.index(variable))
    return day_of_year + 100.0 * variable_index + self.offset

  def extract_date(
      self,
      date: pd.Timestamp,
      target_lat: np.ndarray | None = None,
      target_lon: np.ndarray | None = None,
  ) -> dict[str, np.ndarray] | None:
    """Mirrors the ``extract_date`` contract of the real source classes."""
    del target_lat, target_lon
    key = pd.Timestamp(date).strftime("%Y-%m-%d")
    self.requested.append(key)
    if self.available is not None and key not in self.available:
      return None
    shape = (
        hres_module.NUM_LEAD_DAYS,
        len(self.latitudes),
        len(self.longitudes),
    )
    return {
        variable: np.full(shape, self.value_for(date, variable), np.float32)
        for variable in hres_module.HRES_VARIABLES
    }


@pytest.fixture
def fake_hres_source(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., FakeHRESSource]:
  """Factory that swaps HRES upstream sources for an in-memory fake."""

  def _install(
      available: Iterable[str] | None = None,
      offset: float = 0.0,
  ) -> FakeHRESSource:
    source = FakeHRESSource(available=available, offset=offset)

    def fake_init_worker(
        project: str | None = None,
        wb2_zarr: str = "",
        ecmwf_open_data_bucket: str = "",
        need_wb2: bool = True,
        need_open_data: bool = True,
    ) -> None:
      del project, wb2_zarr, ecmwf_open_data_bucket, need_wb2, need_open_data
      hres_module._worker_wb2 = source  # type: ignore[assignment]
      hres_module._worker_open_data = source  # type: ignore[assignment]
      hres_module._target_lat = source.latitudes
      hres_module._target_lon = source.longitudes

    monkeypatch.setattr(hres_module, "_init_worker", fake_init_worker)
    monkeypatch.setattr(
        hres_module, "WB2_CUTOFF_DATE", pd.Timestamp("2100-01-01")
    )
    return source

  return _install
