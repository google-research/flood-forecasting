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

"""Unit tests for :mod:`multimet.build_hres_archive`."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet import build_hres_archive as hres_module
from multimet import hres_schema, storage
from multimet.build_hres_archive import (
    HRES_ATTRS,
    HRES_VARIABLES,
    MIN_START_DATE,
    NUM_LEAD_DAYS,
    build_arg_parser,
    build_batch_dataset,
    resolve_date_range,
    write_batch_in_place,
    write_batch_to_zarr,
)
from multimet.test.conftest import FAKE_HRES_LATS, FAKE_HRES_LONS

pytestmark = pytest.mark.unit


def make_batch(
    dates: list[str],
    value_for: Callable[[int, str], float] = lambda i, var: float(i),
) -> tuple[list[pd.Timestamp], dict[str, list[np.ndarray]]]:
  """Builds ``(batch_dates, batch_data)`` inputs for ``build_batch_dataset``."""
  timestamps = [pd.Timestamp(d) for d in dates]
  shape = (NUM_LEAD_DAYS, len(FAKE_HRES_LATS), len(FAKE_HRES_LONS))
  batch_data = {
      var: [
          np.full(shape, value_for(i, var), dtype=np.float32)
          for i in range(len(timestamps))
      ]
      for var in HRES_VARIABLES
  }
  return timestamps, batch_data


def make_batch_dataset(
    dates: list[str],
    value_for: Callable[[int, str], float] = lambda i, var: float(i),
) -> xr.Dataset:
  """Convenience wrapper returning a ready-to-write batch dataset."""
  timestamps, batch_data = make_batch(dates, value_for)
  return build_batch_dataset(
      timestamps, batch_data, FAKE_HRES_LATS, FAKE_HRES_LONS
  )


class TestBuildBatchDataset:
  """The on-disk schema."""

  def test_dimension_order(self) -> None:
    dataset = make_batch_dataset(["2025-01-01", "2025-01-02"])

    for var in HRES_VARIABLES:
      assert dataset[var].dims == ("time", "lead_time", "latitude", "longitude")

  def test_variables_are_the_schema_variables(self) -> None:
    dataset = make_batch_dataset(["2025-01-01"])

    assert tuple(dataset.data_vars) == hres_schema.VARIABLES

  def test_variable_attributes_carry_units(self) -> None:
    dataset = make_batch_dataset(["2025-01-01"])

    assert dataset["temperature_2m_mean"].attrs["units"] == "degC"
    assert dataset["total_precipitation_sum"].attrs["units"] == "mm"
    assert dataset["surface_pressure_mean"].attrs["units"] == "kPa"
    assert dataset["surface_net_solar_radiation_mean"].attrs["units"] == "W m-2"

  def test_lead_time_is_one_based(self) -> None:
    dataset = make_batch_dataset(["2025-01-01"])

    np.testing.assert_array_equal(
        dataset["lead_time"].values, np.arange(1, NUM_LEAD_DAYS + 1)
    )
    assert dataset["lead_time"].dtype == np.int32

  def test_time_coordinate_preserves_input_order(self) -> None:
    dates = ["2025-01-05", "2025-01-06", "2025-01-07"]

    dataset = make_batch_dataset(dates)

    assert list(pd.to_datetime(dataset["time"].values)) == [
        pd.Timestamp(d) for d in dates
    ]

  def test_global_attributes_include_schema_version(self) -> None:
    dataset = make_batch_dataset(["2025-01-01"])

    assert dataset.attrs == HRES_ATTRS
    assert dataset.attrs["schema_version"] == hres_schema.SCHEMA_VERSION

  def test_attributes_are_copied_not_shared(self) -> None:
    dataset = make_batch_dataset(["2025-01-01"])

    dataset.attrs["title"] = "mutated"
    dataset["temperature_2m_mean"].attrs["units"] = "K"

    assert HRES_ATTRS["title"] != "mutated"
    assert hres_schema.VARIABLE_ATTRS["temperature_2m_mean"]["units"] == "degC"


class TestWriteBatchToZarr:
  """Initial write and append semantics."""

  def test_initial_write_creates_store(self, tmp_path: Path) -> None:
    target = str(tmp_path / "hres.zarr")

    write_batch_to_zarr(
        make_batch_dataset(["2025-01-01", "2025-01-02"]),
        target,
        is_initial_write=True,
    )

    with xr.open_zarr(target, consolidated=False) as store:
      assert len(store["time"]) == 2
      assert set(store.data_vars) == set(HRES_VARIABLES)

  def test_append_extends_the_time_axis(self, tmp_path: Path) -> None:
    target = str(tmp_path / "hres.zarr")
    write_batch_to_zarr(
        make_batch_dataset(["2025-01-01"]), target, is_initial_write=True
    )

    write_batch_to_zarr(
        make_batch_dataset(["2025-01-02", "2025-01-03"]), target
    )

    with xr.open_zarr(target, consolidated=False) as store:
      assert list(pd.to_datetime(store["time"].values)) == list(
          pd.date_range("2025-01-01", "2025-01-03")
      )

  def test_chunking_isolates_each_run_date(self, tmp_path: Path) -> None:
    target = str(tmp_path / "hres.zarr")

    write_batch_to_zarr(
        make_batch_dataset(["2025-01-01", "2025-01-02"]),
        target,
        is_initial_write=True,
    )

    with xr.open_zarr(target, consolidated=False) as store:
      chunks = store["temperature_2m_mean"].encoding["chunks"]
    assert chunks[0] == 1
    assert chunks[1] == NUM_LEAD_DAYS


class TestWriteBatchInPlace:
  """Overwriting dates that already exist in the store."""

  @pytest.fixture
  def populated_store(self, tmp_path: Path) -> str:
    target = str(tmp_path / "hres.zarr")
    dates = [f"2025-01-0{d}" for d in range(1, 6)]
    write_batch_to_zarr(
        make_batch_dataset(dates, value_for=lambda i, var: float(i)),
        target,
        is_initial_write=True,
    )
    return target

  def test_non_contiguous_dates_are_updated(self, populated_store: str) -> None:
    replacement = make_batch_dataset(
        ["2025-01-01", "2025-01-05"], value_for=lambda i, var: 77.0
    )

    write_batch_in_place(replacement, populated_store)

    with xr.open_zarr(populated_store, consolidated=False) as store:
      values = store["temperature_2m_mean"].values
    assert values[0].min() == pytest.approx(77.0)
    assert values[4].min() == pytest.approx(77.0)
    for index in (1, 2, 3):
      assert values[index].min() == pytest.approx(float(index))

  def test_unknown_date_raises(self, populated_store: str) -> None:
    with pytest.raises(ValueError, match="2026-06-01"):
      write_batch_in_place(make_batch_dataset(["2026-06-01"]), populated_store)


# ---------------------------------------------------------------------------
# ECMWF Open Data reader
# ---------------------------------------------------------------------------

_DATE = pd.Timestamp("2025-07-15")
_PREFIX = "ecmwf-open-data/20250715/00z/ifs/0p25/oper/20250715000000"


class FakeOpenDataFS:
  """Minimal stand-in for ``gcsfs.GCSFileSystem`` over one Open Data run."""

  def __init__(
      self,
      steps: tuple[int, ...] = hres_schema.FORECAST_STEPS,
      drop_param: tuple[str, int] | None = None,
      folder_exists: bool = True,
  ):
    self.steps = steps
    self.drop_param = drop_param
    self.folder_exists = folder_exists
    self.ranges_read: list[tuple[str, int, int]] = []

  def ls(self, path: str, detail: bool = False, refresh: bool = False) -> list:
    del detail, refresh
    if not self.folder_exists:
      raise FileNotFoundError(path)
    names = []
    for step in self.steps:
      names += [f"{_PREFIX}-{step}h-oper-fc.index",
                f"{_PREFIX}-{step}h-oper-fc.grib2"]
    return names

  def open(self, path: str, mode: str = "r") -> io.StringIO:
    del mode
    step = int(path.rsplit("-", 3)[1].removesuffix("h"))
    lines = [json.dumps({"levtype": "pl", "param": "t", "_offset": 0,
                         "_length": 1})]
    for i, param in enumerate(hres_schema.GRIB_PARAMS):
      if (param, step) == self.drop_param:
        continue
      lines.append(json.dumps({
          "levtype": "sfc", "param": param, "step": str(step),
          "_offset": 1000 * i, "_length": 10,
      }))
    return io.StringIO("\n".join(lines) + "\n")

  def cat_ranges(
      self, paths: list[str], starts: list[int], ends: list[int]
  ) -> list[bytes]:
    out = []
    for path, start, end in zip(paths, starts, ends, strict=True):
      self.ranges_read.append((path, start, end))
      step = int(path.rsplit("-", 3)[1].removesuffix("h"))
      param = hres_schema.GRIB_PARAMS[start // 1000]
      out.append(f"{param}:{step}".encode())
    return out


def _fake_decode(
    message: bytes, *, param: str, step: int, date: pd.Timestamp
) -> np.ndarray:
  """Returns a field whose value encodes (param, step)."""
  assert message.decode() == f"{param}:{step}"
  assert date == _DATE
  value = {
      "2t": 273.15 + step,
      "sp": 100000.0,
      "tp": 0.001 * step / 24,
      "ssr": 86400.0 * step / 24,
      "str": -86400.0 * step / 24,
  }[param]
  return np.full((721, 1440), value)


@pytest.fixture
def open_data_source(monkeypatch: pytest.MonkeyPatch) -> Callable[..., object]:
  """Builds an ``ECMWFOpenDataSource`` backed by :class:`FakeOpenDataFS`."""
  monkeypatch.setattr(hres_module, "decode_grib_message", _fake_decode)

  def _make(**fs_kwargs: object) -> hres_module.ECMWFOpenDataSource:
    source = hres_module.ECMWFOpenDataSource.__new__(
        hres_module.ECMWFOpenDataSource
    )
    source.bucket = "ecmwf-open-data"
    source.fs = FakeOpenDataFS(**fs_kwargs)
    return source

  return _make


class TestECMWFOpenDataSource:
  """Reading one run date from Open Data."""

  def test_reads_only_the_required_messages(
      self, open_data_source: Callable[..., object]
  ) -> None:
    source = open_data_source()

    source.extract_date(_DATE)

    assert len(source.fs.ranges_read) == len(hres_schema.required_fields())

  def test_returns_daily_values_in_archive_units(
      self, open_data_source: Callable[..., object]
  ) -> None:
    result = open_data_source().extract_date(_DATE)

    assert tuple(result) == hres_schema.VARIABLES
    assert result["temperature_2m_mean"][0, 0, 0] == pytest.approx(13.5)
    assert result["temperature_2m_max"][9, 0, 0] == pytest.approx(240.0)
    assert result["total_precipitation_sum"][4, 0, 0] == pytest.approx(1.0)
    assert result["surface_pressure_mean"][0, 0, 0] == pytest.approx(100.0)
    assert result["surface_net_solar_radiation_mean"][0, 0, 0] == (
        pytest.approx(1.0)
    )
    assert result["surface_net_thermal_radiation_mean"][0, 0, 0] == (
        pytest.approx(-1.0)
    )

  def test_unpublished_run_is_upstream_missing(
      self, open_data_source: Callable[..., object]
  ) -> None:
    with pytest.raises(storage.UpstreamDataMissingError):
      open_data_source(folder_exists=False).extract_date(_DATE)

  def test_partly_published_run_is_upstream_missing(
      self, open_data_source: Callable[..., object]
  ) -> None:
    steps = tuple(s for s in hres_schema.FORECAST_STEPS if s <= 120)

    with pytest.raises(storage.UpstreamDataMissingError, match="126"):
      open_data_source(steps=steps).extract_date(_DATE)

  def test_missing_variable_in_a_published_file_raises(
      self, open_data_source: Callable[..., object]
  ) -> None:
    source = open_data_source(drop_param=("ssr", 48))

    with pytest.raises(ValueError, match="ssr"):
      source.extract_date(_DATE)


class TestDecodeGribMessage:
  """Decoding real GRIB2 bytes with ecCodes (skipped without ecCodes)."""

  @pytest.fixture
  def eccodes(self) -> object:
    return pytest.importorskip("eccodes")

  def _message(self, eccodes: object, field: np.ndarray, **keys: object) -> bytes:
    gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
      settings = {
          "Ni": 1440,
          "Nj": 721,
          "latitudeOfFirstGridPointInDegrees": 90.0,
          "latitudeOfLastGridPointInDegrees": -90.0,
          "longitudeOfFirstGridPointInDegrees": -180.0,
          "longitudeOfLastGridPointInDegrees": 179.75,
          "iDirectionIncrementInDegrees": 0.25,
          "jDirectionIncrementInDegrees": 0.25,
          "jScansPositively": 0,
          "dataDate": 20250715,
          "dataTime": 0,
          "paramId": 167,
          "step": 24,
      }
      settings.update(keys)
      for key, value in settings.items():
        eccodes.codes_set(gid, key, value)
      eccodes.codes_set_values(gid, field.ravel())
      return eccodes.codes_get_message(gid)
    finally:
      eccodes.codes_release(gid)

  def test_open_data_layout_is_put_on_the_schema_grid(
      self, eccodes: object
  ) -> None:
    lat = hres_schema.LATITUDES.astype(np.float64)
    lon = hres_schema.LONGITUDES.astype(np.float64)
    expected = 250.0 + (lat[:, None] + 90.0) / 4.0 + lon[None, :] / 40.0
    grib_order = np.roll(expected[::-1, :], -720, axis=1)

    result = hres_module.decode_grib_message(
        self._message(eccodes, grib_order), param="2t", step=24, date=_DATE
    )

    np.testing.assert_allclose(result, expected, atol=0.01)

  def test_wrong_step_raises(self, eccodes: object) -> None:
    message = self._message(eccodes, np.full(721 * 1440, 280.0))

    with pytest.raises(ValueError, match="not the one requested"):
      hres_module.decode_grib_message(
          message, param="2t", step=48, date=_DATE
      )


# ---------------------------------------------------------------------------
# Retry and date-range rules
# ---------------------------------------------------------------------------


class TestExtractSingleDate:
  """Missing, failed and successful dates."""

  @staticmethod
  def _install(monkeypatch: pytest.MonkeyPatch, behaviour: object) -> list:
    calls: list[pd.Timestamp] = []

    class Source:

      def extract_date(self, date: pd.Timestamp) -> dict[str, np.ndarray]:
        calls.append(date)
        if isinstance(behaviour, Exception):
          raise behaviour
        return behaviour

    monkeypatch.setattr(hres_module, "_worker_source", Source())
    monkeypatch.setattr(hres_module.time, "sleep", lambda _: None)
    return calls

  def test_success_returns_the_data(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    payload = {"temperature_2m_mean": np.zeros(1)}
    self._install(monkeypatch, payload)

    _, data = hres_module._extract_single_date(_DATE)

    assert data is payload

  def test_unpublished_date_returns_none(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    calls = self._install(
        monkeypatch, storage.UpstreamDataMissingError("not yet")
    )

    assert hres_module._extract_single_date(_DATE) == (_DATE, None)
    assert len(calls) == 1

  def test_data_problem_raises_without_retry(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    calls = self._install(monkeypatch, ValueError("bad grid"))

    with pytest.raises(ValueError, match="bad grid"):
      hres_module._extract_single_date(_DATE, max_retries=3)
    assert len(calls) == 1

  def test_network_error_is_retried_then_raised(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    calls = self._install(monkeypatch, ConnectionError("network down"))

    with pytest.raises(RuntimeError, match="network down"):
      hres_module._extract_single_date(_DATE, max_retries=3)
    assert len(calls) == 3


class TestResolveDateRange:
  """Choice of run dates."""

  def test_explicit_dates(self) -> None:
    start, end = resolve_date_range(
        "2025-01-01", "2025-01-05", default_start=None
    )
    assert (start, end) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05"))

  def test_default_start_is_used_without_start_date(self) -> None:
    start, _ = resolve_date_range(
        None, "2025-01-05", default_start=pd.Timestamp("2025-01-03")
    )
    assert start == pd.Timestamp("2025-01-03")

  def test_end_date_defaults_to_today(self) -> None:
    _, end = resolve_date_range("2025-01-01", None, default_start=None)
    assert end == pd.Timestamp(pd.Timestamp.today().date())

  def test_new_archive_needs_start_date(self) -> None:
    with pytest.raises(ValueError, match="--start_date is required"):
      resolve_date_range(None, "2025-01-05", default_start=None)

  def test_dates_before_open_data_raise(self) -> None:
    day_before = (MIN_START_DATE - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="2024-03-06"):
      resolve_date_range(day_before, "2025-01-05", default_start=None)


class TestCommandLine:
  """``build-hres-archive`` argument parsing."""

  def test_requires_target_zarr(self) -> None:
    with pytest.raises(SystemExit):
      build_arg_parser().parse_args([])

  def test_defaults_with_target_zarr(self) -> None:
    args = build_arg_parser().parse_args(["--target_zarr", "/tmp/out.zarr"])

    assert args.target_zarr == "/tmp/out.zarr"
    assert args.start_date is None
    assert args.end_date is None
    assert args.project is None
    assert args.ecmwf_open_data_bucket == "ecmwf-open-data"
    assert args.batch_size == 10
    assert args.overwrite is False
    assert args.in_place is False
    assert args.failure_log is None

  def test_explicit_values(self) -> None:
    args = build_arg_parser().parse_args([
        "--start_date",
        "2025-01-01",
        "--end_date",
        "2025-01-31",
        "--target_zarr",
        "/tmp/out.zarr",
        "--batch_size",
        "4",
        "--num_workers",
        "2",
        "--in_place",
    ])

    assert args.start_date == "2025-01-01"
    assert args.end_date == "2025-01-31"
    assert args.batch_size == 4
    assert args.num_workers == 2
    assert args.in_place is True

  @pytest.mark.parametrize("flag", ["--wb2_zarr", "--not_a_real_flag"])
  def test_rejects_unknown_flags(self, flag: str) -> None:
    with pytest.raises(SystemExit):
      build_arg_parser().parse_args(["--target_zarr", "/tmp/out.zarr", flag, "x"])
