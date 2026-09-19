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

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet import build_hres_archive as hres_module
from multimet.build_hres_archive import (
    DEFAULT_START_DATE,
    HRES_ATTRS,
    HRES_VARIABLES,
    NUM_LEAD_DAYS,
    WeatherBench2Source,
    build_arg_parser,
    build_batch_dataset,
    deaccumulate,
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


class TestDeaccumulate:
  """Run-cumulative -> per-lead-day differencing."""

  def test_first_lead_day_passes_through(self) -> None:
    accumulated = np.array([5.0, 9.0, 12.0], dtype=np.float32)

    assert deaccumulate(accumulated)[0] == pytest.approx(5.0)

  def test_differences_successive_lead_days(self) -> None:
    accumulated = np.array([5.0, 9.0, 12.0], dtype=np.float32)

    np.testing.assert_allclose(
        deaccumulate(accumulated), [5.0, 4.0, 3.0], rtol=1e-6
    )

  def test_clip_negative_floors_precipitation_noise(self) -> None:
    accumulated = np.array([5.0, 4.9, 7.0], dtype=np.float32)

    np.testing.assert_allclose(
        deaccumulate(accumulated, clip_negative=True),
        [5.0, 0.0, 2.1],
        atol=1e-6,
    )

  def test_radiation_keeps_negative_increments(self) -> None:
    accumulated = np.array([0.0, -10.0, -25.0], dtype=np.float32)

    np.testing.assert_allclose(
        deaccumulate(accumulated), [0.0, -10.0, -15.0], rtol=1e-6
    )

  def test_preserves_shape_and_operates_on_leading_axis(self) -> None:
    accumulated = np.cumsum(
        np.ones((NUM_LEAD_DAYS, 3, 4), dtype=np.float32), axis=0
    )

    daily = deaccumulate(accumulated)

    assert daily.shape == accumulated.shape
    np.testing.assert_allclose(daily, np.ones_like(accumulated), rtol=1e-6)


class TestWeatherBench2Aggregation:
  """The 24h aggregation applied to the WeatherBench 2 Zarr archive."""

  @pytest.fixture
  def source(self) -> WeatherBench2Source:
    lead_hours = np.arange(6, 241, 6, dtype=np.int64)
    lats = np.array([-45.0, 0.0, 45.0], dtype=np.float32)
    lons = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    times = pd.to_datetime(["2020-03-01", "2020-03-02"])

    shape = (len(times), len(lead_hours), len(lats), len(lons))
    temperature = np.broadcast_to(
        lead_hours.astype(np.float32)[None, :, None, None], shape
    ).copy()
    pressure = temperature * 10.0
    precip_24hr = np.broadcast_to(
        lead_hours.astype(np.float32)[None, :, None, None], shape
    ).copy()

    dims = ["time", "prediction_timedelta", "latitude", "longitude"]
    dataset = xr.Dataset(
        data_vars={
            "2m_temperature": (dims, temperature),
            "surface_pressure": (dims, pressure),
            "total_precipitation_24hr": (dims, precip_24hr),
        },
        coords={
            "time": times,
            "prediction_timedelta": lead_hours,
            "latitude": lats,
            "longitude": lons,
        },
    )

    instance = WeatherBench2Source.__new__(WeatherBench2Source)
    instance.ds = dataset
    instance.latitudes = lats
    instance.longitudes = lons
    return instance

  def test_returns_all_archive_variables(
      self, source: WeatherBench2Source
  ) -> None:
    result = source.extract_date(pd.Timestamp("2020-03-01"))

    assert set(result) == set(HRES_VARIABLES)

  def test_every_variable_has_ten_lead_days(
      self, source: WeatherBench2Source
  ) -> None:
    result = source.extract_date(pd.Timestamp("2020-03-01"))

    for values in result.values():
      assert values.shape == (NUM_LEAD_DAYS, 3, 4)
      assert values.dtype == np.float32

  def test_temperature_is_the_mean_of_the_four_six_hourly_steps(
      self, source: WeatherBench2Source
  ) -> None:
    result = source.extract_date(pd.Timestamp("2020-03-01"))

    expected = np.array(
        [d * 24 - 9 for d in range(1, NUM_LEAD_DAYS + 1)], dtype=np.float32
    )
    np.testing.assert_allclose(
        result["temperature_2m"][:, 0, 0], expected, rtol=1e-6
    )

  def test_precipitation_is_taken_from_the_24h_accumulation(
      self, source: WeatherBench2Source
  ) -> None:
    result = source.extract_date(pd.Timestamp("2020-03-01"))

    expected = np.array(
        [d * 24 for d in range(1, NUM_LEAD_DAYS + 1)], dtype=np.float32
    )
    np.testing.assert_allclose(
        result["total_precipitation"][:, 0, 0], expected, rtol=1e-6
    )

  def test_radiation_is_nan_because_wb2_does_not_archive_it(
      self, source: WeatherBench2Source
  ) -> None:
    result = source.extract_date(pd.Timestamp("2020-03-01"))

    assert bool(np.isnan(result["surface_net_solar_radiation"]).all())
    assert bool(np.isnan(result["surface_net_thermal_radiation"]).all())

  def test_radiation_arrays_are_independent_objects(
      self, source: WeatherBench2Source
  ) -> None:
    result = source.extract_date(pd.Timestamp("2020-03-01"))

    result["surface_net_solar_radiation"][0, 0, 0] = 1.0
    assert np.isnan(result["surface_net_thermal_radiation"][0, 0, 0])

  def test_unknown_date_returns_none(
      self, source: WeatherBench2Source
  ) -> None:
    assert source.extract_date(pd.Timestamp("1999-01-01")) is None


class TestBuildBatchDataset:
  """The canonical on-disk schema."""

  def test_dimension_order(self) -> None:
    dataset = make_batch_dataset(["2020-01-01", "2020-01-02"])

    for var in HRES_VARIABLES:
      assert dataset[var].dims == ("time", "lead_time", "latitude", "longitude")

  def test_shape_matches_inputs(self) -> None:
    dataset = make_batch_dataset(["2020-01-01", "2020-01-02", "2020-01-03"])

    assert dataset["temperature_2m"].shape == (
        3,
        NUM_LEAD_DAYS,
        len(FAKE_HRES_LATS),
        len(FAKE_HRES_LONS),
    )

  def test_lead_time_is_one_based(self) -> None:
    dataset = make_batch_dataset(["2020-01-01"])

    np.testing.assert_array_equal(
        dataset["lead_time"].values, np.arange(1, NUM_LEAD_DAYS + 1)
    )
    assert dataset["lead_time"].dtype == np.int32

  def test_time_coordinate_preserves_input_order(self) -> None:
    dates = ["2020-01-05", "2020-01-06", "2020-01-07"]

    dataset = make_batch_dataset(dates)

    assert list(pd.to_datetime(dataset["time"].values)) == [
        pd.Timestamp(d) for d in dates
    ]

  def test_global_attributes_are_attached(self) -> None:
    dataset = make_batch_dataset(["2020-01-01"])

    assert dataset.attrs == HRES_ATTRS

  def test_attributes_are_copied_not_shared(self) -> None:
    dataset = make_batch_dataset(["2020-01-01"])

    dataset.attrs["title"] = "mutated"

    assert HRES_ATTRS["title"] != "mutated"

  def test_all_variables_are_present(self) -> None:
    dataset = make_batch_dataset(["2020-01-01"])

    assert set(dataset.data_vars) == set(HRES_VARIABLES)


class TestWriteBatchToZarr:
  """Initial write and append semantics."""

  def test_initial_write_creates_store(self, tmp_path: Path) -> None:
    target = str(tmp_path / "hres.zarr")

    write_batch_to_zarr(
        make_batch_dataset(["2020-01-01", "2020-01-02"]),
        target,
        is_initial_write=True,
    )

    with xr.open_zarr(target, consolidated=False) as store:
      assert len(store["time"]) == 2
      assert set(store.data_vars) == set(HRES_VARIABLES)

  def test_append_extends_the_time_axis(self, tmp_path: Path) -> None:
    target = str(tmp_path / "hres.zarr")
    write_batch_to_zarr(
        make_batch_dataset(["2020-01-01"]), target, is_initial_write=True
    )

    write_batch_to_zarr(
        make_batch_dataset(["2020-01-02", "2020-01-03"]), target
    )

    with xr.open_zarr(target, consolidated=False) as store:
      assert list(pd.to_datetime(store["time"].values)) == [
          pd.Timestamp("2020-01-01"),
          pd.Timestamp("2020-01-02"),
          pd.Timestamp("2020-01-03"),
      ]

  def test_chunking_isolates_each_init_date(self, tmp_path: Path) -> None:
    target = str(tmp_path / "hres.zarr")

    write_batch_to_zarr(
        make_batch_dataset(["2020-01-01", "2020-01-02"]),
        target,
        is_initial_write=True,
    )

    with xr.open_zarr(target, consolidated=False) as store:
      chunks = store["temperature_2m"].encoding["chunks"]
    assert chunks[0] == 1
    assert chunks[1] == NUM_LEAD_DAYS


class TestWriteBatchInPlace:
  """Overwriting dates that already exist in the store."""

  @pytest.fixture
  def populated_store(self, tmp_path: Path) -> str:
    target = str(tmp_path / "hres.zarr")
    dates = [f"2020-01-0{d}" for d in range(1, 6)]
    write_batch_to_zarr(
        make_batch_dataset(dates, value_for=lambda i, var: float(i)),
        target,
        is_initial_write=True,
    )
    return target

  def test_contiguous_range_is_updated(self, populated_store: str) -> None:
    replacement = make_batch_dataset(
        ["2020-01-02", "2020-01-03"], value_for=lambda i, var: 99.0
    )

    write_batch_in_place(replacement, populated_store)

    with xr.open_zarr(populated_store, consolidated=False) as store:
      values = store["temperature_2m"].values
    assert values[1].min() == pytest.approx(99.0)
    assert values[2].min() == pytest.approx(99.0)

  def test_non_contiguous_dates_are_updated(self, populated_store: str) -> None:
    replacement = make_batch_dataset(
        ["2020-01-01", "2020-01-05"], value_for=lambda i, var: 77.0
    )

    write_batch_in_place(replacement, populated_store)

    with xr.open_zarr(populated_store, consolidated=False) as store:
      values = store["temperature_2m"].values
    assert values[0].min() == pytest.approx(77.0)
    assert values[4].min() == pytest.approx(77.0)

  def test_untouched_dates_are_preserved(self, populated_store: str) -> None:
    replacement = make_batch_dataset(
        ["2020-01-02"], value_for=lambda i, var: 99.0
    )

    write_batch_in_place(replacement, populated_store)

    with xr.open_zarr(populated_store, consolidated=False) as store:
      values = store["temperature_2m"].values
    for index in (0, 2, 3, 4):
      assert values[index].min() == pytest.approx(float(index))

  def test_time_axis_length_is_unchanged(self, populated_store: str) -> None:
    write_batch_in_place(
        make_batch_dataset(["2020-01-03"], value_for=lambda i, var: 1.0),
        populated_store,
    )

    with xr.open_zarr(populated_store, consolidated=False) as store:
      assert len(store["time"]) == 5

  def test_unknown_date_raises(self, populated_store: str) -> None:
    replacement = make_batch_dataset(["2021-06-01"])

    with pytest.raises(ValueError, match="2021-06-01"):
      write_batch_in_place(replacement, populated_store)


class TestSourceDispatch:
  """``_extract_single_date`` routes each date window to the right source."""

  @staticmethod
  def _install(
      monkeypatch: pytest.MonkeyPatch,
  ) -> dict[str, list[pd.Timestamp]]:
    calls: dict[str, list[pd.Timestamp]] = {"wb2": [], "open": []}
    shape = (NUM_LEAD_DAYS, len(FAKE_HRES_LATS), len(FAKE_HRES_LONS))

    class Recorder:

      def __init__(self, key: str):
        self.key = key

      def extract_date(
          self, date: pd.Timestamp, *args: object, **kwargs: object
      ) -> dict[str, np.ndarray]:
        calls[self.key].append(date)
        return {
            var: np.full(shape, 1.0, dtype=np.float32) for var in HRES_VARIABLES
        }

    monkeypatch.setattr(hres_module, "_worker_wb2", Recorder("wb2"))
    monkeypatch.setattr(hres_module, "_worker_open_data", Recorder("open"))
    monkeypatch.setattr(hres_module, "_target_lat", FAKE_HRES_LATS)
    monkeypatch.setattr(hres_module, "_target_lon", FAKE_HRES_LONS)
    return calls

  @pytest.mark.parametrize(
      ("date", "expected"),
      [
          ("2020-06-01", "wb2"),
          ("2023-01-10", "wb2"),
          ("2023-07-13", "open"),
          ("2024-06-01", "open"),
      ],
  )
  def test_routes_to_expected_source(
      self, monkeypatch: pytest.MonkeyPatch, date: str, expected: str
  ) -> None:
    calls = self._install(monkeypatch)

    _, payload = hres_module._extract_single_date(pd.Timestamp(date))

    assert calls[expected] == [pd.Timestamp(date)]
    for other in set(calls) - {expected}:
      assert calls[other] == [], f"{other} source was consulted for {date}"
    assert not np.isnan(payload["temperature_2m"]).any()

  @pytest.mark.parametrize(
      "date",
      ["2023-01-11", "2023-04-15", "2023-07-12"],
  )
  def test_gap_window_emits_nan_slice(
      self, monkeypatch: pytest.MonkeyPatch, date: str
  ) -> None:
    calls = self._install(monkeypatch)

    _, payload = hres_module._extract_single_date(pd.Timestamp(date))

    assert calls["wb2"] == []
    assert calls["open"] == []
    assert set(payload) == set(HRES_VARIABLES)
    for values in payload.values():
      assert bool(np.isnan(values).all())


class TestMissingAndOperationalErrors:
  """Distinguishes upstream missing dates from operational exceptions."""

  def test_upstream_missing_returns_nan_payload(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    class EmptySource:

      def extract_date(self, *args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(hres_module, "_worker_wb2", EmptySource())
    monkeypatch.setattr(hres_module, "_worker_open_data", EmptySource())
    monkeypatch.setattr(hres_module, "_target_lat", FAKE_HRES_LATS)
    monkeypatch.setattr(hres_module, "_target_lon", FAKE_HRES_LONS)
    monkeypatch.setattr(
        hres_module, "WB2_CUTOFF_DATE", pd.Timestamp("2100-01-01")
    )

    date, payload = hres_module._extract_single_date(pd.Timestamp("2020-01-01"))

    assert date == pd.Timestamp("2020-01-01")
    assert set(payload) == set(HRES_VARIABLES)
    for values in payload.values():
      assert values.shape == (
          NUM_LEAD_DAYS,
          len(FAKE_HRES_LATS),
          len(FAKE_HRES_LONS),
      )
      assert bool(np.isnan(values).all())

  def test_operational_exception_raises_instead_of_masking_as_nan(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    class BrokenSource:

      def extract_date(self, *args: object, **kwargs: object) -> None:
        raise ConnectionError("Simulated network failure")

    monkeypatch.setattr(hres_module, "_worker_wb2", BrokenSource())
    monkeypatch.setattr(hres_module, "_target_lat", FAKE_HRES_LATS)
    monkeypatch.setattr(hres_module, "_target_lon", FAKE_HRES_LONS)
    monkeypatch.setattr(
        hres_module, "WB2_CUTOFF_DATE", pd.Timestamp("2100-01-01")
    )
    monkeypatch.setattr(hres_module.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Simulated network failure"):
      hres_module._extract_single_date(
          pd.Timestamp("2020-01-01"), max_retries=2
      )

  def test_nan_slices_are_not_aliased(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    class EmptySource:

      def extract_date(self, *args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(hres_module, "_worker_wb2", EmptySource())
    monkeypatch.setattr(hres_module, "_target_lat", FAKE_HRES_LATS)
    monkeypatch.setattr(hres_module, "_target_lon", FAKE_HRES_LONS)
    monkeypatch.setattr(
        hres_module, "WB2_CUTOFF_DATE", pd.Timestamp("2100-01-01")
    )

    _, payload = hres_module._extract_single_date(pd.Timestamp("2020-01-01"))
    payload["temperature_2m"][0, 0, 0] = 1.0

    assert np.isnan(payload["surface_pressure"][0, 0, 0])


class TestCommandLine:
  """``build-hres-archive`` argument parsing."""

  def test_requires_target_zarr(self) -> None:
    with pytest.raises(SystemExit):
      build_arg_parser().parse_args([])

  def test_defaults_with_target_zarr(self) -> None:
    args = build_arg_parser().parse_args(["--target_zarr", "/tmp/out.zarr"])

    assert args.target_zarr == "/tmp/out.zarr"
    assert args.project is None
    assert args.start_date == DEFAULT_START_DATE
    assert args.batch_size == 10
    assert args.overwrite is False
    assert args.in_place is False

  def test_end_date_defaults_to_today(self) -> None:
    args = build_arg_parser().parse_args(["--target_zarr", "/tmp/out.zarr"])

    assert pd.Timestamp(args.end_date) >= pd.Timestamp(DEFAULT_START_DATE)

  def test_explicit_values(self) -> None:
    args = build_arg_parser().parse_args([
        "--start_date",
        "2023-01-01",
        "--end_date",
        "2023-01-31",
        "--target_zarr",
        "/tmp/out.zarr",
        "--batch_size",
        "4",
        "--num_workers",
        "2",
        "--in_place",
    ])

    assert args.start_date == "2023-01-01"
    assert args.end_date == "2023-01-31"
    assert args.target_zarr == "/tmp/out.zarr"
    assert args.batch_size == 4
    assert args.num_workers == 2
    assert args.in_place is True

  def test_rejects_unknown_flags(self) -> None:
    with pytest.raises(SystemExit):
      build_arg_parser().parse_args(
          ["--target_zarr", "/tmp/out.zarr", "--not_a_real_flag"]
      )
