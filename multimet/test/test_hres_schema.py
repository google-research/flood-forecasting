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

"""Unit tests for :mod:`multimet.hres_schema`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet import hres_schema as schema

pytestmark = pytest.mark.unit

SHAPE = (3, 4)


class TestStepsAndLeadDays:
  """The forecast step set and the lead-day windows."""

  def test_step_set_matches_open_data(self) -> None:
    expected = list(range(3, 145, 3)) + list(range(150, 241, 6))
    assert list(schema.FORECAST_STEPS) == expected
    assert len(schema.FORECAST_STEPS) == 64

  @pytest.mark.parametrize(
      ("step", "day"), [(3, 1), (24, 1), (27, 2), (144, 6), (150, 7), (240, 10)]
  )
  def test_lead_day_of_step(self, step: int, day: int) -> None:
    assert schema.lead_day_of_step(step) == day

  @pytest.mark.parametrize("step", [0, -3, 246])
  def test_steps_outside_ten_days_raise(self, step: int) -> None:
    with pytest.raises(ValueError):
      schema.lead_day_of_step(step)

  def test_eight_samples_for_days_1_to_6_and_four_after(self) -> None:
    counts = [len(schema.steps_for_lead_day(d)) for d in schema.LEAD_TIMES]
    assert counts == [8] * 6 + [4] * 4

  def test_windows_cover_every_step_once(self) -> None:
    covered = [s for d in schema.LEAD_TIMES for s in schema.steps_for_lead_day(d)]
    assert covered == list(schema.FORECAST_STEPS)

  def test_required_fields(self) -> None:
    fields = schema.required_fields()
    assert len(fields) == 2 * 64 + 3 * 10
    assert ("tp", 24) in fields
    assert ("tp", 27) not in fields
    assert ("2t", 27) in fields


def _schema_field() -> np.ndarray:
  """A field whose value encodes its own position: lat * 1000 + lon."""
  return (
      schema.LATITUDES.astype(np.float64)[:, None] * 1000.0
      + schema.LONGITUDES.astype(np.float64)[None, :]
  )


def _geometry(**overrides: float) -> schema.GribGeometry:
  values = {
      "ni": 1440,
      "nj": 721,
      "lat_first": 90.0,
      "lat_last": -90.0,
      "lon_first": 180.0,
      "lon_last": 179.75,
      "i_scans_negatively": 0,
      "j_scans_positively": 0,
      "j_points_are_consecutive": 0,
  }
  values.update(overrides)
  return schema.GribGeometry(**values)


class TestReorientToSchemaGrid:
  """GRIB layouts are read from the message and put on the schema grid."""

  def test_open_data_layout_north_to_south_from_minus_180(self) -> None:
    expected = _schema_field()
    # Open Data: rows north to south, columns from -180 (encoded as 180).
    grib = np.roll(expected[::-1, :], -720, axis=1)

    result = schema.reorient_to_schema_grid(grib.ravel(), _geometry())

    np.testing.assert_array_equal(result, expected)

  def test_minus_180_written_as_negative(self) -> None:
    expected = _schema_field()
    grib = np.roll(expected[::-1, :], -720, axis=1)

    result = schema.reorient_to_schema_grid(
        grib.ravel(), _geometry(lon_first=-180.0)
    )

    np.testing.assert_array_equal(result, expected)

  def test_north_to_south_from_zero(self) -> None:
    expected = _schema_field()
    grib = expected[::-1, :]

    result = schema.reorient_to_schema_grid(
        grib.ravel(), _geometry(lon_first=0.0, lon_last=359.75)
    )

    np.testing.assert_array_equal(result, expected)

  def test_south_to_north_is_left_as_is(self) -> None:
    expected = _schema_field()

    result = schema.reorient_to_schema_grid(
        expected.ravel(),
        _geometry(
            lat_first=-90.0,
            lat_last=90.0,
            j_scans_positively=1,
            lon_first=0.0,
            lon_last=359.75,
        ),
    )

    np.testing.assert_array_equal(result, expected)

  def test_east_to_west_columns(self) -> None:
    expected = _schema_field()
    grib = expected[::-1, ::-1]

    result = schema.reorient_to_schema_grid(
        grib.ravel(),
        _geometry(lon_first=359.75, lon_last=0.0, i_scans_negatively=1),
    )

    np.testing.assert_array_equal(result, expected)

  @pytest.mark.parametrize(
      "overrides",
      [
          {"ni": 3600, "nj": 1801},
          {"j_scans_positively": 1},
          {"lat_first": 89.75},
          {"lon_last": 179.5},
          {"lon_first": 180.1, "lon_last": 179.85},
          {"j_points_are_consecutive": 1},
      ],
  )
  def test_unexpected_layouts_raise(self, overrides: dict[str, float]) -> None:
    values = np.zeros(1440 * 721)
    with pytest.raises(ValueError):
      schema.reorient_to_schema_grid(values, _geometry(**overrides))


def _fill_aggregator(
    aggregator: schema.DailyAggregator,
    skip: tuple[str, int] | None = None,
) -> None:
  """Adds synthetic fields with simple, known daily answers.

  2t = 273.15 + step (K), sp = 100000 + step (Pa), tp accumulates 2 mm/day,
  ssr accumulates 100 W m-2, str accumulates -50 W m-2.
  """
  for param, step in sorted(schema.required_fields()):
    if (param, step) == skip:
      continue
    value = {
        "2t": 273.15 + step,
        "sp": 100000.0 + step,
        "tp": 0.002 * step / 24,
        "ssr": 100.0 * 86400 * step / 24,
        "str": -50.0 * 86400 * step / 24,
    }[param]
    aggregator.add(param, step, np.full(SHAPE, value))


class TestDailyAggregator:
  """Daily statistics and unit conversion."""

  @pytest.fixture
  def result(self) -> dict[str, np.ndarray]:
    aggregator = schema.DailyAggregator(SHAPE)
    _fill_aggregator(aggregator)
    return aggregator.finalize()

  def test_returns_every_variable_as_float32(
      self, result: dict[str, np.ndarray]
  ) -> None:
    assert tuple(result) == schema.VARIABLES
    for values in result.values():
      assert values.shape == (10,) + SHAPE
      assert values.dtype == np.float32

  def test_temperature_mean_min_max_in_celsius(
      self, result: dict[str, np.ndarray]
  ) -> None:
    for d in schema.LEAD_TIMES:
      steps = schema.steps_for_lead_day(int(d))
      i = int(d) - 1
      assert result["temperature_2m_mean"][i, 0, 0] == pytest.approx(
          np.mean(steps), abs=1e-4
      )
      assert result["temperature_2m_min"][i, 0, 0] == pytest.approx(
          min(steps), abs=1e-4
      )
      assert result["temperature_2m_max"][i, 0, 0] == pytest.approx(
          max(steps), abs=1e-4
      )

  def test_pressure_mean_in_kpa(self, result: dict[str, np.ndarray]) -> None:
    steps = schema.steps_for_lead_day(1)
    assert result["surface_pressure_mean"][0, 0, 0] == pytest.approx(
        (100000.0 + np.mean(steps)) / 1000.0, rel=1e-6
    )

  def test_precipitation_is_daily_sum_in_mm(
      self, result: dict[str, np.ndarray]
  ) -> None:
    np.testing.assert_allclose(
        result["total_precipitation_sum"], 2.0, rtol=1e-5
    )

  def test_radiation_is_daily_mean_flux(
      self, result: dict[str, np.ndarray]
  ) -> None:
    np.testing.assert_allclose(
        result["surface_net_solar_radiation_mean"], 100.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        result["surface_net_thermal_radiation_mean"], -50.0, rtol=1e-5
    )

  def test_missing_field_raises(self) -> None:
    aggregator = schema.DailyAggregator(SHAPE)
    _fill_aggregator(aggregator, skip=("2t", 27))

    assert aggregator.missing_fields() == [("2t", 27)]
    with pytest.raises(ValueError, match="missing"):
      aggregator.finalize()

  def test_repeated_field_raises(self) -> None:
    aggregator = schema.DailyAggregator(SHAPE)
    aggregator.add("2t", 3, np.zeros(SHAPE))
    with pytest.raises(ValueError, match="twice"):
      aggregator.add("2t", 3, np.zeros(SHAPE))

  @pytest.mark.parametrize(("param", "step"), [("tp", 27), ("2t", 1), ("q", 3)])
  def test_unused_field_raises(self, param: str, step: int) -> None:
    with pytest.raises(ValueError, match="not used"):
      schema.DailyAggregator(SHAPE).add(param, step, np.zeros(SHAPE))

  def test_wrong_shape_raises(self) -> None:
    with pytest.raises(ValueError, match="shape"):
      schema.DailyAggregator(SHAPE).add("2t", 3, np.zeros((2, 2)))

  def test_nan_input_gives_nan_output(self) -> None:
    aggregator = schema.DailyAggregator(SHAPE)
    for param, step in sorted(schema.required_fields()):
      field = np.full(SHAPE, 1.0 if param != "2t" else 280.0)
      if param in schema.ACCUMULATED_PARAMS:
        field = field * step
      if (param, step) == ("2t", 6):
        field[1, 1] = np.nan
      aggregator.add(param, step, field)

    result = aggregator.finalize()

    for name in ("temperature_2m_mean", "temperature_2m_min",
                 "temperature_2m_max"):
      assert np.isnan(result[name][0, 1, 1])
      assert not np.isnan(result[name][0, 0, 0])
      assert not np.isnan(result[name][1, 1, 1])

  def test_tiny_negative_precipitation_is_zero(self) -> None:
    aggregator = schema.DailyAggregator(SHAPE)
    for param, step in sorted(schema.required_fields()):
      field = np.zeros(SHAPE)
      if param == "tp" and step == 24:
        field[:] = 0.001  # 1 mm on day 1
      elif param == "tp" and step > 24:
        field[:] = 0.001 - 0.00005  # -0.05 mm on day 2, then 0
      aggregator.add(param, step, field)

    precip = aggregator.finalize()["total_precipitation_sum"]

    assert precip[0, 0, 0] == pytest.approx(1.0)
    assert precip[1, 0, 0] == 0.0
    assert precip.min() >= 0.0

  def test_large_negative_precipitation_raises(self) -> None:
    aggregator = schema.DailyAggregator(SHAPE)
    for param, step in sorted(schema.required_fields()):
      field = np.zeros(SHAPE)
      if param == "tp":
        field[:] = 0.001 if step == 24 else 0.0  # -1 mm on day 2
      aggregator.add(param, step, field)

    with pytest.raises(ValueError, match="not increasing"):
      aggregator.finalize()


def _schema_dataset(**attr_overrides: str) -> xr.Dataset:
  lats = np.array([-45.0, 45.0], dtype=np.float32)
  lons = np.array([0.0, 180.0], dtype=np.float32)
  shape = (1, 10, 2, 2)
  ds = xr.Dataset(
      {
          name: (
              ["time", "lead_time", "latitude", "longitude"],
              np.zeros(shape, np.float32),
              schema.variable_attrs(name),
          )
          for name in schema.VARIABLES
      },
      coords={
          "time": [pd.Timestamp("2025-01-01")],
          "lead_time": schema.LEAD_TIMES,
          "latitude": lats,
          "longitude": lons,
      },
      attrs=dict(schema.GLOBAL_ATTRS),
  )
  ds.attrs.update(attr_overrides)
  return ds


class TestFindSchemaProblems:
  """Detects archives that new dates must not be added to."""

  LATS = np.array([-45.0, 45.0], dtype=np.float32)
  LONS = np.array([0.0, 180.0], dtype=np.float32)

  def test_matching_archive_has_no_problems(self) -> None:
    ds = _schema_dataset()
    assert schema.find_schema_problems(ds, self.LATS, self.LONS) == []

  def test_missing_schema_version(self) -> None:
    ds = _schema_dataset()
    del ds.attrs["schema_version"]
    problems = schema.find_schema_problems(ds, self.LATS, self.LONS)
    assert any("schema_version" in p for p in problems)

  def test_wrong_units(self) -> None:
    ds = _schema_dataset()
    ds["temperature_2m_mean"].attrs["units"] = "K"
    problems = schema.find_schema_problems(ds, self.LATS, self.LONS)
    assert any("temperature_2m_mean has units 'K'" in p for p in problems)

  def test_old_variable_names(self) -> None:
    ds = _schema_dataset().rename({"temperature_2m_mean": "temperature_2m"})
    problems = schema.find_schema_problems(ds, self.LATS, self.LONS)
    assert any("Variables are" in p for p in problems)

  def test_flipped_latitude(self) -> None:
    ds = _schema_dataset()
    problems = schema.find_schema_problems(ds, self.LATS[::-1], self.LONS)
    assert any("latitude" in p for p in problems)

  def test_default_grid_is_the_full_schema_grid(self) -> None:
    problems = schema.find_schema_problems(_schema_dataset())
    assert any("latitude" in p for p in problems)
    assert any("longitude" in p for p in problems)


class TestUpdateSourceRecords:
  """The per-date-range source attribute."""

  def test_new_record(self) -> None:
    records = schema.update_source_records(
        [], "open", ["2025-01-02", "2025-01-01", "2025-01-03"]
    )
    assert records == [
        {"source": "open", "first_date": "2025-01-01", "last_date": "2025-01-03"}
    ]

  def test_extends_existing_range(self) -> None:
    existing = [
        {"source": "internal", "first_date": "2016-01-01",
         "last_date": "2025-09-30"},
        {"source": "open", "first_date": "2025-10-01",
         "last_date": "2025-10-02"},
    ]
    records = schema.update_source_records(existing, "open", ["2025-10-03"])
    assert records[-1] == {
        "source": "open", "first_date": "2025-10-01", "last_date": "2025-10-03"
    }
    assert records[0]["last_date"] == "2025-09-30"

  def test_rewritten_dates_move_to_the_new_source(self) -> None:
    existing = [
        {"source": "internal", "first_date": "2025-01-01",
         "last_date": "2025-01-05"},
    ]
    records = schema.update_source_records(existing, "open", ["2025-01-03"])
    assert records == [
        {"source": "internal", "first_date": "2025-01-01",
         "last_date": "2025-01-02"},
        {"source": "open", "first_date": "2025-01-03",
         "last_date": "2025-01-03"},
        {"source": "internal", "first_date": "2025-01-04",
         "last_date": "2025-01-05"},
    ]
