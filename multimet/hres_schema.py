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

"""Canonical schema and aggregation logic for the HRES daily archive.

Every pipeline that writes dates to the HRES daily archive (both the internal
historical builder and the public ECMWF Open Data updater) must use this module
so that all dates share an identical specification:

* Grid: 0.25 degree global regular lat-lon, latitude ascending (-90 to 90),
  longitude 0 to 359.75.
* Forecast steps: the ECMWF Open Data 0.25 degree step schedule
  (3-hourly +3h..+144h, 6-hourly +150h..+240h).
* Daily aggregation rules and unit conversions for each variable.
* Variable names, units, and metadata attributes.
* The ``schema_version`` root attribute enforced before appending updates.

All functions in this module are pure NumPy and perform no file or network I/O.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# Increment when any change below alters stored values, coordinates, or units.
SCHEMA_VERSION = "1"

# ---------------------------------------------------------------------------
# Grid and time axes
# ---------------------------------------------------------------------------

GRID_STEP_DEGREES = 0.25
NUM_LATITUDES = 721
NUM_LONGITUDES = 1440
LATITUDES = np.linspace(-90.0, 90.0, NUM_LATITUDES, dtype=np.float32)
LONGITUDES = np.linspace(0.0, 359.75, NUM_LONGITUDES, dtype=np.float32)

NUM_LEAD_DAYS = 10
LEAD_TIMES = np.arange(1, NUM_LEAD_DAYS + 1, dtype=np.int32)

# Forecast steps (hours after the 00 UTC initialization) published by ECMWF
# Open Data at 0.25 degree: 3-hourly up to 144 h, then 6-hourly up to 240 h.
# Only these steps are used across all sources so daily statistics are uniform.
FORECAST_STEPS: tuple[int, ...] = tuple(range(3, 145, 3)) + tuple(
    range(150, 241, 6)
)

# 24-hour boundary steps used to difference cumulative forecast fields.
ACCUMULATION_STEPS: tuple[int, ...] = tuple(
    24 * d for d in range(1, NUM_LEAD_DAYS + 1)
)

SECONDS_PER_DAY = 86400.0
KELVIN_TO_CELSIUS = 273.15

# Daily precipitation is computed by differencing two GRIB-packed cumulative
# fields, which can produce small negative rounding noise on dry days. Values
# in [-0.1 mm, 0) are clamped to 0; values below -0.1 mm raise ValueError.
PRECIP_NEGATIVE_TOLERANCE_MM = 0.1

# ECMWF GRIB shortName identifiers.
INSTANT_PARAMS: tuple[str, ...] = ("2t", "sp")
ACCUMULATED_PARAMS: tuple[str, ...] = ("tp", "ssr", "str")
GRIB_PARAMS: tuple[str, ...] = INSTANT_PARAMS + ACCUMULATED_PARAMS


def lead_day_of_step(step: int) -> int:
  """Returns the 1-based lead day (1..10) containing forecast ``step`` (hours).

  Lead day ``d`` covers steps where ``24 * (d - 1) < step <= 24 * d``.
  """
  if step <= 0 or step > 24 * NUM_LEAD_DAYS:
    raise ValueError(f"Step {step} h is outside lead days 1..{NUM_LEAD_DAYS}.")
  return (step - 1) // 24 + 1


def steps_for_lead_day(lead_day: int) -> tuple[int, ...]:
  """Returns the instantaneous forecast steps belonging to ``lead_day``."""
  return tuple(s for s in FORECAST_STEPS if lead_day_of_step(s) == lead_day)


def required_fields() -> frozenset[tuple[str, int]]:
  """Returns the complete set of ``(grib_param, step)`` pairs for one run."""
  instant = {(p, s) for p in INSTANT_PARAMS for s in FORECAST_STEPS}
  accumulated = {(p, s) for p in ACCUMULATED_PARAMS for s in ACCUMULATION_STEPS}
  return frozenset(instant | accumulated)


# ---------------------------------------------------------------------------
# Variables and attributes
# ---------------------------------------------------------------------------

_INSTANT_RULE = (
    "Computed over instantaneous forecast steps in 24*(d-1) < step <= 24*d"
    " hours: 8 steps (3-hourly) for lead days 1-6 and 4 steps (6-hourly) for"
    " lead days 7-10."
)
_ACCUM_RULE = (
    "De-accumulated over each 24-hour lead window as step 24*d minus step"
    " 24*(d-1) (with step 0 equal to 0)."
)

VARIABLE_ATTRS: dict[str, dict[str, str]] = {
    "temperature_2m_mean": {
        "units": "degC",
        "long_name": "Daily mean 2 m air temperature",
        "grib_param": "2t",
        "description": f"Arithmetic mean of 2t. {_INSTANT_RULE}",
    },
    "temperature_2m_min": {
        "units": "degC",
        "long_name": "Daily minimum 2 m air temperature",
        "grib_param": "2t",
        "description": (
            f"Minimum of 2t across sub-daily forecast steps. {_INSTANT_RULE}"
        ),
    },
    "temperature_2m_max": {
        "units": "degC",
        "long_name": "Daily maximum 2 m air temperature",
        "grib_param": "2t",
        "description": (
            f"Maximum of 2t across sub-daily forecast steps. {_INSTANT_RULE}"
        ),
    },
    "total_precipitation_sum": {
        "units": "mm",
        "long_name": "Daily total precipitation",
        "grib_param": "tp",
        "description": (
            f"{_ACCUM_RULE} Values in [-{PRECIP_NEGATIVE_TOLERANCE_MM} mm, 0)"
            " from GRIB packing noise are clamped to 0."
        ),
    },
    "surface_pressure_mean": {
        "units": "kPa",
        "long_name": "Daily mean surface pressure",
        "grib_param": "sp",
        "description": f"Arithmetic mean of sp. {_INSTANT_RULE}",
    },
    "surface_net_solar_radiation_mean": {
        "units": "W m-2",
        "long_name": "Daily mean surface net solar radiation",
        "grib_param": "ssr",
        "description": (
            f"{_ACCUM_RULE} Divided by 86400 s. Positive downward."
        ),
    },
    "surface_net_thermal_radiation_mean": {
        "units": "W m-2",
        "long_name": "Daily mean surface net thermal radiation",
        "grib_param": "str",
        "description": (
            f"{_ACCUM_RULE} Divided by 86400 s. Positive downward (typically"
            " negative due to net longwave cooling)."
        ),
    },
}

VARIABLES: tuple[str, ...] = tuple(VARIABLE_ATTRS)

GLOBAL_ATTRS: dict[str, Any] = {
    "title": "Open-MultiMet ECMWF IFS HRES daily surface forecast archive",
    "description": (
        "Daily aggregated surface forecasts for lead days 1-10 from the 00 UTC"
        " ECMWF IFS HRES run on a 0.25 degree global grid."
    ),
    "schema_version": SCHEMA_VERSION,
    "spatial_resolution": "0.25 degree",
    "time_definition": "Forecast initialization date (00 UTC run).",
    "lead_time_definition": (
        "Lead day d covers the 24-hour window (run + 24*(d-1) h, run + 24*d h]."
    ),
    "forecast_steps_hours": list(FORECAST_STEPS),
    "license": "CC-BY-4.0",
    "institution": "ECMWF / Open-MultiMet",
}

# Root Zarr attribute recording provenance by contiguous date range.
SOURCE_ATTR = "source_by_date_range"


def variable_attrs(name: str) -> dict[str, str]:
  """Returns a copy of the metadata attributes for variable ``name``."""
  return dict(VARIABLE_ATTRS[name])


# ---------------------------------------------------------------------------
# Grid orientation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GribGeometry:
  """Grid geometry metadata extracted from a single GRIB message.

  Field names correspond to the equivalent ecCodes grid keys.
  """

  ni: int
  nj: int
  lat_first: float
  lat_last: float
  lon_first: float
  lon_last: float
  i_scans_negatively: int
  j_scans_positively: int
  j_points_are_consecutive: int


def reorient_to_schema_grid(
    values: np.ndarray, geometry: GribGeometry
) -> np.ndarray:
  """Reorients a decoded 1-D GRIB payload onto the canonical schema grid.

  The target schema grid has latitude ascending from -90 to 90 and longitude
  from 0 to 359.75. Orientation and starting longitude are determined strictly
  from ``geometry``; any non-global or non-0.25-degree layout raises
  ``ValueError``.

  Args:
    values: 1-D array of decoded GRIB values in message storage order.
    geometry: Grid geometry read from the same GRIB message.

  Returns:
    2-D array of shape ``(721, 1440)`` aligned with ``(LATITUDES, LONGITUDES)``.
  """
  g = geometry
  if (g.ni, g.nj) != (NUM_LONGITUDES, NUM_LATITUDES):
    raise ValueError(
        f"Expected a {NUM_LONGITUDES} x {NUM_LATITUDES} grid, got"
        f" {g.ni} x {g.nj}."
    )
  if g.j_points_are_consecutive:
    raise ValueError("Column-major GRIB storage is not supported.")
  if values.size != g.ni * g.nj:
    raise ValueError(f"Got {values.size} values for a {g.ni} x {g.nj} grid.")
  if sorted((g.lat_first, g.lat_last)) != [-90.0, 90.0]:
    raise ValueError(
        f"Latitudes {g.lat_first} .. {g.lat_last} do not cover -90 .. 90."
    )
  north_to_south = g.lat_first > g.lat_last
  if north_to_south == bool(g.j_scans_positively):
    raise ValueError(
        "GRIB latitude order does not match jScansPositively:"
        f" first={g.lat_first}, last={g.lat_last},"
        f" jScansPositively={g.j_scans_positively}."
    )

  field = np.asarray(values).reshape(g.nj, g.ni)
  if north_to_south:
    field = field[::-1, :]

  west_lon = g.lon_first
  east_lon = g.lon_last
  if g.i_scans_negatively:
    field = field[:, ::-1]
    west_lon, east_lon = east_lon, west_lon
  span = (east_lon - west_lon) % 360.0
  if not np.isclose(span, 360.0 - GRID_STEP_DEGREES):
    raise ValueError(
        f"Longitudes {g.lon_first} .. {g.lon_last} are not a global 0.25"
        " degree grid."
    )
  shift = (west_lon % 360.0) / GRID_STEP_DEGREES
  if not np.isclose(shift, round(shift)):
    raise ValueError(f"First longitude {west_lon} is not on the 0.25 grid.")
  # Column j holds longitude west_lon + 0.25 * j, which maps to schema column
  # (shift + j) mod 1440.
  return np.roll(field, int(round(shift)) % NUM_LONGITUDES, axis=1)


# ---------------------------------------------------------------------------
# Daily statistics
# ---------------------------------------------------------------------------


class DailyAggregator:
  """Aggregates sub-daily GRIB fields for one run into daily archive variables.

  Call :meth:`add` once for every ``(param, step)`` pair in
  :func:`required_fields` (in any order) with fields already reoriented onto the
  schema grid and in raw GRIB units (K, Pa, m, J m-2), then call
  :meth:`finalize`. Missing, unexpected, or duplicate fields raise
  ``ValueError``. NaN inputs propagate to NaN outputs.
  """

  def __init__(
      self, shape: tuple[int, int] = (NUM_LATITUDES, NUM_LONGITUDES)
  ):
    self.shape = shape
    days = (NUM_LEAD_DAYS,) + shape
    self._sum = {p: np.zeros(days, dtype=np.float64) for p in INSTANT_PARAMS}
    self._min = np.full(days, np.inf, dtype=np.float64)
    self._max = np.full(days, -np.inf, dtype=np.float64)
    self._accumulated: dict[str, dict[int, np.ndarray]] = {
        p: {} for p in ACCUMULATED_PARAMS
    }
    self._required = required_fields()
    self._seen: set[tuple[str, int]] = set()

  def add(self, param: str, step: int, field: np.ndarray) -> None:
    """Adds one GRIB field."""
    key = (param, step)
    if key not in self._required:
      raise ValueError(f"Field {param} at step {step} h is not used.")
    if key in self._seen:
      raise ValueError(f"Field {param} at step {step} h was added twice.")
    if field.shape != self.shape:
      raise ValueError(
          f"Field {param} at step {step} h has shape {field.shape},"
          f" expected {self.shape}."
      )
    self._seen.add(key)
    if param in INSTANT_PARAMS:
      day = lead_day_of_step(step) - 1
      self._sum[param][day] += field
      if param == "2t":
        np.minimum(self._min[day], field, out=self._min[day])
        np.maximum(self._max[day], field, out=self._max[day])
    else:
      self._accumulated[param][step] = np.asarray(field, dtype=np.float64)

  def missing_fields(self) -> list[tuple[str, int]]:
    """Returns the required fields that have not been added yet."""
    return sorted(self._required - self._seen)

  def _daily_differences(self, param: str) -> np.ndarray:
    fields = self._accumulated[param]
    daily = np.empty((NUM_LEAD_DAYS,) + self.shape, dtype=np.float64)
    previous = np.zeros(self.shape, dtype=np.float64)
    for d, step in enumerate(ACCUMULATION_STEPS):
      daily[d] = fields[step] - previous
      previous = fields[step]
    return daily

  def finalize(self) -> dict[str, np.ndarray]:
    """Returns the daily variables, each of shape ``(10, lat, lon)``."""
    missing = self.missing_fields()
    if missing:
      raise ValueError(
          f"{len(missing)} required GRIB fields are missing, for example"
          f" {missing[:5]}."
      )
    counts = np.array(
        [len(steps_for_lead_day(d)) for d in LEAD_TIMES], dtype=np.float64
    )[:, None, None]

    precip_mm = self._daily_differences("tp") * 1000.0
    too_negative = precip_mm < -PRECIP_NEGATIVE_TOLERANCE_MM
    if too_negative.any():
      raise ValueError(
          f"{int(too_negative.sum())} daily precipitation values are below"
          f" -{PRECIP_NEGATIVE_TOLERANCE_MM} mm (lowest"
          f" {float(np.nanmin(precip_mm)):.3f} mm). The accumulated input is"
          " not increasing."
      )
    precip_mm = np.where(precip_mm < 0.0, 0.0, precip_mm)

    result = {
        "temperature_2m_mean": self._sum["2t"] / counts - KELVIN_TO_CELSIUS,
        "temperature_2m_min": self._min - KELVIN_TO_CELSIUS,
        "temperature_2m_max": self._max - KELVIN_TO_CELSIUS,
        "total_precipitation_sum": precip_mm,
        "surface_pressure_mean": self._sum["sp"] / counts / 1000.0,
        "surface_net_solar_radiation_mean": (
            self._daily_differences("ssr") / SECONDS_PER_DAY
        ),
        "surface_net_thermal_radiation_mean": (
            self._daily_differences("str") / SECONDS_PER_DAY
        ),
    }
    return {name: result[name].astype(np.float32) for name in VARIABLES}


# ---------------------------------------------------------------------------
# Archive checks and source records
# ---------------------------------------------------------------------------


def find_schema_problems(
    ds: xr.Dataset,
    latitudes: np.ndarray = LATITUDES,
    longitudes: np.ndarray = LONGITUDES,
) -> list[str]:
  """Checks ``ds`` against the HRES archive schema and returns any mismatches.

  An empty list indicates that ``ds`` is compatible and can be updated safely.
  """
  problems = []
  found_version = ds.attrs.get("schema_version")
  if found_version != SCHEMA_VERSION:
    problems.append(
        f"schema_version is {found_version!r}, expected {SCHEMA_VERSION!r}."
    )
  found_vars = set(ds.data_vars)
  if found_vars != set(VARIABLES):
    problems.append(
        f"Variables are {sorted(found_vars)}, expected {sorted(VARIABLES)}."
    )
  for name in sorted(found_vars & set(VARIABLES)):
    units = ds[name].attrs.get("units")
    if units != VARIABLE_ATTRS[name]["units"]:
      problems.append(
          f"{name} has units {units!r}, expected"
          f" {VARIABLE_ATTRS[name]['units']!r}."
      )
  for coord, expected in (
      ("lead_time", LEAD_TIMES),
      ("latitude", latitudes),
      ("longitude", longitudes),
  ):
    if coord not in ds.coords:
      problems.append(f"Coordinate {coord} is missing.")
      continue
    found = np.asarray(ds[coord].values)
    if found.shape != expected.shape or not np.allclose(found, expected):
      problems.append(f"Coordinate {coord} does not match the schema grid.")
  return problems


def _date_ranges(dates: Iterable[str]) -> list[tuple[str, str]]:
  """Groups ISO date strings into contiguous daily ``(first, last)`` ranges."""
  ordered = sorted(pd.Timestamp(d) for d in set(dates))
  ranges: list[tuple[str, str]] = []
  for date in ordered:
    if ranges and date - pd.Timestamp(ranges[-1][1]) == pd.Timedelta(days=1):
      ranges[-1] = (ranges[-1][0], date.strftime("%Y-%m-%d"))
    else:
      ranges.append((date.strftime("%Y-%m-%d"),) * 2)
  return ranges


def update_source_records(
    records: Sequence[Mapping[str, str]],
    source: str,
    dates: Iterable[str],
) -> list[dict[str, str]]:
  """Updates provenance records to attribute ``dates`` to ``source``.

  Args:
    records: Existing provenance entries, each with keys ``"source"``,
      ``"first_date"``, and ``"last_date"``.
    source: Provenance label for the newly written ``dates``.
    dates: ISO date strings (``YYYY-MM-DD``) written in this update.

  Returns:
    Consolidated provenance records sorted by ``(first_date, source)``, with
    ``dates`` removed from any prior source so every date has a single source.
  """
  new_dates = set(dates)
  by_source: dict[str, set[str]] = {}
  for record in records:
    days = pd.date_range(record["first_date"], record["last_date"], freq="D")
    by_source.setdefault(record["source"], set()).update(
        d.strftime("%Y-%m-%d") for d in days
    )
  for name in by_source:
    by_source[name] -= new_dates
  by_source.setdefault(source, set()).update(new_dates)

  updated = [
      {"source": name, "first_date": first, "last_date": last}
      for name, days in by_source.items()
      for first, last in _date_ranges(days)
  ]
  return sorted(updated, key=lambda r: (r["first_date"], r["source"]))
