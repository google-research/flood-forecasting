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

"""Metrics must handle the column-shaped series accepted by validation."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from googlehydrology.evaluation import metrics


def make_series() -> tuple[xr.DataArray, xr.DataArray]:
    """Return positive hydrographs with well-separated observed peaks."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    t = np.linspace(0, 4 * np.pi, 365)
    values = (
        10
        + 5 * np.sin(t)
        + 3 * np.cos(2 * t)
        + np.maximum(0, 15 * np.sin(4 * t))
    )
    obs = xr.DataArray(values, dims='date', coords={'date': dates}, name='flow')
    sim = xr.DataArray(
        values * 0.9 + 0.2 * np.sin(3 * t),
        dims='date',
        coords={'date': dates},
        name='flow',
    )
    return obs, sim


def column(value: xr.DataArray) -> xr.DataArray:
    """Add a trailing singleton discharge-feature dimension."""
    return value.expand_dims({'feature': ['discharge']}, axis=1)


@pytest.mark.parametrize(
    'metric',
    [
        metrics.nse,
        metrics.mse,
        metrics.rmse,
        metrics.alpha_nse,
        metrics.beta_nse,
        metrics.beta_kge,
        metrics.kge,
        metrics.pearsonr,
        metrics.fdc_fhv,
        metrics.fdc_fms,
        metrics.fdc_flv,
        metrics.mean_peak_timing,
        metrics.missed_peaks,
        metrics.mean_absolute_percentage_peak_error,
    ],
)
def test_column_series_match_vector_metrics(
    metric: Callable[[xr.DataArray, xr.DataArray], float],
) -> None:
    """Check every scalar metric against its one-dimensional counterpart."""
    obs, sim = make_series()
    expected = metric(obs, sim)
    actual = metric(column(obs), column(sim))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


def test_masking_keeps_pairing_timestamps_and_original_inputs() -> None:
    """Preserve paired observations, dates, and the caller-owned arrays."""
    obs, sim = make_series()
    obs.data[[2, 7]] = np.nan
    sim.data[[3, 7]] = np.nan
    obs_col, sim_col = column(obs), column(sim)
    obs_before, sim_before = obs_col.copy(deep=True), sim_col.copy(deep=True)
    actual_obs, actual_sim = metrics._mask_valid(obs_col, sim_col)  # noqa: SLF001
    expected_obs, expected_sim = metrics._mask_valid(obs, sim)  # noqa: SLF001
    xr.testing.assert_identical(actual_obs, expected_obs)
    xr.testing.assert_identical(actual_sim, expected_sim)
    xr.testing.assert_identical(obs_col, obs_before)
    xr.testing.assert_identical(sim_col, sim_before)


@pytest.mark.parametrize('size', [0, 1])
def test_short_time_axis_is_not_removed(size: int) -> None:
    """Keep an empty or singleton time axis after removing the feature axis."""
    obs, sim = make_series()
    obs, sim = column(obs[:size]), column(sim[:size])
    actual_obs, actual_sim = metrics._mask_valid(obs, sim)  # noqa: SLF001
    assert actual_obs.dims == ('date',)
    assert actual_sim.dims == ('date',)
    assert actual_obs.shape == (size,)


def test_missing_values_match_through_all_metrics_dispatcher() -> None:
    """Check the public metric dispatcher after removing missing pairs."""
    obs, sim = make_series()
    obs.data[10:15] = np.nan
    sim.data[12:17] = np.nan
    expected = metrics.calculate_all_metrics(obs, sim)
    actual = metrics.calculate_all_metrics(column(obs), column(sim))
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-12, equal_nan=True
        )


def test_multiple_feature_columns_are_still_rejected() -> None:
    """Continue rejecting arrays containing more than one feature."""
    data = xr.DataArray(np.ones((5, 2)), dims=('date', 'feature'))
    with pytest.raises(
        RuntimeError, match='Metrics only defined for time series'
    ):
        metrics.mse(data, data)
