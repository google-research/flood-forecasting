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

"""Regression tests for undersampled flow-duration-curve sections."""

import numpy as np
import pytest
import xarray as xr

from googlehydrology.evaluation import metrics


@pytest.mark.parametrize(
    ('size', 'fraction'),
    [(1, 0.3), (2, 0.2), (10, 0.04), (10, 0.05), (1, 0.8), (4, 0.3)],
)
def test_low_flow_tail_needs_two_samples(size: int, fraction: float) -> None:
    """Empty or single-point tails cannot describe low-flow volume bias."""
    obs = xr.DataArray(np.arange(1, size + 1, dtype=float))
    sim = obs**2
    with np.errstate(divide='raise', invalid='raise'):
        assert np.isnan(metrics.fdc_flv(obs, sim, l=fraction))


def test_low_flow_count_uses_pairwise_valid_samples() -> None:
    """Missing observations or predictions must be removed before counting."""
    obs = xr.DataArray([1.0, 2.0, 3.0, np.nan, 5.0, np.nan])
    sim = xr.DataArray([2.0, np.nan, 9.0, 4.0, np.nan, 6.0])
    assert np.isnan(metrics.fdc_flv(obs, sim, l=0.2))


@pytest.mark.parametrize('fraction', [0.4, 0.5, 0.8])
def test_low_flow_valid_tail_matches_formula(fraction: float) -> None:
    """Sufficient tails keep the existing formula and ignore larger flows."""
    obs = xr.DataArray([32.0, 2.0, 8.0, 1.0, 16.0, 4.0])
    sim = xr.DataArray([48.0, 3.0, 16.0, 2.0, 24.0, 6.0])
    count = round(fraction * obs.size)
    observed = np.log(np.sort(obs.values)[:count])
    simulated = np.log(np.sort(sim.values)[:count])
    observed_volume = np.sum(observed - observed[0])
    simulated_volume = np.sum(simulated - simulated[0])
    expected = (
        -100 * (simulated_volume - observed_volume) / (observed_volume + 1e-6)
    )
    assert metrics.fdc_flv(obs, sim, l=fraction) == pytest.approx(expected)
    sim.values[0] = 1e6
    assert metrics.fdc_flv(obs, sim, l=fraction) == pytest.approx(expected)


@pytest.mark.parametrize(
    ('size', 'lower', 'upper'),
    [(1, 0.2, 0.7), (2, 0.2, 0.9), (2, 0.1, 0.2), (3, 0.1, 0.9)],
)
def test_middle_flow_needs_distinct_in_bounds_ranks(
    size: int, lower: float, upper: float
) -> None:
    """Rounding must not index past the curve or reuse the same rank."""
    obs = xr.DataArray(np.arange(1, size + 1, dtype=float))
    assert np.isnan(metrics.fdc_fms(obs, obs**2, lower=lower, upper=upper))


@pytest.mark.parametrize('size', [1, 10, 25])
def test_empty_high_flow_tail_is_undefined_without_warning(size: int) -> None:
    """An empty high-flow section returns NaN without dividing zero by zero."""
    obs = xr.DataArray(np.arange(1, size + 1, dtype=float))
    with np.errstate(divide='raise', invalid='raise'):
        assert np.isnan(metrics.fdc_fhv(obs, obs**2))


def test_dispatcher_handles_single_valid_sample() -> None:
    """The public metric dispatcher must not crash on a short window."""
    obs = xr.DataArray([np.nan, 2.0, 3.0])
    sim = xr.DataArray([1.0, 4.0, np.nan])
    with np.errstate(divide='raise', invalid='raise'):
        result = metrics.calculate_metrics(obs, sim, ['FMS', 'FHV', 'FLV'])
    assert all(np.isnan(value) for value in result.values())
