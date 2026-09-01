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

"""Unit tests for googlehydrology.evaluation.metrics."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import googlehydrology.evaluation.metrics as metrics
from googlehydrology.utils.errors import AllNaNError


@pytest.fixture
def sample_timeseries():
    """Create synthetic observed and simulated series with a datetime coord."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    t = np.linspace(0, 4 * np.pi, 365)
    # Synthetic hydrograph with distinct peaks and baseflow
    obs_values = (
        10.0
        + 5.0 * np.sin(t)
        + 3.0 * np.cos(2 * t)
        + np.maximum(0, 15.0 * np.sin(4 * t))
    )
    obs_values = np.maximum(0.1, obs_values)  # Ensure positive discharge

    # Add small noise for simulated series
    rng = np.random.default_rng(seed=42)
    sim_values = obs_values + rng.normal(0, 0.5, size=len(obs_values))
    sim_values = np.maximum(0.0, sim_values)

    obs = xr.DataArray(obs_values, coords={'date': dates}, dims=['date'])
    sim = xr.DataArray(sim_values, coords={'date': dates}, dims=['date'])
    return obs, sim


@pytest.mark.unit
def test_get_available_metrics():
    available = metrics.get_available_metrics()
    assert isinstance(available, list)
    expected_metrics = [
        'NSE', 'MSE', 'RMSE', 'KGE', 'Alpha-NSE', 'Pearson-r',
        'Beta-KGE', 'Beta-NSE', 'FHV', 'FMS', 'FLV',
        'Peak-Timing', 'Missed-Peaks', 'Peak-MAPE',
    ]
    for m in expected_metrics:
        assert m in available


@pytest.mark.unit
def test_validate_inputs_mismatched_shape():
    obs = xr.DataArray(np.array([1.0, 2.0, 3.0]))
    sim = xr.DataArray(np.array([1.0, 2.0]))
    with pytest.raises(
        RuntimeError, match='Shapes of observations and simulations must match'
    ):
        metrics._validate_inputs(obs, sim)


@pytest.mark.unit
def test_validate_inputs_multi_dim():
    obs = xr.DataArray(np.ones((5, 2)))
    sim = xr.DataArray(np.ones((5, 2)))
    with pytest.raises(
        RuntimeError, match='Metrics only defined for time series'
    ):
        metrics._validate_inputs(obs, sim)


@pytest.mark.unit
def test_all_nan_error():
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    obs = xr.DataArray(
        np.full(10, np.nan), coords={'date': dates}, dims=['date']
    )
    sim = xr.DataArray(np.ones(10), coords={'date': dates}, dims=['date'])

    with pytest.raises(AllNaNError, match='All observed values are NaN'):
        metrics.calculate_all_metrics(obs, sim)

    obs_valid = xr.DataArray(
        np.ones(10), coords={'date': dates}, dims=['date']
    )
    sim_all_nan = xr.DataArray(
        np.full(10, np.nan), coords={'date': dates}, dims=['date']
    )

    with pytest.raises(AllNaNError, match='All simulated values are NaN'):
        metrics.calculate_all_metrics(obs_valid, sim_all_nan)


@pytest.mark.unit
def test_metrics_perfect_prediction(sample_timeseries):
    obs, _ = sample_timeseries
    sim = obs.copy()

    assert np.isclose(metrics.nse(obs, sim), 1.0)
    assert np.isclose(metrics.mse(obs, sim), 0.0)
    assert np.isclose(metrics.rmse(obs, sim), 0.0)
    assert np.isclose(metrics.kge(obs, sim), 1.0)
    assert np.isclose(metrics.alpha_nse(obs, sim), 1.0)
    assert np.isclose(metrics.beta_nse(obs, sim), 0.0)
    assert np.isclose(metrics.beta_kge(obs, sim), 1.0)
    assert np.isclose(metrics.pearsonr(obs, sim), 1.0)
    assert np.isclose(metrics.fdc_fhv(obs, sim), 0.0, atol=1e-5)
    assert np.isclose(metrics.fdc_fms(obs, sim), 0.0, atol=1e-5)
    assert np.isclose(metrics.fdc_flv(obs, sim), 0.0, atol=1e-5)
    assert np.isclose(metrics.mean_peak_timing(obs, sim, resolution='1D'), 0.0)
    assert np.isclose(metrics.missed_peaks(obs, sim, resolution='1D'), 0.0)
    assert np.isclose(
        metrics.mean_absolute_percentage_peak_error(obs, sim), 0.0
    )


@pytest.mark.unit
def test_metrics_mean_prediction(sample_timeseries):
    obs, _ = sample_timeseries
    sim = xr.DataArray(
        np.full_like(obs.values, obs.mean().values),
        coords=obs.coords,
        dims=obs.dims,
    )

    # For constant mean prediction, NSE = 0
    assert np.isclose(metrics.nse(obs, sim), 0.0, atol=1e-6)
    assert np.isclose(metrics.beta_nse(obs, sim), 0.0, atol=1e-6)
    assert np.isclose(metrics.beta_kge(obs, sim), 1.0, atol=1e-6)
    assert np.isclose(metrics.alpha_nse(obs, sim), 0.0, atol=1e-6)


@pytest.mark.unit
def test_metrics_with_nans(sample_timeseries):
    obs, sim = sample_timeseries
    obs_nan = obs.copy()
    sim_nan = sim.copy()

    # Introduce some NaN values
    obs_nan.values[10:20] = np.nan
    sim_nan.values[15:25] = np.nan

    # Calculate metrics with NaNs
    res_nan = metrics.calculate_all_metrics(obs_nan, sim_nan, resolution='1D')
    assert isinstance(res_nan, dict)
    assert np.isfinite(res_nan['NSE'])
    assert np.isfinite(res_nan['RMSE'])
    assert np.isfinite(res_nan['KGE'])


@pytest.mark.unit
def test_kge_custom_weights(sample_timeseries):
    obs, sim = sample_timeseries
    # Custom weights
    kge_val = metrics.kge(obs, sim, weights=[0.5, 0.25, 0.25])
    assert np.isfinite(kge_val)

    # Invalid weights length
    with pytest.raises(
        ValueError, match='Weights of the KGE must be a list of three values'
    ):
        metrics.kge(obs, sim, weights=[1.0, 1.0])


@pytest.mark.unit
def test_kge_and_pearsonr_short_series():
    obs = xr.DataArray([1.0])
    sim = xr.DataArray([1.0])
    assert np.isnan(metrics.kge(obs, sim))
    assert np.isnan(metrics.pearsonr(obs, sim))


@pytest.mark.unit
def test_fdc_bounds_errors(sample_timeseries):
    obs, sim = sample_timeseries

    # FMS invalid bounds
    with pytest.raises(ValueError, match='upper and lower have to be in range'):
        metrics.fdc_fms(obs, sim, lower=-0.1, upper=0.7)
    with pytest.raises(ValueError, match='upper and lower have to be in range'):
        metrics.fdc_fms(obs, sim, lower=0.2, upper=1.5)
    with pytest.raises(
        ValueError, match='The lower threshold has to be smaller than the upper'
    ):
        metrics.fdc_fms(obs, sim, lower=0.8, upper=0.5)

    # FHV invalid fraction
    with pytest.raises(ValueError, match='h has to be in range'):
        metrics.fdc_fhv(obs, sim, h=-0.1)
    with pytest.raises(ValueError, match='h has to be in range'):
        metrics.fdc_fhv(obs, sim, h=1.2)

    # FLV invalid fraction
    with pytest.raises(ValueError, match='l has to be in range'):
        metrics.fdc_flv(obs, sim, l=-0.1)
    with pytest.raises(ValueError, match='l has to be in range'):
        metrics.fdc_flv(obs, sim, l=1.2)


@pytest.mark.unit
def test_fdc_empty_series():
    obs = xr.DataArray([], coords={'date': []}, dims=['date'])
    sim = xr.DataArray([], coords={'date': []}, dims=['date'])
    assert np.isnan(metrics.fdc_fms(obs, sim))
    assert np.isnan(metrics.fdc_fhv(obs, sim))
    assert np.isnan(metrics.fdc_flv(obs, sim))
    assert np.isnan(metrics.mean_absolute_percentage_peak_error(obs, sim))


@pytest.mark.unit
def test_peak_metrics_timing_and_missed():
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    # Generate signal with peaks spaced > 100 days apart
    values_obs = np.zeros(500)
    values_obs[100] = 50.0
    values_obs[250] = 60.0
    values_obs[400] = 70.0

    # Sim shifted by 2 days on one peak
    values_sim = np.zeros(500)
    values_sim[102] = 48.0
    values_sim[250] = 59.0
    values_sim[400] = 68.0

    obs = xr.DataArray(values_obs, coords={'date': dates}, dims=['date'])
    sim = xr.DataArray(values_sim, coords={'date': dates}, dims=['date'])

    timing_error = metrics.mean_peak_timing(obs, sim, window=5, resolution='1D')
    assert np.isfinite(timing_error)
    assert timing_error > 0.0

    # Test missed peaks
    missed_frac = metrics.missed_peaks(
        obs, sim, window=5, resolution='1D', percentile=90
    )
    assert missed_frac == 0.0

    # If sim has no peaks
    sim_flat = xr.DataArray(
        np.zeros(500), coords={'date': dates}, dims=['date']
    )
    assert (
        metrics.missed_peaks(
            obs, sim_flat, window=5, resolution='1D', percentile=90
        )
        == 1.0
    )


@pytest.mark.unit
def test_calculate_metrics_dispatcher(sample_timeseries):
    obs, sim = sample_timeseries
    selected = [
        'NSE', 'kge', 'RMSE', 'beta-kge', 'beta-nse', 'alpha-nse', 'pearson-r',
        'fhv', 'fms', 'flv', 'peak-timing', 'missed-peaks', 'peak-mape', 'mse',
    ]
    res = metrics.calculate_metrics(obs, sim, metrics=selected, resolution='1D')
    assert len(res) == len(selected)

    # Test 'all' keyword
    res_all = metrics.calculate_metrics(
        obs, sim, metrics=['all'], resolution='1D'
    )
    assert len(res_all) >= 13

    # Test unknown metric error
    with pytest.raises(RuntimeError, match='Unknown metric invalid_metric'):
        metrics.calculate_metrics(obs, sim, metrics=['invalid_metric'])
