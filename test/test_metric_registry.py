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

"""All metric entry points share names, defaults, and coordinate handling."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from googlehydrology.evaluation import metrics
from googlehydrology.utils.errors import AllNaNError

NAMES = (
    'NSE',
    'MSE',
    'RMSE',
    'KGE',
    'Alpha-NSE',
    'Pearson-r',
    'Beta-KGE',
    'Beta-NSE',
    'FHV',
    'FMS',
    'FLV',
    'Peak-Timing',
    'Missed-Peaks',
    'Peak-MAPE',
)


def _hydrographs(resolution: str = '1D') -> tuple[xr.DataArray, xr.DataArray]:
    """Use real peak detection with unambiguous, well-separated events."""
    dates = pd.date_range('2020-01-01', periods=500, freq=resolution)
    coords = {
        'observation_time': ('step', dates),
        'other_time': ('step', dates + pd.Timedelta(days=30)),
    }
    obs = xr.DataArray(np.full(500, 2.0), dims=['step'], coords=coords)
    sim = obs.copy(deep=True)
    obs.data[[100, 250, 400]] = [50.0, 60.0, 70.0]
    sim.data[[102, 250, 400]] = [48.0, 59.0, 68.0]
    return obs, sim


def _direct_results(
    obs: xr.DataArray,
    sim: xr.DataArray,
    resolution: str,
) -> dict[str, float]:
    """Evaluate each existing public formula without registry dispatch."""
    return {
        'NSE': metrics.nse(obs, sim),
        'MSE': metrics.mse(obs, sim),
        'RMSE': metrics.rmse(obs, sim),
        'KGE': metrics.kge(obs, sim),
        'Alpha-NSE': metrics.alpha_nse(obs, sim),
        'Pearson-r': metrics.pearsonr(obs, sim),
        'Beta-KGE': metrics.beta_kge(obs, sim),
        'Beta-NSE': metrics.beta_nse(obs, sim),
        'FHV': metrics.fdc_fhv(obs, sim),
        'FMS': metrics.fdc_fms(obs, sim),
        'FLV': metrics.fdc_flv(obs, sim),
        'Peak-Timing': metrics.mean_peak_timing(
            obs,
            sim,
            resolution=resolution,
            datetime_coord='observation_time',
        ),
        'Missed-Peaks': metrics.missed_peaks(
            obs,
            sim,
            resolution=resolution,
            datetime_coord='observation_time',
        ),
        'Peak-MAPE': metrics.mean_absolute_percentage_peak_error(obs, sim),
    }


@pytest.mark.unit
@pytest.mark.parametrize('resolution', ['1D', '1h', '6h'])
@pytest.mark.parametrize(
    'selection', [None, ['all'], ['NSE', 'all'], list(NAMES)]
)
@pytest.mark.parametrize('with_gaps', [False, True])
def test_all_metrics_match_direct_formulas(
    resolution: str,
    selection: list[str] | None,
    *,
    with_gaps: bool,
) -> None:
    """Include missed peaks and retain explicit datetime and resolution."""
    obs, sim = _hydrographs(resolution)
    if with_gaps:
        obs.data[25:29] = np.nan
        sim.data[27:31] = np.nan
    original_obs, original_sim = obs.copy(deep=True), sim.copy(deep=True)
    kwargs = {'resolution': resolution, 'datetime_coord': 'observation_time'}
    if selection is None:
        actual = metrics.calculate_all_metrics(obs, sim, **kwargs)
    else:
        requested = selection.copy()
        actual = metrics.calculate_metrics(obs, sim, requested, **kwargs)
        assert requested == selection
    expected = _direct_results(obs, sim, resolution)
    assert list(actual) == list(NAMES)
    np.testing.assert_array_equal(
        list(actual.values()), list(expected.values())
    )
    assert actual['Missed-Peaks'] == (1 / 3 if resolution == '1D' else 0.0)
    xr.testing.assert_identical(obs, original_obs)
    xr.testing.assert_identical(sim, original_sim)


@pytest.mark.unit
@pytest.mark.parametrize('case', [str.lower, str.upper, str.swapcase])
def test_metric_names_remain_case_insensitive(
    case: Callable[[str], str],
) -> None:
    """Return canonical names in requested order, with duplicates collapsed."""
    obs, sim = _hydrographs()
    requested = [case(name) for name in reversed(NAMES)] + ['nse']
    actual = metrics.calculate_metrics(
        obs,
        sim,
        requested,
        datetime_coord='observation_time',
    )
    expected = _direct_results(obs, sim, '1D')
    assert list(actual) == list(reversed(NAMES))
    for name, value in actual.items():
        np.testing.assert_equal(value, expected[name])


@pytest.mark.unit
def test_available_names_are_an_independent_ordered_list() -> None:
    """Changing a returned list must not alter later dispatch or discovery."""
    returned = metrics.get_available_metrics()
    assert returned == list(NAMES)
    returned.clear()
    assert metrics.get_available_metrics() == list(NAMES)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['unknown', 'ALL', ''])
def test_unknown_metric_error_is_preserved(name: str) -> None:
    """Preserve the current case-sensitive all sentinel and unknown errors."""
    obs, sim = _hydrographs()
    with pytest.raises(RuntimeError, match=f'Unknown metric {name}'):
        metrics.calculate_metrics(obs, sim, [name])


@pytest.mark.unit
@pytest.mark.parametrize('selection', [None, ['all'], ['NSE'], []])
@pytest.mark.parametrize('missing', ['observed', 'simulated'])
def test_all_nan_rejection_is_preserved(
    selection: list[str] | None,
    missing: str,
) -> None:
    """Do not weaken missing-data checks for all, selected, or empty lists."""
    obs, sim = _hydrographs()
    (obs if missing == 'observed' else sim).data[:] = np.nan
    function = (
        metrics.calculate_all_metrics
        if selection is None
        else metrics.calculate_metrics
    )
    arguments = (obs, sim) if selection is None else (obs, sim, selection)
    with pytest.raises(AllNaNError, match=f'All {missing} values are NaN'):
        function(*arguments)


@pytest.mark.unit
def test_scalar_metrics_do_not_require_datetime_coordinates() -> None:
    """Keep scalar-only and empty requests independent of time metadata."""
    obs = xr.DataArray([1.0, 2.0, 3.0])
    sim = xr.DataArray([1.0, 3.0, 2.0])
    assert metrics.calculate_metrics(obs, sim, []) == {}
    assert metrics.calculate_metrics(obs, sim, ['mse']) == {'MSE': 2 / 3}
