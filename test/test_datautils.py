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

"""Unit tests for datautils functions. """
import pandas as pd
import pytest

from googlehydrology.datautils.utils import (get_frequency_factor, infer_frequency, sort_frequencies, _ME_FREQ,
                                             _QE_FREQ, _YE_FREQ)


def test_sort_frequencies():
    """Test the sorting of frequencies. """
    assert sort_frequencies(['1D', '1h', '2D', '3h']) == ['2D', '1D', '3h', '1h']
    assert sort_frequencies(['1' + _ME_FREQ, '1' + _YE_FREQ]) == ['1' + _YE_FREQ, '1' + _ME_FREQ]
    assert sort_frequencies(['1D', '48h']) == ['48h', '1D']
    assert sort_frequencies(['1D']) == ['1D']
    assert sort_frequencies([]) == []

    pytest.raises(ValueError, sort_frequencies, ['1D', '1XYZ'])  # not a frequency


def test_infer_frequency():
    """Test the logic to infer frequencies. """
    assert infer_frequency(pd.date_range('2000-01-01', '2000-01-10', freq='D')) == '1D'
    assert infer_frequency(pd.date_range('2000-01-01', '2000-01-10', freq='48h')) == '2D'
    assert infer_frequency(pd.date_range('2000-01-01', '2000-03-01', freq='W-MON')) == '7D'

    # just a single date
    pytest.raises(ValueError, infer_frequency, [pd.to_datetime('2000-01-01')])
    # frequency of zero
    pytest.raises(ValueError, infer_frequency,
                  [pd.to_datetime('2000-01-01'),
                   pd.to_datetime('2000-01-01'),
                   pd.to_datetime('2000-01-01')])
    # irregular dates
    pytest.raises(ValueError, infer_frequency,
                  [pd.to_datetime('2000-01-01'),
                   pd.to_datetime('2000-01-02'),
                   pd.to_datetime('2000-01-04')])


def test_get_frequency_factor():
    """Test the logic that calculates the ratio between two frequencies. """
    assert get_frequency_factor('1h', '1h') == 1
    assert get_frequency_factor('1A', '1' + _YE_FREQ) == 1
    assert get_frequency_factor('1' + _YE_FREQ, '4' + _QE_FREQ) == 1
    assert get_frequency_factor('1h', '1D') == 1 / 24
    assert get_frequency_factor('1D', '1h') == 24
    assert get_frequency_factor('2D', '12h') == 4
    assert get_frequency_factor('1W', '1D') == 7
    assert get_frequency_factor('1W-MON', '1D') == 7
    assert get_frequency_factor('1' + _YE_FREQ, '1' + _ME_FREQ) == 12
    assert get_frequency_factor('0D', '0h') == 1

    pytest.raises(ValueError, get_frequency_factor, '1YS', '1' + _ME_FREQ)  # year-start vs. month-end
    pytest.raises(ValueError, get_frequency_factor, '1' + _QE_FREQ, '1W')  # quarter vs. week
    pytest.raises(ValueError, get_frequency_factor, '1XYZ', '1D')  # not a frequency
    pytest.raises(ValueError, get_frequency_factor, '1' + _YE_FREQ,
                  '1D')  # disallowed because to_timedelta('1Y') is deprecated
    pytest.raises(ValueError, get_frequency_factor, '1' + _ME_FREQ,
                  '1D')  # disallowed because to_timedelta('1M') is deprecated
    pytest.raises(NotImplementedError, get_frequency_factor, '-1D', '1h')  # we should never need negative frequencies
