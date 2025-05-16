from typing import List, Optional
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from unittest.mock import patch, MagicMock

from neuralhydrology.datautils.validate_samples import (
    _flatten_feature_groups,
    extract_feature_groups,
    validate_samples,
    validate_samples_for_nan_handling,
    validate_samples_any_all_group,
    validate_samples_all_any_group,
    validate_samples_any,
    validate_samples_all,
    validate_sequence_all,
    validate_sequence_any
)

# --- Pytest Fixtures and Helper Functions ---

@pytest.fixture
def sample_dates_fixture():
    """Fixture for sample_dates (pd.DatetimeIndex)."""
    return pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])

@pytest.fixture
def basins_fixture():
    """Fixture for basin names."""
    return ['basin_A', 'basin_B', 'basin_C']

def create_test_dataset(
    data_vars: dict,
    basins: List[str],
    dates: pd.DatetimeIndex,
    lead_times: Optional[List[int]] = None
) -> xr.Dataset:
    """Helper to create a flexible xarray Dataset for testing."""
    coords = {'basin': basins, 'date': dates}
    dims = ['basin', 'date']

    if lead_times is not None:
        coords['lead_time'] = lead_times
        dims.append('lead_time')

    dataset = xr.Dataset()
    for var_name, data_array_data in data_vars.items():
        # Adjust data_array_data shape to match dims
        if lead_times is not None:
            # Assuming data_array_data is (num_basins, num_dates, num_lead_times)
            data_array = xr.DataArray(
                np.array(data_array_data).reshape(len(basins), len(dates), len(lead_times)),
                coords=coords,
                dims=dims
            )
        else:
            # Assuming data_array_data is (num_basins, num_dates)
            data_array = xr.DataArray(
                np.array(data_array_data).reshape(len(basins), len(dates)),
                coords={'basin': basins, 'date': dates},
                dims=['basin', 'date']
            )
        dataset[var_name] = data_array
    return dataset

# --- Tests for _flatten_feature_groups ---

def test_flatten_feature_groups_list_of_lists():
    """Test _flatten_feature_groups with a list of lists."""
    groups = [['a', 'b'], ['c', 'd']]
    expected = ['a', 'b', 'c', 'd']
    assert _flatten_feature_groups(groups) == expected

def test_flatten_feature_groups_flat_list():
    """Test _flatten_feature_groups with an already flat list."""
    groups = ['a', 'b', 'c']
    expected = ['a', 'b', 'c']
    assert _flatten_feature_groups(groups) == expected

def test_flatten_feature_groups_mixed_list_raises_error():
    """Test _flatten_feature_groups with a mixed list (should behave as flat)."""
    # The function's logic is `if isinstance(groups[0], list)`.
    # If groups[0] is not a list, it returns groups as is.
    # This test confirms that behavior, not an error.
    groups = ['a', ['b', 'c']]
    expected = ['a', ['b', 'c']] # It will not flatten if the first element is not a list
    with pytest.raises(ValueError, match='A mix of lists and features was supplied as feature groups.'):
        _flatten_feature_groups(groups)

def test_flatten_feature_groups_non_list_input_raises_value_error():
    """Test _flatten_feature_groups with non-list input."""
    with pytest.raises(ValueError, match='Feature groups must be supplied as a list.'):
        _flatten_feature_groups("not_a_list")

def test_flatten_feature_groups_empty_list():
    """Test _flatten_feature_groups with an empty list."""
    assert _flatten_feature_groups([]) == []

# --- Tests for extract_feature_groups ---

def test_extract_feature_groups_success():
    """Test extract_feature_groups with successful extraction."""
    groups = [['f1', 'f2'], ['f3', 'f4'], ['f5']]
    features = ['f1', 'f2', 'f3', 'f4', 'f5']
    expected = [['f1', 'f2'], ['f3', 'f4'], ['f5']]
    assert extract_feature_groups(groups, features) == expected

def test_extract_feature_groups_no_groups_extracted():
    """Test extract_feature_groups when no groups match."""
    groups = [['f1', 'f2'], ['f3', 'f4']]
    features = ['f5', 'f6']
    with pytest.raises(ValueError, match='No groups were extracted.'):
        extract_feature_groups(groups, features)

def test_extract_feature_groups_partial_groups_raises_error():
    """Test extract_feature_groups when some features are partially grouped."""
    groups = [['f1', 'f2', 'f_partial'], ['f3', 'f4']]
    features = ['f1', 'f3', 'f4'] # f_partial is in a group but not in features
    with pytest.raises(ValueError, match='There appear to be mixed groups in the dataset.'):
        extract_feature_groups(groups, features)

def test_extract_feature_groups_missing_features_raises_error():
    """Test extract_feature_groups when not all features are in groups."""
    groups = [['f1', 'f2'], ['f3', 'f4']]
    features = ['f1', 'f2', 'f3', 'f4', 'f5'] # f5 is missing from groups
    with pytest.raises(ValueError, match='Not all features are in feature groups: '):
        extract_feature_groups(groups, features)

def test_extract_feature_groups_empty_features_list():
    """Test extract_feature_groups with an empty features list."""
    groups = [['f1', 'f2']]
    features = []
    expected_groups = []
    assert extract_feature_groups(groups, features) == expected_groups

def test_extract_feature_groups_empty_groups_list_raises_error():
    """Test extract_feature_groups with an empty groups list."""
    groups = []
    features = ['f1']
    with pytest.raises(ValueError, match='No groups were extracted.'):
        extract_feature_groups(groups, features)

# --- Tests for validate_samples_any ---

def test_validate_samples_any_no_lead_time_some_nan(basins_fixture, sample_dates_fixture):
    """Test validate_samples_any without lead_time, some NaNs."""
    data_vars = {
        'var1': [[1, 2, np.nan, 4, 5], [6, 7, 8, 9, 10], [11, np.nan, 13, 14, 15]],
        'var2': [[1, 2, 3, 4, 5], [np.nan, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    mask = validate_samples_any(dataset)
    expected_mask = xr.DataArray(
        [[True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True]],
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)


def test_validate_samples_any_with_lead_time_some_nan(basins_fixture, sample_dates_fixture):
    """Test validate_samples_any with lead_time, some NaNs."""
    lead_times = [0, 1, 2]
    data_vars = {
        'var1': np.array([
            [[1, 2, 3], [4, np.nan, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], # basin_A
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [np.nan, np.nan, np.nan]], # basin_B (last date all NaN)
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], # basin_C
        ]),
        'var2': np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [np.nan, np.nan, np.nan]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        ])
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture, lead_times)
    mask = validate_samples_any(dataset)
    expected_mask_data = np.array([
        [True, True, True, True, True],
        [True, True, True, True, False], # basin_B, date=2020-01-05 is False
        [True, True, True, True, True]
    ])
    expected_mask = xr.DataArray(
        expected_mask_data,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

def test_validate_samples_any_no_nan(basins_fixture, sample_dates_fixture):
    """Test validate_samples_any with no NaNs."""
    data_vars = {
        'var1': np.ones((len(basins_fixture), len(sample_dates_fixture))),
        'var2': np.ones((len(basins_fixture), len(sample_dates_fixture)))
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    mask = validate_samples_any(dataset)
    expected_mask = xr.DataArray(
        True,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

# --- Tests for validate_samples_all ---

def test_validate_samples_all_no_lead_time_some_nan(basins_fixture, sample_dates_fixture):
    """Test validate_samples_all without lead_time, some NaNs."""
    data_vars = {
        'var1': [[1, 2, np.nan, 4, 5], [6, 7, 8, 9, 10], [11, np.nan, 13, 14, 15]],
        'var2': [[1, 2, 3, 4, 5], [np.nan, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    mask = validate_samples_all(dataset)
    expected_mask_data = np.array([
        [True, True, False, True, True],
        [False, True, True, True, True],
        [True, False, True, True, True]
    ])
    expected_mask = xr.DataArray(
        expected_mask_data,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

def test_validate_samples_all_with_lead_time_some_nan(basins_fixture, sample_dates_fixture):
    """Test validate_samples_all with lead_time, some NaNs."""
    lead_times = [0, 1, 2]
    data_vars = {
        'var1': np.array([
            [[1, 2, 3], [4, np.nan, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], # basin_A (date 2 has NaN)
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [np.nan, np.nan, np.nan]], # basin_B (last date all NaN)
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], # basin_C
        ]),
        'var2': np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [np.nan, np.nan, np.nan]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        ])
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture, lead_times)
    mask = validate_samples_all(dataset)
    expected_mask_data = np.array([
        [True, False, True, True, True], # basin_A, date=2020-01-02 is False
        [True, True, True, True, False], # basin_B, date=2020-01-05 is False
        [True, True, True, True, True]
    ])
    expected_mask = xr.DataArray(
        expected_mask_data,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

def test_validate_samples_all_no_nan(basins_fixture, sample_dates_fixture):
    """Test validate_samples_all with no NaNs."""
    data_vars = {
        'var1': np.ones((len(basins_fixture), len(sample_dates_fixture))),
        'var2': np.ones((len(basins_fixture), len(sample_dates_fixture)))
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    mask = validate_samples_all(dataset)
    expected_mask = xr.DataArray(
        True,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

# --- Tests for validate_samples_any_all_group ---

def test_validate_samples_any_all_group_success(basins_fixture, sample_dates_fixture):
    """Test validate_samples_any_all_group where at least one group is all valid."""
    feature_groups = [['f1', 'f2'], ['f3', 'f4']]
    data_vars = {
        'f1': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f2': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f3': [[1, 2, 3, np.nan, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f4': [[1, 2, 3, np.nan, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    mask = validate_samples_any_all_group(dataset, feature_groups)
    # Result is ANY group is valid. So, as long as Group 0 is valid, the result is True.
    expected_mask = xr.DataArray(
        True,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

def test_validate_samples_any_all_group_no_valid_group(basins_fixture, sample_dates_fixture):
    """Test validate_samples_any_all_group where no group is all valid."""
    feature_groups = [['f1', 'f2'], ['f3', 'f4']]
    data_vars = {
        'f1': [[1, 2, 3, np.nan, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f2': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], # f2 is all valid
        'f3': [[1, 2, 3, np.nan, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f4': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]] # f4 is all valid
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    mask = validate_samples_any_all_group(dataset, feature_groups)

    # Group 0 (f1, f2): f1 has NaN at date=2020-01-04 for basin_A. validate_samples_all(dataset[['f1','f2']]) will be False for basin_A, date=2020-01-04.
    # Group 1 (f3, f4): f3 has NaN at date=2020-01-04 for basin_A. validate_samples_all(dataset[['f3','f4']]) will be False for basin_A, date=2020-01-04.
    # Since both groups are False for basin_A, date=2020-01-04, the final result for that sample should be False.
    expected_mask_data = np.array([
        [True, True, True, False, True], # basin_A, date=2020-01-04 is False
        [True, True, True, True, True],
        [True, True, True, True, True]
    ])
    expected_mask = xr.DataArray(
        expected_mask_data,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

def test_validate_samples_any_all_group_empty_feature_groups(basins_fixture, sample_dates_fixture):
    """Test validate_samples_any_all_group with empty feature_groups."""
    feature_groups = []
    data_vars = {
        'f1': np.ones((len(basins_fixture), len(sample_dates_fixture))),
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    with pytest.raises(ValueError, match='No feature groups provided.'):
        validate_samples_any_all_group(dataset, feature_groups)

# --- Tests for validate_samples_all_any_group ---

def test_validate_samples_all_any_group_success(basins_fixture, sample_dates_fixture):
    """Test validate_samples_all_any_group where all groups pass ANY check."""
    feature_groups = [['f1', 'f2'], ['f3', 'f4']]
    data_vars = {
        'f1': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f2': [[1, 2, 3, np.nan, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], # f2 has NaN
        'f3': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f4': [[1, 2, 3, np.nan, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]] # f4 has NaN
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    mask = validate_samples_all_any_group(dataset, feature_groups)

    # Group 0 (f1, f2): f1 is all valid, f2 has NaN at date=2020-01-04 for basin_A.
    # validate_samples_any(dataset[['f1','f2']]) will be True for basin_A, date=2020-01-04 because f1 is valid.
    # Group 1 (f3, f4): f3 is all valid, f4 has NaN at date=2020-01-04 for basin_A.
    # validate_samples_any(dataset[['f3','f4']]) will be True for basin_A, date=2020-01-04 because f3 is valid.
    # Since both groups pass the ANY check, the final result is True.
    expected_mask = xr.DataArray(
        True,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask, expected_mask)

def test_validate_samples_all_any_group_one_group_fails_any(basins_fixture, sample_dates_fixture):
    """Test validate_samples_all_any_group where one group fails ANY check."""
    feature_groups = [['f1', 'f2'], ['f3', 'f4']]    
    # Group 0 (f1, f2): f1 is all valid, f2 is all valid. validate_samples_any is True.
    # Group 1 (f3, f4): f3 & f4 have NaN at date=2020-01-04 for basin_A, and only f3 has NaN for basin_A, date=2020-01-04.
    data_vars_fail = {
        'f1': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f2': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f3': [[np.nan, 2, 3, np.nan, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        'f4': [[np.nan, np.nan, np.nan, 4, np.nan], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    }
    dataset_fail = create_test_dataset(data_vars_fail, basins_fixture, sample_dates_fixture)
    mask_fail = validate_samples_all_any_group(dataset_fail, feature_groups)
    expected_mask_data_fail = np.array([
        [False, True, True, True, True], # basin_A, date=2020-01-01 is False
        [True, True, True, True, True],
        [True, True, True, True, True]
    ])
    expected_mask_fail = xr.DataArray(
        expected_mask_data_fail,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(mask_fail, expected_mask_fail)

def test_validate_samples_all_any_group_empty_feature_groups(basins_fixture, sample_dates_fixture):
    """Test validate_samples_all_any_group with empty feature_groups."""
    feature_groups = []
    data_vars = {
        'f1': np.ones((len(basins_fixture), len(sample_dates_fixture))),
    }
    dataset = create_test_dataset(data_vars, basins_fixture, sample_dates_fixture)
    with pytest.raises(ValueError, match='No feature groups provided.'):
        validate_samples_all_any_group(dataset, feature_groups)

# --- Tests for validate_samples_for_nan_handling ---
@patch('neuralhydrology.datautils.validate_samples.validate_samples_all')
@patch('neuralhydrology.datautils.validate_samples.validate_samples_any')
@patch('neuralhydrology.datautils.validate_samples.validate_samples_any_all_group')
@patch('neuralhydrology.datautils.validate_samples.validate_samples_all_any_group')
def test_validate_samples_for_nan_handling_dispatch(
    mock_all_any_group, mock_any_all_group, mock_any, mock_all,
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples_for_nan_handling dispatches to correct function."""
    dummy_dataset = create_test_dataset({'var': np.ones((len(basins_fixture), len(sample_dates_fixture)))},
                                         basins_fixture, sample_dates_fixture)
    dummy_feature_groups = [['var']]
    mock_return_mask = xr.DataArray(True, coords={'basin': basins_fixture[0], 'date': sample_dates_fixture[0]}, dims=['basin', 'date'])

    mock_all.return_value = mock_return_mask
    mock_any.return_value = mock_return_mask
    mock_any_all_group.return_value = mock_return_mask
    mock_all_any_group.return_value = mock_return_mask

    # Test None
    validate_samples_for_nan_handling(dummy_dataset, None, dummy_feature_groups)
    mock_all.assert_called_once_with(dummy_dataset)
    mock_all.reset_mock()

    # Test 'input_replacing'
    validate_samples_for_nan_handling(dummy_dataset, 'input_replacing', dummy_feature_groups)
    mock_any.assert_called_once_with(dummy_dataset)
    mock_any.reset_mock()

    # Test 'masked_mean'
    validate_samples_for_nan_handling(dummy_dataset, 'masked_mean', dummy_feature_groups)
    mock_any_all_group.assert_called_once_with(dummy_dataset, dummy_feature_groups)
    mock_any_all_group.reset_mock()

    # Test 'attention'
    validate_samples_for_nan_handling(dummy_dataset, 'attention', dummy_feature_groups)
    mock_any_all_group.assert_called_once_with(dummy_dataset, dummy_feature_groups)
    mock_any_all_group.reset_mock()

    # Test 'unioning'
    validate_samples_for_nan_handling(dummy_dataset, 'unioning', dummy_feature_groups)
    mock_all_any_group.assert_called_once_with(dummy_dataset, dummy_feature_groups)
    mock_all_any_group.reset_mock()

    # Test unrecognized method
    with pytest.raises(ValueError, match='Unrecognized NaN-handling method: unknown_method.'):
        validate_samples_for_nan_handling(dummy_dataset, 'unknown_method', dummy_feature_groups)

# --- Tests for validate_sequence_all ---

def test_validate_sequence_all_basic(sample_dates_fixture):
    """Test validate_sequence_all with a simple mask."""
    mask_data = np.array([True, True, False, True, True])
    mask = xr.DataArray(mask_data, coords={'date': sample_dates_fixture}, dims=['date'])
    seq_length = 2
    shift_right = 0
    result = validate_sequence_all(mask, seq_length, shift_right)

    # Expected:
    # [T, T, F, T, T] (mask)
    # Rolling window of 2, all true:
    # [~,T] -> F
    # [T,T] -> T
    # [T,F] -> F
    # [F,T] -> F
    # [T,T] -> T
    # Result before shift: [T, F, F, T, F (NaN from rolling)]
    # After shift(0) and fillna(False): [T, F, F, T, False]
    expected_data = np.array([False, True, False, False, True])
    expected_mask = xr.DataArray(expected_data, coords={'date': sample_dates_fixture}, dims=['date'])
    xr.testing.assert_equal(result, expected_mask)

def test_validate_sequence_all_shift_right(sample_dates_fixture):
    """Test validate_sequence_all with positive shift_right."""
    mask_data = np.array([True, True, True, False, True])
    mask = xr.DataArray(mask_data, coords={'date': sample_dates_fixture}, dims=['date'])
    seq_length = 3
    shift_right = 1 # Shift right by 1 means the window ends 1 step earlier
    result = validate_sequence_all(mask, seq_length, shift_right)

    # Mask: [T, T, T, F, T]
    # Rolling(3, all):
    # [~,T,T] -> F
    # [T,T,T] -> T
    # [T,T,F] -> F
    # [T,F,T] -> F
    expected_data = np.array([False, True, False, False, False])
    expected_mask = xr.DataArray(expected_data, coords={'date': sample_dates_fixture}, dims=['date'])
    xr.testing.assert_equal(result, expected_mask)

def test_validate_sequence_all_shift_left(sample_dates_fixture):
    """Test validate_sequence_all with negative shift_right (shift left)."""
    mask_data = np.array([True, True, True, False, True])
    mask = xr.DataArray(mask_data, coords={'date': sample_dates_fixture}, dims=['date'])
    seq_length = 2
    shift_right = -1 # Shift left by 1 means the window starts 1 step later
    result = validate_sequence_all(mask, seq_length, shift_right)

    # Mask: [T, T, T, F, T]
    # Rolling(2, all) result (right-aligned):
    # [~,~] -> F
    # [~,T] -> F
    # [T,T] -> T (at 01-02)
    # [T,T] -> T (at 01-03)
    # [T,F] -> F (at 01-04)
    expected_data = np.array([False, False, True, True, False])
    expected_mask = xr.DataArray(expected_data, coords={'date': sample_dates_fixture}, dims=['date'])
    xr.testing.assert_equal(result, expected_mask)

# --- Tests for validate_sequence_any ---

def test_validate_sequence_any_basic(sample_dates_fixture):
    """Test validate_sequence_any with a simple mask."""
    mask_data = np.array([False, False, True, False, False])
    mask = xr.DataArray(mask_data, coords={'date': sample_dates_fixture}, dims=['date'])
    seq_length = 3
    shift_right = 0
    result = validate_sequence_any(mask, seq_length, shift_right)

    # Mask: [F, F, T, F, F]
    # Rolling(3, any):
    # [F,F,T] -> T (at 01-03)
    # [F,T,F] -> T (at 01-04)
    # [T,F,F] -> T (at 01-05)
    # Result before shift: [F, F, T, T, T] (aligned to dates 01-01 to 01-05)
    expected_data = np.array([False, False, True, True, True])
    expected_mask = xr.DataArray(expected_data, coords={'date': sample_dates_fixture}, dims=['date'])
    xr.testing.assert_equal(result, expected_mask)

def test_validate_sequence_any_shift_right(sample_dates_fixture):
    """Test validate_sequence_any with positive shift_right."""
    mask_data = np.array([False, False, True, False, False])
    mask = xr.DataArray(mask_data, coords={'date': sample_dates_fixture}, dims=['date'])
    seq_length = 3
    shift_right = 1
    result = validate_sequence_any(mask, seq_length, shift_right)

    # Mask: [F, F, T, F, F]
    # Rolling(3, any) result (right-aligned):
    # [~,F,F] -> F (at 01-03)
    # [F,F,T] -> T (at 01-04)
    # [F,T,F] -> T (at 01-05)
    # [T,T,T] -> T (at 01-05)
    # [T,T,~] -> F Even with an ANY check, we do not return sequences that run past the end of the dates.
    expected_data = np.array([False, True, True, True, False])
    expected_mask = xr.DataArray(expected_data, coords={'date': sample_dates_fixture}, dims=['date'])
    xr.testing.assert_equal(result, expected_mask)

# --- Tests for validate_samples (main function) ---

# Mock dependencies for validate_samples
@patch('neuralhydrology.datautils.validate_samples.validate_samples_all')
@patch('neuralhydrology.datautils.validate_samples.validate_samples_for_nan_handling')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_all')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_any')
@patch('neuralhydrology.datautils.validate_samples._flatten_feature_groups')
@patch('neuralhydrology.datautils.validate_samples.extract_feature_groups')
def test_validate_samples_no_features_raises_error(
    mock_extract_feature_groups, mock_flatten_feature_groups,
    mock_validate_sequence_any, mock_validate_sequence_all,
    mock_validate_samples_for_nan_handling, mock_validate_samples_all,
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples raises ValueError if no feature lists are provided."""
    dummy_dataset = create_test_dataset({'var': np.ones((len(basins_fixture), len(sample_dates_fixture)))},
                                         basins_fixture, sample_dates_fixture)
    with pytest.raises(ValueError, match='At least one feature list is required to validate samples.'):
        validate_samples(
            is_train=False,
            dataset=dummy_dataset,
            sample_dates=sample_dates_fixture,
            nan_handling_method=None,
            feature_groups=[]
        )
    # Ensure no internal validation functions were called
    mock_validate_samples_all.assert_not_called()
    mock_validate_samples_for_nan_handling.assert_not_called()
    mock_validate_sequence_all.assert_not_called()
    mock_validate_sequence_any.assert_not_called()
    mock_flatten_feature_groups.assert_not_called()
    mock_extract_feature_groups.assert_not_called()

@patch('neuralhydrology.datautils.validate_samples.validate_samples_any')
@patch('neuralhydrology.datautils.validate_samples.validate_samples_all')
@patch('neuralhydrology.datautils.validate_samples.validate_samples_for_nan_handling')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_all')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_any')
@patch('neuralhydrology.datautils.validate_samples._flatten_feature_groups')
@patch('neuralhydrology.datautils.validate_samples.extract_feature_groups')
def test_validate_samples_static_features_only(
    mock_extract_feature_groups, mock_flatten_feature_groups,
    mock_validate_sequence_any, mock_validate_sequence_all,
    mock_validate_samples_for_nan_handling, mock_validate_samples_all,
    mock_validate_samples_any, basins_fixture, sample_dates_fixture
):
    """Test validate_samples with only static features."""
    dummy_dataset = create_test_dataset({'static_var': np.ones((len(basins_fixture), len(sample_dates_fixture)))},
                                         basins_fixture, sample_dates_fixture)
    mock_mask = xr.DataArray(True, coords={'basin': basins_fixture, 'date': sample_dates_fixture}, dims=['basin', 'date'])
    mock_validate_samples_all.return_value = mock_mask

    result_mask, masks = validate_samples(
        is_train=False,
        dataset=dummy_dataset,
        sample_dates=sample_dates_fixture,
        nan_handling_method=None,
        feature_groups=[],
        static_features=['static_var']
    )

    mock_validate_samples_all.assert_called_once_with(dataset=dummy_dataset[['static_var']])
    mock_validate_samples_any.assert_not_called()
    mock_validate_sequence_any.assert_not_called()
    mock_validate_sequence_all.assert_not_called()
    mock_flatten_feature_groups.assert_not_called()
    mock_extract_feature_groups.assert_not_called()
    assert len(masks) == 2
    assert masks[0].name == 'statics'
    assert masks[1].name == 'dates'
    xr.testing.assert_equal(result_mask, mock_mask)

@patch('neuralhydrology.datautils.validate_samples.validate_samples_for_nan_handling')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_all')
@patch('neuralhydrology.datautils.validate_samples.extract_feature_groups', return_value=[['h1', 'h2']])
def test_validate_samples_hindcast_features(
    mock_extract_feature_groups,
    mock_validate_sequence_all,
    mock_validate_samples_for_nan_handling,
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples with hindcast features."""
    dummy_dataset = create_test_dataset({'h1': np.ones((len(basins_fixture), len(sample_dates_fixture))),
                                         'h2': np.ones((len(basins_fixture), len(sample_dates_fixture)))},
                                         basins_fixture, sample_dates_fixture)
    mock_mask = xr.DataArray(True, coords={'basin': basins_fixture, 'date': sample_dates_fixture}, dims=['basin', 'date'])
    mock_validate_samples_for_nan_handling.return_value = mock_mask
    mock_validate_sequence_all.return_value = mock_mask

    result_mask, masks = validate_samples(
        is_train=False,
        dataset=dummy_dataset,
        sample_dates=sample_dates_fixture,
        nan_handling_method='input_replacing',
        feature_groups=[['h1', 'h2']],
        hindcast_features=['h1', 'h2'],
        seq_length=5
    )

    mock_extract_feature_groups.assert_called_once_with([['h1', 'h2']], ['h1', 'h2'])
    mock_validate_samples_for_nan_handling.assert_called_once_with(
        dataset=dummy_dataset[['h1', 'h2']],
        nan_handling_method='input_replacing',
        feature_groups=[['h1', 'h2']]
    )
    mock_validate_sequence_all.assert_called_once_with(
        mask=mock_mask, seq_length=5, shift_right=0
    )
    assert len(masks) == 2
    assert masks[0].name == 'hindcasts'
    assert masks[1].name == 'dates'
    xr.testing.assert_equal(result_mask, mock_mask)

@patch('neuralhydrology.datautils.validate_samples.validate_samples_for_nan_handling')
@patch('neuralhydrology.datautils.validate_samples.extract_feature_groups', return_value=[['f1', 'f2']])
def test_validate_samples_forecast_features(
    mock_extract_feature_groups,
    mock_validate_samples_for_nan_handling, 
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples with forecast features."""
    lead_times = [0, 1, 2]
    dummy_dataset = create_test_dataset({'f1': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times))),
                                         'f2': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times)))},
                                         basins_fixture, sample_dates_fixture, lead_times)
    mock_mask = xr.DataArray(True, coords={'basin': basins_fixture, 'date': sample_dates_fixture}, dims=['basin', 'date'])
    mock_validate_samples_for_nan_handling.return_value = mock_mask

    result_mask, masks = validate_samples(
        is_train=False,
        dataset=dummy_dataset,
        sample_dates=sample_dates_fixture,
        nan_handling_method='input_replacing',
        feature_groups=[['f1', 'f2']],
        forecast_features=['f1', 'f2']
    )

    mock_extract_feature_groups.assert_called_once_with([['f1', 'f2']], ['f1', 'f2'])
    mock_validate_samples_for_nan_handling.assert_called_once_with(
        dataset=dummy_dataset[['f1', 'f2']],
        nan_handling_method='input_replacing',
        feature_groups=[['f1', 'f2']]
    )
    assert len(masks) == 2
    assert masks[0].name == 'forecasts'
    assert masks[1].name == 'dates'
    xr.testing.assert_equal(result_mask, mock_mask)

@patch('neuralhydrology.datautils.validate_samples.validate_samples_for_nan_handling')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_all')
@patch('neuralhydrology.datautils.validate_samples.extract_feature_groups', return_value=[['f1', 'f2']])
def test_validate_samples_forecast_features_with_overlap(
    mock_extract_feature_groups,
    mock_validate_sequence_all,
    mock_validate_samples_for_nan_handling,
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples with forecast features and overlap."""
    lead_times = [0, 1, 2]
    dummy_dataset = create_test_dataset({'f1': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times))),
                                         'f2': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times)))},
                                         basins_fixture, sample_dates_fixture, lead_times)
    mock_mask = xr.DataArray(True, coords={'basin': basins_fixture, 'date': sample_dates_fixture}, dims=['basin', 'date'])
    mock_validate_samples_for_nan_handling.side_effect = [mock_mask, mock_mask] # For forecast and forecast_overlap calls
    mock_validate_sequence_all.return_value = mock_mask

    result_mask, masks = validate_samples(
        is_train=False,
        dataset=dummy_dataset,
        sample_dates=sample_dates_fixture,
        nan_handling_method='input_replacing',
        feature_groups=[['f1', 'f2']],
        forecast_features=['f1', 'f2'],
        forecast_overlap=3,
        min_lead_time=1
    )

    assert mock_extract_feature_groups.call_count == 1 # Called once for forecast_features
    assert mock_validate_samples_for_nan_handling.call_count == 2 # Once for forecast, once for forecast_overlap
    assert mock_validate_sequence_all.call_count == 1 # Once for forecast_overlap
    assert len(masks) == 3
    assert masks[0].name == 'forecasts'
    assert masks[1].name == 'forecast_overlap'
    assert masks[2].name == 'dates'
    xr.testing.assert_equal(result_mask, mock_mask)

@patch('neuralhydrology.datautils.validate_samples.validate_samples_any')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_any')
def test_validate_samples_target_features_train(
    mock_validate_sequence_any,
    mock_validate_samples_any, 
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples with target features in training mode."""
    lead_times = [0, 1, 2]
    dummy_dataset = create_test_dataset({'t1': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times)))},
                                         basins_fixture, sample_dates_fixture, lead_times)
    mock_mask = xr.DataArray(True, coords={'basin': basins_fixture, 'date': sample_dates_fixture}, dims=['basin', 'date'])
    mock_validate_samples_any.return_value = mock_mask
    mock_validate_sequence_any.return_value = mock_mask

    result_mask, masks = validate_samples(
        is_train=True,
        dataset=dummy_dataset,
        sample_dates=sample_dates_fixture,
        nan_handling_method=None,
        feature_groups=[],
        target_features=['t1'],
        predict_last_n=2,
        lead_time=1
    )

    mock_validate_samples_any.assert_called_once_with(dataset=dummy_dataset[['t1']])
    mock_validate_sequence_any.assert_called_once_with(
        mask=mock_mask, seq_length=2, shift_right=1
    )
    assert len(masks) == 2
    assert masks[0].name == 'targets'
    assert masks[1].name == 'dates'
    xr.testing.assert_equal(result_mask, mock_mask)

@patch('neuralhydrology.datautils.validate_samples.validate_samples_any')
@patch('neuralhydrology.datautils.validate_samples.validate_sequence_any')
def test_validate_samples_target_features_inference(
    mock_validate_sequence_any,
    mock_validate_samples_any,
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples with target features in inference mode (should not validate targets)."""
    lead_times = [0, 1, 2]
    dummy_dataset = create_test_dataset({'t1': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times)))},
                                         basins_fixture, sample_dates_fixture, lead_times)

    result_mask, masks = validate_samples(
        is_train=False, # Key difference
        dataset=dummy_dataset,
        sample_dates=sample_dates_fixture,
        nan_handling_method=None,
        feature_groups=[],
        target_features=['t1'],
        predict_last_n=2,
        lead_time=1
    )

    mock_validate_samples_any.assert_not_called()
    mock_validate_sequence_any.assert_not_called()
    assert len(masks) == 1
    # The result_mask should be a full True mask based on sample_dates and basin_fixture
    expected_mask = xr.DataArray(
        True,
        coords={'basin': basins_fixture, 'date': sample_dates_fixture},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(result_mask, expected_mask)


# --- Test ValueError conditions in validate_samples ---

def test_validate_samples_hindcast_missing_seq_length_raises_error(
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples raises ValueError if seq_length is missing for hindcasts."""
    dummy_dataset = create_test_dataset({'h1': np.ones((len(basins_fixture), len(sample_dates_fixture)))},
                                         basins_fixture, sample_dates_fixture)
    with pytest.raises(ValueError, match='Sequence length is required when validating hindcast data.'):
        validate_samples(
            is_train=False,
            dataset=dummy_dataset,
            sample_dates=sample_dates_fixture,
            nan_handling_method=None,
            feature_groups=[],
            hindcast_features=['h1'],
            seq_length=None # Missing seq_length
        )

@patch('neuralhydrology.datautils.validate_samples.validate_samples_for_nan_handling')
def test_validate_samples_forecast_overlap_missing_min_lead_time_raises_error(
    mock_validate_samples_for_nan_handling,
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples raises ValueError if min_lead_time is missing for forecast overlap."""
    lead_times = [0, 1, 2]
    dummy_dataset = create_test_dataset({'f1': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times)))},
                                         basins_fixture, sample_dates_fixture, lead_times)
    mock_mask = xr.DataArray(True, coords={'basin': basins_fixture, 'date': sample_dates_fixture}, dims=['basin', 'date'])
    mock_validate_samples_for_nan_handling.return_value = mock_mask

    with pytest.raises(ValueError, match='`min_lead_time`is required when validating a forecast overlap sequence.'):
        validate_samples(
            is_train=False,
            dataset=dummy_dataset,
            sample_dates=sample_dates_fixture,
            nan_handling_method=None,
            feature_groups=[['f1']],
            forecast_features=['f1'],
            forecast_overlap=3,
            min_lead_time=None # Missing min_lead_time
        )

def test_validate_samples_target_missing_predict_last_n_raises_error(
    basins_fixture, sample_dates_fixture
):
    """Test validate_samples raises ValueError if predict_last_n is missing for targets in training mode."""
    lead_times = [0, 1, 2]
    dummy_dataset = create_test_dataset({'t1': np.ones((len(basins_fixture), len(sample_dates_fixture), len(lead_times)))},
                                         basins_fixture, sample_dates_fixture, lead_times)
    with pytest.raises(ValueError, match='Target sequence length is required when validating target data.'):
        validate_samples(
            is_train=True, # Training mode
            dataset=dummy_dataset,
            sample_dates=sample_dates_fixture,
            nan_handling_method=None,
            feature_groups=[],
            target_features=['t1'],
            predict_last_n=None # Missing predict_last_n
        )
