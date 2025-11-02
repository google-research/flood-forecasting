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

"""Unit tests for configutils functions. """
from googlehydrology.utils.configutils import flatten_feature_list, group_features_list

def test_flatten_feature_list_empty_list_input():
    """Tests behavior with an empty list input."""
    assert flatten_feature_list([]) == []

def test_flatten_feature_list_simple_list_input():
    """Tests a standard list of strings (list[str])."""
    data = ["aa", "bb", "cc"]
    expected = ["aa", "bb", "cc"]
    assert flatten_feature_list(data) == expected

def test_flatten_feature_list_list_of_lists_input():
    """Tests a list of lists of strings (list[list[str]])."""
    data = [["aa", "bb"], ["cc"], ["dd", "ee"]]
    expected = ["aa", "bb", "cc", "dd", "ee"]
    assert flatten_feature_list(data) == expected

def test_flatten_feature_list_dictionary_input():
    """Tests a dictionary with string keys and list values (dict[str, list[str]])."""
    data = {
        "11": ["aa", "bb"],
        "22": ["cc"],
        "33": ["dd", "ee"]
    }
    expected = ["aa", "bb", "cc", "dd", "ee"]
    assert flatten_feature_list(data) == expected

def test_flatten_feature_list_empty_dictionary_input():
    """Tests behavior with an empty dictionary."""
    assert flatten_feature_list({}) == []

def test_flatten_feature_list_dictionary_with_empty_list_value():
    """Tests a dictionary containing an empty list."""
    data = {
        "11": ["aa", "bb"],
        "22": []
    }
    expected = ["aa", "bb"]
    assert flatten_feature_list(data) == expected

def test_flatten_feature_list_list_of_empty_lists():
    """Tests a list that contains only empty lists."""
    data = [[], [], []]
    expected = []
    assert flatten_feature_list(data) == expected

def test_group_features_list_normal_case():
    """Tests a typical case with multiple groups and items."""
    features = ['temp_max', 'temp_min', 'wind_speed', 'wind_dir', 'temp_avg']
    expected = {
        'temp': {'temp_max', 'temp_min', 'temp_avg'},
        'wind': {'wind_speed', 'wind_dir'},
    }
    assert group_features_list(features) == expected

def test_group_features_list_empty_list():
    """Tests what happens when an empty list is provided."""
    assert group_features_list([]) == {}

def test_group_features_list_single_item():
    """Tests a list with only one item."""
    features = ['pressure_sea']
    expected = {'pressure': {'pressure_sea'}}
    assert group_features_list(features) == expected

def test_group_features_list_all_same_prefix():
    """Tests when all items share the same prefix."""
    features = ['cloud_high', 'cloud_mid', 'cloud_low']
    expected = {'cloud': {'cloud_high', 'cloud_mid', 'cloud_low'}}
    assert group_features_list(features) == expected

def test_group_features_list_no_underscores():
    """Tests features that do not contain an underscore. 
        The entire string should become the prefix."""
    features = ['temperature', 'wind', 'pressure']
    expected = {
        'temperature': {'temperature'},
        'wind': {'wind'},
        'pressure': {'pressure'},
    }
    assert group_features_list(features) == expected

def test_group_features_list_multiple_underscores():
    """Tests that only the first part before the *first* underscore is used."""
    features = ['data_v1_max', 'data_v1_min', 'data_v2_mean']
    expected = {'data': {'data_v1_max', 'data_v1_min', 'data_v2_mean'}}
    assert group_features_list(features) == expected

def test_group_features_list_nested_list_normal_case():
    """Tests a typical case with multiple groups and items."""
    features = [['temp_max', 'temp_min', 'temp_avg'], ['wind_speed', 'wind_dir']]
    expected = {
        'temp': {'temp_max', 'temp_min', 'temp_avg'},
        'wind': {'wind_speed', 'wind_dir'},
    }
    assert group_features_list(features) == expected

def test_group_features_list_nested_list_repeating_item():
    """Tests a nested list with a repeating item."""
    features = [['pressure_sea', 'pressure_sea']]
    expected = {'pressure': {'pressure_sea'}}
    assert group_features_list(features) == expected

def test_group_features_list_nested_list_all_same_prefix():
    """Tests when all items share the same prefix."""
    features = [['cloud_high', 'cloud_mid', 'cloud_low']]
    expected = {'cloud': {'cloud_high', 'cloud_mid', 'cloud_low'}}
    assert group_features_list(features) == expected

def test_group_features_list_nested_list_no_underscores():
    """Tests features that do not contain an underscore. 
        The entire first string should become the prefix."""
    features = [['temperature', 'wind', 'pressure']]
    expected = {
        'temperature': {'temperature', 'wind', 'pressure'},
    }
    assert group_features_list(features) == expected

def test_group_features_list_nested_list_multiple_underscores():
    """Tests that only the first part before the *first* underscore is used."""
    features = [['data_v1_max', 'data_v1_min', 'data_v2_mean']]
    expected = {'data': {'data_v1_max', 'data_v1_min', 'data_v2_mean'}}
    assert group_features_list(features) == expected

def test_group_features_list_dict_normal_case():
    """Tests a typical case with multiple groups and items."""
    features = {'temp': ['temp_max', 'temp_min', 'temp_avg'], 'wind': ['wind_speed', 'wind_dir']}
    expected = {
        'temp': {'temp_max', 'temp_min', 'temp_avg'},
        'wind': {'wind_speed', 'wind_dir'},
    }
    assert group_features_list(features) == expected

def test_group_features_list_empty_dict():
    """Tests what happens when an empty dict is provided."""
    assert group_features_list({}) == {}

def test_group_features_list_dict_repeating_item():
    """Tests a dict with a repeating item."""
    features = {'group': ['pressure_sea', 'pressure_sea']}
    expected = {'group': {'pressure_sea'}}
    assert group_features_list(features) == expected

def test_group_features_list_dict_all_same_prefix():
    """Tests when all items share the same prefix."""
    features = {'cloud': ['cloud_high', 'cloud_mid', 'cloud_low']}
    expected = {'cloud': {'cloud_high', 'cloud_mid', 'cloud_low'}}
    assert group_features_list(features) == expected

def test_group_features_list_dict_no_underscores():
    """Tests features that do not contain an underscore. 
        The entire string should become the prefix."""
    features = {'group': ['temperature', 'wind', 'pressure']}
    expected = {'group': {'temperature', 'wind', 'pressure'}}
    assert group_features_list(features) == expected
