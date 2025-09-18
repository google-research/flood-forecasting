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
from neuralhydrology.utils.configutils import flatten_feature_list

def test_empty_list_input():
    """Tests behavior with an empty list input."""
    assert flatten_feature_list([]) == []

def test_simple_list_input():
    """Tests a standard list of strings (list[str])."""
    data = ["aa", "bb", "cc"]
    expected = ["aa", "bb", "cc"]
    assert flatten_feature_list(data) == expected

def test_list_of_lists_input():
    """Tests a list of lists of strings (list[list[str]])."""
    data = [["aa", "bb"], ["cc"], ["dd", "ee"]]
    expected = ["aa", "bb", "cc", "dd", "ee"]
    assert flatten_feature_list(data) == expected

def test_dictionary_input():
    """Tests a dictionary with string keys and list values (dict[str, list[str]])."""
    data = {
        "11": ["aa", "bb"],
        "22": ["cc"],
        "33": ["dd", "ee"]
    }
    expected = ["aa", "bb", "cc", "dd", "ee"]
    assert flatten_feature_list(data) == expected

def test_empty_dictionary_input():
    """Tests behavior with an empty dictionary."""
    assert flatten_feature_list({}) == []

def test_dictionary_with_empty_list_value():
    """Tests a dictionary containing an empty list."""
    data = {
        "11": ["aa", "bb"],
        "22": []
    }
    expected = ["aa", "bb"]
    assert flatten_feature_list(data) == expected

def test_list_of_empty_lists():
    """Tests a list that contains only empty lists."""
    data = [[], [], []]
    expected = []
    assert flatten_feature_list(data) == expected

