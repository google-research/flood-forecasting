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

import itertools

import more_itertools


def flatten_feature_list(
    data: list[str] | list[list[str]] | dict[str, list[str]],
) -> list[str]:
    if not data:
        return []
    if isinstance(data, dict):
        return list(itertools.chain.from_iterable(data.values()))
    if isinstance(data, list) and isinstance(data[0], list):
        return list(itertools.chain.from_iterable(data))
    return list(data)


def group_features_list(
    features: list[str] | list[list[str]] | dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Groups list of features according to the prefix of each feature.
    For lists, assumes the prefix is written with an underscore for the first feature name.
    """
    # TODO (future) :: Simplify types to only dict.
    # TODO (future) :: Simplify impl and make it more resilient (check all values etc)
    if not features:
        return {}
    if isinstance(features, dict):
        return {key: _unique(value) for key, value in features.items()}
    if isinstance(features, list) and isinstance(features[0], list):
        return {_prefix(sublist[0]): _unique(sublist) for sublist in features}
    if isinstance(features, list) and all(isinstance(e, str) for e in features):
        result = {}
        for feature in features:
            result.setdefault(_prefix(feature), []).append(feature)
        for feature, items in result.items():
            result[feature] = _unique(items)
        return result
    raise ValueError(f'Unsupported type {features}')


def _prefix(feature: str) -> str:
    return feature.partition('_')[0]


def _unique(features: list[str]) -> list[str]:
    return list(more_itertools.unique_everseen(features))
