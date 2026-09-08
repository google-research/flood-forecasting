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

"""Tests for HandoffForecastLSTM feature ordering."""

import torch

from googlehydrology.modelzoo.handoff_forecast_lstm import (
    _concat_dynamic_features,
)


def test_concat_dynamic_features_uses_model_feature_order() -> None:
    data = {
        'temperature': torch.tensor([[[2.0], [4.0]]]),
        'precipitation': torch.tensor([[[1.0], [3.0]]]),
    }

    result = _concat_dynamic_features(
        data, ['precipitation', 'temperature']
    )

    expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    assert torch.equal(result, expected)
