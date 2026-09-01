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

"""Unit tests for googlehydrology.utils.lstm_utils."""

import pytest
import torch
import torch.nn as nn

from googlehydrology.utils.config import WeightInitOpt
from googlehydrology.utils.lstm_utils import _forget_gate_slice, lstm_init


@pytest.mark.unit
def test_forget_gate_slice():
    lstm = nn.LSTM(input_size=10, hidden_size=20)
    sl = _forget_gate_slice(lstm)
    assert sl == slice(20, 40)


@pytest.mark.unit
def test_lstm_init():
    lstm = nn.LSTM(input_size=10, hidden_size=20)
    lstm_init(
        lstms=[lstm],
        forget_bias=1.5,
        weight_opts=[
            WeightInitOpt.LSTM_IH_XAVIER,
            WeightInitOpt.LSTM_HH_ORTHOGONAL,
        ],
    )
    # Check forget bias initialized
    sl = _forget_gate_slice(lstm)
    assert torch.allclose(lstm.bias_hh_l0.data[sl], torch.tensor(1.5))
