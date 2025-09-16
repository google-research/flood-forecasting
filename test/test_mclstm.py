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

import torch

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.mclstm import MCLSTM


def test_mass_conservation():
    torch.manual_seed(111)

    # create minimal config required for model initialization
    config = Config({
        'dynamic_inputs': ['tmin(C)', 'tmax(C)'],
        'hidden_size': 10,
        'initial_forget_bias': 0,
        'mass_inputs': ['prcp(mm/day)'],
        'model': 'mclstm',
        'target_variables': ['QObs(mm/d)']
    })
    model = MCLSTM(config)

    # create random inputs
    data = {
        # [batch size, sequence length, total number of time series inputs]
        'x_d': {k: torch.rand((1, 25, 1)) for k in config.dynamic_inputs + config.mass_inputs}
    }

    # get model outputs and intermediate states
    output = model(data)

    # the total mass within the system at each time step is the cumsum over the outgoing mass + the current cell state
    cumsum_system = output["m_out"].sum(-1).cumsum(-1) + output["c"].sum(-1)

    # the accumulated mass of the inputs at each time step
    cumsum_input = data["x_d"]['prcp(mm/day)'][:, :, 0].cumsum(-1)

    # check if the total mass is conserved at every timestep of the forward pass
    assert torch.allclose(cumsum_system, cumsum_input)
