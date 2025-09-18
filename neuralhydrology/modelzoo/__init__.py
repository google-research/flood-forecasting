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
import torch.nn as nn

from neuralhydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from neuralhydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from neuralhydrology.utils.config import Config

SINGLE_FREQ_MODELS = ["handoff_forecast_lstm"]
AUTOREGRESSIVE_MODELS = []


torch.compile(mode="max-autotune")
def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """
    if cfg.model.lower() in SINGLE_FREQ_MODELS and len(cfg.use_frequencies) > 1:
        raise ValueError(f"Model {cfg.model} does not support multiple frequencies.")

    if cfg.model.lower() not in AUTOREGRESSIVE_MODELS and cfg.autoregressive_inputs:
        raise ValueError(f"Model {cfg.model} does not support autoregression.")

    if cfg.mass_inputs:
        raise ValueError(f"The use of 'mass_inputs' with {cfg.model} is not supported.")

    if cfg.model.lower() == "handoff_forecast_lstm":
        model = HandoffForecastLSTM(cfg=cfg)
    elif cfg.model.lower() == "mean_embedding_forecast_lstm":
        model = MeanEmbeddingForecastLSTM(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
