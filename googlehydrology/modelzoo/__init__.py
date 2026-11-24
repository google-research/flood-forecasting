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

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config


def get_model_class(cfg: Config) -> type[BaseModel]:
    """Return the model class specified by ``cfg.model`` without instantiating.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    type[BaseModel]
        The model class corresponding to ``cfg.model``.
    """
    model_name = cfg.model.lower()
    if model_name == 'handoff_forecast_lstm':
        return HandoffForecastLSTM
    if model_name == 'mean_embedding_forecast_lstm':
        return MeanEmbeddingForecastLSTM
    raise NotImplementedError(
        f'{cfg.model} not implemented or not linked in `get_model()`'
    )


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
    model = get_model_class(cfg)(cfg=cfg)

    if cfg.compile:
        return torch.compile(model, mode='max-autotune')
    return model
