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

"""Helpers shared by the data assimilation engine tests.

They sit on top of the ``tiny_mean_embedding_model`` and
``tiny_mean_embedding_data`` conftest builders: a model with ``SEQ_LENGTH``
observed and ``LEAD_TIME`` forecast steps, targets that are the prior
prediction plus a constant shift, and an ``AssimilationConfig`` built through
a real ``Config``.
"""

from collections.abc import Callable

import torch

from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.utils.config import Config

SEQ_LENGTH = 8
LEAD_TIME = 2
WINDOW = 4
BATCH_SIZE = 3
SHIFT = 1.0
ALL_COMPONENTS = [
    'static_embedding',
    'hindcast_embedding',
    'forecast_embedding',
]
DYNAMIC_COMPONENTS = ['hindcast_embedding', 'forecast_embedding']
# Defaults of the DA block; tests override single keys.
DA_DEFAULTS = {
    'assimilation_components': ALL_COMPONENTS,
    'assimilation_window': WINDOW,
    'epochs': 50,
    'initial_learning_rate': 0.1,
    'regularization_weight': 1e-3,
    'loss': 'MSE',
}


def build_model(
    builder: Callable[..., MeanEmbeddingForecastLSTM], **options: object
) -> MeanEmbeddingForecastLSTM:
    """Build the tiny model with the DA test geometry."""
    return builder(seq_length=SEQ_LENGTH, lead_time=LEAD_TIME, **options)


def da_config(model: BaseModel, **overrides: object) -> AssimilationConfig:
    """Return the ``AssimilationConfig`` of the run config plus a DA block."""
    options = dict(model.cfg.as_dict())
    options['assimilation_config'] = {**DA_DEFAULTS, **overrides}
    return Config(options).assimilation_config


def make_data(
    model: BaseModel,
    maker: Callable[..., dict],
    *,
    batch_size: int = BATCH_SIZE,
    shift: float = SHIFT,
    nan_at: dict[str, list[int]] | None = None,
) -> dict:
    """Random inputs; targets are the prior point prediction plus ``shift``.

    Targets cover ``SEQ_LENGTH`` steps ending at the last forecast step, as
    in the datasets, i.e. they are offset by ``LEAD_TIME`` from the model
    output. ``per_basin_target_stds`` (ones) is included for the NSE loss.
    """
    data = maker(model.cfg, batch_size=batch_size, nan_at=nan_at)
    with torch.no_grad():
        prior = model.point_prediction(model(data))
    data['y'] = prior[:, LEAD_TIME:] + shift
    n_targets = prior.shape[-1]
    data['per_basin_target_stds'] = torch.ones(batch_size, 1, n_targets)
    return data


def prior_output(model: BaseModel, data: dict) -> dict[str, torch.Tensor]:
    """Unassimilated model output including the embeddings."""
    with torch.no_grad():
        return model(data, return_embeddings=True)


def assimilate(
    model: BaseModel, data: dict, **overrides: object
) -> dict[str, torch.Tensor]:
    """Run DA with the default DA block updated by ``overrides``."""
    return Assimilation(da_config(model, **overrides)).assimilate(model, data)


def window_slice(model_len: int) -> slice:
    """Window on the model time axis: ends ``LEAD_TIME`` before the end."""
    return slice(model_len - LEAD_TIME - WINDOW, model_len - LEAD_TIME)


def observed(data: dict) -> torch.Tensor:
    """Observations inside the window, [B, WINDOW, n_targets]."""
    y = data['y']
    return y[:, y.shape[1] - LEAD_TIME - WINDOW : y.shape[1] - LEAD_TIME]


def window_error(
    model: BaseModel, output: dict[str, torch.Tensor], data: dict
) -> torch.Tensor:
    """Per-sequence MSE of the point prediction inside the window, [B]."""
    point = model.point_prediction(output)
    pred = point[:, window_slice(point.shape[1])]
    return ((pred - observed(data)) ** 2).flatten(1).nanmean(dim=1)


def select_rows(data: dict, rows: list[int]) -> dict:
    """Return the batch restricted to ``rows`` (nested dicts included)."""
    return {
        key: (
            {name: tensor[rows] for name, tensor in value.items()}
            if isinstance(value, dict)
            else value[rows]
        )
        for key, value in data.items()
    }
