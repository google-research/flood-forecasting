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

import logging
from collections.abc import Iterable

import torch

import googlehydrology.training.loss as loss
from googlehydrology.training import regularization
from googlehydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_optimizer(
    model: torch.nn.Module | Iterable[torch.Tensor] | Iterable[dict],
    cfg: Config,
    *,
    is_gpu: bool = False,
) -> torch.optim.Optimizer:
    """Get specific optimizer object, depending on the run configuration.

    Parameters
    ----------
    model : torch.nn.Module | Iterable[torch.Tensor] | Iterable[dict]
        The model to be optimized, or the parameters to optimize directly: an
        iterable of tensors or of torch param-group dicts (e.g. tensors
        optimized during data assimilation). Param groups may set their own
        ``lr``; otherwise ``cfg.initial_learning_rate`` is used.
    cfg : Config
        The run configuration.
    is_gpu : bool, optional
        Whether to use the fused implementation, where available.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object that can be used for model training.
    """
    params = model.parameters() if isinstance(model, torch.nn.Module) else model
    if cfg.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=cfg.initial_learning_rate, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=cfg.initial_learning_rate, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=cfg.initial_learning_rate, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'asgd':
        optimizer = torch.optim.ASGD(params, lr=cfg.initial_learning_rate)
    elif cfg.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=cfg.initial_learning_rate)
    elif cfg.optimizer.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(
            params, lr=cfg.initial_learning_rate, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta(
            params,
            lr=cfg.initial_learning_rate,
        )
    elif cfg.optimizer.lower() == 'adamax':
        optimizer = torch.optim.Adamax(params, lr=cfg.initial_learning_rate)
    else:
        raise NotImplementedError(
            f'{cfg.optimizer} not implemented or not linked in `get_optimizer()`'
        )

    return optimizer


def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.

    Currently supported are 'MSE', 'NSE', 'RMSE', 'CMALLoss' (or 'CMAL').

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss.BaseLoss
        A new loss instance that implements the loss specified in the config or, if different, the loss required by the
        head.
    """
    if cfg.loss.lower() == 'mse':
        loss_obj = loss.MaskedMSELoss(cfg)
    elif cfg.loss.lower() == 'nse':
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == 'rmse':
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif cfg.loss.lower() in ['cmalloss', 'cmal']:
        loss_obj = loss.MaskedCMALLoss(cfg)
    else:
        raise NotImplementedError(
            f'{cfg.loss} not implemented or not linked in `get_loss_obj()`'
        )

    return loss_obj


def get_regularization_obj(
    cfg: Config,
) -> list[regularization.BaseRegularization]:
    """Get list of regularization objects.

    Currently supported are 'forecast_overlap' and 'bg_embedding'.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    list[regularization.BaseRegularization]
        List of regularization objects that will be added to the loss during training.
    """
    regularization_modules = []
    for reg_item in cfg.regularization:
        if isinstance(reg_item, str):
            reg_name = reg_item
            reg_weight = 1.0
        else:
            reg_name, reg_weight = reg_item
        if reg_name == 'forecast_overlap':
            regularization_modules.append(
                regularization.ForecastOverlapMSERegularization(
                    cfg=cfg, weight=reg_weight
                )
            )
        elif reg_name == 'bg_embedding':
            regularization_modules.append(
                regularization.BackgroundEmbeddingRegularization(
                    cfg=cfg, weight=reg_weight
                )
            )
        else:
            raise NotImplementedError(
                f'{reg_name} not implemented or not linked in `get_regularization_obj()`.'
            )

    return regularization_modules
