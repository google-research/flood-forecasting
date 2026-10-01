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
    model_or_params: torch.nn.Module | Iterable[torch.Tensor] | Iterable[dict],
    cfg: Config,
    *,
    is_gpu: bool = False,
) -> torch.optim.Optimizer:
    """Get specific optimizer object, depending on the run configuration.

    Supported ``cfg.optimizer`` values (case-insensitive) are 'adam', 'adamw',
    'sgd', 'adagrad' and 'adadelta' (fused on GPU where available) as well as
    'asgd', 'rmsprop' and 'adamax'.

    Parameters
    ----------
    model_or_params : torch.nn.Module | Iterable[torch.Tensor] | Iterable[dict]
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

    Raises
    ------
    NotImplementedError
        If ``cfg.optimizer`` is not one of the supported optimizers.
    """
    params = (
        model_or_params.parameters()
        if isinstance(model_or_params, torch.nn.Module)
        else model_or_params
    )
    name = cfg.optimizer.lower()
    fused = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'adagrad': torch.optim.Adagrad,
    }
    unfused = {
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'adadelta': torch.optim.Adadelta,
        'adamax': torch.optim.Adamax,
    }
    if name in fused:
        return fused[name](params, lr=cfg.initial_learning_rate, fused=is_gpu)
    if name in unfused:
        return unfused[name](params, lr=cfg.initial_learning_rate)
    raise NotImplementedError(
        f'{cfg.optimizer} not implemented or not linked in `get_optimizer()`'
    )


def get_loss_obj(cfg: Config, *, per_sequence: bool = False) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.

    Currently supported are 'MSE', 'NSE', 'RMSE', 'CMALLoss' (or 'CMAL').

    Parameters
    ----------
    cfg : Config
        The run configuration.
    per_sequence : bool, optional
        Average within each sequence before averaging over sequences (see
        `loss.BaseLoss`). Supported by 'MSE', 'NSE' and 'CMAL'; 'RMSE' does
        not support it and raises ``ValueError`` if it is requested. Default
        False (pooled mean, as used for training).

    Returns
    -------
    loss.BaseLoss
        A new loss instance that implements the loss specified in the config or, if different, the loss required by the
        head.
    """
    if cfg.loss.lower() == 'mse':
        loss_obj = loss.MaskedMSELoss(cfg, per_sequence=per_sequence)
    elif cfg.loss.lower() == 'nse':
        loss_obj = loss.MaskedNSELoss(cfg, per_sequence=per_sequence)
    elif cfg.loss.lower() == 'rmse':
        if per_sequence:
            raise ValueError(
                'per_sequence reduction is not supported for the RMSE loss.'
            )
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif cfg.loss.lower() in ['cmalloss', 'cmal']:
        loss_obj = loss.MaskedCMALLoss(cfg, per_sequence=per_sequence)
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
