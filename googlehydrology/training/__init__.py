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

import torch

import googlehydrology.training.loss as loss
from googlehydrology.training import regularization
from googlehydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


from typing import Iterable, Union


def get_optimizer(
    model: Union[torch.nn.Module, Iterable[torch.Tensor]], cfg: Config, *, is_gpu: bool = False
) -> torch.optim.Optimizer:
    """Get specific optimizer object, depending on the run configuration.

    Parameters
    ----------
    model : torch.nn.Module or Iterable[torch.Tensor]
        The model to be optimized or an iterable of parameters/tensors.
    cfg : Config
        The run configuration.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object that can be used for model training or data assimilation.
    """
    params = model.parameters() if hasattr(model, 'parameters') else model

    # Resolve learning rate
    lr = getattr(cfg, 'initial_learning_rate', None)
    if lr is None:
        if hasattr(cfg, 'learning_rate'):
            lr_val = cfg.learning_rate
            lr = lr_val[0] if isinstance(lr_val, dict) else lr_val
        else:
            lr = 0.01  # default fallback

    if cfg.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=lr, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=lr, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=lr, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'asgd':
        optimizer = torch.optim.ASGD(
            params, lr=lr
        )
    elif cfg.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params, lr=lr
        )
    elif cfg.optimizer.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(
            params, lr=lr, fused=is_gpu
        )
    elif cfg.optimizer.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta(
            params,
            lr=lr,
        )
    elif cfg.optimizer.lower() == 'adamax':
        optimizer = torch.optim.Adamax(
            params, lr=lr
        )
    else:
        raise NotImplementedError(
            f'{cfg.optimizer} not implemented or not linked in `get_optimizer()`'
        )

    return optimizer


def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.

    Currently supported are 'MSE', 'NSE', 'RMSE', 'CMALLoss'.

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
            f'{cfg.loss} not implemented or not linked in `get_loss()`'
        )

    return loss_obj


def get_regularization_obj(
    cfg: Config,
) -> list[regularization.BaseRegularization]:
    """Get list of regularization objects.

    Currently, only the 'tie_frequencies' regularization is implemented.

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
        else:
            raise NotImplementedError(
                f'{reg_name} not implemented or not linked in `get_regularization_obj()`.'
            )

    return regularization_modules
