#!/usr/bin/env python
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

import argparse
import logging
import sys
from pathlib import Path

import cachey
import dask
import dask.cache
import torch
import tqdm.dask
import xarray

from googlehydrology.utils.tqdm import AutoRefreshTqdm

# make sure code directory is in path, even if the package is not installed using the setup.py
sys.path.append(str(Path(__file__).parent.parent))
from googlehydrology.evaluation.evaluate import start_evaluation
from googlehydrology.training.train import start_training
from googlehydrology.utils.config import Config
from googlehydrology.utils.logging_utils import setup_logging


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        choices=['train', 'continue_training', 'finetune', 'evaluate', 'infer'],
    )
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--run-dir', type=str)
    parser.add_argument(
        '--epoch',
        type=int,
        help='Epoch, of which the model should be evaluated',
    )
    parser.add_argument(
        '--period',
        type=str,
        choices=['train', 'validation', 'test'],
        default='test',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        help="GPU id to use. Overrides config argument 'device'. Use a value < 0 for CPU.",
    )
    args = vars(parser.parse_args())

    if (args['mode'] in ['train', 'finetune']) and (
        args['config_file'] is None
    ):
        raise ValueError('Missing path to config file')

    if (args['mode'] == 'continue_training') and (args['run_dir'] is None):
        raise ValueError('Missing path to run directory file')

    if (args['mode'] in ['evaluate', 'infer']) and (args['run_dir'] is None):
        raise ValueError('Missing path to run directory')

    return args


def _main():
    args = _get_args()
    config = Config(
        Path(args['config_file'] or Path(args['run_dir']) / 'config.yml')
    )

    if (args['run_dir'] is not None) and (
        args['mode'] in ['evaluate', 'infer']
    ):
        setup_logging(
            str(Path(args['run_dir']) / 'output.log'),
            config.logging_level,
            config.print_warnings_once,
        )

    torch.autograd.set_detect_anomaly(config.detect_anomaly)

    dask_config = {
        'scheduler': 'threads',
        'shuffle': 'p2p',
    }
    if config.use_swap_memory is not None:
        dask_config['distributed.p2p.storage.disk'] = config.use_swap_memory
    dask.config.set(dask_config)

    if config.cache.enabled:
        dask.cache.Cache(cachey.Cache(config.cache.byte_limit)).register()

    if config.logging_level <= logging.DEBUG:
        tqdm.dask.TqdmCallback(
            mininterval=2,
            unit='task',
            desc='compute',
            unit_scale=True,
            dynamic_ncols=True,
            tqdm_class=AutoRefreshTqdm,
        ).register()

    # engines netcdf4 and h5netcdf fail parallelizing anyway
    xarray.set_options(file_cache_maxsize=1)

    if args['mode'] == 'train':
        start_run(config=config, gpu=args['gpu'])
    elif args['mode'] == 'continue_training':
        continue_run(
            run_dir=Path(args['run_dir']),
            config_file=Path(args['config_file'])
            if args['config_file'] is not None
            else None,
            gpu=args['gpu'],
        )
    elif args['mode'] == 'finetune':
        finetune(config_file=Path(args['config_file']), gpu=args['gpu'])
    elif args['mode'] in ['evaluate', 'infer']:
        config.inference_mode = args['mode'] == 'infer'
        if config.inference_mode:
            config.tester_skip_obs_all_nan = False
        eval_run(
            config,
            run_dir=Path(args['run_dir']),
            period=args['period'],
            epoch=args['epoch'],
            gpu=args['gpu'],
        )
    else:
        raise RuntimeError(f'Unknown mode {args["mode"]}')


def start_run(config: Config, gpu: int = None):
    """Start training a model.

    Parameters
    ----------
    config: Config
        Config object from a config file (.yml), defining the settings for the specific run.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

    """
    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f'cuda:{gpu}'
    if gpu is not None and gpu < 0:
        config.device = 'cpu'

    start_training(config)


def continue_run(run_dir: Path, config_file: Path = None, gpu: int = None):
    """Continue model training.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.
    config_file : Path, optional
        Path to an additional config file. Each config argument in this file will overwrite the original run config.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

    """
    # load config from base run and overwrite all elements with an optional new config
    base_config = Config(run_dir / 'config.yml')

    if config_file is not None:
        base_config.update_config(config_file)
        base_config.run_dir = run_dir

    base_config.is_continue_training = True

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        base_config.device = f'cuda:{gpu}'
    if gpu is not None and gpu < 0:
        base_config.device = 'cpu'

    start_training(base_config)


def finetune(config_file: Path = None, gpu: int = None):
    """Finetune a pre-trained model.

    Parameters
    ----------
    config_file : Path, optional
        Path to an additional config file. Each config argument in this file will overwrite the original run config.
        The config file for finetuning must contain the argument `base_run_dir`, pointing to the folder of the
        pre-trained model, as well as 'finetune_modules' to indicate which model parts will be trained during
        fine-tuning.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

    """
    # load finetune config and check for a non-empty list of finetune_modules
    temp_config = Config(config_file)
    if not temp_config.finetune_modules:
        raise ValueError(
            "For finetuning, at least one model part has to be specified by 'finetune_modules'."
        )

    # extract base run dir, load base run config and combine with the finetune arguments
    config = Config(temp_config.base_run_dir / 'config.yml')
    config.update_config({'run_dir': None, 'experiment_name': None})
    config.update_config(config_file)
    config.is_finetuning = True

    # if the base run was a continue_training run, we need to override the continue_training flag from its config.
    config.is_continue_training = False

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f'cuda:{gpu}'
    if gpu is not None and gpu < 0:
        config.device = 'cpu'

    start_training(config)


def eval_run(
    config: Config,
    run_dir: Path,
    period: str,
    epoch: int = None,
    gpu: int = None,
):
    """Start evaluating a trained model.

    Parameters
    ----------
    config: Config
        Config object from a config file (.yml), defining the settings for the specific run.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    epoch : int, optional
        Define a specific epoch to use. By default, the weights of the last epoch are used.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value less than zero indicates CPU.
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

    """
    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f'cuda:{gpu}'
    if gpu is not None and gpu < 0:
        config.device = 'cpu'

    start_evaluation(cfg=config, run_dir=run_dir, epoch=epoch, period=period)


if __name__ == '__main__':
    _main()
