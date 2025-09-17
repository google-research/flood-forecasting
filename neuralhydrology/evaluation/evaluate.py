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

from pathlib import Path

from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config


def start_evaluation(cfg: Config, run_dir: Path, epoch: int = None, period: str = "test"):
    """Start evaluation of a trained network

    Parameters
    ----------
    cfg : Config
        The run configuration, read from the run directory.
    run_dir : Path
        Path to the run directory.
    epoch : int, optional
        Define a specific epoch to evaluate. By default, the weights of the last epoch are used.
    period : {'train', 'validation', 'test'}, optional
        The period to evaluate, by default 'test'.

    """
    tester = get_tester(cfg=cfg, run_dir=run_dir, period=period, init_model=True)
    tester.evaluate(epoch=epoch, save_results=True, save_all_output=cfg.save_all_output, metrics=cfg.metrics)
