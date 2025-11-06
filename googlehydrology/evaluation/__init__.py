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

from googlehydrology.evaluation.tester import (
    BaseTester,
    RegressionTester,
    UncertaintyTester,
)
from googlehydrology.utils.config import Config


def get_tester(
    cfg: Config, run_dir: Path, period: str, init_model: bool
) -> BaseTester:
    """Get specific tester class objects depending on the model (head) type.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.

    Returns
    -------
    BaseTester
        `RegressionTester` if the model head is 'regression'. `UncertaintyTester` if the model head is one of
        {'cmal', 'cmal_deterministic'} or if the evaluation is run in MC-Dropout mode.
    """
    if cfg.mc_dropout or cfg.head.lower() in ['cmal', 'cmal_deterministic']:
        Tester = UncertaintyTester
    # MC-LSTM is a special case, where the head returns an empty string but the model is trained as regression model.
    elif cfg.head.lower() in ['regression', '']:
        Tester = RegressionTester
    else:
        NotImplementedError(
            f'No evaluation method implemented for {cfg.head} head'
        )

    return Tester(
        cfg=cfg, run_dir=run_dir, period=period, init_model=init_model
    )
