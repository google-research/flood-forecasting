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

from googlehydrology.training.basetrainer import BaseTrainer
from googlehydrology.utils.config import Config


def start_training(cfg: Config):
    """Start model training.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    """
    # MC-LSTM is a special case, where the head returns an empty string but the model is trained as regression model.
    if cfg.head.lower() in ['regression', 'gmm', 'umal', 'cmal', 'cmal_deterministic', '']:
        trainer = BaseTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")
    trainer.initialize_training()
    trainer.train_and_validate()
