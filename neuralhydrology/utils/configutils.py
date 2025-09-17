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

"""Utility script to generate config files from a base config and a defined set of variations"""
import itertools
from pathlib import Path
from typing import Dict

from neuralhydrology.utils.config import Config


def create_config_files(base_config_path: Path, modify_dict: Dict[str, list], output_dir: Path):
    """Create configs, given a base config and a dictionary of parameters to modify.
    
    This function will create one config file for each combination of parameters defined in the modify_dict.
    
    Parameters
    ----------
    base_config_path : Path
        Path to a base config file (.yml)
    modify_dict : dict
        Dictionary, mapping from parameter names to lists of possible parameter values.
    output_dir : Path 
        Path to a folder where the generated configs will be stored
    """
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # load base config as dictionary
    base_config = Config(base_config_path)
    experiment_name = base_config.experiment_name
    option_names = list(modify_dict.keys())

    # iterate over each possible combination of hyper parameters
    for i, options in enumerate(itertools.product(*[val for val in modify_dict.values()])):

        base_config.update_config(dict(zip(option_names, options)))

        # create a unique run name
        name = experiment_name
        for key, val in zip(option_names, options):
            name += f"_{key}{val}"
        base_config.update_config({"experiment_name": name})

        base_config.dump_config(output_dir, f"config_{i+1}.yml")

    print(f"Finished. Configs are stored in {output_dir}")


def flatten_feature_list(data: list[str] | list[list[str]]| dict[str, list[str]]) -> list[str]:
    if not data:
        return []
    if isinstance(data, dict):
        return list(itertools.chain.from_iterable(data.values()))
    if isinstance(data, list) and isinstance(data[0], list):
        return list(itertools.chain.from_iterable(data))
    return list(data)