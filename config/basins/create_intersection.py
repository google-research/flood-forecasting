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

#
# Creates intersections between caravan and multimet gauges.
#
# Supply --basins=<int list> to take overall approx a number of basins,
# where each one comprises its size divided by the number of datasets
# that intersect that are available. For example, generate 500 basins
# where given 4 datasets available, the 500 basins will be composed of
# 125 basins each from a dataset.
#
# The --batch=<int> specifies which batch number to write, e.g.
# a file name may be 500_0 or 500_1 etc. This allows to replace a
# batch or generate new ones.
#
# Example invocation:
# $ config/basins/create_intersection.py --basins=100 --basins=200 --batch=0
#
# NOTE: to simplify, output may not be exactly the num of basins specified.

import itertools
import math
import os
from pathlib import Path
import random

from absl import app
from absl import flags

PATH = Path(os.path.dirname(__file__))  # Abs path for .../config/basins.

NUM_BASINS = flags.DEFINE_multi_integer(
    'basins',
    default=None,
    help='Number of basins files to genereate. Supply 0 to include all.',
    required=True,
)

BATCH = flags.DEFINE_integer(
    'batch',
    default=None,
    help='The batch number of the results.',
    required=True,
)


def main(unused_argv):
    caravan = read_gauges(PATH / 'caravan_gauges.txt')
    multimet = read_gauges(PATH / 'multimet_gauges.txt')
    all = sorted(caravan & multimet)
    datasets = {
        k: tuple(v) for k, v in itertools.groupby(all, key=extract_dataset)
    }

    for num_basins in NUM_BASINS.value:
        with open(
            PATH / 'intersections' / f'{num_basins}_{BATCH.value}.txt', 'w'
        ) as f:
            for basins in datasets.values():
                # TODO: Divide dataset sizes correctly for the case requesting
                #       a larger total than some datasets' size. Because then
                #       the division needs not be equal, needs to take more
                #       basins from the larger datasets accordingly.
                #       For example, to generate 5000, need to input 12000.
                k = min(len(basins), math.ceil(num_basins / len(datasets)))
                res = random.sample(basins, k=k)
                f.writelines(f'{e}\n' for e in res or basins)


def read_gauges(file: Path) -> set[str]:
    with open(file) as f:
        return set(e.strip() for e in f.readlines() if e.strip())


def extract_dataset(e: str) -> str:
    return e.partition('_')[0]


if __name__ == '__main__':
    app.run(main)
