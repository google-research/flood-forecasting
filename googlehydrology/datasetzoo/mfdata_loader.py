"""Load basin data (mfdataset) and output its pickle serialization to stdout."""

import pickle
import sys
from pathlib import Path

from absl import app, flags

from googlehydrology.datasetzoo.caravan import load_caravan_timeseries_together

DATA_DIR = flags.DEFINE_string(
    'data_dir',
    default=None,
    help='Path to the root directory of Caravan.',
    required=True,
)

BASINS = flags.DEFINE_list(
    'basins',
    default=None,
    help='List of basin ids to load.',
    required=True,
)

TARGET_FEATURES = flags.DEFINE_list(
    'target_features',
    default=None,
    help='Target variables to select.',
    required=True,
)


def main(unused_argv):
    dataset = load_caravan_timeseries_together(
        Path(DATA_DIR.value), BASINS.value, TARGET_FEATURES.value
    )
    serialized = pickle.dumps(dataset)
    sys.stdout.buffer.write(serialized)


if __name__ == '__main__':
    app.run(main)
