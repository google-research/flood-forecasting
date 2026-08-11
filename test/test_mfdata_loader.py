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

"""Unit tests for googlehydrology.datasetzoo.mfdata_loader."""

from unittest.mock import patch

from absl import flags
from absl.testing import flagsaver
import pytest
import xarray as xr

from googlehydrology.datasetzoo import mfdata_loader

FLAGS = flags.FLAGS


@pytest.mark.unit
def test_mfdata_loader_main():
    mock_ds = xr.Dataset({'streamflow': [1.0, 2.0]})

    with (
        flagsaver.flagsaver(),
        patch(
            'googlehydrology.datasetzoo.mfdata_loader.'
            'load_caravan_timeseries_together',
            return_value=mock_ds,
        ) as mock_load,
        patch('sys.stdout.buffer.write') as mock_write,
    ):
        FLAGS([
            'mfdata_loader.py',
            '--data_dir=/tmp/data',
            '--basins=b1',
            '--target_features=streamflow',
        ])
        mfdata_loader.main([])
        mock_load.assert_called_once()
        mock_write.assert_called_once()
