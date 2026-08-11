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

"""Unit tests for googlehydrology.evaluation.plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from googlehydrology.evaluation.plots import percentile_plot, regression_plot


@pytest.mark.unit
def test_percentile_plot():
    y = np.linspace(0, 10, 50)
    # y_hat has shape [50, 100] (50 timesteps, 100 samples)
    y_hat = np.random.normal(loc=y[:, None], scale=1.0, size=(50, 100))

    fig, ax = percentile_plot(y=y, y_hat=y_hat, title='Test Percentiles')
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == 'Test Percentiles'
    plt.close(fig)


@pytest.mark.unit
def test_regression_plot():
    y = np.linspace(0, 10, 50)
    y_hat = y + np.random.normal(0, 0.2, size=50)

    fig, ax = regression_plot(y=y, y_hat=y_hat, title='Test Regression')
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == 'Test Regression'
    plt.close(fig)
