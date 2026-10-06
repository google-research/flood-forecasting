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

"""Exercise deterministic head summaries through the public evaluation path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from googlehydrology.datasetzoo.multimet import SampleIndexer
from googlehydrology.evaluation import tester as tester_module
from googlehydrology.utils.config import Config


def evaluate_summaries(
    monkeypatch, head, reduction, samples, observations, clip=False
):
    config = Config(
        {
            'head': head,
            'tester_sample_reduction': reduction,
            'batch_size': 4,
            'predict_last_n': 2,
            'seq_length': 2,
            'target_variables': ['q'],
            'log_n_figures': 0,
            'validate_n_random_basins': 1,
            'mc_dropout': False,
            'inference_mode': False,
            'lazy_load': False,
            'clip_targets_to_zero': ['q'] if clip else [],
        }
    )
    sample_count = samples.shape[0]
    tester = tester_module.UncertaintyTester.__new__(
        tester_module.UncertaintyTester
    )
    tester.cfg = config
    tester.period = 'test'
    tester.init_model = False
    tester.basins = ['basin']
    tester._disable_pbar = True
    # Isolate dataset I/O and model inference; the production evaluator still
    # constructs xarray results, unscales, reduces and computes the real metric.
    tester.dataset = SimpleNamespace(
        frequencies=['1D'],
        _sample_index=SampleIndexer(
            (('basin', np.zeros(sample_count, dtype=int)),)
        ),
        collate_fn=None,
        scaler=SimpleNamespace(unscale=lambda ds: ds * 2 + 3),
    )
    ends = np.arange('2020-01-02', '2020-01-06', dtype='datetime64[D]')[
        :sample_count
    ]
    dates = np.stack([ends - np.timedelta64(1, 'D'), ends], axis=1)
    data = {
        'basin': 'basin',
        'preds': {'1D': np.repeat(samples[:, None, None, :], 2, axis=1)},
        'obs': {'1D': np.repeat(observations[:, None, None], 2, axis=1)},
        'dates': {'1D': dates},
        'mean_losses': {},
    }
    tester._evaluate = lambda *args: iter([data])
    tester._ensure_no_previous_results_saved = lambda *args: None
    saved = {}
    tester._save_incremental_results = lambda basin, **kwargs: saved.update(
        kwargs['results']
    )
    monkeypatch.setattr(
        tester_module, 'MultimetDataLoader', lambda *args, **kwargs: [data]
    )
    tester.evaluate(
        model=torch.nn.Identity(), metrics=['MSE'], save_results=False
    )
    return saved['1D']


@pytest.mark.parametrize('head', ['cmal_deterministic', 'CMAL_DETERMINISTIC'])
@pytest.mark.parametrize('reduction,index', [('mean', 0), ('median', 5)])
@pytest.mark.parametrize('location', [-20.0, 20.0])
def test_mean_and_median_select_their_summary_entries(
    monkeypatch, head, reduction, index, location
):
    deciles = np.arange(1.0, 10.0)
    samples = np.stack([np.r_[location + i, deciles + i] for i in range(4)])
    result = evaluate_summaries(
        monkeypatch, head, reduction, samples, samples[:, index]
    )
    assert result['MSE'] == pytest.approx(0.0)
    assert result['xr'].sizes['samples'] == 10
    np.testing.assert_array_equal(
        result['xr']['q_sim'].values[:, 0], samples * 2 + 3
    )


@pytest.mark.parametrize('head', ['cmal', 'gmm', 'umal'])
@pytest.mark.parametrize('reduction', ['mean', 'median'])
def test_random_samples_keep_existing_reductions(monkeypatch, head, reduction):
    samples = np.array(
        [[1.0, 3.0, 20.0], [2.0, 4.0, 21.0], [3.0, 5.0, 22.0], [4.0, 6.0, 23.0]]
    )
    expected = getattr(np, reduction)(samples, axis=-1)
    result = evaluate_summaries(monkeypatch, head, reduction, samples, expected)
    assert result['MSE'] == pytest.approx(0.0, abs=1e-12)


def test_target_clipping_applies_to_selected_summary(monkeypatch):
    samples = np.tile(np.arange(-10.0, 0.0), (4, 1))
    result = evaluate_summaries(
        monkeypatch,
        'cmal_deterministic',
        'mean',
        samples,
        np.full(4, -1.5),
        clip=True,
    )
    assert result['MSE'] == pytest.approx(0.0)
    np.testing.assert_array_equal(
        result['xr']['q_sim'].values[:, 0], samples * 2 + 3
    )


def test_invalid_reduction_keeps_the_existing_error(monkeypatch):
    with pytest.raises(KeyError):
        evaluate_summaries(
            monkeypatch,
            'cmal_deterministic',
            'unknown',
            np.ones((4, 10)),
            np.ones(4),
        )
