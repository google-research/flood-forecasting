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

from math import ceil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from googlehydrology.datasetzoo.multimet import SampleIndexer
from googlehydrology.evaluation.metrics import get_available_metrics
from googlehydrology.evaluation.tester import RegressionTester
from googlehydrology.evaluation.utils import BasinBatchSampler


@pytest.fixture
def fixture():
    sample_index = SampleIndexer(  # sample index -> basin metadata
        (
            (
                'basin',
                np.array(
                    [
                        # basin 101: 7 samples, 3 batches
                        *[101, 101, 101, 101, 101, 101, 101],
                        # basin 102: 6 samples, 2 batches
                        *[102, 102, 102, 102, 102, 102],
                        # basin 103: 2 samples, 1 batch
                        *[103, 103],
                    ]
                ),
            ),
            (
                'date',
                np.array(
                    [
                        # basin 101: 7 samples, 3 batches
                        *[1, 2, 3, 4, 5, 6, 7],
                        # basin 102: 6 samples, 2 batches
                        *[1, 2, 3, 4, 5, 6],
                        # basin 103: 2 samples, 1 batch
                        *[1, 2],
                    ]
                ),
            ),
        )
    )

    expected_groups = {  # basin id -> expected sample indexes
        101: [0, 1, 2, 3, 4, 5, 6],
        102: [7, 8, 9, 10, 11, 12],
        103: [13, 14],
    }

    return {
        'sample_index': sample_index,
        'expected_groups': expected_groups,
        'total_basins': 3,
        'total_samples': 15,
    }


def test_init_groups_basins(fixture):
    """Test grouping all sample indices by their basin id."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=np.array([])
    )

    indices = np.concatenate(list(sampler))

    groups = fixture['expected_groups']
    expected_indices = groups[101] + groups[102] + groups[103]
    np.testing.assert_array_equal(indices, expected_indices)


def test_init_groups_basins_subset(fixture):
    """Test grouping all sample indices by their basin id."""
    sampler = BasinBatchSampler(
        fixture['sample_index'],
        batch_size=3,
        basins_indexes=np.array([102, 103]),
    )

    indices = np.concatenate(list(sampler))

    groups = fixture['expected_groups']
    expected_indices = groups[102] + groups[103]
    np.testing.assert_array_equal(indices, expected_indices)


def test_num_batches(fixture):
    """Test _num_batches is total num batches for an epoc (accounting for partial batch)."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=np.array([])
    )

    expected_num_batches = ceil(7 / 3) + ceil(6 / 3) + ceil(2 / 3)

    assert sampler._num_batches == expected_num_batches


def test_len_returns_num_batches(fixture):
    """Test __len__ returns total num batches for an epoc (accounting for partial batch)."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=np.array([])
    )

    expected_num_batches = ceil(7 / 3) + ceil(6 / 3) + ceil(2 / 3)

    assert len(sampler) == expected_num_batches


def test_iter_yields_all_samples_once(fixture):
    """Test iterating results in all sample indices once per epoc."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=np.array([])
    )

    indices = {i for batch in sampler for i in batch}

    assert indices == set(fixture['sample_index'].keys())


def test_one_basin_per_batch(fixture):
    """Test every batch contains samples belonging to only one basin."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=np.array([])
    )

    basinss = [
        {fixture['sample_index'][i]['basin'] for i in batch}
        for batch in sampler
    ]
    assert all(len(basins) == 1 for basins in basinss)


def test_sampler_with_single_basin(fixture):
    """Test that the sampler works with one basin."""
    sample_index = SampleIndexer(
        (
            ('basin', np.array([101, 101, 101, 101, 101, 101, 101])),
            ('date', np.array([1, 2, 3, 4, 5, 6, 7])),
        ),
    )
    sampler = BasinBatchSampler(
        sample_index, batch_size=3, basins_indexes=np.array([])
    )

    indices = {idx for batch in sampler for idx in batch}

    assert len(sampler) == ceil(7 / 3)
    assert indices == set(sample_index.keys())


def test_sampler_with_batch_size_larger_than_samples():
    """Test behavior when a basin has fewer samples than the batch size."""
    sample_index = SampleIndexer((('basin', np.array([201, 201])),))
    sampler = BasinBatchSampler(
        sample_index, batch_size=5, basins_indexes=np.array([])
    )
    batches = list(sampler)

    assert len(sampler) == 1
    assert len(batches) == 1
    assert batches[0] == (0, 1)


def test_evaluate_synchronizes_configured_cuda_device():
    """_evaluate must synchronize `self.device`, not only the default GPU."""
    from unittest.mock import MagicMock, patch
    import torch
    from googlehydrology.evaluation.tester import (
        RegressionTester,
        _values_to_cpu,
    )

    tensors = {'1D': torch.tensor([[1.5, 2.5]])}
    cpu_tensors = _values_to_cpu(tensors)
    assert cpu_tensors['1D'].device.type == 'cpu'
    torch.testing.assert_close(cpu_tensors['1D'], tensors['1D'])

    tester = object.__new__(RegressionTester)
    tester.device = torch.device('cuda:2')
    tester.cfg = MagicMock(
        predict_last_n=2,
        hot_start_path=None,
        save_state=False,
    )
    tester.period = 'test'
    tester.loss_obj = lambda preds, data: (
        torch.tensor(0.25),
        {'loss': torch.tensor(0.25)},
    )

    class _FakeDataset:
        _basins = ['basin_A']

    batch = {
        'basin_index': torch.tensor([0, 0]),
        'date': np.array(
            [
                ['2020-01-01', '2020-01-02'],
                ['2020-01-02', '2020-01-03'],
            ],
            dtype='datetime64[D]',
        ),
        'x_d': {'f1': torch.ones((2, 2, 1))},
        'y': torch.full((2, 2, 1), 3.0),
    }

    class _FakeLoader:
        dataset = _FakeDataset()

        def __iter__(self):
            return iter([batch])

    model = MagicMock()
    model.pre_model_hook.side_effect = lambda d, is_train: d
    model.return_value = {'y_hat': torch.full((2, 2, 1), 4.0)}

    with (
        patch.object(torch.Tensor, 'to', lambda self, *a, **kw: self),
        patch('googlehydrology.evaluation.tester.autocast'),
        patch('torch.cuda.synchronize') as mock_sync,
    ):
        results = list(
            tester._evaluate(
                model=model,
                loader=_FakeLoader(),
                basins=['basin_A'],
                frequencies=['1D'],
            )
        )

    mock_sync.assert_called_once_with(torch.device('cuda:2'))
    assert len(results) == 1
    assert results[0]['basin'] == 'basin_A'
    torch.testing.assert_close(
        results[0]['preds']['1D'], torch.full((2, 2, 1), 4.0)
    )
    torch.testing.assert_close(
        results[0]['obs']['1D'], torch.full((2, 2, 1), 3.0)
    )


def test_save_incremental_results_with_per_target_metrics(
    tmp_path: Path,
) -> None:
    """`metrics` may map each target to its own list of metrics."""
    tester = object.__new__(RegressionTester)
    tester.period = 'test'
    tester.cfg = MagicMock(
        metrics={'q1': ['NSE', 'KGE'], 'q2': ['NSE']},
        target_variables=['q1', 'q2'],
        inference_mode=False,
    )
    tester._parent_directory_for_results = lambda _: tmp_path

    expected_a = {'q1_NSE': 0.5, 'q1_KGE': 0.4, 'q2_NSE': 0.3}
    expected_b = {'q1_NSE': 0.7, 'q1_KGE': 0.6, 'q2_NSE': 0.8}
    tester._save_incremental_results(
        'basin_A',
        results={'1D': expected_a},
        states={},
        save_results=False,
        epoch=1,
    )
    tester._save_incremental_results(
        'basin_B',
        results={'1D': expected_b},
        states={},
        save_results=False,
        epoch=1,
    )

    df = pd.read_csv(tmp_path / 'test_metrics.csv', index_col=0)
    assert list(df.columns) == ['q1_NSE', 'q1_KGE', 'q2_NSE', 'q2_KGE']
    assert list(df.index) == ['basin_A', 'basin_B']
    for col, val in expected_a.items():
        assert df.loc['basin_A', col] == val
    for col, val in expected_b.items():
        assert df.loc['basin_B', col] == val
    assert pd.isna(df.loc['basin_A', 'q2_KGE'])
    assert pd.isna(df.loc['basin_B', 'q2_KGE'])


@pytest.mark.parametrize('metric_spec', ['all', ['all']])
def test_save_incremental_results_with_dict_all_metrics(
    tmp_path: Path, metric_spec: str | list[str]
) -> None:
    """Dict values of `'all'` or `['all']` expand without splitting chars."""
    tester = object.__new__(RegressionTester)
    tester.period = 'test'
    tester.cfg = MagicMock(
        metrics={'streamflow': metric_spec},
        target_variables=['streamflow'],
        inference_mode=False,
    )
    tester._parent_directory_for_results = lambda _: tmp_path

    expected_nse = 0.65
    tester._save_incremental_results(
        'basin_A',
        results={'1D': {'NSE': expected_nse}},
        states={},
        save_results=False,
        epoch=1,
    )

    df = pd.read_csv(tmp_path / 'test_metrics.csv', index_col=0)
    assert list(df.columns) == get_available_metrics()
    assert df.loc['basin_A', 'NSE'] == expected_nse


def test_save_incremental_results_skips_empty_per_target_metrics(
    tmp_path: Path,
) -> None:
    """A metrics dict with only empty lists writes no metrics CSV."""
    tester = object.__new__(RegressionTester)
    tester.period = 'test'
    tester.cfg = MagicMock(
        metrics={'q1': [], 'q2': []},
        target_variables=['q1', 'q2'],
        inference_mode=False,
    )
    tester._parent_directory_for_results = lambda _: tmp_path

    tester._save_incremental_results(
        'basin_A',
        results={'1D': {}},
        states={},
        save_results=False,
        epoch=1,
    )

    assert not (tmp_path / 'test_metrics.csv').exists()


def test_save_incremental_results_rejects_non_iterable_metric_value(
    tmp_path: Path,
) -> None:
    """Non-iterable dict values in `metrics` fail loudly with TypeError."""
    tester = object.__new__(RegressionTester)
    tester.period = 'test'
    tester.cfg = MagicMock(
        metrics={'q1': None},
        target_variables=['q1'],
        inference_mode=False,
    )
    tester._parent_directory_for_results = lambda _: tmp_path

    with pytest.raises(TypeError):
        tester._save_incremental_results(
            'basin_A',
            results={'1D': {'NSE': 0.5}},
            states={},
            save_results=False,
            epoch=1,
        )
