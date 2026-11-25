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

from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path
from types import MappingProxyType
from typing import get_type_hints
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import Dataset

from googlehydrology.datasetzoo.multimet import Multimet, SampleIndexer
from googlehydrology.evaluation import metrics as eval_metrics
from googlehydrology.evaluation.metrics import get_available_metrics
from googlehydrology.evaluation.tester import BaseTester, RegressionTester
from googlehydrology.evaluation.utils import (
    BasinBatchSampler,
    metrics_to_dataframe,
)
from googlehydrology.utils.config import Config


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


class _IdentityScaler:
    """No-op scaler for synthetic RegressionTester evaluation tests."""

    def unscale(self, dataset: xr.Dataset) -> xr.Dataset:
        """Return the dataset unchanged."""
        return dataset


class _SyntheticEvalDataset(Dataset):
    """Minimal Dataset compatible with RegressionTester and Multimet."""

    collate_fn = staticmethod(Multimet.collate_fn)

    def __init__(self, obs_by_basin: dict[str, np.ndarray]) -> None:
        self._basins = list(obs_by_basin)
        self.frequencies = ['1D']
        self.scaler = _IdentityScaler()
        self._samples: list[dict[str, object]] = []

        basin_ids: list[int] = []
        date_ids: list[int] = []
        start_date = np.datetime64('2020-01-01', 'D')
        for basin_idx, basin in enumerate(self._basins):
            obs = np.asarray(obs_by_basin[basin], dtype=np.float32)
            for step, obs_val in enumerate(obs):
                basin_ids.append(basin_idx)
                date_ids.append(step)
                pair_dates = np.array(
                    [
                        start_date + np.timedelta64(step, 'D'),
                        start_date + np.timedelta64(step + 1, 'D'),
                    ],
                    dtype='datetime64[D]',
                )
                target = np.full((2, 1), obs_val, dtype=np.float32)
                self._samples.append(
                    {
                        'basin_index': np.array(basin_idx, dtype=np.int64),
                        'date': pair_dates,
                        'x_d': {'precip': np.ones((2, 1), dtype=np.float32)},
                        'y': target,
                    }
                )

        self._sample_index = SampleIndexer(
            (
                ('basin', np.array(basin_ids, dtype=np.int64)),
                ('date', np.array(date_ids, dtype=np.int64)),
            )
        )

    def __len__(self) -> int:
        """Return total number of synthetic samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Return a single synthetic sample."""
        return self._samples[idx]


class _AffineEvalModel(torch.nn.Module):
    """Deterministic model returning `scale * y + offset` or all-NaN outputs."""

    def __init__(
        self,
        *,
        scale: float = 0.9,
        offset: float = 0.3,
        all_nan: bool = False,
    ) -> None:
        super().__init__()
        self._scale = scale
        self._offset = offset
        self._all_nan = all_nan

    def reset_state(self) -> None:
        """No-op state reset matching BaseModel interface."""

    def pre_model_hook(
        self, data: dict[str, object], *, is_train: bool = False
    ) -> dict[str, object]:
        """Pass batch data through unchanged."""
        del is_train
        return data

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return deterministic predictions with the same shape as `y`."""
        y = data['y']
        if self._all_nan:
            return {'y_hat': torch.full_like(y, float('nan'))}
        return {'y_hat': y * self._scale + self._offset}


class _RecordingLogger:
    """Minimal experiment logger capturing step-level metric dicts."""

    def __init__(self) -> None:
        self.steps: list[dict[str, object]] = []

    def log_step(self, **kwargs: object) -> None:
        """Record each logged key-value dictionary."""
        self.steps.append(dict(kwargs))


def _make_regression_tester(
    tmp_path: Path,
    *,
    cfg_metrics: object,
    obs_by_basin: dict[str, np.ndarray] | None = None,
) -> tuple[RegressionTester, dict[str, np.ndarray]]:
    """Construct a real RegressionTester backed by a synthetic Dataset."""
    if obs_by_basin is None:
        obs_by_basin = {
            'basin_A': np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32),
            'basin_B': np.array([2.0, 5.0, 3.0, 6.0], dtype=np.float32),
        }
    basin_file = tmp_path / 'test_basins.txt'
    basin_file.write_text('\n'.join(obs_by_basin) + '\n', encoding='utf-8')
    cfg = Config(
        {
            'device': 'cpu',
            'loss': 'MSE',
            'metrics': cfg_metrics,
            'target_variables': ['streamflow'],
            'test_basin_file': str(basin_file),
            'batch_size': 4,
            'predict_last_n': 2,
            'seq_length': 2,
            'inference_mode': False,
            'log_n_figures': 0,
            'verbose': 0,
            'lazy_load': False,
            'logging_level': 'WARNING',
            'clip_targets_to_zero': [],
            'mc_dropout': False,
            'validate_n_random_basins': len(obs_by_basin),
            'tester_skip_obs_all_nan': False,
        }
    )
    dataset = _SyntheticEvalDataset(obs_by_basin)
    with patch(
        'googlehydrology.evaluation.tester.get_dataset', return_value=dataset
    ):
        tester = RegressionTester(
            cfg, run_dir=tmp_path, period='test', init_model=False
        )
    return tester, obs_by_basin


def _expected_metric_values(
    obs_by_basin: dict[str, np.ndarray],
    *,
    scale: float = 0.9,
    offset: float = 0.3,
) -> dict[str, dict[str, float]]:
    """Compute exact expected NSE and KGE for the affine synthetic model."""
    expected: dict[str, dict[str, float]] = {}
    for basin, obs_vals in obs_by_basin.items():
        sim_vals = obs_vals * np.float32(scale) + np.float32(offset)
        obs_da = xr.DataArray(obs_vals)
        sim_da = xr.DataArray(sim_vals)
        expected[basin] = {
            'NSE': float(eval_metrics.nse(obs_da, sim_da)),
            'KGE': float(eval_metrics.kge(obs_da, sim_da)),
        }
    return expected


@pytest.mark.unit
def test_metrics_to_dataframe_canonicalizes_list_str_all_and_generator() -> (
    None
):
    """Canonicalize lists, bare strings, 'all', and single-use generators."""
    nse_a, kge_a = 0.82, 0.74
    nse_b, kge_b = 0.65, 0.58
    results = {
        'basin_A': {'1D': {'NSE': nse_a, 'KGE': kge_a}},
        'basin_B': {'1D': {'NSE': nse_b, 'KGE': kge_b}},
    }

    df_list = metrics_to_dataframe(results, ['nse', 'kge', 'NSE'], ['q'])
    assert list(df_list.columns) == ['NSE', 'KGE']
    assert df_list.loc['basin_A', 'NSE'] == nse_a
    assert df_list.loc['basin_A', 'KGE'] == kge_a
    assert df_list.loc['basin_B', 'NSE'] == nse_b
    assert df_list.loc['basin_B', 'KGE'] == kge_b

    df_str = metrics_to_dataframe(results, 'nse', ['q'])
    assert list(df_str.columns) == ['NSE']
    assert df_str.loc['basin_A', 'NSE'] == nse_a
    assert df_str.loc['basin_B', 'NSE'] == nse_b

    all_names = eval_metrics.get_available_metrics()
    for all_spec in ('all', ['all']):
        df_all = metrics_to_dataframe(results, all_spec, ['q'])
        assert list(df_all.columns) == all_names
        assert 'all' not in df_all.columns
        assert df_all.loc['basin_A', 'NSE'] == nse_a
        assert df_all.loc['basin_B', 'KGE'] == kge_b
        assert pd.isna(df_all.loc['basin_A', 'RMSE'])

    df_gen = metrics_to_dataframe(
        results, (name for name in ['nse', 'kge']), ['q']
    )
    assert list(df_gen.columns) == ['NSE', 'KGE']
    assert df_gen.loc['basin_A', 'NSE'] == nse_a
    assert df_gen.loc['basin_B', 'KGE'] == kge_b


@pytest.mark.unit
def test_metrics_to_dataframe_multi_target_and_multi_frequency() -> None:
    """Case-insensitive names resolve cleanly across targets and frequencies."""
    q1_nse_1d, q1_kge_1d = 0.81, 0.72
    q2_nse_1d, q2_kge_1d = 0.63, 0.54
    q1_nse_1h, q2_nse_1h = 0.77, 0.68

    multi_target = {
        'basin_A': {
            '1D': {
                'q1_NSE': q1_nse_1d,
                'q1_KGE': q1_kge_1d,
                'q2_NSE': q2_nse_1d,
                'q2_KGE': q2_kge_1d,
            }
        }
    }
    df_mt = metrics_to_dataframe(multi_target, ['nse', 'Kge'], ['q1', 'q2'])
    assert list(df_mt.columns) == ['q1_NSE', 'q1_KGE', 'q2_NSE', 'q2_KGE']
    assert df_mt.loc['basin_A', 'q1_NSE'] == q1_nse_1d
    assert df_mt.loc['basin_A', 'q2_KGE'] == q2_kge_1d

    multi_freq = {
        'basin_A': {
            '1D': {'NSE_1D': q1_nse_1d, 'KGE_1D': q1_kge_1d},
            '1h': {'NSE_1h': q1_nse_1h, 'KGE_1h': q2_kge_1d},
        }
    }
    df_mf = metrics_to_dataframe(multi_freq, ['nse', 'kge'], ['q1'])
    assert list(df_mf.columns) == ['NSE_1D', 'KGE_1D', 'NSE_1h', 'KGE_1h']
    assert df_mf.loc['basin_A', 'NSE_1D'] == q1_nse_1d
    assert df_mf.loc['basin_A', 'NSE_1h'] == q1_nse_1h

    combined = {
        'basin_A': {
            '1D': {'q1_NSE_1D': q1_nse_1d, 'q2_NSE_1D': q2_nse_1d},
            '1h': {'q1_NSE_1h': q1_nse_1h, 'q2_NSE_1h': q2_nse_1h},
        }
    }
    df_both = metrics_to_dataframe(combined, 'nse', ['q1', 'q2'])
    assert list(df_both.columns) == [
        'q1_NSE_1D',
        'q2_NSE_1D',
        'q1_NSE_1h',
        'q2_NSE_1h',
    ]
    assert df_both.loc['basin_A', 'q1_NSE_1D'] == q1_nse_1d
    assert df_both.loc['basin_A', 'q2_NSE_1h'] == q2_nse_1h


@pytest.mark.unit
@pytest.mark.parametrize(
    ('bad_metrics', 'expected_error'),
    [
        ({'streamflow': ['NSE']}, TypeError),
        (MappingProxyType({'streamflow': ['NSE']}), TypeError),
        (b'NSE', TypeError),
        (b'', TypeError),
        (bytearray(), TypeError),
        (['NSE', 123], TypeError),
        (None, TypeError),
        (42, TypeError),
        ('', RuntimeError),
        ('totally_bogus', RuntimeError),
        (['all', 'bogus'], RuntimeError),
    ],
)
def test_metrics_to_dataframe_rejects_invalid_inputs(
    bad_metrics: object,
    expected_error: type[Exception],
) -> None:
    """Invalid metric specs raise TypeError or RuntimeError without NaN cols."""
    results = {'basin_A': {'1D': {'NSE': 0.8}}}
    with pytest.raises(expected_error):
        metrics_to_dataframe(results, bad_metrics, ['q'])  # type: ignore[arg-type]


@pytest.mark.unit
def test_evaluate_defaults_overrides_strings_generators_and_mappings(
    tmp_path: Path,
) -> None:
    """Public BaseTester.evaluate handles None, [], lists, str, gen, Mapping."""
    tester, obs_by_basin = _make_regression_tester(
        tmp_path, cfg_metrics=['nse']
    )
    expected = _expected_metric_values(obs_by_basin)
    model = _AffineEvalModel()
    csv_path = tmp_path / 'test' / 'model_epoch001' / 'test_metrics.csv'

    # 1. metrics=None defaults to self.cfg.metrics=['nse'] -> canonical 'NSE'.
    logger_default = _RecordingLogger()
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=None,
        model=model,
        experiment_logger=logger_default,
    )
    metric_logs = [s for s in logger_default.steps if 'NSE' in s]
    assert len(metric_logs) == len(obs_by_basin)
    assert all(list(step) == ['NSE'] for step in metric_logs)
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.columns) == ['NSE']
    for basin in obs_by_basin:
        assert df.loc[basin, 'NSE'] == pytest.approx(
            expected[basin]['NSE'], rel=1e-6
        )

    # 2. metrics=[] skips metric computation and removes/omits the CSV.
    logger_empty = _RecordingLogger()
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=[],
        model=model,
        experiment_logger=logger_empty,
    )
    assert not csv_path.exists()
    assert all('NSE' not in step for step in logger_empty.steps)

    # 3. metrics=['kge'] overrides cfg.metrics=['nse'] cleanly.
    logger_kge = _RecordingLogger()
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=['kge'],
        model=model,
        experiment_logger=logger_kge,
    )
    kge_logs = [step for step in logger_kge.steps if 'KGE' in step]
    assert len(kge_logs) == len(obs_by_basin)
    assert all(list(step) == ['KGE'] for step in kge_logs)
    assert all('NSE' not in step for step in logger_kge.steps)
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.columns) == ['KGE']
    for basin, step in zip(obs_by_basin, kge_logs, strict=True):
        assert step['KGE'] == pytest.approx(expected[basin]['KGE'], rel=1e-6)
        assert df.loc[basin, 'KGE'] == pytest.approx(
            expected[basin]['KGE'], rel=1e-6
        )

    # 4. Bare string metrics='nse' works without character splitting.
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics='nse',
        model=model,
    )
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.columns) == ['NSE']
    for basin in obs_by_basin:
        assert df.loc[basin, 'NSE'] == pytest.approx(
            expected[basin]['NSE'], rel=1e-6
        )

    # 5. Single-use generator materializes once and works across all basins.
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=(name for name in ['nse', 'kge']),
        model=model,
    )
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.columns) == ['NSE', 'KGE']
    for basin in obs_by_basin:
        assert df.loc[basin, 'NSE'] == pytest.approx(
            expected[basin]['NSE'], rel=1e-6
        )
        assert df.loc[basin, 'KGE'] == pytest.approx(
            expected[basin]['KGE'], rel=1e-6
        )

    # 6. Empty generator writes no header-only CSV.
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=(name for name in ()),
        model=model,
    )
    assert not csv_path.exists()

    # 7. Non-dict Mapping works end-to-end through evaluate() and CSV writer.
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=MappingProxyType({'streamflow': 'nse'}),
        model=model,
    )
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.columns) == ['NSE']
    for basin in obs_by_basin:
        assert df.loc[basin, 'NSE'] == pytest.approx(
            expected[basin]['NSE'], rel=1e-6
        )


@pytest.mark.unit
def test_evaluate_handles_yaml_null_cfg_metrics(tmp_path: Path) -> None:
    """YAML `metrics: null` (`cfg.metrics` is None) skips metrics cleanly."""
    tester, _ = _make_regression_tester(tmp_path, cfg_metrics=None)
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=None,
        model=_AffineEvalModel(),
    )
    csv_path = tmp_path / 'test' / 'model_epoch001' / 'test_metrics.csv'
    assert not csv_path.exists()


@pytest.mark.unit
def test_evaluate_all_nan_simulation_logs_and_saves_canonical_keys(
    tmp_path: Path,
) -> None:
    """AllNaNError fallback must log and save canonical metric keys."""
    tester, obs_by_basin = _make_regression_tester(
        tmp_path, cfg_metrics=['nse', 'kge']
    )
    logger = _RecordingLogger()
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=['nse', 'kge'],
        model=_AffineEvalModel(all_nan=True),
        experiment_logger=logger,
    )
    metric_logs = [step for step in logger.steps if 'NSE' in step]
    assert len(metric_logs) == len(obs_by_basin)
    for step in metric_logs:
        assert list(step) == ['NSE', 'KGE']
        assert np.isnan(step['NSE'])
        assert np.isnan(step['KGE'])

    csv_path = tmp_path / 'test' / 'model_epoch001' / 'test_metrics.csv'
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.columns) == ['NSE', 'KGE']
    assert list(df.index) == list(obs_by_basin)
    assert df.isna().all().all()


@pytest.mark.unit
@pytest.mark.parametrize(
    ('bad_metrics', 'expected_error', 'expected_match'),
    [
        ('', RuntimeError, r'^Unknown metric $'),
        ('totally_bogus', RuntimeError, r'^Unknown metric totally_bogus$'),
        (['all', 'bogus'], RuntimeError, r'^Unknown metric bogus$'),
        (False, TypeError, r'metrics must be'),
        (0, TypeError, r'metrics must be'),
        (b'', TypeError, r'metrics must be'),
        (['NSE', 123], TypeError, r'Metric names must be strings'),
    ],
)
def test_evaluate_rejects_falsy_and_invalid_metrics_fast(
    tmp_path: Path,
    bad_metrics: object,
    expected_error: type[Exception],
    expected_match: str,
) -> None:
    """Invalid/falsy metrics fail before model checks or deleting prior CSVs."""
    tester, _ = _make_regression_tester(tmp_path, cfg_metrics=['nse'])
    sentinel_dir = tmp_path / 'test' / 'model_epoch001'
    sentinel_dir.mkdir(parents=True)
    sentinel_csv = sentinel_dir / 'test_metrics.csv'
    sentinel_csv.write_text('basin,NSE\nbasin_A,0.5\n', encoding='utf-8')

    with pytest.raises(expected_error, match=expected_match):
        tester.evaluate(
            epoch=1,
            save_results=False,
            metrics=bad_metrics,  # type: ignore[arg-type]
            model=None,
        )
    assert sentinel_csv.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ('bad_mapping', 'expected_error', 'expected_match'),
    [
        (
            {'streamflow': 'totally_bogus'},
            RuntimeError,
            r'^Unknown metric totally_bogus$',
        ),
        ({'streamflow': 123}, TypeError, r'Metric names must be strings'),
    ],
)
def test_evaluate_rejects_invalid_mapping_values_when_obs_all_nan(
    tmp_path: Path,
    bad_mapping: dict[str, object],
    expected_error: type[Exception],
    expected_match: str,
) -> None:
    """Invalid mapping metric values fail loudly even when obs is all-NaN."""
    obs_all_nan = {'basin_A': np.full(4, np.nan, dtype=np.float32)}
    tester, _ = _make_regression_tester(
        tmp_path, cfg_metrics=['nse'], obs_by_basin=obs_all_nan
    )
    with pytest.raises(expected_error, match=expected_match):
        tester.evaluate(
            epoch=1,
            save_results=False,
            metrics=bad_mapping,  # type: ignore[arg-type]
            model=_AffineEvalModel(),
        )
    csv_path = tmp_path / 'test' / 'model_epoch001' / 'test_metrics.csv'
    assert not csv_path.exists()


@pytest.mark.unit
def test_config_metrics_type_hints_are_valid_and_consistent() -> None:
    """Config.metrics getter, setter, and BaseTester.evaluate carry hints."""
    expected_cfg = str | list[str] | dict[str, str | list[str]]
    getter_hints = get_type_hints(Config.metrics.fget)
    setter_hints = get_type_hints(Config.metrics.fset)
    assert getter_hints['return'] == expected_cfg
    assert setter_hints['metrics'] == expected_cfg
    assert setter_hints['return'] is type(None)

    eval_hints = get_type_hints(BaseTester.evaluate)
    expected_eval = (
        str | Sequence[str] | Mapping[str, str | Sequence[str]] | None
    )
    assert eval_hints['metrics'] == expected_eval


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
