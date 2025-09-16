from math import ceil
import pytest
from unittest.mock import MagicMock

from neuralhydrology.evaluation.utils import BasinBatchSampler


@pytest.fixture
def fixure():
    sample_index = {  # sample index -> basin metadata
        # basin 101: 7 samples, 3 batches
        0: {"basin": 101, "date": 1},
        1: {"basin": 101, "date": 2},
        2: {"basin": 101, "date": 3},
        3: {"basin": 101, "date": 4},
        4: {"basin": 101, "date": 5},
        5: {"basin": 101, "date": 6},
        6: {"basin": 101, "date": 7},
        # basin 102: 6 samples, 2 batches
        7: {"basin": 102, "date": 1},
        8: {"basin": 102, "date": 2},
        9: {"basin": 102, "date": 3},
        10: {"basin": 102, "date": 4},
        11: {"basin": 102, "date": 5},
        12: {"basin": 102, "date": 6},
        # basin 103: 2 samples, 1 batch
        13: {"basin": 103, "date": 1},
        14: {"basin": 103, "date": 2},
    }

    expected_groups = {  # basin id -> expected sample indexes
        101: [0, 1, 2, 3, 4, 5, 6],
        102: [7, 8, 9, 10, 11, 12],
        103: [13, 14],
    }

    return {
        "sample_index": sample_index,
        "expected_groups": expected_groups,
        "total_basins": 3,
        "total_samples": 15,
    }


def test_init_groups_basins(fixure):
    """Test grouping all sample indices by their basin id."""
    sampler = BasinBatchSampler(fixure["sample_index"], batch_size=3)

    assert sampler._basin_indices == fixure["expected_groups"]


def test_num_batches(fixure):
    """Test _num_batches is total num batches for an epoc (accounting for partial batch)."""
    sampler = BasinBatchSampler(fixure["sample_index"], batch_size=3)

    expected_num_batches = ceil(7 / 3) + ceil(6 / 3) + ceil(2 / 3)

    assert sampler._num_batches == expected_num_batches


def test_len_returns_num_batches(fixure):
    """Test __len__ returns total num batches for an epoc (accounting for partial batch)."""
    sampler = BasinBatchSampler(fixure["sample_index"], batch_size=3)

    expected_num_batches = ceil(7 / 3) + ceil(6 / 3) + ceil(2 / 3)

    assert len(sampler) == expected_num_batches


def test_iter_yields_all_samples_once(fixure):
    """Test iterating results in all sample indices once per epoc."""
    sampler = BasinBatchSampler(fixure["sample_index"], batch_size=3)

    indices = {i for batch in sampler for i in batch}

    assert indices == set(fixure["sample_index"].keys())


def test_one_basin_per_batch(fixure):
    """Test every batch contains samples belonging to only one basin."""
    sampler = BasinBatchSampler(fixure["sample_index"], batch_size=3)

    basinss = [{fixure["sample_index"][i]["basin"] for i in batch} for batch in sampler]
    assert all(len(basins) == 1 for basins in basinss)


def test_sampler_with_single_basin(fixure):
    """Test that the sampler works with one basin."""
    sample_index = {
        k: v for k, v in fixure["sample_index"].items() if v["basin"] == 101
    }
    sampler = BasinBatchSampler(sample_index, batch_size=3)

    indices = {idx for batch in sampler for idx in batch}

    assert len(sampler) == ceil(7 / 3)
    assert indices == set(sample_index.keys())


def test_sampler_with_batch_size_larger_than_samples():
    """Test behavior when a basin has fewer samples than the batch size."""
    sample_index = {0: {"basin": 201}, 1: {"basin": 201}}
    sampler = BasinBatchSampler(sample_index, batch_size=5)
    batches = list(sampler)

    assert len(sampler) == 1
    assert len(batches) == 1
    assert batches[0] == [0, 1]
