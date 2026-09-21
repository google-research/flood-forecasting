# Copyright 2026 Google LLC
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

"""End-to-end integration tests for the gridded archive builders."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet.build_cpc_archive import CPC_VARIABLE, build_cpc_archive
from multimet.build_hres_archive import (
    HRES_VARIABLES,
    NUM_LEAD_DAYS,
    build_hres_archive,
)
from multimet.test.conftest import (
    FAKE_HRES_LATS,
    FAKE_HRES_LONS,
    FakeHRESSource,
)

pytestmark = pytest.mark.integration


def open_store(path: str) -> xr.Dataset:
  """Loads a local Zarr store fully into memory so it can be closed."""
  with xr.open_zarr(path, consolidated=False) as store:
    return store.load()


class TestCPCArchiveEndToEnd:
  """Full ``build_cpc_archive`` runs against a local store."""

  def test_single_year_build(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-05")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    assert len(store["time"]) == 5
    assert store[CPC_VARIABLE].shape == (5, 360, 720)
    assert list(pd.to_datetime(store["time"].values)) == list(
        pd.date_range("2020-01-01", "2020-01-05")
    )

  def test_multi_year_build_is_chronological(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-12-29", "2020-12-31")
    write_psl_year(2021, "2021-01-01", "2021-01-03")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2021,
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert len(times) == 6
    assert times.is_monotonic_increasing
    assert times[0] == pd.Timestamp("2020-12-29")
    assert times[-1] == pd.Timestamp("2021-01-03")

  def test_values_land_on_the_right_dates(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-04")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    for i, day in enumerate(pd.date_range("2020-01-01", "2020-01-04")):
      slice_values = store[CPC_VARIABLE].sel(time=day).values
      assert float(np.nanmin(slice_values)) == pytest.approx(1.0 + i)
      assert float(np.nanmax(slice_values)) == pytest.approx(1.0 + i)

  def test_date_filters_bound_the_build(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-10")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        start_date="2020-01-03",
        end_date="2020-01-06",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert times[0] == pd.Timestamp("2020-01-03")
    assert times[-1] == pd.Timestamp("2020-01-06")
    assert len(times) == 4

  def test_resume_appends_only_new_days(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-10")
    target = str(tmp_path / "cpc.zarr")
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-04",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-08",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert list(times) == list(pd.date_range("2020-01-01", "2020-01-08"))
    assert len(set(times)) == len(times)

  def test_resume_does_not_disturb_existing_values(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-06")
    target = str(tmp_path / "cpc.zarr")
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-03",
        cache_dir=str(psl_cache),
        num_workers=1,
    )
    before = open_store(target)[CPC_VARIABLE].isel(time=slice(0, 3)).values

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-06",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    after = open_store(target)[CPC_VARIABLE].isel(time=slice(0, 3)).values
    np.testing.assert_array_equal(before, after)

  def test_overwrite_rebuilds_from_scratch(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    target = str(tmp_path / "cpc.zarr")
    write_psl_year(2020, "2020-01-01", "2020-01-06")
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    write_psl_year(2020, "2020-01-01", "2020-01-02", fill_value=50.0)
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        cache_dir=str(psl_cache),
        num_workers=1,
        overwrite=True,
    )

    store = open_store(target)
    assert len(store["time"]) == 2
    assert float(store[CPC_VARIABLE].isel(time=0).min()) == pytest.approx(50.0)

  def test_missing_cells_survive_as_nan(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(
        2020, "2020-01-01", "2020-01-02", missing_cells=[(0, 0, 0), (1, 5, 5)]
    )
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    assert int(np.isnan(store[CPC_VARIABLE].values).sum()) == 2

  def test_cleanup_cache_removes_downloads(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    netcdf = write_psl_year(2020, "2020-01-01", "2020-01-02")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        cache_dir=str(psl_cache),
        cleanup_cache=True,
        num_workers=1,
    )

    assert not netcdf.exists()
    assert len(open_store(target)["time"]) == 2

  @pytest.mark.slow
  def test_parallel_workers_produce_the_same_store(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-03")
    write_psl_year(2021, "2021-01-01", "2021-01-03")
    serial_target = str(tmp_path / "serial.zarr")
    parallel_target = str(tmp_path / "parallel.zarr")

    build_cpc_archive(
        target_zarr=serial_target,
        start_year=2020,
        end_year=2021,
        cache_dir=str(psl_cache),
        num_workers=1,
    )
    build_cpc_archive(
        target_zarr=parallel_target,
        start_year=2020,
        end_year=2021,
        cache_dir=str(psl_cache),
        num_workers=2,
    )

    serial = open_store(serial_target)
    parallel = open_store(parallel_target)
    assert len(parallel["time"]) == 6
    xr.testing.assert_identical(serial, parallel)


class TestHRESArchiveEndToEnd:
  """Full ``build_hres_archive`` runs against a local store."""

  def test_build_writes_every_requested_date(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "hres.zarr")

    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-05",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    store = open_store(target)
    assert list(pd.to_datetime(store["time"].values)) == list(
        pd.date_range("2020-01-01", "2020-01-05")
    )

  def test_store_has_the_canonical_schema(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "hres.zarr")

    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-03",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    store = open_store(target)
    assert set(store.data_vars) == set(HRES_VARIABLES)
    for var in HRES_VARIABLES:
      assert store[var].dims == ("time", "lead_time", "latitude", "longitude")
      assert store[var].shape == (
          3,
          NUM_LEAD_DAYS,
          len(FAKE_HRES_LATS),
          len(FAKE_HRES_LONS),
      )
    np.testing.assert_array_equal(
        store["lead_time"].values, np.arange(1, NUM_LEAD_DAYS + 1)
    )

  def test_a_trailing_partial_batch_is_flushed(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "hres.zarr")

    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-05",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    assert len(open_store(target)["time"]) == 5

  def test_values_land_on_the_right_dates(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    source = fake_hres_source()
    target = str(tmp_path / "hres.zarr")

    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-04",
        target_zarr=target,
        batch_size=3,
        num_workers=1,
    )

    store = open_store(target)
    for day in pd.date_range("2020-01-01", "2020-01-04"):
      for var in HRES_VARIABLES:
        actual = float(store[var].sel(time=day).min())
        assert actual == pytest.approx(source.value_for(day, var))

  def test_batch_size_does_not_change_the_result(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    one_shot = str(tmp_path / "one_shot.zarr")
    batched = str(tmp_path / "batched.zarr")

    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-07",
        target_zarr=one_shot,
        batch_size=100,
        num_workers=1,
    )
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-07",
        target_zarr=batched,
        batch_size=2,
        num_workers=1,
    )

    xr.testing.assert_identical(open_store(one_shot), open_store(batched))

  def test_interior_missing_upstream_dates_become_nan_and_heal_on_resume(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    """Interior gap becomes NaN and is recorded in missing_dates, then healed."""
    # First build: 2020-01-02 is missing (interior gap), and 2020-01-04..05 are
    # trailing future dates (not available yet).
    fake_hres_source(available=["2020-01-01", "2020-01-03"])
    target = str(tmp_path / "hres.zarr")

    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-05",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    store = open_store(target)
    # Trailing missing dates (2020-01-04..05) were NOT written!
    assert list(pd.to_datetime(store["time"].values)) == list(
        pd.date_range("2020-01-01", "2020-01-03")
    )
    gap = store["temperature_2m"].sel(time="2020-01-02").values
    assert bool(np.isnan(gap).all())
    assert store.attrs.get("missing_dates") == ["2020-01-02"]

    # Second build (resume): now 2020-01-02 and 2020-01-04 are available!
    # Resume heals 2020-01-02 in-place AND appends 2020-01-04!
    source = fake_hres_source(
        available=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    )
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-05",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    healed = open_store(target)
    assert list(pd.to_datetime(healed["time"].values)) == list(
        pd.date_range("2020-01-01", "2020-01-04")
    )
    assert not bool(
        np.isnan(healed["temperature_2m"].sel(time="2020-01-02")).any()
    )
    assert float(
        healed["temperature_2m"].sel(time="2020-01-02").min()
    ) == pytest.approx(
        source.value_for(pd.Timestamp("2020-01-02"), "temperature_2m")
    )
    assert healed.attrs.get("missing_dates") == []

  def test_resume_appends_only_new_dates(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "hres.zarr")
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-03",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    source = fake_hres_source()
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-06",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    store = open_store(target)
    assert list(pd.to_datetime(store["time"].values)) == list(
        pd.date_range("2020-01-01", "2020-01-06")
    )
    assert source.requested == ["2020-01-04", "2020-01-05", "2020-01-06"]

  def test_resume_is_a_no_op_when_already_complete(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "hres.zarr")
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-03",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    source = fake_hres_source()
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-03",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    assert source.requested == []
    assert len(open_store(target)["time"]) == 3

  def test_overwrite_rebuilds_from_scratch(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "hres.zarr")
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-05",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    source = fake_hres_source(offset=1000.0)
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-02",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
        overwrite=True,
    )

    store = open_store(target)
    assert len(store["time"]) == 2
    expected = source.value_for(pd.Timestamp("2020-01-01"), "temperature_2m")
    assert float(store["temperature_2m"].isel(time=0).min()) == pytest.approx(
        expected
    )

  def test_in_place_rewrites_existing_dates(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "hres.zarr")
    build_hres_archive(
        start_date="2020-01-01",
        end_date="2020-01-05",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
    )

    corrected = fake_hres_source(offset=500.0)
    build_hres_archive(
        start_date="2020-01-02",
        end_date="2020-01-03",
        target_zarr=target,
        batch_size=2,
        num_workers=1,
        in_place=True,
    )

    store = open_store(target)
    assert len(store["time"]) == 5
    for day in pd.date_range("2020-01-02", "2020-01-03"):
      expected = corrected.value_for(day, "temperature_2m")
      assert float(
          store["temperature_2m"].sel(time=day).min()
      ) == pytest.approx(expected)
    original = FakeHRESSource()
    for day in (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05")):
      expected = original.value_for(day, "temperature_2m")
      assert float(
          store["temperature_2m"].sel(time=day).min()
      ) == pytest.approx(expected)

  def test_in_place_on_a_missing_store_raises(
      self, fake_hres_source: Callable[..., FakeHRESSource], tmp_path: Path
  ) -> None:
    fake_hres_source()
    target = str(tmp_path / "does_not_exist.zarr")

    with pytest.raises(ValueError, match="in_place"):
      build_hres_archive(
          start_date="2020-01-01",
          end_date="2020-01-02",
          target_zarr=target,
          num_workers=1,
          in_place=True,
      )
