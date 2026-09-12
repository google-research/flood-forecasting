# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and integration tests for MultiMet Dask massively parallel extraction."""

import os
import pathlib
import shutil
import tempfile
import distributed
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from multimet.config import PRODUCT_BANDS, Product
from multimet.dask_runner import (
    extract_product_dask,
    init_dask_client,
)
from multimet.geometry import load_basin_geometries
from multimet.runner import extract_multimet_serial
from multimet.zarr_writer import MultiMetZarrWriter

_TEST_DIR = pathlib.Path(__file__).parent
_TEST_BASINS_PATH = _TEST_DIR / "test_data/shapefiles/us/us_basin_shapes.geojson"


@pytest.fixture(scope="module")
def dask_client():
  """Creates a local Dask cluster with 2 workers for the test session."""
  cluster = distributed.LocalCluster(
      n_workers=2,
      threads_per_worker=1,
      processes=True,
      dashboard_address=None,
  )
  client = distributed.Client(cluster)
  yield client
  client.close()
  cluster.close()


def test_dask_concurrent_direct_writes_nowcast(dask_client):
  """Verifies that multiple Dask workers can concurrently write nowcast days to Zarr."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    writer = MultiMetZarrWriter(tmp_dir)
    basins = [f"basin_{i}" for i in range(25)]
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    store_path = writer.initialize_zarr_store(Product.CPC, basins, dates)

    def _write_day(day_idx: int) -> int:
      z = zarr.open_group(store_path, mode="r+")
      vals = np.full(len(basins), float(day_idx * 10 + 1), dtype=np.float32)
      z["cpc_precipitation"][:, day_idx] = vals
      return day_idx

    futures = [dask_client.submit(_write_day, i) for i in range(10)]
    results = [f.result() for f in futures]
    assert len(results) == 10

    writer.consolidate_metadata(Product.CPC)

    # Validate output
    ds = xr.open_zarr(store_path)
    for day_idx in range(10):
      expected = float(day_idx * 10 + 1)
      actual = ds["cpc_precipitation"].values[:, day_idx]
      assert np.all(actual == expected), f"Mismatch at day {day_idx}"


def test_dask_concurrent_direct_writes_forecast(dask_client):
  """Verifies that multiple Dask workers can concurrently write 3D forecast days."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    writer = MultiMetZarrWriter(tmp_dir)
    basins = [f"gauge_{i}" for i in range(15)]
    dates = pd.date_range("2021-06-01", periods=6, freq="D")
    store_path = writer.initialize_zarr_store(Product.HRES, basins, dates)

    def _write_forecast_day(day_idx: int) -> int:
      z = zarr.open_group(store_path, mode="r+")
      for band in PRODUCT_BANDS[Product.HRES]:
        vals = np.full((len(basins), 10), float(day_idx + 0.5), dtype=np.float32)
        z[band][:, day_idx, :] = vals
      return day_idx

    futures = [dask_client.submit(_write_forecast_day, i) for i in range(6)]
    results = [f.result() for f in futures]
    assert len(results) == 6

    writer.consolidate_metadata(Product.HRES)

    ds = xr.open_zarr(store_path)
    for band in PRODUCT_BANDS[Product.HRES]:
      for day_idx in range(6):
        expected = float(day_idx + 0.5)
        actual = ds[band].values[:, day_idx, :]
        assert np.all(actual == expected), f"Mismatch in {band} at day {day_idx}"


def test_dask_cpc_numerical_equivalence_with_serial(dask_client):
  """Verifies Dask parallel extraction produces identical numerical results to serial."""
  if not _TEST_BASINS_PATH.exists():
    pytest.skip(f"Test basins GeoJSON not found at {_TEST_BASINS_PATH}")

  with tempfile.TemporaryDirectory() as tmp_dir:
    dir_serial = os.path.join(tmp_dir, "serial")
    dir_dask = os.path.join(tmp_dir, "dask")

    start_date = "2020-01-01"
    end_date = "2020-01-02"

    # 1. Run serial extraction
    stores_serial = extract_multimet_serial(
        basins=_TEST_BASINS_PATH,
        output_dir=dir_serial,
        products=["CPC"],
        start_date=start_date,
        end_date=end_date,
        source="public",
    )
    store_serial = stores_serial["CPC"]

    # 2. Run Dask extraction
    store_dask = extract_product_dask(
        product="CPC",
        basins=_TEST_BASINS_PATH,
        output_dir=dir_dask,
        start_date=start_date,
        end_date=end_date,
        client=dask_client,
        source="public",
        use_bounding_box=True,
    )

    # 3. Compare datasets
    ds_s = xr.open_zarr(store_serial)
    ds_d = xr.open_zarr(store_dask)

    assert list(ds_s["basin"].values) == list(ds_d["basin"].values)
    assert len(ds_s["date"]) == len(ds_d["date"])

    val_s = ds_s["cpc_precipitation"].values
    val_d = ds_d["cpc_precipitation"].values

    np.testing.assert_allclose(val_s, val_d, rtol=1e-5, atol=1e-5, equal_nan=True)
    max_diff = float(np.nanmax(np.abs(val_s - val_d)))
    assert max_diff == 0.0, f"Expected 0.0 diff, got {max_diff}"


def test_dask_resumption_skips_completed_chunks(dask_client):
  """Verifies that resume=True detects existing chunks and only processes missing days."""
  if not _TEST_BASINS_PATH.exists():
    pytest.skip(f"Test basins GeoJSON not found at {_TEST_BASINS_PATH}")

  with tempfile.TemporaryDirectory() as tmp_dir:
    # 1. Run full 3-day extraction
    extract_product_dask(
        product="CPC",
        basins=_TEST_BASINS_PATH,
        output_dir=tmp_dir,
        start_date="2020-01-01",
        end_date="2020-01-03",
        client=dask_client,
        source="public",
        use_bounding_box=True,
        overwrite=True,
    )

    store_path = MultiMetZarrWriter(tmp_dir).get_store_path(Product.CPC)
    ds1 = xr.open_zarr(store_path)
    val1 = ds1["cpc_precipitation"].values.copy()
    assert val1.shape == (5, 3)
    assert np.all(~np.isnan(val1))

    # 2. Run again with resume=True over same date range
    # It should detect all 3 days are already written and skip re-computation
    extract_product_dask(
        product="CPC",
        basins=_TEST_BASINS_PATH,
        output_dir=tmp_dir,
        start_date="2020-01-01",
        end_date="2020-01-03",
        client=dask_client,
        source="public",
        use_bounding_box=True,
        resume=True,
    )

    ds2 = xr.open_zarr(store_path)
    val2 = ds2["cpc_precipitation"].values
    np.testing.assert_array_equal(val1, val2)


def test_dask_multi_file_basins(dask_client, tmp_path):
  """Verifies that Dask pipeline runs smoothly with multiple basin geometry files."""
  if not _TEST_BASINS_PATH.exists():
    pytest.skip(f"Test basins GeoJSON not found at {_TEST_BASINS_PATH}")

  full_gdf = gpd.read_file(str(_TEST_BASINS_PATH))
  p1 = tmp_path / "part1.geojson"
  p2 = tmp_path / "part2.geojson"
  full_gdf.iloc[:2].to_file(str(p1), driver="GeoJSON")
  full_gdf.iloc[2:].to_file(str(p2), driver="GeoJSON")

  out_dir = tmp_path / "zarr_out"
  out_dir.mkdir()

  store_path = extract_product_dask(
      product="CPC",
      basins=[str(p1), str(p2)],
      output_dir=str(out_dir),
      start_date="2020-01-01",
      end_date="2020-01-01",
      client=dask_client,
      source="public",
      use_bounding_box=True,
      overwrite=True,
  )

  ds = xr.open_zarr(store_path)
  assert len(ds["basin"]) == 5
  assert ds["cpc_precipitation"].shape == (5, 1)
  assert np.all(~np.isnan(ds["cpc_precipitation"].values))


def test_dask_store_exists_and_overwrite_fsspec(tmp_path):
  """Verifies that fsspec store detection and removal works for both local and cloud URIs."""
  import fsspec

  # 1. Local filesystem
  local_store = str(tmp_path / "local_test.zarr")
  fs_local, local_path = fsspec.core.url_to_fs(local_store)
  assert not fs_local.exists(local_path)
  fs_local.makedirs(local_path, exist_ok=True)
  fs_local.touch(f"{local_path}/zarr.json")
  assert fs_local.exists(f"{local_path}/zarr.json")
  fs_local.rm(local_path, recursive=True)
  assert not fs_local.exists(local_path)

  # 2. Remote / memory filesystem URI (representing gs://, s3://)
  remote_store = "memory://test_bucket/remote_test.zarr"
  fs_remote, remote_path = fsspec.core.url_to_fs(remote_store)
  assert not fs_remote.exists(remote_path)
  fs_remote.makedirs(remote_path, exist_ok=True)
  fs_remote.touch(f"{remote_path}/zarr.json")
  assert fs_remote.exists(f"{remote_path}/zarr.json")
  fs_remote.rm(remote_path, recursive=True)
  assert not fs_remote.exists(remote_path)


def test_gcp_project_autodetection_and_configuration(monkeypatch):
  """Tests GCP project autodetection precedence and fsspec/env configuration."""
  from multimet.gcp import auto_detect_gcp_project, configure_gcp_project
  import fsspec.config

  # 1. Explicit project takes precedence
  assert auto_detect_gcp_project("explicit-proj-123") == "explicit-proj-123"

  # 2. Environment variable fallback
  for k in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_QUOTA_PROJECT", "CLOUDSDK_CORE_PROJECT", "GCP_PROJECT", "GCLOUD_PROJECT"):
    monkeypatch.delenv(k, raising=False)

  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj-456")
  assert auto_detect_gcp_project() == "env-proj-456"

  # 3. Test configure_gcp_project updates environment and fsspec config
  proj = configure_gcp_project("configured-proj-789")
  assert proj == "configured-proj-789"
  assert os.environ.get("GOOGLE_CLOUD_PROJECT") == "configured-proj-789"
  assert os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT") == "configured-proj-789"
  assert fsspec.config.conf.get("gs", {}).get("project") == "configured-proj-789"
  assert fsspec.config.conf.get("gcs", {}).get("project") == "configured-proj-789"



