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

"""Unit tests for real-time meteorological forcing fetcher (Cold-Start & Hot-Start)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet.build_hres_archive import decode_grib2_message
from multimet.config import PRODUCT_BANDS, Product
from multimet.geometry import load_basin_geometries
from multimet.hres import (
    OPEN_DATA_GRID_SHAPE,
    OPEN_DATA_LEAD_STEPS,
    HRESExtractor,
    find_latest_hres_open_data_date,
)
from multimet.realtime import (
    DEFAULT_COLDSTART_LOOKBACK_DAYS,
    RealtimeForcingFetcher,
    build_arg_parser,
    fetch_realtime_multimet,
    inspect_store_last_valid_date,
    main as realtime_main,
    read_hot_start_state_date,
)
from multimet.zarr_writer import MultiMetZarrWriter

pytestmark = pytest.mark.unit


@pytest.fixture
def basins_gdf() -> gpd.GeoDataFrame:
  path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  return load_basin_geometries(path)


class FakeECMWFOpenDataFS:
  """In-memory mock of gs://ecmwf-open-data with synthetic GRIB2 payloads."""

  def __init__(self, available_dates: Sequence[str]):
    self.available_dates = {
        pd.to_datetime(d).strftime("%Y%m%d") for d in available_dates
    }
    self.requested_index_paths: List[str] = []
    self.requested_grib_ranges: List[Tuple[str, int, int]] = []
    self._params = ("2t", "sp", "tp", "ssr", "str")

  def _parse_date_and_step(self, path: str) -> Tuple[str, int]:
    fname = path.rsplit("/", 1)[-1]
    date_str = fname[:8]
    step_part = fname.split("-")[1]  # e.g. "24h"
    step = int(step_part.rstrip("h"))
    return date_str, step

  def exists(self, path: str) -> bool:
    date_str, _ = self._parse_date_and_step(path)
    return date_str in self.available_dates

  def _make_index_bytes(self, step: int) -> bytes:
    lines = []
    for idx, param in enumerate(self._params):
      lines.append(
          json.dumps({
              "param": param,
              "levtype": "sfc",
              "step": str(step),
              "_offset": idx * 100,
              "_length": 100,
          })
      )
    return ("\n".join(lines) + "\n").encode("utf-8")

  def cat(
      self, paths: Sequence[str], on_error: str = "raise"
  ) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in paths:
      self.requested_index_paths.append(p)
      date_str, step = self._parse_date_and_step(p)
      if date_str not in self.available_dates:
        if on_error == "return":
          out[p] = FileNotFoundError(p)
          continue
        raise FileNotFoundError(p)
      out[p] = self._make_index_bytes(step)
    return out

  def cat_ranges(
      self, paths: Sequence[str], starts: Sequence[int], ends: Sequence[int]
  ) -> List[bytes]:
    blobs: List[bytes] = []
    for p, s, e in zip(paths, starts, ends):
      self.requested_grib_ranges.append((p, s, e))
      _, step = self._parse_date_and_step(p)
      param_idx = s // 100
      param = self._params[param_idx]
      payload = json.dumps({"param": param, "step": step}).encode("utf-8")
      blobs.append(payload)
    return blobs


def _fake_decode_grib2(
    raw_bytes: bytes,
    expected_shape: Tuple[int, int],
    *,
    param: str = "",
    context: str = "",
) -> np.ndarray:
  """Synthetic decoder returning deterministic physical values in raw WMO units.

  Also sets southern-hemisphere rows (raw row > 360, i.e. lat < 0 before flip)
  to a distinct sentinel so tests can verify the latitude flip ``arr[::-1, :]``
  aligns northern-hemisphere US basins (lat ~ 40N) with raw rows < 360.
  """
  del param, context
  meta = json.loads(raw_bytes.decode("utf-8"))
  p = meta["param"]
  step = int(meta["step"])
  lead_day = step // 24  # 1..10

  if p == "2t":
    # 293.15 K -> 20.0 degC in Northern Hemisphere; 250.0 K in Southern Hemisphere
    val_nh = 293.15
    val_sh = 250.0
  elif p == "sp":
    # 101325 Pa -> 101.325 kPa
    val_nh = 101325.0
    val_sh = 50000.0
  elif p == "tp":
    # Cumulative 0.005 m per lead day -> 5.0 mm/day after deaccumulation
    val_nh = 0.005 * lead_day
    val_sh = 0.0
  elif p == "ssr":
    # Cumulative 200 W/m^2 * 86400 s per lead day -> 200.0 W/m^2
    val_nh = 200.0 * 86400.0 * lead_day
    val_sh = 0.0
  elif p == "str":
    # Cumulative -50 W/m^2 * 86400 s per lead day -> -50.0 W/m^2
    val_nh = -50.0 * 86400.0 * lead_day
    val_sh = 0.0
  else:
    raise ValueError(f"Unexpected param {p}")

  arr = np.full(expected_shape, val_nh, dtype=np.float32)
  # In raw ECMWF GRIB2, row 0 is +90N and row 720 is -90S.
  # Rows 361..720 are Southern Hemisphere (< 0 lat).
  arr[361:, :] = val_sh
  return arr


def test_find_latest_hres_open_data_date():
  """Verifies walking backwards from reference_date to find newest published 00z run."""
  fs = FakeECMWFOpenDataFS(available_dates=["2026-09-25", "2026-09-26"])
  latest = find_latest_hres_open_data_date(
      reference_date="2026-09-28",
      max_lookback_days=5,
      fs=fs,
  )
  assert latest == pd.Timestamp("2026-09-26")

  with pytest.raises(FileNotFoundError, match="No published ECMWF Open Data"):
    find_latest_hres_open_data_date(
        reference_date="2026-10-15",
        max_lookback_days=3,
        fs=fs,
    )


def test_hres_open_data_extraction_and_spinup_1d_optimization(
    basins_gdf, monkeypatch
):
  """Verifies unit conversion, lat-flip, and 10x spinup_only_before optimization."""
  monkeypatch.setattr(
      "multimet.hres.decode_grib2_message", _fake_decode_grib2
  )
  fs = FakeECMWFOpenDataFS(
      available_dates=["2026-09-25", "2026-09-26", "2026-09-27"]
  )
  ext = HRESExtractor(source="open_data", fs=fs)

  # Extract 3 days where 2026-09-25 and 2026-09-26 are spin-up (lead 1D only)
  # and 2026-09-27 is the forecast issue date (all 10 leads).
  ds = ext.extract_for_basins(
      basins_gdf,
      start_date="2026-09-25",
      end_date="2026-09-28",  # 2026-09-28 is unpublished -> should be NaN
      spinup_only_before="2026-09-27",
  )

  assert ds["hres_temperature_2m"].shape == (len(basins_gdf), 4, 10)

  # 1. Spin-up days (indices 0 and 1: 2026-09-25, 2026-09-26):
  #    Only lead_time=1D (index 0) was downloaded; leads 2D..10D are strictly NaN.
  for d_idx in (0, 1):
    assert np.allclose(
        ds["hres_temperature_2m"].values[:, d_idx, 0], 20.0, atol=1e-2
    )
    assert np.allclose(
        ds["hres_surface_pressure"].values[:, d_idx, 0], 101.325, atol=1e-2
    )
    assert np.allclose(
        ds["hres_total_precipitation"].values[:, d_idx, 0], 5.0, atol=1e-2
    )
    assert np.allclose(
        ds["hres_surface_net_solar_radiation"].values[:, d_idx, 0],
        200.0,
        atol=1e-2,
    )
    assert np.allclose(
        ds["hres_surface_net_thermal_radiation"].values[:, d_idx, 0],
        -50.0,
        atol=1e-2,
    )
    assert np.all(np.isnan(ds["hres_temperature_2m"].values[:, d_idx, 1:]))
    assert np.all(np.isnan(ds["hres_total_precipitation"].values[:, d_idx, 1:]))
    assert np.allclose(
        ds["hres_missing_fraction"].values[:, d_idx, 0], 0.0, atol=1e-4
    )
    assert np.allclose(
        ds["hres_missing_fraction"].values[:, d_idx, 1:], 1.0, atol=1e-4
    )

  # 2. Forecast issue date (index 2: 2026-09-27): all 10 lead days are populated!
  assert np.allclose(
      ds["hres_temperature_2m"].values[:, 2, :], 20.0, atol=1e-2
  )
  assert np.allclose(
      ds["hres_surface_pressure"].values[:, 2, :], 101.325, atol=1e-2
  )
  assert np.allclose(
      ds["hres_total_precipitation"].values[:, 2, :], 5.0, atol=1e-2
  )
  assert np.allclose(
      ds["hres_surface_net_solar_radiation"].values[:, 2, :], 200.0, atol=1e-2
  )
  assert np.allclose(
      ds["hres_surface_net_thermal_radiation"].values[:, 2, :],
      -50.0,
      atol=1e-2,
  )
  assert np.allclose(
      ds["hres_missing_fraction"].values[:, 2, :], 0.0, atol=1e-4
  )

  # 3. Unpublished trailing date (index 3: 2026-09-28): all NaN and missing_fraction=1.0
  assert np.all(np.isnan(ds["hres_temperature_2m"].values[:, 3, :]))
  assert np.allclose(
      ds["hres_missing_fraction"].values[:, 3, :], 1.0, atol=1e-4
  )

  # Verify index file count: 1 step for 09-25 + 1 step for 09-26 + 10 steps for 09-27 + 10 steps checked for 09-28
  # Confirming spin-up days only fetched step=24h!
  sep25_steps = [
      p for p in fs.requested_index_paths if "/20260925/" in p
  ]
  sep26_steps = [
      p for p in fs.requested_index_paths if "/20260926/" in p
  ]
  sep27_steps = [
      p for p in fs.requested_index_paths if "/20260927/" in p
  ]
  assert len(sep25_steps) == 1
  assert len(sep26_steps) == 1
  assert len(sep27_steps) == 10


def test_read_hot_start_state_date_from_file_and_dir(tmp_path):
  """Verifies reading saved hot-start timestamps from .npz files and directories."""
  state_dir = tmp_path / "hot_start"
  state_dir.mkdir()

  f1 = state_dir / "state_basin_1.npz"
  f2 = state_dir / "state_basin_2.npz"
  np.savez(f1, h=np.zeros((1, 64)), date=np.array("2026-09-24"))
  np.savez(f2, h=np.zeros((1, 64)), last_date=np.array("2026-09-22"))

  assert read_hot_start_state_date(f1) == pd.Timestamp("2026-09-24")
  # Directory returns the minimum (earliest) date across all basin states
  assert read_hot_start_state_date(state_dir) == pd.Timestamp("2026-09-22")


def _make_synthetic_nowcast_ds(
    basins: Sequence[str],
    start_date: str,
    end_date: str,
    band_name: str,
    fill_val: float = 4.0,
    trailing_nan_days: int = 0,
) -> xr.Dataset:
  dates = pd.date_range(start_date, end_date, freq="D")
  arr = np.full((len(basins), len(dates)), fill_val, dtype=np.float32)
  if trailing_nan_days > 0:
    arr[:, -trailing_nan_days:] = np.nan
  return xr.Dataset(
      data_vars={band_name: (["basin", "date"], arr)},
      coords={"basin": list(basins), "date": dates.values},
  )


def test_coldstart_and_hotstart_end_to_end_workflow(
    tmp_path, basins_gdf, monkeypatch
):
  """Tests Cold-Start initialization followed by Hot-Start incremental append & NaN healing."""
  monkeypatch.setattr(
      "multimet.hres.decode_grib2_message", _fake_decode_grib2
  )
  basin_ids = [str(b) for b in basins_gdf.index]
  out_dir = tmp_path / "realtime_forcing"

  # Day 1: Cold-Start with lookback_days=4 ending on reference_date="2026-09-25"
  # Suppose on 2026-09-25, IMERG and CPC have a 1-day publication lag (so 2026-09-25 is NaN).
  all_dates = [
      f"2026-09-{d:02d}" for d in range(21, 28)
  ]
  fs = FakeECMWFOpenDataFS(available_dates=all_dates)

  def _mock_imerg_extract(self, basins_gdf, start_date, end_date, **kwargs):
    del self, kwargs
    # On the first run (ending 2026-09-25), pretend 2026-09-25 is trailing NaN (fill_val=4.0),
    # and on the second run (ending 2026-09-27), upstream has published 8.0!
    is_first_run = pd.to_datetime(end_date) == pd.Timestamp("2026-09-25")
    trailing = 1 if is_first_run else 0
    val = 4.0 if is_first_run else 8.0
    return _make_synthetic_nowcast_ds(
        [str(b) for b in basins_gdf.index],
        start_date,
        end_date,
        "imerg_precipitation",
        fill_val=val,
        trailing_nan_days=trailing,
    )

  def _mock_cpc_extract(self, basins_gdf, start_date, end_date, **kwargs):
    del self, kwargs
    is_first_run = pd.to_datetime(end_date) == pd.Timestamp("2026-09-25")
    trailing = 1 if is_first_run else 0
    val = 3.0 if is_first_run else 6.0
    return _make_synthetic_nowcast_ds(
        [str(b) for b in basins_gdf.index],
        start_date,
        end_date,
        "cpc_precipitation",
        fill_val=val,
        trailing_nan_days=trailing,
    )

  monkeypatch.setattr(
      "multimet.imerg.IMERGExtractor.extract_for_basins", _mock_imerg_extract
  )
  monkeypatch.setattr(
      "multimet.cpc.CPCExtractor.extract_for_basins", _mock_cpc_extract
  )

  # 1. Execute Cold-Start up to 2026-09-25 (with lookback_days=4 -> 2026-09-21..2026-09-25)
  cold_res = fetch_realtime_multimet(
      basins=basins_gdf,
      output_dir=out_dir,
      mode="coldstart",
      reference_date="2026-09-25",
      lookback_days=4,
      full_forecast_days=1,
      hres_fs=fs,
  )
  assert set(cold_res.keys()) == {"HRES", "IMERG", "CPC"}
  assert cold_res.reference_date == pd.Timestamp("2026-09-25")
  assert cold_res.product_windows["HRES"] == (
      pd.Timestamp("2026-09-21"),
      pd.Timestamp("2026-09-25"),
  )

  # Verify on-disk Zarr stores after Cold-Start
  with xr.open_zarr(cold_res["HRES"]) as ds_hres_1:
    assert len(ds_hres_1["date"]) == 5
    # 2026-09-21..24 have lead 1D valid and leads 2D..10D strictly NaN
    assert np.all(~np.isnan(ds_hres_1["hres_temperature_2m"].values[:, :4, 0]))
    assert np.all(np.isnan(ds_hres_1["hres_temperature_2m"].values[:, :4, 1:]))
    assert np.allclose(ds_hres_1["hres_missing_fraction"].values[:, :4, 0], 0.0)
    assert np.allclose(ds_hres_1["hres_missing_fraction"].values[:, :4, 1:], 1.0)
    # 2026-09-25 (index 4) has all 10 lead days valid with missing_fraction=0.0!
    assert np.all(~np.isnan(ds_hres_1["hres_temperature_2m"].values[:, 4, :]))
    assert np.allclose(ds_hres_1["hres_missing_fraction"].values[:, 4, :], 0.0)

  with xr.open_zarr(cold_res["IMERG"]) as ds_imerg_1:
    assert len(ds_imerg_1["date"]) == 5
    # 2026-09-25 (index 4) is strictly NaN due to simulated 1-day lag
    assert np.allclose(ds_imerg_1["imerg_precipitation"].values[:, :4], 4.0)
    assert np.all(np.isnan(ds_imerg_1["imerg_precipitation"].values[:, 4]))
    assert np.allclose(ds_imerg_1["imerg_missing_fraction"].values[:, :4], 0.0)
    assert np.allclose(ds_imerg_1["imerg_missing_fraction"].values[:, 4], 1.0)

  # Confirm inspect_store_last_valid_date backs up over the trailing NaN on 2026-09-25
  writer = MultiMetZarrWriter(out_dir)
  assert inspect_store_last_valid_date(writer, Product.IMERG, basin_ids) == pd.Timestamp(
      "2026-09-24"
  )
  assert inspect_store_last_valid_date(writer, Product.HRES, basin_ids) == pd.Timestamp(
      "2026-09-25"
  )

  # 2. Execute Hot-Start 2 days later on reference_date="2026-09-27":
  #    - IMERG & CPC should automatically plan [2026-09-24, 2026-09-27], healing 2026-09-25
  #      with the newly published 8.0 and appending 2026-09-26 and 2026-09-27!
  #    - HRES should automatically plan [2026-09-25, 2026-09-27], preserving the existing
  #      full 10-lead forecast on 2026-09-25, adding 1D spin-up on 2026-09-26, and adding
  #      full 10-lead forecast on 2026-09-27!
  hot_res = fetch_realtime_multimet(
      basins=basins_gdf,
      output_dir=out_dir,
      mode="hotstart",
      reference_date="2026-09-27",
      full_forecast_days=1,
      hres_fs=fs,
  )
  assert hot_res.product_windows["IMERG"] == (
      pd.Timestamp("2026-09-24"),
      pd.Timestamp("2026-09-27"),
  )
  assert hot_res.product_windows["CPC"] == (
      pd.Timestamp("2026-09-24"),
      pd.Timestamp("2026-09-27"),
  )
  assert hot_res.product_windows["HRES"] == (
      pd.Timestamp("2026-09-25"),
      pd.Timestamp("2026-09-27"),
  )

  with xr.open_zarr(hot_res["IMERG"]) as ds_imerg_2:
    assert len(ds_imerg_2["date"]) == 7  # 2026-09-21 .. 2026-09-27
    # 2026-09-21..24 preserved as 4.0; 2026-09-25 (index 4) healed to 8.0; 26, 27 are 8.0!
    assert np.allclose(ds_imerg_2["imerg_precipitation"].values[:, :4], 4.0)
    assert np.allclose(ds_imerg_2["imerg_precipitation"].values[:, 4:], 8.0)
    assert np.allclose(ds_imerg_2["imerg_missing_fraction"].values, 0.0)

  with xr.open_zarr(hot_res["HRES"]) as ds_hres_2:
    assert len(ds_hres_2["date"]) == 7  # 2026-09-21 .. 2026-09-27
    # 2026-09-25 (index 4) STILL has all 10 lead days valid (missing_fraction=0.0, not clobbered by 1D spin-up!)
    assert np.all(~np.isnan(ds_hres_2["hres_temperature_2m"].values[:, 4, :]))
    assert np.allclose(ds_hres_2["hres_missing_fraction"].values[:, 4, :], 0.0)
    # 2026-09-26 (index 5) has lead 1D valid
    assert np.all(~np.isnan(ds_hres_2["hres_temperature_2m"].values[:, 5, 0]))
    # 2026-09-27 (index 6) has all 10 lead days valid!
    assert np.all(~np.isnan(ds_hres_2["hres_temperature_2m"].values[:, 6, :]))


def test_default_coldstart_lookback_is_365_days(tmp_path, basins_gdf):
  """Verifies that coldstart mode plans a 365-day spin-up window by default."""
  fetcher = RealtimeForcingFetcher(tmp_path)
  basin_ids = [str(b) for b in basins_gdf.index]
  ref_dt = pd.Timestamp("2026-09-27")
  start_dt, end_dt = fetcher.plan_product_window(
      Product.HRES,
      basin_ids=basin_ids,
      reference_date=ref_dt,
      mode="coldstart",
  )
  assert end_dt == ref_dt
  assert (end_dt - start_dt).days == DEFAULT_COLDSTART_LOOKBACK_DAYS == 365
  assert start_dt == pd.Timestamp("2025-09-27")


def test_cli_arg_parser_and_main(tmp_path, monkeypatch):
  """Verifies CLI argument parsing and execution via python -m multimet.realtime."""
  geojson_path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  out_dir = tmp_path / "cli_out"

  parser = build_arg_parser()
  args = parser.parse_args([
      "--basins_path",
      str(geojson_path),
      "--output_dir",
      str(out_dir),
      "--mode",
      "coldstart",
      "--reference_date",
      "2026-09-27",
      "--lookback_days",
      "2",
      "--products",
      "HRES",
  ])
  assert args.mode == "coldstart"
  assert args.reference_date == "2026-09-27"
  assert args.lookback_days == 2

  monkeypatch.setattr(
      "multimet.hres.decode_grib2_message", _fake_decode_grib2
  )
  fs = FakeECMWFOpenDataFS(
      available_dates=["2026-09-25", "2026-09-26", "2026-09-27"]
  )
  monkeypatch.setattr(
      "multimet.hres.HRESExtractor._get_open_data_fs", lambda self: fs
  )

  res = realtime_main([
      "--basins_path",
      str(geojson_path),
      "--output_dir",
      str(out_dir),
      "--mode",
      "coldstart",
      "--reference_date",
      "2026-09-27",
      "--lookback_days",
      "2",
      "--products",
      "HRES",
  ])
  assert "HRES" in res
  summary = res.summary()
  assert summary["mode"] == "coldstart"
  assert summary["reference_date"] == "2026-09-27"
  assert summary["start_date"] == "2026-09-25"
  assert summary["end_date"] == "2026-09-27"
