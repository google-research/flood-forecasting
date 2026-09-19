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

"""Tests for Zarr target classification and storage helpers."""

from __future__ import annotations

import pytest

from multimet.storage import is_remote_target, resolve_zarr_target

pytestmark = pytest.mark.unit


class TestRemoteTargets:
  """Locations with explicit remote URI schemes."""

  def test_explicit_gs_url_is_remote(self) -> None:
    assert resolve_zarr_target('gs://example-bucket/daily.zarr') == (
        'gs://example-bucket/daily.zarr',
        True,
    )

  @pytest.mark.parametrize('url', ['s3://bucket/x.zarr', 'az://c/x.zarr'])
  def test_other_schemes_are_passed_through(self, url: str) -> None:
    assert resolve_zarr_target(url) == (url, True)


class TestLocalTargets:
  """Locations on the local filesystem (never implicitly rewritten to gs://)."""

  def test_posix_absolute_path(self) -> None:
    assert resolve_zarr_target('/tmp/out/daily.zarr') == (
        '/tmp/out/daily.zarr',
        False,
    )

  @pytest.mark.parametrize(
      'path',
      ['./daily.zarr', '../daily.zarr', 'daily.zarr', 'output/daily.zarr'],
  )
  def test_relative_paths_stay_local(self, path: str) -> None:
    assert resolve_zarr_target(path) == (path, False)

  def test_home_relative_path(self) -> None:
    assert resolve_zarr_target('~/daily.zarr') == ('~/daily.zarr', False)

  def test_file_url_is_unwrapped(self) -> None:
    assert resolve_zarr_target('file:///tmp/daily.zarr') == (
        '/tmp/daily.zarr',
        False,
    )


class TestWindowsTargets:
  """Regression tests for Windows-style paths."""

  @pytest.mark.parametrize(
      'path',
      [
          r'C:\Users\runneradmin\AppData\Local\Temp\pytest-0\daily.zarr',
          r'C:/Users/runneradmin/daily.zarr',
          r'D:\data\daily.zarr',
          r'z:\data\daily.zarr',
      ],
  )
  def test_drive_qualified_paths_are_local(self, path: str) -> None:
    assert resolve_zarr_target(path) == (path, False)

  def test_unc_path_is_local(self) -> None:
    path = r'\\fileserver\share\daily.zarr'
    assert resolve_zarr_target(path) == (path, False)

  def test_relative_windows_path_is_local(self) -> None:
    path = r'.\output\daily.zarr'
    assert resolve_zarr_target(path) == (path, False)

  def test_drive_letter_is_not_mistaken_for_a_uri_scheme(self) -> None:
    assert is_remote_target(r'C:\data\daily.zarr') is False


class TestValidation:

  def test_empty_target_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='non-empty'):
      resolve_zarr_target('')
