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

"""Shared utilities for MultiMet data workflows."""

from multimet.utils.gcs import (
    download_directory_from_gcs,
    download_file_from_gcs,
    download_hydroatlas_from_gcs,
    gcs_path_exists,
    is_gcs_path,
    normalize_gcs_path,
    read_bytes_from_gcs,
    strip_gcs_prefix,
    sync_gcs_directory,
    upload_file_to_gcs,
    upload_to_gcs,
)

__all__ = [
    "download_directory_from_gcs",
    "download_file_from_gcs",
    "download_hydroatlas_from_gcs",
    "gcs_path_exists",
    "is_gcs_path",
    "normalize_gcs_path",
    "read_bytes_from_gcs",
    "strip_gcs_prefix",
    "sync_gcs_directory",
    "upload_file_to_gcs",
    "upload_to_gcs",
]
