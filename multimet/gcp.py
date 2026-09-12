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

"""Google Cloud Platform (GCP) authentication and project resolution utilities."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional
import urllib.request

logger = logging.getLogger(__name__)


def auto_detect_gcp_project(explicit_project: Optional[str] = None) -> Optional[str]:
  """Auto-detects the GCP project from environment, GCE metadata, gcloud, or ADC.

  Resolution precedence:
    1. Explicitly provided `explicit_project` argument.
    2. Environment variables (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_QUOTA_PROJECT`,
       `CLOUDSDK_CORE_PROJECT`, `GCP_PROJECT`, `GCLOUD_PROJECT`).
    3. Google Compute Engine (GCE) local instance metadata server.
    4. Local `gcloud config get-value project` CLI command.
    5. `google.auth.default()` project resolution.

  Args:
    explicit_project: Optional project ID passed directly by caller.

  Returns:
    Discovered project ID string, or None if not on GCP and unconfigured.
  """
  if explicit_project and str(explicit_project).strip():
    return str(explicit_project).strip()

  for env_key in (
      "GOOGLE_CLOUD_PROJECT",
      "GOOGLE_CLOUD_QUOTA_PROJECT",
      "CLOUDSDK_CORE_PROJECT",
      "GCP_PROJECT",
      "GCLOUD_PROJECT",
  ):
    val = os.environ.get(env_key)
    if val and val.strip():
      return val.strip()

  # 1. Query GCE instance metadata server (available on Google Cloud VMs without credentials)
  try:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/project/project-id",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=1.0) as resp:
      proj = resp.read().decode("utf-8").strip()
      if proj:
        logger.debug("Auto-detected GCP project from GCE metadata server: %s", proj)
        return proj
  except Exception:
    pass

  # 2. Query gcloud CLI configuration
  if shutil.which("gcloud"):
    try:
      proj = subprocess.check_output(
          ["gcloud", "config", "get-value", "project"],
          stderr=subprocess.DEVNULL,
          text=True,
          timeout=2.0,
      ).strip()
      if proj and proj != "(unset)":
        logger.debug("Auto-detected GCP project from gcloud config: %s", proj)
        return proj
    except Exception:
      pass

  # 3. Query Application Default Credentials via google-auth
  try:
    import google.auth
    _, proj = google.auth.default()
    if proj:
      logger.debug("Auto-detected GCP project from google.auth: %s", proj)
      return proj
  except Exception:
    pass

  return None


def configure_gcp_project(project: Optional[str] = None) -> Optional[str]:
  """Detects and activates GCP project context across environment, fsspec, and gcsfs.

  Automatically configures environment variables (GOOGLE_CLOUD_PROJECT,
  GOOGLE_CLOUD_QUOTA_PROJECT) and fsspec storage configurations so that
  GCS calls (including parallel chunk writes and overwrites) carry the correct
  billing/quota project context and avoid 403 Forbidden errors on user credentials.

  Args:
    project: Optional explicit project ID. If None, auto_detect_gcp_project() is used.

  Returns:
    Configured project ID, or None if none could be detected.
  """
  detected = auto_detect_gcp_project(project)
  if not detected:
    return None

  # Configure environment variables for libraries that read from env
  os.environ["GOOGLE_CLOUD_PROJECT"] = detected
  os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = detected
  os.environ["CLOUDSDK_CORE_PROJECT"] = detected

  # Configure fsspec for gs / gcs protocols
  try:
    import fsspec.config
    fsspec.config.conf.setdefault("gs", {})["project"] = detected
    fsspec.config.conf.setdefault("gcs", {})["project"] = detected
  except Exception as e:
    logger.debug("Failed to set fsspec config: %s", e)

  return detected
