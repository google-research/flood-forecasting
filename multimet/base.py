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

from __future__ import annotations

import abc
import os
from typing import Dict, List, Mapping, Optional, Sequence

import geopandas as gpd
import pandas as pd
import xarray as xr

from multimet.config import (
    PRODUCT_BANDS,
    PRODUCT_TYPES,
    Product,
    ProductType,
)


class BaseExtractor(abc.ABC):
  """Abstract base class for all MultiMet forcing extractors."""

  def __init__(self, product: Product, data_dir: Optional[str] = None):
    self.product = product
    self.product_type = PRODUCT_TYPES[product]
    self.expected_bands = PRODUCT_BANDS[product]
    self.data_dir = data_dir

  @abc.abstractmethod
  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[str | pd.Timestamp] = None,
      end_date: Optional[str | pd.Timestamp] = None,
  ) -> xr.Dataset:
    """Extracts forcing data for all basins in the GeoDataFrame.

    Args:
      basins_gdf: GeoDataFrame indexed by basin_id with geometry in EPSG:4326.
      start_date: Optional start date filter.
      end_date: Optional end date filter.

    Returns:
      xr.Dataset matching the exact MultiMet schema for this product.
    """
    pass
