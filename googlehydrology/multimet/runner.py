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

"""Local serial execution runner for MultiMet meteorological forcing extraction.

Enables extraction of the 5 core meteorological products (ERA5-Land, CPC, IMERG,
HRES, and GraphCast) over arbitrary catchment geometries and time intervals
in a local serial workflow without distributed dependencies.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd
import xarray as xr

from googlehydrology.multimet.base import BaseExtractor
from googlehydrology.multimet.config import Product
from googlehydrology.multimet.cpc import CPCExtractor
from googlehydrology.multimet.era5_land import ERA5LandExtractor
from googlehydrology.multimet.geometry import load_basin_geometries
from googlehydrology.multimet.graphcast import GraphCastExtractor
from googlehydrology.multimet.hres import HRESExtractor
from googlehydrology.multimet.imerg import IMERGExtractor
from googlehydrology.multimet.zarr_writer import MultiMetZarrWriter
from googlehydrology.multimet.zonal import ZonalWeightMatrix

logger = logging.getLogger(__name__)

PRODUCT_MAP: Dict[str, tuple[Product, type[BaseExtractor]]] = {
    "CPC": (Product.CPC, CPCExtractor),
    "ERA5_LAND": (Product.ERA5_LAND, ERA5LandExtractor),
    "IMERG": (Product.IMERG, IMERGExtractor),
    "HRES": (Product.HRES, HRESExtractor),
    "GRAPHCAST": (Product.GRAPHCAST, GraphCastExtractor),
}


def extract_multimet_serial(
    basins: Union[str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any]],
    output_dir: Union[str, os.PathLike],
    products: Optional[Sequence[Union[str, Product]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = "2020-01-01",
    end_date: Optional[Union[str, pd.Timestamp]] = "2020-01-02",
    source: str = "public",
    id_column: Optional[str] = None,
    overwrite: bool = False,
    weights_cache: Optional[str] = None,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
) -> Dict[str, str]:
  """Runs local serial extraction for requested meteorological forcing products.

  Args:
    basins: Catchment geometry source (file path, GeoDataFrame, or GeoJSON dict).
    output_dir: Directory where extracted consolidated Zarr stores will be saved.
    products: Products to extract (defaults to all 5 core products:
      CPC, ERA5_LAND, IMERG, HRES, GRAPHCAST).
    start_date: Start date string (YYYY-MM-DD) or Timestamp.
    end_date: End date string (YYYY-MM-DD) or Timestamp.
    source: Mode: 'public' (WeatherBench 2 / NOAA PSL / NASA / ECMWF Open Data)
      or 'local' (local files / custom directory).
    id_column: Optional column name for gauge/basin identifiers in geometries.
    overwrite: Whether to overwrite existing basins in destination Zarr stores.
    weights_cache: Optional path to .npz file for loading/saving weights.
    earthdata_username: Optional NASA Earthdata Login username.
    earthdata_password: Optional NASA Earthdata Login password.
    earthdata_token: Optional NASA Earthdata Bearer token.
    netrc_path: Optional path to custom .netrc file.

  Returns:
    Dictionary mapping product name to the output Zarr store path.
  """
  os.makedirs(output_dir, exist_ok=True)
  basins_gdf = load_basin_geometries(basins, id_column=id_column)
  basin_ids = list(basins_gdf.index)

  if products is None:
    target_prods = list(PRODUCT_MAP.keys())
  else:
    target_prods = []
    for p in products:
      name = p.value if isinstance(p, Product) else str(p).upper()
      target_prods.append(name)

  writer = MultiMetZarrWriter(output_dir)
  output_stores: Dict[str, str] = {}

  loaded_weights: Optional[ZonalWeightMatrix] = None
  if weights_cache is not None and os.path.exists(weights_cache):
    try:
      loaded_weights = ZonalWeightMatrix.load(weights_cache)
      logger.info("Loaded precomputed weight matrix from %s", weights_cache)
    except Exception as e:
      logger.warning("Failed to load weights cache at %s: %s", weights_cache, e)

  for prod_name in target_prods:
    if prod_name not in PRODUCT_MAP:
      raise ValueError(
          f"Unsupported product '{prod_name}'. Supported products: "
          f"{list(PRODUCT_MAP.keys())}"
      )

    prod_enum, extractor_cls = PRODUCT_MAP[prod_name]
    logger.info("Starting extraction for %s [%s to %s]...", prod_name, start_date, end_date)

    if prod_name == "CPC":
      src = "psl" if source in ("public", "auto") else ("binary" if source == "local" else source)
      extractor = CPCExtractor(source=src)
    elif prod_name == "IMERG":
      src = "gesdisc" if source in ("public", "auto") else ("h5" if source == "local" else source)
      extractor = IMERGExtractor(
          source=src,
          username=earthdata_username,
          password=earthdata_password,
          token=earthdata_token,
          netrc_path=netrc_path,
      )
    elif prod_name == "ERA5_LAND":
      src = "wb2" if source in ("public", "auto") else ("grib" if source == "local" else source)
      extractor = ERA5LandExtractor(source=src)
    elif prod_name == "HRES":
      src = "wb2" if source in ("public", "auto") else source
      extractor = HRESExtractor(source=src)
    elif prod_name == "GRAPHCAST":
      src = "wb2" if source in ("public", "auto") else source
      extractor = GraphCastExtractor(source=src)
    else:
      extractor = extractor_cls()

    weights_matrix = loaded_weights
    if weights_matrix is not None:
      # Verify compatibility with extractor's coordinate resolution
      if weights_matrix.grid_shape != (len(extractor.lats), len(extractor.lons)):
        weights_matrix = None

    ds = extractor.extract_for_basins(
        basins_gdf,
        start_date=start_date,
        end_date=end_date,
        weights_matrix=weights_matrix,
    )

    store_path = writer.write_or_append(
        ds,
        prod_enum,
        overwrite_existing_basins=overwrite,
    )
    writer.consolidate_metadata(prod_enum)
    output_stores[prod_name] = store_path
    logger.info("Successfully extracted %s to %s", prod_name, store_path)

  return output_stores


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="MultiMet Serial Meteorological Forcing Extractor",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--basins_path",
      type=str,
      required=True,
      help="Path to GeoJSON or Shapefile catchment boundaries.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Directory to save extracted consolidated Zarr stores.",
  )
  parser.add_argument(
      "--products",
      type=str,
      default="CPC,ERA5_LAND,IMERG,HRES,GRAPHCAST",
      help="Comma-separated product list to extract.",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default="2020-01-01",
      help="Start date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default="2020-01-02",
      help="End date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--id_column",
      type=str,
      default=None,
      help="Column name for basin ID in geometry file.",
  )
  parser.add_argument(
      "--source",
      type=str,
      default="public",
      help="Source mode: 'public' (WeatherBench 2, NOAA, NASA) or 'local'.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite existing basins in output Zarr store.",
  )
  parser.add_argument(
      "--weights_cache",
      type=str,
      default=None,
      help="Path to precomputed/cached .npz weights archive.",
  )
  parser.add_argument(
      "--earthdata_username",
      type=str,
      default=None,
      help="NASA Earthdata Login username.",
  )
  parser.add_argument(
      "--earthdata_password",
      type=str,
      default=None,
      help="NASA Earthdata Login password.",
  )
  parser.add_argument(
      "--earthdata_token",
      type=str,
      default=None,
      help="NASA Earthdata Bearer token.",
  )
  parser.add_argument(
      "--netrc_path",
      type=str,
      default=None,
      help="Custom path to .netrc file for Earthdata credentials.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )
  parser = _build_parser()
  args = parser.parse_args(argv)

  prods = [p.strip() for p in args.products.split(",") if p.strip()]
  t0 = time.time()
  print(f"▶ Starting MultiMet serial extraction for: {prods}")
  stores = extract_multimet_serial(
      basins=args.basins_path,
      output_dir=args.output_dir,
      products=prods,
      start_date=args.start_date,
      end_date=args.end_date,
      source=args.source,
      id_column=args.id_column,
      overwrite=args.overwrite,
      weights_cache=args.weights_cache,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
  )
  print(f"\n✓ Completed extraction of {len(stores)} products in {time.time() - t0:.2f}s:")
  for prod, store_path in stores.items():
    print(f"  • {prod:12s} -> {store_path}")


if __name__ == "__main__":
  main()
