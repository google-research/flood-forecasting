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

from multimet.base import BaseExtractor
from multimet.config import Product
from multimet.cpc import CPCExtractor
from multimet.era5_land import ERA5LandExtractor
from multimet.geometry import load_basin_geometries
from multimet.gcp import configure_gcp_project
from multimet.graphcast import GraphCastExtractor
from multimet.hres import HRESExtractor
from multimet.imerg import IMERGExtractor
from multimet.zarr_writer import MultiMetZarrWriter
from multimet.zonal import ZonalWeightMatrix

logger = logging.getLogger(__name__)

PRODUCT_MAP: Dict[str, tuple[Product, type[BaseExtractor]]] = {
    "CPC": (Product.CPC, CPCExtractor),
    "ERA5_LAND": (Product.ERA5_LAND, ERA5LandExtractor),
    "IMERG": (Product.IMERG, IMERGExtractor),
    "HRES": (Product.HRES, HRESExtractor),
    "GRAPHCAST": (Product.GRAPHCAST, GraphCastExtractor),
}


def _parse_product_uri_pairs(
    pairs: Optional[Sequence[str]],
) -> Dict[str, str]:
  """Parses repeatable ``PRODUCT=URI`` CLI arguments into an uppercase dict."""
  out: Dict[str, str] = {}
  if not pairs:
    return out
  for item in pairs:
    if "=" not in item:
      raise ValueError(
          f"Invalid --archive-store entry {item!r}; expected format PRODUCT=URI "
          "(e.g. CPC=gs://.../daily_surface.zarr)."
      )
    k, v = item.split("=", 1)
    k = k.strip().upper()
    v = v.strip()
    if not k or not v:
      raise ValueError(
          f"Invalid --archive-store entry {item!r}; both PRODUCT and URI must be non-empty."
      )
    out[k] = v
  return out


def extract_multimet_serial(
    basins: Union[
        str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any], Sequence[Any]
    ],
    output_dir: Union[str, os.PathLike],
    products: Optional[Sequence[Union[str, Product]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    source: str = "public",
    archive_stores: Optional[Mapping[str, str]] = None,
    data_dirs: Optional[Mapping[str, str]] = None,
    id_column: Optional[str] = None,
    overwrite: bool = False,
    weights_cache: Optional[str] = None,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
    gcp_project: Optional[str] = None,
) -> Dict[str, str]:
  """Runs local serial extraction for requested meteorological forcing products.

  Args:
    basins: Catchment geometry source (file path, GeoDataFrame, or GeoJSON dict).
    output_dir: Directory where extracted consolidated Zarr stores will be saved.
    products: Products to extract (defaults to all 5 core products:
      CPC, ERA5_LAND, IMERG, HRES, GRAPHCAST).
    start_date: Required start date string (YYYY-MM-DD) or Timestamp.
    end_date: Required end date string (YYYY-MM-DD) or Timestamp.
    source: Source mode: 'archive' (gridded Zarr archives), 'public'/'upstream'
      (third-party agency HTTP/Zarr feeds), or 'local'.
    archive_stores: Mapping of product name (e.g. ``"CPC"``, ``"ERA5_LAND"``,
      ``"IMERG"``, ``"HRES"``) to its explicit gridded archive Zarr URI or path.
    data_dirs: Optional per-product data directory / URI mapping for non-archive
      sources.
    id_column: Optional column name for gauge/basin identifiers in geometries.
    overwrite: Whether to overwrite existing basins in destination Zarr stores.
    weights_cache: Optional path to .npz file for loading/saving weights.
    earthdata_username: Optional NASA Earthdata Login username.
    earthdata_password: Optional NASA Earthdata Login password.
    earthdata_token: Optional NASA Earthdata Bearer token.
    netrc_path: Optional path to custom .netrc file.
    gcp_project: Optional Google Cloud project ID for GCS quota/billing.

  Returns:
    Dictionary mapping product name to the output Zarr store path.
  """
  if start_date is None or end_date is None:
    raise ValueError(
        "extract_multimet_serial requires both start_date and end_date to be "
        "explicitly provided; default placeholder dates are not permitted."
    )

  norm_archive_stores: Dict[str, str] = {
      (k.value if isinstance(k, Product) else str(k).upper()): str(v)
      for k, v in (archive_stores or {}).items()
  }
  norm_data_dirs: Dict[str, str] = {
      (k.value if isinstance(k, Product) else str(k).upper()): str(v)
      for k, v in (data_dirs or {}).items()
  }

  all_paths = [str(output_dir)]
  if isinstance(basins, (str, os.PathLike)):
    all_paths.append(str(basins))
  elif isinstance(basins, (list, tuple, set)):
    all_paths.extend(str(x) for x in basins)
  if weights_cache:
    all_paths.append(str(weights_cache))
  all_paths.extend(norm_archive_stores.values())
  all_paths.extend(norm_data_dirs.values())

  if any(p.startswith(("gs://", "gcs://")) for p in all_paths) or gcp_project:
    gcp_project = configure_gcp_project(gcp_project)
    if gcp_project:
      logger.info(
          "Configured Google Cloud project for GCS operations: %s", gcp_project
      )

  if not str(output_dir).startswith(("gs://", "gcs://")):
    os.makedirs(output_dir, exist_ok=True)
  basins_gdf = load_basin_geometries(basins, id_column=id_column)

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
    loaded_weights = ZonalWeightMatrix.load(weights_cache)
    logger.info("Loaded precomputed weight matrix from %s", weights_cache)

  source_lower = source.lower().strip()
  is_archive_mode = source_lower in (
      "archive",
      "gridded_archive",
      "zarr_archive",
  )

  for prod_name in target_prods:
    if prod_name not in PRODUCT_MAP:
      raise ValueError(
          f"Unsupported product '{prod_name}'. Supported products: "
          f"{list(PRODUCT_MAP.keys())}"
      )

    prod_enum, extractor_cls = PRODUCT_MAP[prod_name]
    logger.info(
        "Starting extraction for %s [%s to %s]...",
        prod_name,
        start_date,
        end_date,
    )

    prod_archive_uri = norm_archive_stores.get(
        prod_name, norm_data_dirs.get(prod_name)
    )

    if prod_name == "ERA5_LAND":
      # ERA5-Land is strictly archive-only.
      extractor = ERA5LandExtractor(
          data_dir=prod_archive_uri,
          source="archive",
      )
    elif is_archive_mode or prod_name in norm_archive_stores:
      if not prod_archive_uri:
        raise ValueError(
            f"Product {prod_name} in archive mode requires an explicit store "
            f"URI via archive_stores[{prod_name!r}] or --archive-store "
            f"{prod_name}=<URI>."
        )
      if prod_name == "CPC":
        extractor = CPCExtractor(data_dir=prod_archive_uri, source="archive")
      elif prod_name == "IMERG":
        extractor = IMERGExtractor(data_dir=prod_archive_uri, source="archive")
      elif prod_name == "HRES":
        extractor = HRESExtractor(data_dir=prod_archive_uri, source="archive")
      elif prod_name == "GRAPHCAST":
        extractor = GraphCastExtractor(
            data_dir=prod_archive_uri, source="archive"
        )
      else:
        extractor = extractor_cls(data_dir=prod_archive_uri)
    elif prod_name == "CPC":
      src = (
          "psl"
          if source_lower in ("public", "auto", "upstream")
          else ("binary" if source_lower == "local" else source_lower)
      )
      extractor = CPCExtractor(
          data_dir=norm_data_dirs.get(prod_name), source=src
      )
    elif prod_name == "IMERG":
      src = (
          "gesdisc"
          if source_lower in ("public", "auto", "upstream")
          else ("h5" if source_lower == "local" else source_lower)
      )
      extractor = IMERGExtractor(
          data_dir=norm_data_dirs.get(prod_name),
          source=src,
          username=earthdata_username,
          password=earthdata_password,
          token=earthdata_token,
          netrc_path=netrc_path,
      )
    elif prod_name == "HRES":
      src = (
          "wb2"
          if source_lower in ("public", "auto", "upstream")
          else source_lower
      )
      extractor = HRESExtractor(
          data_dir=norm_data_dirs.get(prod_name), source=src
      )
    elif prod_name == "GRAPHCAST":
      src = (
          "wb2"
          if source_lower in ("public", "auto", "upstream")
          else source_lower
      )
      extractor = GraphCastExtractor(
          data_dir=norm_data_dirs.get(prod_name), source=src
      )
    else:
      extractor = extractor_cls(data_dir=norm_data_dirs.get(prod_name))

    weights_matrix = loaded_weights
    if weights_matrix is not None:
      if weights_matrix.grid_shape != (
          len(extractor.lats),
          len(extractor.lons),
      ):
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
      nargs="+",
      required=True,
      help=(
          "Path(s) to GeoJSON or Shapefile catchment boundaries. "
          "Accepts one or more files, glob patterns, directories, or comma-separated strings."
      ),
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
      default="CPC,ERA5_LAND,IMERG,HRES",
      help="Comma-separated product list to extract.",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      required=True,
      help="Required start date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      required=True,
      help="Required end date (YYYY-MM-DD).",
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
      help=(
          "Source mode: 'archive' (gridded Zarr archives via --archive-store), "
          "'public'/'upstream' (NOAA PSL, NASA GES DISC, etc.), or 'local'."
      ),
  )
  parser.add_argument(
      "--archive-store",
      "--archive_store",
      dest="archive_stores",
      action="append",
      default=None,
      metavar="PRODUCT=URI",
      help=(
          "Explicit gridded archive Zarr store URI for a product (repeatable), "
          "e.g. --archive-store CPC=gs://.../CPC/daily_surface.zarr."
      ),
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite existing basins in output Zarr store.",
  )
  parser.add_argument(
      "--append",
      action="store_true",
      help="Append to existing Zarr store (either new dates or new basins).",
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
  parser.add_argument(
      "--gcp_project",
      type=str,
      default=None,
      help="Optional Google Cloud project ID for GCS quota/billing. Auto-detected if omitted.",
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
  archive_stores = _parse_product_uri_pairs(args.archive_stores)
  t0 = time.time()
  print(f"▶ Starting MultiMet serial extraction for: {prods}")
  stores = extract_multimet_serial(
      basins=args.basins_path,
      output_dir=args.output_dir,
      products=prods,
      start_date=args.start_date,
      end_date=args.end_date,
      source=args.source,
      archive_stores=archive_stores,
      id_column=args.id_column,
      overwrite=args.overwrite,
      weights_cache=args.weights_cache,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
      gcp_project=args.gcp_project,
  )
  print(
      f"\n✓ Completed extraction of {len(stores)} products in"
      f" {time.time() - t0:.2f}s:"
  )
  for prod, store_path in stores.items():
    print(f"  • {prod:12s} -> {store_path}")


if __name__ == "__main__":
  main()
