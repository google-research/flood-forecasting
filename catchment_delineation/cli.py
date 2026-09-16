"""Command-Line Interface for DEM Catchment Delineation.

Allows users to delineate upstream drainage catchment boundaries from a DEM by
supplying a single coordinate pair, a list of coordinates, or a CSV file.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catchment_delineation.config import GCS_TILES_URI, get_default_cache_dir
from catchment_delineation.delineator import CatchmentCoverageError, DemDelineator
from catchment_delineation.gcs import download_tile_from_gcs
from catchment_delineation.tiles import (
    is_coord_in_coverage,
    is_tile_in_coverage,
    latlon_to_tile_key,
    list_available_tiles,
    tile_key_to_filename,
)

logger = logging.getLogger("catchment_delineation.cli")


def _delineate_worker(
    task: Tuple[float, float, Optional[str], Optional[str], str, Optional[str], int, int],
) -> Optional[Dict[str, Any]]:
  lat, lon, cid, tiles_dir, gcs_uri, cache_dir, snap_window, max_cells = task
  try:
    delineator = DemDelineator(
        tiles_dir=Path(tiles_dir) if tiles_dir else None,
        gcs_uri=gcs_uri,
        cache_dir=cache_dir,
        cache_tiles=True,
    )
    return delineator.delineate(
        lat=lat,
        lon=lon,
        catchment_id=cid,
        snap_window_cells=snap_window,
        max_cells=max_cells,
    )
  except CatchmentCoverageError as e:
    logger.warning("Catchment coverage abort for %s (%.4f, %.4f): %s", cid, lat, lon, e)
    return None
  except Exception as e:
    logger.error("Error delineating %s (%.4f, %.4f): %s", cid, lat, lon, e)
    return None


def parse_coord_str(coord_str: str) -> Tuple[float, float]:
  """Parses a coordinate string like '39.6828,-88.7729' into (lat, lon)."""
  parts = coord_str.strip().split(",")
  if len(parts) != 2:
    parts = coord_str.strip().split()
  if len(parts) != 2:
    raise ValueError(
        f"Invalid coordinate format '{coord_str}'. Expected 'lat,lon'."
    )
  return float(parts[0]), float(parts[1])


def load_coords_from_file(
    file_path: Path,
    lat_col_arg: Optional[str] = None,
    lon_col_arg: Optional[str] = None,
    id_col_arg: Optional[str] = None,
) -> Tuple[List[Tuple[float, float]], List[Optional[str]]]:
  """Parses lat/lon coordinates and optional IDs from CSV or Parquet file."""
  if str(file_path).endswith((".parquet", ".geoparquet")):
    import pandas as pd
    df = pd.read_parquet(file_path)
    fieldnames = list(df.columns)

    lat_col = lat_col_arg or next(
        (c for c in fieldnames if any(k in c.lower() for k in ("lat", "latitude", "y"))),
        None,
    )
    lon_col = lon_col_arg or next(
        (c for c in fieldnames if any(k in c.lower() for k in ("lon", "longitude", "long", "lng", "x"))),
        None,
    )
    id_col = id_col_arg or next(
        (c for c in fieldnames if any(k in c.lower() for k in ("id", "gauge_id", "station_id", "catchment_id", "hybas_id", "name"))),
        None,
    )
    if not id_col and fieldnames and (fieldnames[0] == "" or "unnamed" in fieldnames[0].lower()):
      id_col = fieldnames[0]

    if not lat_col or not lon_col:
      raise ValueError(f"Parquet file must have latitude and longitude columns. Found: {fieldnames}")

    coords = [(float(r[lat_col]), float(r[lon_col])) for _, r in df.iterrows()]
    ids = [str(r[id_col]) if id_col and pd.notna(r[id_col]) else None for _, r in df.iterrows()]
    return coords, ids

  coords = []
  ids = []
  with open(file_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
      raise ValueError(f"CSV file '{file_path}' is empty or has no header.")

    fieldnames = list(reader.fieldnames)
    lat_col = lat_col_arg or next(
        (c for c in fieldnames if any(k in c.lower() for k in ("lat", "latitude", "y"))),
        None,
    )
    lon_col = lon_col_arg or next(
        (c for c in fieldnames if any(k in c.lower() for k in ("lon", "longitude", "long", "lng", "x"))),
        None,
    )
    id_col = id_col_arg or next(
        (c for c in fieldnames if any(k in c.lower() for k in ("id", "gauge_id", "station_id", "catchment_id", "hybas_id", "name"))),
        None,
    )
    if not id_col and fieldnames and (fieldnames[0] == "" or "unnamed" in fieldnames[0].lower()):
      id_col = fieldnames[0]

    if not lat_col or not lon_col:
      raise ValueError(
          f"CSV file must have latitude and longitude columns. Found: {fieldnames}"
      )

    for row in reader:
      lat_val = float(row[lat_col].strip())
      lon_val = float(row[lon_col].strip())
      id_val = row[id_col].strip() if id_col and row.get(id_col) else None
      coords.append((lat_val, lon_val))
      ids.append(id_val)

  return coords, ids


def load_coords_from_csv(
    csv_path: Path,
    lat_col_arg: Optional[str] = None,
    lon_col_arg: Optional[str] = None,
    id_col_arg: Optional[str] = None,
) -> Tuple[List[Tuple[float, float]], List[Optional[str]]]:
  return load_coords_from_file(csv_path, lat_col_arg, lon_col_arg, id_col_arg)


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(
      description=(
          "DEM Watershed Catchment Delineation Tool.\n"
          "Delineates upstream drainage basin polygons from high-resolution "
          "D8 flow direction rasters (HydroSHEDS/MERIT) for arbitrary coordinates."
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  coord_group = parser.add_argument_group("Coordinate Input Options")
  coord_group.add_argument(
      "--lat",
      type=float,
      help="Latitude of outlet pour point.",
  )
  coord_group.add_argument(
      "--lon",
      type=float,
      help="Longitude of outlet pour point.",
  )
  coord_group.add_argument(
      "--coords",
      nargs="+",
      help=(
          "One or more coordinate pairs specified as 'lat,lon' strings "
          "(e.g., --coords 39.6828,-88.7729 40.4,-86.9)."
      ),
  )
  coord_group.add_argument(
      "--csv",
      type=str,
      help="Path to CSV or Parquet file with latitude and longitude columns.",
  )
  coord_group.add_argument(
      "--id",
      type=str,
      default=None,
      help="Custom catchment ID (for single coordinate delineation).",
  )
  coord_group.add_argument(
      "--workers",
      "-w",
      type=int,
      default=1,
      help="Number of parallel worker processes to use for batch delineation.",
  )
  coord_group.add_argument(
      "--lat-col",
      type=str,
      default=None,
      help="Name of latitude column in CSV/Parquet file.",
  )
  coord_group.add_argument(
      "--lon-col",
      type=str,
      default=None,
      help="Name of longitude column in CSV/Parquet file.",
  )
  coord_group.add_argument(
      "--id-col",
      type=str,
      default=None,
      help="Name of catchment/gauge ID column in CSV/Parquet file.",
  )

  config_group = parser.add_argument_group("Algorithm & Tile Settings")
  config_group.add_argument(
      "--tiles-dir",
      type=str,
      default=None,
      help=(
          "Optional custom directory containing 5x5 degree DEM flow-direction .npy tiles. "
          "If omitted, tiles are automatically retrieved from the gs bucket "
          f"({GCS_TILES_URI})."
      ),
  )
  config_group.add_argument(
      "--snap-window",
      type=int,
      default=12,
      help="Snap search window half-width in cells (default: 12 cells ~1.1 km).",
  )
  config_group.add_argument(
      "--max-cells",
      type=int,
      default=50000000,
      help="Maximum upstream cells safety limit (default: 50,000,000).",
  )

  output_group = parser.add_argument_group("Output Options")
  output_group.add_argument(
      "-o",
      "--output",
      type=str,
      default=None,
      help="Output path for GeoJSON (.geojson, .json) or GeoParquet (.geoparquet, .parquet) file. If omitted, prints GeoJSON to stdout.",
  )
  output_group.add_argument(
      "--clean-cache",
      action="store_true",
      help="Automatically clean up downloaded local DEM tiles from cache after delineation completes.",
  )
  output_group.add_argument(
      "--pretty",
      action="store_true",
      help="Format output JSON with indentation.",
  )
  output_group.add_argument(
      "--list-tiles",
      action="store_true",
      help="List available DEM tiles in the tiles directory and exit.",
  )

  args = parser.parse_args(argv)

  tiles_dir = Path(args.tiles_dir) if args.tiles_dir else None

  if args.list_tiles:
    tiles = list_available_tiles(tiles_dir)
    target_dir = tiles_dir if tiles_dir else get_default_cache_dir()
    print(f"DEM Tiles Directory: {target_dir} (Total: {len(tiles)} tiles)")
    for t in tiles:
      print(f"  {t}")
    return 0

  # Collect target coordinates
  coords_to_process: List[Tuple[float, float]] = []
  ids_to_process: List[Optional[str]] = []

  if args.lat is not None and args.lon is not None:
    coords_to_process.append((args.lat, args.lon))
    ids_to_process.append(args.id)

  if args.coords:
    for c_str in args.coords:
      coords_to_process.append(parse_coord_str(c_str))
      ids_to_process.append(None)

  if args.csv:
    csv_coords, csv_ids = load_coords_from_file(
        Path(args.csv),
        lat_col_arg=args.lat_col,
        lon_col_arg=args.lon_col,
        id_col_arg=args.id_col,
    )
    coords_to_process.extend(csv_coords)
    ids_to_process.extend(csv_ids)

  if not coords_to_process:
    parser.print_help(sys.stderr)
    sys.stderr.write(
        "\nError: No coordinates provided. Specify --lat and --lon, --coords, or --csv.\n"
    )
    return 1

  delineator = DemDelineator(tiles_dir=tiles_dir)

  # Run delineation
  try:
    if len(coords_to_process) == 1 and not args.coords and not args.csv:
      # Single feature output
      lat, lon = coords_to_process[0]
      cid = ids_to_process[0]
      result = delineator.delineate(
          lat=lat,
          lon=lon,
          snap_window_cells=args.snap_window,
          max_cells=args.max_cells,
          catchment_id=cid,
      )
    elif args.workers > 1 and len(coords_to_process) > 1:
      # Parallel multi-worker delineation
      if not tiles_dir:
        cache_path = get_default_cache_dir()
        cache_path.mkdir(parents=True, exist_ok=True)
        needed_tile_keys = set()
        for lat, lon in coords_to_process:
          if is_coord_in_coverage(lat, lon):
            tk = latlon_to_tile_key(lat, lon)
            if is_tile_in_coverage(tk[0], tk[1]):
              needed_tile_keys.add(tk)
        missing_tiles = [
            tk
            for tk in sorted(needed_tile_keys)
            if not (cache_path / tile_key_to_filename(tk[0], tk[1])).exists()
        ]
        if missing_tiles:
          print(
              f"Pre-caching {len(missing_tiles)} required DEM tiles in main process before spawning {args.workers} workers...",
              file=sys.stderr,
          )
          def _dl(tk: Tuple[int, int]):
            try:
              download_tile_from_gcs(
                  lat_top=tk[0],
                  lon_left=tk[1],
                  target_dir=cache_path,
                  source_uri=GCS_TILES_URI,
              )
            except Exception as e:
              logger.warning("Could not pre-cache tile %s: %s", tile_key_to_filename(tk[0], tk[1]), e)
          with ThreadPoolExecutor(max_workers=min(32, len(missing_tiles))) as pool:
            list(pool.map(_dl, missing_tiles))

      tasks = [
          (
              lat,
              lon,
              cid,
              str(tiles_dir) if tiles_dir else None,
              GCS_TILES_URI,
              None,
              args.snap_window,
              args.max_cells,
          )
          for (lat, lon), cid in zip(coords_to_process, ids_to_process)
      ]
      features = []
      total = len(tasks)
      print(
          f"Delineating {total} catchments in parallel using {args.workers} workers...",
          file=sys.stderr,
      )
      import multiprocessing
      mp_ctx = multiprocessing.get_context("spawn")
      with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as pool:
        futures = {pool.submit(_delineate_worker, t): i for i, t in enumerate(tasks)}
        completed = 0
        for fut in as_completed(futures):
          completed += 1
          if completed % max(1, total // 20) == 0 or completed == total:
            pct = (completed / total) * 100.0
            print(
                f"Progress: [{completed}/{total}] catchments evaluated ({pct:.1f}%)",
                file=sys.stderr,
            )
          res_feat = fut.result()
          if res_feat is not None:
            features.append(res_feat)
      result = {"type": "FeatureCollection", "features": features}
    else:
      # Multiple features -> FeatureCollection
      result = delineator.delineate_batch(
          coords=coords_to_process,
          ids=ids_to_process,
          snap_window_cells=args.snap_window,
          max_cells=args.max_cells,
      )
  except CatchmentCoverageError as e:
    sys.stderr.write(f"\nCatchment Delineation Aborted: {e}\n")
    return 1
  finally:
    if args.clean_cache:
      cache_p = tiles_dir if tiles_dir else get_default_cache_dir()
      if cache_p.exists():
        import shutil
        shutil.rmtree(cache_p, ignore_errors=True)

  if args.output and args.output != "-":
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".parquet", ".geoparquet"):
      import geopandas as gpd
      feats = result["features"] if result.get("type") == "FeatureCollection" else [result]
      gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
      gdf.to_parquet(out_path)
    else:
      indent = 2 if args.pretty else None
      json_output = json.dumps(result, indent=indent)
      with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_output)
        f.write("\n")
    print(
        f"Successfully delineated {len(coords_to_process)} catchment(s) to {out_path}",
        file=sys.stderr,
    )
  else:
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))

  return 0


if __name__ == "__main__":
  sys.exit(main())
