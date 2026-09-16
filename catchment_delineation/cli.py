"""Command-Line Interface for DEM Catchment Delineation.

Allows users to delineate upstream drainage catchment boundaries from a DEM by
supplying a single coordinate pair, a list of coordinates, or a CSV file.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from catchment_delineation.config import GCS_TILES_URI, get_default_cache_dir
from catchment_delineation.delineator import CatchmentCoverageError, DemDelineator
from catchment_delineation.tiles import list_available_tiles


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


def load_coords_from_csv(
    csv_path: Path,
) -> Tuple[List[Tuple[float, float]], List[Optional[str]]]:
  """Parses lat/lon coordinates and optional IDs from a CSV file."""
  coords = []
  ids = []
  with open(csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
      raise ValueError(f"CSV file '{csv_path}' is empty or has no header.")

    # Find latitude column
    lat_col = next(
        (c for c in reader.fieldnames if c.lower() in ("lat", "latitude", "y")),
        None,
    )
    # Find longitude column
    lon_col = next(
        (
            c
            for c in reader.fieldnames
            if c.lower() in ("lon", "longitude", "long", "x")
        ),
        None,
    )
    # Optional ID column
    id_col = next(
        (
            c
            for c in reader.fieldnames
            if c.lower()
            in ("id", "gauge_id", "station_id", "catchment_id", "name")
        ),
        None,
    )

    if not lat_col or not lon_col:
      raise ValueError(
          f"CSV file must have latitude and longitude columns. Found: {reader.fieldnames}"
      )

    for row in reader:
      lat_val = float(row[lat_col].strip())
      lon_val = float(row[lon_col].strip())
      id_val = row[id_col].strip() if id_col and row.get(id_col) else None
      coords.append((lat_val, lon_val))
      ids.append(id_val)

  return coords, ids


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
      help="Path to CSV file with latitude and longitude columns.",
  )
  coord_group.add_argument(
      "--id",
      type=str,
      default=None,
      help="Custom catchment ID (for single coordinate delineation).",
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
      default=4,
      help="Snap search window half-width in cells (default: 4 cells ~360m).",
  )
  config_group.add_argument(
      "--max-cells",
      type=int,
      default=5000000,
      help="Maximum upstream cells safety limit (default: 5,000,000).",
  )

  output_group = parser.add_argument_group("Output Options")
  output_group.add_argument(
      "-o",
      "--output",
      type=str,
      default=None,
      help="Output path for GeoJSON file. If omitted, prints GeoJSON to stdout.",
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
    csv_coords, csv_ids = load_coords_from_csv(Path(args.csv))
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

  indent = 2 if args.pretty else None
  json_output = json.dumps(result, indent=indent)

  if args.output and args.output != "-":
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(json_output)
      f.write("\n")
    print(
        f"Successfully delineated {len(coords_to_process)} catchment(s) to {out_path}",
        file=sys.stderr,
    )
  else:
    print(json_output)

  return 0


if __name__ == "__main__":
  sys.exit(main())
