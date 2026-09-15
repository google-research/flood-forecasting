# Catchment Delineation from DEM Flow Direction

High-performance, pure DEM flow-direction watershed delineation module. Performs authentic reverse-flow BFS graph traversal on high-resolution (90m / 3 arc-second) D8 flow-direction matrices (HydroSHEDS / MERIT Hydro) with seamless cross-tile boundary routing.

## Features

- **Pure DEM D8 Flow Routing**: Authentic reverse-flow traversal on 3 arc-second (~90m) D8 matrices.
- **Cross-Tile Routing**: Seamlessly handles river basins crossing arbitrary 5x5 degree tile boundaries without boundary artifacts.
- **Channel Snapping**: Automatically snaps clicked outlet coordinates to the nearest stream channel using bounded BFS upstream connectivity scoring.
- **Memory-Mapped I/O**: Direct `np.load(..., mmap_mode="r")` streaming from disk for instant retrieval and minimal memory footprint.
- **Vectorization**: Uses run-length row fusion and Shapely unary union to produce clean, simplified GeoJSON polygons.
- **Batch Processing**: Supports single coordinates, coordinate lists, or batch CSV files.
- **CLI & Python API**: Zero complex server dependencies; run as a CLI or import directly.

---

## Installation & Requirements

Ensure `numpy` and `shapely` are installed in your environment:

```bash
pip install numpy shapely
```

To install as an editable package within `googlehydrology`:

```bash
pip install -e .
```

---

## Data Setup (DEM Tiles)

In this open source repository, DEM data comes exclusively from the Google Cloud Storage bucket:
```bash
gs://open-multimet/data/DEMs/tiles_5deg/
```

Required 5x5 degree tiles (`.npy`) are automatically retrieved from the gs bucket and cached locally in `~/.cache/googlehydrology/dem/`.

### Custom User Paths
Users can supply their own local tile directory if desired using the `--tiles-dir` command line flag or `tiles_dir` parameter in Python:
- **CLI**: `delineate-catchment --lat 39.6828 --lon -88.7729 --tiles-dir /path/to/custom/tiles`
- **Python**: `DemDelineator(tiles_dir="/path/to/custom/tiles")`

When a custom path is supplied, tiles are loaded strictly from that path. There is no other candidate path searching.

To check available cached tiles on your system:

```bash
python -m catchment_delineation --list-tiles
```

---

## Python API Usage

### 1. Delineating a Single Catchment

```python
from catchment_delineation import DemDelineator, delineate_dem

# Using the DemDelineator class
delineator = DemDelineator()
watershed = delineator.delineate(lat=39.6828, lon=-88.7729)

print("Catchment ID:", watershed["properties"]["catchment_id"])
print("Area (km²):", watershed["properties"]["area_km2"])
print("Upstream Cells:", watershed["properties"]["upstream_cells_count"])
print("Geometry type:", watershed["geometry"]["type"])

# Or using the convenience function
watershed = delineate_dem(lat=39.6828, lon=-88.7729)
```

### 2. Delineating Multiple Catchments

```python
from catchment_delineation import DemDelineator

delineator = DemDelineator()
coords = [(39.6828, -88.7729), (40.4172, -86.8858)]
feature_collection = delineator.delineate_batch(coords)

for feat in feature_collection["features"]:
    props = feat["properties"]
    print(f"{props['catchment_id']}: {props['area_km2']} km²")
```

---

## CLI Usage

You can run the delineation tool directly via `python -m catchment_delineation` or via the installed console script `delineate-catchment`.

### Single Coordinate Point

```bash
# Output GeoJSON to stdout
python -m catchment_delineation --lat 39.6828 --lon -88.7729 --pretty

# Save GeoJSON to file
python -m catchment_delineation --lat 39.6828 --lon -88.7729 -o dalton_city_basin.geojson
```

### Multiple Coordinate Pairs

```bash
python -m catchment_delineation \
  --coords "39.6828,-88.7729" "40.4172,-86.8858" \
  -o multi_basins.geojson
```

### Batch Processing from CSV

Given a CSV file `gauges.csv`:
```csv
id,latitude,longitude
USGS_1,39.6828,-88.7729
USGS_2,40.4172,-86.8858
```

Run:
```bash
python -m catchment_delineation --csv gauges.csv -o delineated_basins.geojson
```

### Options Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--lat` | Latitude of outlet point | None |
| `--lon` | Longitude of outlet point | None |
| `--coords` | One or more `lat,lon` pairs | None |
| `--csv` | Path to CSV with coordinate columns | None |
| `-o`, `--output` | Output GeoJSON filepath | stdout |
| `--tiles-dir` | Path to directory containing `.npy` tiles | Auto-detected |
| `--snap-window` | Channel snap search half-window in cells | `4` (~360m) |
| `--max-cells` | Safety limit for BFS traversal cells | `5000000` |
| `--id` | Custom catchment ID | Auto-generated |
| `--pretty` | Pretty-print output JSON | False |
| `--list-tiles` | List available tiles in directory | False |
