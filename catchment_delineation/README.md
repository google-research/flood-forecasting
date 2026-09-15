# Catchment Delineation from DEM Flow Direction

High-performance, pure DEM flow-direction watershed delineation module for `googlehydrology` and OpenHydroNet. Performs authentic reverse-flow BFS graph traversal on high-resolution (90m / 3 arc-second) D8 flow-direction matrices with seamless cross-tile boundary routing.

---

## 1. What This Module Does

- **Pure DEM D8 Flow Routing**: Performs exact reverse-flow breadth-first search (BFS) graph traversal on 3 arc-second (~90m at equator, 1/1200° cell size) D8 flow direction matrices (HydroSHEDS v1.4 / MERIT Hydro).
- **Seamless Multi-Tile Traversal**: Dynamically traverses across 5°×5° tile boundaries without edge truncation or boundary artifacts, reconstructing the true natural watershed basin geometry regardless of river size.
- **Channel Snapping**: Automatically snaps clicked or gauge coordinates to the nearest stream channel outlet cell using a bounded local BFS connectivity scoring algorithm.
- **Accurate Geodesic Area**: Sums raster cell ground footprints with exact latitude scaling (`lat_scale * lon_scale`) to compute geodesic drainage area in $\text{km}^2$.
- **Fast Run-Length Vectorization**: Combines row run-length raster interval fusion with Shapely `unary_union` and adaptive boundary simplification to produce clean, valid GeoJSON Polygon / MultiPolygon geometries.
- **Zero Heavy Dependencies**: Requires only `numpy` and `shapely` (no GDAL, GIS servers, or heavy C-extensions).

---

## 2. Data Sources & Storage Paths

In this open-source repository, DEM data comes exclusively from the official Google Cloud Storage bucket:

```text
gs://open-multimet/data/DEMs/tiles_5deg/
```

### Where the Paths Are

| Resource | Path / URI | Description |
| :--- | :--- | :--- |
| **GCS Bucket (Remote)** | `gs://open-multimet/data/DEMs/tiles_5deg/` | Hosted **763** pre-sliced 5°×5° D8 flow-direction tiles (`uint8`, shape `(6000, 6000)`, ~34 MB each) covering all habitable continents |
| **GCS Master DEMs** | `gs://open-multimet/data/DEMs/{na,sa,eu,af,as,au}_dir_3s.tif` | Full continental HydroSHEDS 3-arc-second master flow direction GeoTIFFs |
| **GCS Benchmark Catalog**| `gs://open-multimet/data/DEMs/benchmark_basins_1000.parquet` | Stratified global evaluation catalog of 1,200 validated reference catchments |
| **GCS Elevation** | `gs://open-multimet/data/DEMs/elevation_tiles_5deg/` | Conditioned elevation tiles (`int16`, 119 files, 8.0 GB) |
| **Local Cache** | `~/.cache/googlehydrology/dem/` | Default local directory where required tiles are cached automatically on first use |
| **Custom Path** | `--tiles-dir <path>` or `tiles_dir="<path>"` | Optional user-supplied directory containing local `.npy` tiles |

### Automatic Retrieval & Strict Path Handling

- **Default Behavior**: When no custom path is provided, `DemDelineator` checks the local cache (`~/.cache/googlehydrology/dem/`). If a required tile is not present locally, it is downloaded on demand directly from `gs://open-multimet/data/DEMs/tiles_5deg/`.
- **Custom User Directory**: Users can specify `--tiles-dir /path/to/my/tiles` on the command line or pass `tiles_dir="/path/to/my/tiles"` to `DemDelineator`. When specified, tiles are loaded strictly from that path.
- **No Path Searching**: There is no candidate path scanning or fallback searching. Tiles are sourced strictly from the GCS bucket or from the user's explicit path.

---

## 3. Installation

Ensure dependencies are installed in your Python environment:

```bash
pip install numpy shapely
```

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

---

## 4. How to Use: Python API

### A. Direct Access via `googlehydrology`

```python
import googlehydrology

# Delineate a single catchment (e.g., Dalton City, IL)
basin = googlehydrology.delineate_dem(lat=39.6828, lon=-88.7729)

print("Catchment ID:", basin["properties"]["catchment_id"])
print("Area (km²):", basin["properties"]["area_km2"])
print("Upstream Cells:", basin["properties"]["upstream_cells_count"])
print("Geometry Type:", basin["geometry"]["type"])
```

### B. Using `DemDelineator` Class

```python
from catchment_delineation import DemDelineator

# Uses official GCS bucket with local cache (~/.cache/googlehydrology/dem/)
delineator = DemDelineator()

# Delineate a single point
feature = delineator.delineate(
    lat=39.6828,
    lon=-88.7729,
    snap_window_cells=4,  # Half-width of snap search window (~360m)
    catchment_id="USGS_05592500",
)

# Output is a standard GeoJSON Feature dict
print(feature["properties"])
```

### C. Batch Processing Multiple Coordinates

```python
from catchment_delineation import DemDelineator

delineator = DemDelineator()

coords = [(39.6828, -88.7729), (40.4172, -86.8858)]
ids = ["DALTON_CITY", "LAFAYETTE"]

feature_collection = delineator.delineate_batch(coords, ids=ids)

for feat in feature_collection["features"]:
  props = feat["properties"]
  print(f"{props['catchment_id']}: {props['area_km2']:.1f} km²")
```

### D. Using a Custom Local Tiles Directory

```python
from catchment_delineation import DemDelineator

# Strict local loading (no GCS download or fallback searching)
delineator = DemDelineator(tiles_dir="/path/to/custom/tiles")
feature = delineator.delineate(lat=39.6828, lon=-88.7729)
```

---

## 5. How to Use: Command-Line Interface (CLI)

The package installs the `delineate-catchment` console script (or use `python -m catchment_delineation`).

### Single Coordinate

```bash
# Output GeoJSON directly to stdout
delineate-catchment --lat 39.6828 --lon -88.7729 --pretty

# Save GeoJSON to file
delineate-catchment --lat 39.6828 --lon -88.7729 -o dalton_city.geojson
```

### Multiple Coordinates

```bash
delineate-catchment \
  --coords "39.6828,-88.7729" "40.4172,-86.8858" \
  --pretty -o multi_basins.geojson
```

### Batch Coordinates from CSV

Given `gauges.csv`:
```csv
id,latitude,longitude
USGS_05592500,39.6828,-88.7729
USGS_03335500,40.4172,-86.8858
```

Run:
```bash
delineate-catchment --csv gauges.csv -o delineated_basins.geojson
```

### Using a Custom Local Tiles Directory

```bash
delineate-catchment \
  --lat 39.6828 --lon -88.7729 \
  --tiles-dir /path/to/my/tiles \
  -o custom_basin.geojson
```

### List Available Cached Tiles

```bash
delineate-catchment --list-tiles
```

---

## 6. CLI Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--lat` | float | None | Latitude of pour point outlet |
| `--lon` | float | None | Longitude of pour point outlet |
| `--coords` | string(s) | None | One or more space-separated `"lat,lon"` pairs |
| `--csv` | path | None | CSV file with latitude, longitude, and optional ID columns |
| `--id` | string | None | Custom catchment ID for single-coordinate runs |
| `--tiles-dir` | path | `None` | Custom tile directory. If omitted, tiles are downloaded from `gs://open-multimet/data/DEMs/tiles_5deg/` |
| `--snap-window` | int | `4` | Search window half-width in cells (~360m at 90m resolution) |
| `--max-cells` | int | `5000000` | Traversal safety limit for maximum upstream raster cells |
| `-o`, `--output` | path | stdout | Output GeoJSON file path |
| `--pretty` | flag | False | Pretty-print output JSON with 2-space indentation |
| `--list-tiles` | flag | False | Print available `.npy` tile files and exit |

---

## 7. Global Benchmarking Suite

The package includes a comprehensive global benchmarking runner (`benchmark-catchment`) to evaluate delineation accuracy against official reference catchment polygons:

- **1,200 Balanced Global Basins**: Bundled dataset stratified equally across all 6 continents (200 each in Africa, Asia, Europe, North America, South America, Oceania), all 4 hemisphere quadrants (NW, NE, SW, SE), and 5 size tiers (micro to macro).
- **Core Spatial Metrics**: Computes Intersection-over-Union (IoU / Jaccard Index), Dice similarity coefficient, relative area bias ($\Delta \text{Area} \%$), and stream snapping distances.
- **Hermetic Cloud Execution**: Automatically pulls required 5°×5° tiles on demand from `gs://open-multimet/data/DEMs/tiles_5deg/`.

### Running the Global Benchmark

```bash
# Run full benchmark across 1,000 global basins using 16 workers
benchmark-catchment --samples 1000 --workers 16 -o benchmark_results.csv

# Run specific continents (e.g. Europe and Africa)
benchmark-catchment --continents Europe Africa --workers 8

# Filter by basin size tiers
benchmark-catchment --size-tiers 1_micro 2_small 3_medium 4_large 5_macro
```

### Benchmark CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--samples` | int | `1000` | Number of basins to evaluate (max 1200) |
| `--continents` | strings | None | Subset continents (`Africa`, `Asia`, `Europe`, `North America`, `South America`, `Oceania`) |
| `--size-tiers` | strings | None | Subset size tiers (`1_micro`, `2_small`, `3_medium`, `4_large`, `5_macro`) |
| `--workers` | int | `8` | Number of parallel worker processes |
| `--tiles-dir` | path | `None` | Optional local DEM tile folder (defaults to GCS auto-download) |
| `-o`, `--output` | path | `benchmark_results.csv` | Output file path (.csv or .parquet) for detailed per-basin metrics |

---

## 8. Testing

Run the automated unit test suite with pytest:

```bash
pytest test/test_catchment_delineation.py -v
```
