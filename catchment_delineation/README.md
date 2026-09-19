# Catchment Delineation

The `catchment_delineation` package finds the boundary (polygon) and drainage area ($\text{km}^2$) of the watershed that drains to any latitude and longitude point.

You give it:
1. One or more `(latitude, longitude)` coordinates (for example, the location of a streamflow gauge), and
2. The path to a folder (or Google Cloud Storage `gs://` bucket) containing 5°×5° flow-direction `.npy` tiles.

It traces every grid cell that flows into your point and saves the resulting watershed boundary as a **GeoJSON**, **GeoParquet**, or **ESRI Shapefile**.

---

## 1. Key Principles

* **You provide all file paths:** Nothing is hardcoded. The tool only reads from the tile folder or `gs://` bucket you specify, and only writes to the output path you specify.
* **No silent fallbacks or made-up numbers:**
  * If a required map tile is missing or unreadable, the program stops immediately with an error.
  * If a river basin extends past the latitude limits of the elevation map (`-56°S` to `60°N`), it aborts rather than saving a chopped-off polygon.
  * In batch runs over a CSV table, any gauge with missing (`NaN`) or out-of-bounds coordinates is recorded with `geometry = None` and `area = NaN` (never filled with `0.0`).
* **Optional expected drainage area hint (`--expected-area` / `--area-col`):**
  Streamflow gauge coordinates are often slightly off the center of a river on a 90-meter grid — especially on wide rivers (like the Amazon or Danube) or when coordinates were rounded to two decimal places. If you know the approximate drainage area reported by a water agency, you can pass it as a hint. The tool will look for the river channel that matches that area (within `±50%` by default). **If no nearby river matches your area hint, the tool logs a clear error and refuses to save a polygon.**

---

## 2. What Data You Need

The delineator uses 3-arc-second (~90 m resolution) ESRI D8 flow-direction grids sliced into 5°×5° `.npy` files (`uint8`, shape `6000 × 6000`), named by their top-left (north-west) corner, such as `n40w090.npy` (covering `35°N–40°N`, `90°W–85°W`).

Each cell stores which of its 8 neighbors water flows into (`1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE`).

You can either:
* Store the `.npy` tiles in a local folder and pass `--tiles-dir /path/to/tiles`, or
* Point to a Google Cloud Storage bucket with `--gcs-uri gs://my-bucket/tiles_5deg` and provide a local folder `--cache-dir /path/to/cache` where downloaded tiles should be stored.

*(If you have continental HydroSHEDS GeoTIFFs such as `na_dir_3s.tif`, you can slice them into 5°×5° `.npy` tiles yourself using `python scripts/slice_continental_dems.py --input-tifs na_dir_3s.tif --out-dir /path/to/tiles`.)*

---

## 3. Command-Line Usage (`delineate-catchment`)

After installing the repository (`pip install -e .` inside the `googlehydrology` Conda environment), you can run `delineate-catchment` from your terminal.

### A. Single Point (Local Tiles)

```bash
delineate-catchment \
  --lat 39.6828 \
  --lon -88.7729 \
  --tiles-dir /path/to/tiles_5deg \
  -o dalton_city.geojson \
  --pretty
```

### B. Single Point with an Expected Area Hint

If your gauge sits on the bank of a wide river and you know its approximate drainage area (for example, `101,750 km²`), pass `--expected-area`:

```bash
delineate-catchment \
  --lat 48.25 \
  --lon 16.30 \
  --expected-area 101750 \
  --tiles-dir /path/to/tiles_5deg \
  -o danube_vienna.geojson
```

If no river channel near `(48.25, 16.30)` has an area within `±50%` of `101,750 km²`, the command prints `[AREA HINT FAILURE]` to `stderr`, exits with code `1`, and does not create `danube_vienna.geojson`.

### C. Reading Tiles from Google Cloud Storage

When reading tiles from a `gs://` bucket, pass both `--gcs-uri` and `--cache-dir`. Add `--clean-cache` if you want the tool to delete the tiles it downloaded when the run finishes (it will only delete the files it created, never any pre-existing files):

```bash
delineate-catchment \
  --lat 39.6828 \
  --lon -88.7729 \
  --gcs-uri gs://your-bucket/tiles_5deg \
  --cache-dir /tmp/dem_tile_cache \
  --clean-cache \
  -o dalton_city.geojson
```

### D. Batch Delineation from a CSV or Parquet Table

Prepare a CSV or Parquet file with explicit latitude and longitude columns (such as `latitude` and `longitude`, or `gauge_lat` and `gauge_lon`):

```csv
gauge_id,latitude,longitude,expected_area_km2
camels_01013500,47.2374,-68.5826,2252.7
camels_03335500,40.4172,-86.8858,18821.0
```

Run across multiple CPU cores with `--workers` and optionally pass `--area-col`:

```bash
delineate-catchment \
  --csv gauges.csv \
  --area-col expected_area_km2 \
  --tiles-dir /path/to/tiles_5deg \
  --workers 8 \
  -o basins.geoparquet
```

### E. Saving in Standard Caravan Folder Structure (`--preserve-caravan-dirs`)

If your `gauge_id` values follow the Caravan `<subdataset>_<id>` naming convention (such as `camels_01013500` or `grdc_6340110`), adding `--preserve-caravan-dirs` organizes the output files into the standard Caravan `shapefiles/<subdataset>/` hierarchy:

```bash
delineate-catchment \
  --csv gauges.csv \
  --tiles-dir /path/to/tiles_5deg \
  --output-dir /path/to/caravan_output \
  --preserve-caravan-dirs \
  --format all \
  --workers 8
```

This writes:
* `/path/to/caravan_output/shapefiles/camels/camels_basin_shapes.geoparquet`
* `/path/to/caravan_output/shapefiles/camels/camels_basin_shapes.geojson`
* `/path/to/caravan_output/shapefiles/camels/camels_basin_shapes.shp` (plus `.shx`, `.dbf`, `.prj`, `.cpg`)

---

## 4. Python API Usage

### Single Point

```python
from catchment_delineation import DemDelineator

delineator = DemDelineator(tiles_dir='/path/to/tiles_5deg')

feature = delineator.delineate(
    lat=39.6828,
    lon=-88.7729,
    catchment_id='USGS_05592500',
    expected_area_km2=480.0,  # optional area hint in km²
)

print('Catchment ID:', feature['properties']['catchment_id'])
print('Area (km²):', feature['properties']['area_km2'])
print('Upstream Cells:', feature['properties']['upstream_cells_count'])
```

### Multiple Points (Batch)

```python
from catchment_delineation import DemDelineator

delineator = DemDelineator(tiles_dir='/path/to/tiles_5deg')

coords = [(39.6828, -88.7729), (40.4172, -86.8858)]
ids = ['camels_05592500', 'camels_03335500']
expected_areas = [480.0, 18821.0]  # optional

feature_collection = delineator.delineate_batch(
    coords=coords,
    ids=ids,
    expected_areas_km2=expected_areas,
)

for feat in feature_collection['features']:
    props = feat['properties']
    print(props['catchment_id'], props['area_km2'], props['status'])
```

---

## 5. CLI Options Reference (`delineate-catchment`)

| Option | Type | Default | What It Does |
| :--- | :--- | :--- | :--- |
| `--lat` | float | `None` | Latitude of the outlet point |
| `--lon` | float | `None` | Longitude of the outlet point |
| `--coords` | string(s) | `None` | Space-separated `"lat,lon"` coordinate pairs |
| `--csv` | path / `gs://` | `None` | Path to a CSV/Parquet table (or a Caravan directory containing `attributes_other_*.csv`) |
| `--id` | string | `None` | Custom ID for a single-point run |
| `--lat-col` | string | auto | Exact latitude column name in `--csv` |
| `--lon-col` | string | auto | Exact longitude column name in `--csv` |
| `--id-col` | string | auto | Exact ID column name in `--csv` |
| `--expected-area` | float | `None` | Optional expected drainage area ($\text{km}^2$) for single-point snapping |
| `--area-col` | string | `None` | Optional column in `--csv` containing expected drainage area ($\text{km}^2$) |
| `--area-tolerance` | float | `0.50` | Allowed relative difference around expected area (`0.50` = `±50%`) |
| `--tiles-dir` | path / `gs://` | `None` | Folder (or `gs://` URI) containing 5°×5° `.npy` flow-direction tiles |
| `--gcs-uri` | `gs://` URI | `None` | Google Cloud Storage URI containing 5°×5° `.npy` tiles (requires `--cache-dir`) |
| `--cache-dir` | path | `None` | Local folder used to store tiles downloaded from `--gcs-uri` |
| `--snap-window` | int | `12` | Half-width of initial channel search window in grid cells (`12` cells ≈ `1.1 km`) |
| `--max-cells` | int | `None` | Optional hard cap on upstream cells (default `None`: no river is ever cut off) |
| `--workers`, `-w` | int | `1` | Number of parallel CPU processes for batch runs |
| `-o`, `--output` | path / `gs://` | stdout | Output file path (`.geojson`, `.geoparquet`, `.parquet`, `.shp`) |
| `--output-dir` | path / `gs://` | `None` | Output directory for partitioned outputs |
| `--preserve-caravan-dirs` | flag | `False` | Save outputs under `<output-dir>/shapefiles/<subdataset>/<subdataset>_basin_shapes.*` |
| `--format` | choice | `all` | Which file formats to write in `--output-dir`: `all`, `geoparquet`, `geojson`, `shp` |
| `--clean-cache` | flag | `False` | Delete only the `.npy` tile files downloaded during this run |
| `--pretty` | flag | `False` | Indent JSON output for easier reading |
| `--list-tiles` | flag | `False` | List `.npy` tile files in `--tiles-dir` and exit |

---

## 6. Benchmarking Against Reference Polygons (`benchmark-catchment`)

If you have a reference Parquet table of known watershed polygons (with columns `gauge_id`, `continent`, `hemisphere`, `size_tier`, `latitude`, `longitude`, `reference_area_km2`, `geometry_wkt`), you can compare the delineator's polygons against the reference polygons using `benchmark-catchment`:

```bash
benchmark-catchment \
  --dataset /path/to/benchmark_basins_1000.parquet \
  --tiles-dir /path/to/tiles_5deg \
  --workers 16 \
  --output benchmark_results.csv
```

On the 1,200-basin global evaluation set (`1,127` basins within the `-56°` to `60°` DEM domain across 6 continents):
* **Median Intersection-over-Union (IoU):** `0.976`
* **Median Dice Score:** `0.988`
* **Share of basins with IoU ≥ 0.80:** `97.4%`
* **Median Absolute Area Error:** `1.8%`

---

## 7. Running the Unit Tests

```bash
pytest test/test_catchment_delineation.py -v
```

All 19 tests create temporary tiles inside pytest's temporary folder (`tmp_path`) and do not require internet access or cloud credentials.
