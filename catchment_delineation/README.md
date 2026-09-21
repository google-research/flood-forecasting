# Catchment Delineation (`catchment_delineation`)

This module creates watershed boundary polygons and calculates drainage areas ($\text{km}^2$) for streamflow gauges using 90-meter (3-arc-second) flow-direction maps.

You can use it from the command line (`delineate-catchment`) or from Python (`DemDelineator`) to prepare basin boundary polygons for Caravan static attribute extraction and MultiMet weather forcing extraction.

---

## 1. How It Works (Brief Overview)

Given the latitude and longitude of a river gauge, the tool produces its watershed polygon in three steps:

1. **Snaps to the river channel:** Gauge coordinates recorded by water agencies are often a few hundred meters off the center of the river on a digital map. The tool searches nearby 90-meter pixels (`12` cells ≈ `1.1 km` by default) to place the point on the river channel. If you also provide an approximate expected drainage area (`--expected-area` or `--area-col`), the tool searches up to `80` cells (`~7.2 km`) to find the river channel matching that area.
2. **Traces upstream water flow:** Each 90-meter pixel in a D8 flow-direction map records which of its 8 neighboring pixels water flows into. Starting from the snapped river pixel, the tool follows water flow upstream through every pixel that drains into the gauge. When a river crosses from one 5°×5° map tile (`6000 × 6000` pixels) into neighboring tiles, those tiles are loaded automatically so rivers are never cut off at tile edges.
3. **Saves the polygon:** All upstream pixels are merged into a single boundary polygon and written to **GeoJSON**, **GeoParquet**, or **ESRI Shapefile** along with the calculated drainage area in square kilometers ($\text{km}^2$).

---

## 2. Preparing Map Tiles

Before running `delineate-catchment`, you need a folder (either on your computer or in Google Cloud Storage `gs://`) containing 5°×5° flow-direction `.npy` files (for example, `n35w090_dir.npy`).

* **If you already have a folder or `gs://` bucket of `.npy` tiles:** Pass that folder directly using `--tiles-dir /path/to/tiles_5deg` (or `--gcs-uri gs://... --cache-dir /tmp/tile_cache`).
* **If you are starting from raw HydroSHEDS or MERIT GeoTIFF files:** Run `scripts/slice_continental_dems.py` once to slice the continental `.tif` rasters into 5°×5° `.npy` tiles:

```bash
python scripts/slice_continental_dems.py \
  --source-dir /path/to/raw_hydrosheds_tifs \
  --output-dir /path/to/tiles_5deg
```

---

## 3. Quick Start Examples

### A. Delineate a Single Gauge

```bash
delineate-catchment \
  --lat 39.6828 \
  --lon -88.7729 \
  --id USGS_05592500 \
  --expected-area 480.0 \
  --tiles-dir /path/to/tiles_5deg \
  -o basin.geojson \
  --pretty
```

> **Tip:** Whenever your water agency publishes an approximate drainage area for a gauge, pass it with `--expected-area` (in $\text{km}^2$). This ensures the gauge snaps to the main river rather than a small nearby creek.

### B. Delineate Many Gauges from a CSV or Parquet File

Create a CSV or Parquet table (for example, `gauges.csv`):

```csv
gauge_id,latitude,longitude,area_km2
camels_01013500,47.2374,-68.5826,2252.7
camels_03335500,40.4172,-86.8858,18821.0
```

Run `delineate-catchment` across multiple CPU cores using `--workers`:

```bash
delineate-catchment \
  --csv gauges.csv \
  --area-col area_km2 \
  --tiles-dir /path/to/tiles_5deg \
  --workers 8 \
  -o basins.geoparquet
```

### C. Save in Standard Caravan Folder Structure

If your `gauge_id` column uses the Caravan naming format `<subdataset>_<id>` (such as `camels_01013500` or `grdc_6340110`), add `--preserve-caravan-dirs` and `--output-dir`:

```bash
delineate-catchment \
  --csv gauges.csv \
  --area-col area_km2 \
  --tiles-dir /path/to/tiles_5deg \
  --output-dir /path/to/caravan_dataset \
  --preserve-caravan-dirs \
  --format all \
  --workers 8
```

This creates the exact folder structure expected by Caravan and MultiMet tools:

```text
/path/to/caravan_dataset/
└── shapefiles/
    └── camels/
        ├── camels_basin_shapes.geoparquet
        ├── camels_basin_shapes.geojson
        ├── camels_basin_shapes.shp
        ├── camels_basin_shapes.shx
        ├── camels_basin_shapes.dbf
        ├── camels_basin_shapes.prj
        └── camels_basin_shapes.cpg
```

### D. Read Map Tiles from Google Cloud Storage (`gs://`)

When your `.npy` tiles live in a Google Cloud Storage bucket, pass both `--gcs-uri` and a local `--cache-dir` where downloaded tiles can be stored. Add `--clean-cache` to delete the downloaded tiles automatically when the command finishes:

```bash
delineate-catchment \
  --csv gauges.csv \
  --area-col area_km2 \
  --gcs-uri gs://your-bucket/tiles_5deg \
  --cache-dir /tmp/dem_tile_cache \
  --clean-cache \
  --workers 8 \
  -o basins.geoparquet
```

### E. Use from Python

```python
from catchment_delineation import DemDelineator

delineator = DemDelineator(tiles_dir="/path/to/tiles_5deg")

feature = delineator.delineate(
    lat=39.6828,
    lon=-88.7729,
    catchment_id="USGS_05592500",
    expected_area_km2=480.0,  # optional expected area in km²
)

print("Gauge ID:", feature["properties"]["catchment_id"])
print("Area (km²):", feature["properties"]["area_km2"])
```

---

## 4. What to Watch Out For (Common Pitfalls)

1. **You must provide all file paths yourself (no hidden defaults)**
   The tool never guesses file locations and never falls back to default paths. Always pass your tile location (`--tiles-dir` or `--gcs-uri` + `--cache-dir`) and your output destination (`-o` or `--output-dir`).

2. **Gauges on wide rivers need `--expected-area` or `--area-col`**
   On a 90-meter map, a wide river (such as the Danube, Mississippi, or Amazon) spans many grid cells across its width. A gauge coordinate near the riverbank can sit closer to a tiny creek on the bank than to the main river channel in the middle of the river.
   * Passing `--expected-area` (or `--area-col` for CSV tables) tells the tool to find the nearby channel whose drainage area matches your expected area (within `±50%` by default, controlled by `--area-tolerance`).
   * **Loud error if no river matches:** If you supply an expected area and no river within `~7.2 km` matches that area, the tool **will not guess or output a bad polygon**. It logs a `[AREA HINT FAILURE]` error and refuses to output a polygon (`CatchmentAreaMismatchError`).

3. **Latitude limit (`-56°S` to `60°N`)**
   The 90-meter HydroSHEDS maps cover latitudes from `-56°S` to `60°N`.
   * If a gauge is north of `60°N` (such as northern Scandinavia, Alaska, or northern Canada), it is outside map coverage.
   * If a gauge sits south of `60°N` (for example at `59.8°N`) but its upstream headwaters cross north of `60°N`, the tool stops rather than cutting the river off at the `60°N` border. In batch runs, these gauges are marked `status: "out_of_coverage"` with `geometry: null`.

4. **All upstream map tiles must be in your tile folder**
   Large rivers can start hundreds of kilometers away and cross several 5°×5° map tiles. If your `--tiles-dir` contains the tile for the gauge location but is missing an upstream tile that drains into that river, the tool stops with an error instead of returning an incomplete polygon.

5. **Coordinate order and units**
   * **Coordinates:** Standard decimal degrees (`EPSG:4326` / WGS84), with **latitude first** (`-56` to `60`) and **longitude second** (`-180` to `180`).
   * **Drainage areas:** All area inputs (`--expected-area`, `--area-col`) and outputs (`area_km2`, `area`) are in **square kilometers ($\text{km}^2$)**.

---

## 5. Complete Command-Line Arguments (`delineate-catchment`)

### Coordinate Input Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--lat` | float | `None` | Latitude of a single river gauge or outlet point in decimal degrees (e.g., `39.6828`). Must be used with `--lon`. |
| `--lon` | float | `None` | Longitude of a single river gauge or outlet point in decimal degrees (e.g., `-88.7729`). Must be used with `--lat`. |
| `--id` | string | `None` | Custom gauge ID when running a single point with `--lat` and `--lon` (e.g., `USGS_05592500`). If omitted, an ID is generated from the coordinates. |
| `--coords` | string(s) | `None` | One or more space-separated `"lat,lon"` pairs for running a few points without a CSV file (e.g., `--coords "39.68,-88.77" "40.42,-86.89"`). |
| `--csv` | path / URI | `None` | Path (local or `gs://`) to a CSV or Parquet table of gauge coordinates, or a Caravan directory containing `attributes/` tables. |
| `--lat-col` | string | Auto | Name of the latitude column in `--csv`. Only needed if your column is not named `latitude`, `lat`, `gauge_lat`, `caravan:gauge_lat`, `outlet_lat`, or `pour_point_lat`. |
| `--lon-col` | string | Auto | Name of the longitude column in `--csv`. Only needed if your column is not named `longitude`, `lon`, `long`, `lng`, `gauge_lon`, `caravan:gauge_lon`, `outlet_lon`, or `pour_point_lon`. |
| `--id-col` | string | Auto | Name of the gauge ID column in `--csv`. Only needed if your column is not named `gauge_id`, `catchment_id`, `station_id`, `hybas_id`, `id`, or `caravan:gauge_id`. |
| `--workers`, `-w` | int | `1` | Number of CPU processes to run in parallel during batch runs (e.g., `--workers 8`). |

### Map Tile & River Snapping Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--tiles-dir` | path / URI | `None` | Folder (local path or `gs://` URI) containing 5°×5° `.npy` flow-direction tiles. You must provide either `--tiles-dir` or `--gcs-uri`. |
| `--gcs-uri` | URI | `None` | Google Cloud Storage folder (`gs://...`) containing 5°×5° `.npy` tiles. Must be used with `--cache-dir`. |
| `--cache-dir` | path | `None` | Local folder where tiles downloaded from Google Cloud Storage are stored. Required whenever reading tiles from `gs://`. |
| `--snap-window` | int | `12` | Half-width of the search box (in 90-meter pixels) around the gauge coordinate used to snap onto the nearest river channel (`12` cells ≈ `1.1 km`). |
| `--expected-area` | float | `None` | Optional expected drainage area in $\text{km}^2$ for a single gauge (or applied to all gauges if `--area-col` is not set). Searches up to `80` cells (`~7.2 km`) for a matching river channel; raises an error if none matches. |
| `--area-col` | string | `None` | Optional column name in `--csv` containing the expected drainage area in $\text{km}^2$ for each gauge. |
| `--area-tolerance` | float | `0.50` | Allowed relative difference between delineated area and expected area (`0.50` = within `±50%`, i.e., `0.5×` to `1.5×`). |
| `--max-cells` | int | `None` | Optional upper limit on upstream 90-meter pixels traced. Default is `None` (no limit — large rivers are never cut off). |

### Output & Utility Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-o`, `--output` | path / URI | `stdout` | Output file path (`.geojson`, `.json`, `.geoparquet`, `.parquet`, or `.shp`). Prints GeoJSON to the terminal if omitted. |
| `--output-dir` | path / URI | `None` | Output root folder for multi-format or Caravan-structured outputs. |
| `--preserve-caravan-dirs` | flag | `False` | Groups gauges by the prefix in `gauge_id` (before the first `_`) and writes files into `<output-dir>/shapefiles/<subdataset>/<subdataset>_basin_shapes.*`. |
| `--format` | choice | `all` | File format(s) to write when using `--output-dir` or `--preserve-caravan-dirs`: `all`, `geoparquet`, `parquet`, `geojson`, or `shp`. |
| `--pretty` | flag | `False` | Formats GeoJSON output with indentation and line breaks. |
| `--clean-cache` | flag | `False` | Deletes only the `.npy` tile files downloaded into `--cache-dir` during this run. |
| `--list-tiles` | flag | `False` | Lists all `.npy` tile files available in `--tiles-dir` and exits. |

---

## 6. Benchmark CLI (`benchmark-catchment`) & Accuracy Summary

You can evaluate delineation accuracy against a reference dataset of published gauge polygons using `benchmark-catchment`:

```bash
benchmark-catchment \
  --dataset /path/to/benchmark_basins_1000.parquet \
  --tiles-dir /path/to/tiles_5deg \
  --workers 16 \
  --output benchmark_results.csv
```

### Benchmark CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | path / URI | Required | Path (`.parquet` or `gs://`) to the benchmark reference dataset. |
| `--tiles-dir` | path / URI | `None` | Local folder (or `gs://` URI) containing 5°×5° `.npy` flow-direction tiles. |
| `--gcs-uri` | URI | `None` | Google Cloud Storage URI containing 5°×5° `.npy` tiles (requires `--cache-dir`). |
| `--cache-dir` | path | `None` | Local folder for storing tiles downloaded from `--gcs-uri`. |
| `--output` | path | `None` | Optional file path (`.csv` or `.parquet`) to save per-basin metrics. |
| `--workers` | int | `8` | Number of parallel worker processes. |
| `--samples` | int | `None` | Optional number of basins to randomly sample (balanced across continents and basin sizes). |
| `--continents` | string(s) | `None` | Filter benchmark to specific continents (e.g., `--continents Europe "North America"`). |
| `--size-tiers` | string(s) | `None` | Filter benchmark to specific basin size buckets (`1_micro`, `2_small`, `3_medium`, `4_large`, `5_macro`). |
| `--snap-window` | int | `12` | Search window half-width in 90-meter pixels around each gauge. |
| `--no-area-hint` | flag | `False` | Disable using `reference_area_km2` as `--expected-area` during the benchmark (enabled by default). |
| `--area-tolerance` | float | `0.50` | Allowed relative tolerance around expected area (`0.50` = `±50%`). |
| `--clean-cache` | flag | `False` | Delete only the `.npy` tile files downloaded during the benchmark run. |

### Global 1,200-Basin Benchmark Results

Evaluated across `1,200` global gauges (`1,127` within the `[-56°S, 60°N]` HydroSHEDS coverage bounds, plus `73` gauges above `60°N` flagged as `out_of_coverage`):

| Basin Size Bucket | In-Coverage Basins | Median IoU | Median Dice | Basins with $\text{IoU} \ge 0.80$ | Median Area Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1_micro** ($< 10\text{ km}^2$) | 132 | **0.916** | **0.956** | **87.1%** (115 / 132) | 5.29% |
| **2_small** ($10\text{–}100\text{ km}^2$) | 172 | **0.955** | **0.977** | **98.3%** (169 / 172) | 2.68% |
| **3_medium** ($100\text{–}1,000\text{ km}^2$) | 222 | **0.975** | **0.988** | **98.2%** (218 / 222) | 1.88% |
| **4_large** ($1,000\text{–}10,000\text{ km}^2$) | 282 | **0.984** | **0.992** | **98.9%** (279 / 282) | 1.38% |
| **5_macro** ($> 10,000\text{ km}^2$) | 319 | **0.991** | **0.995** | **99.4%** (317 / 319) | 0.96% |
| **All In-Coverage** | **1,127** | **0.976** | **0.988** | **97.4%** (1,098 / 1,127) | **1.80%** |
