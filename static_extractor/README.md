# Extracting Static Attributes for Watersheds (`static_extractor`)

To predict river flow in a watershed, OpenHydroNet needs a table of unchanging ("static") facts about that watershed—such as its area, elevation, slope, soil type, land cover, and long-term average weather.

If you are working with watersheds from the published [Caravan](https://www.nature.com/articles/s41597-023-01975-w) dataset, those tables are already provided. If you want to run models on **your own watersheds**, this tool takes a map file of your watershed boundaries and creates a Caravan-compatible CSV table for you.

It calculates these attributes using the same community datasets and methods used by Caravan:
- **[HydroATLAS](https://www.hydrosheds.org/hydroatlas) (BasinATLAS v10, Level 12):** Geography, elevation, slope, soil, land cover, lakes, and human footprint.
- **[ERA5-Land](https://cds.climate.copernicus.eu/) (1981–2020):** Long-term climate averages (precipitation, temperature, evaporation, aridity, snow fraction, and wet/dry spell frequency).

---

## Before You Start

### 1. What Your Input File Needs
- **File format:** A map file containing polygon boundaries for one or more watersheds in **GeoJSON** (`.geojson`), **Shapefile** (`.shp`), or **GeoPackage** (`.gpkg`) format.
- **Coordinates:** Standard latitude and longitude (`EPSG:4326` / WGS84).
- **Watershed ID column:** Each polygon should have a column with a unique name or ID. By default, the tool looks for a column named `gauge_id`, `catchment_id`, `id`, or `basin_id`. If your file uses a different column name, pass `--id-column <column_name>`.

### 2. Automatic Data Download (First Run)
You do not need to download HydroATLAS or ERA5-Land files ahead of time. The first time you run the tool, it automatically downloads the needed reference files (~5.5 GB) from Google Cloud Storage and saves them in `~/.cache/googlehydrology/` so future runs are fast.
- If you want the tool to delete this local cache folder when it finishes, add `--clean-cache`.

### 3. Choosing `--era5-source` (`hybas` vs. `gridded`)
Every run requires the `--era5-source` flag to tell the tool how to calculate climate statistics:

| Option | How It Works | Speed | When to Use It |
| :--- | :--- | :--- | :--- |
| **`--era5-source hybas`** | Combines pre-calculated climate summaries from the standard HydroATLAS sub-basins that overlap your watershed. *(Four ERA5-Land evaporation columns are still read from the daily weather grid when reachable).* | Fast | **Recommended for most users.** Works well whenever your watersheds are roughly the size of standard river sub-basins (~100 $\text{km}^2$) or larger. |
| **`--era5-source gridded`** | Reads 40 years (1981–2020) of daily ERA5-Land weather grids over your exact polygon boundary and calculates every climate number from scratch. | Slower (several seconds per basin) | Use this if your watersheds are very small or have custom boundaries that do not follow natural river sub-basins. |

---

## Command-Line Usage

### 1. Extract Attributes for One File (`extract-caravan-static`)

Use `extract-caravan-static` (or its alias `extract-static-attributes`) when you have a single file of watershed polygons and want a single CSV table of attributes.

```bash
# Using pre-calculated sub-basin climate summaries
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv \
    --era5-source hybas

# Recalculating climate numbers directly from daily ERA5-Land grids
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv \
    --era5-source gridded

# Specifying a custom ID column and running across 8 CPU cores
extract-caravan-static \
    --input /path/to/basins.shp \
    --output /path/to/attributes.csv \
    --era5-source hybas \
    --id-column station_id \
    --workers 8
```

#### All Flags for `extract-caravan-static`

| Flag | Required? | Default | What It Does |
| :--- | :--- | :--- | :--- |
| `--input`, `-i` | **Yes** | — | Path to your input watershed boundary file (`.geojson`, `.shp`, or `.gpkg`). |
| `--output`, `-o` | **Yes** | — | Path where the output CSV file will be saved. |
| `--era5-source` | **Yes** | — | How to calculate ERA5 climate numbers: `hybas` (from pre-calculated sub-basin tables) or `gridded` (from daily ERA5-Land grids). |
| `--id-column` | No | Auto-detected | Name of the column in your input file that holds the watershed ID. If omitted, checks `gauge_id`, `catchment_id`, `id`, and `basin_id`. |
| `--workers`, `-w` | No | `1` | Number of CPU processes to run in parallel. Increase this (for example, `-w 8`) when extracting many watersheds. |
| `--min-overlap-threshold` | No | `0.0` | Minimum overlap area in $\text{km}^2$ required for a sub-basin fragment to be included (unless the watershed covers more than 50% of that sub-basin). Useful for ignoring tiny border slivers along the edge of a polygon. |
| `--gridded-era5-uri` | No | Public GCS store | Custom Google Cloud Storage (`gs://...`) URI or local folder path for the daily ERA5-Land Zarr dataset. |
| `--gdb-path`, `-g` | No | Local cache | Path to an existing local copy of `BasinATLAS_v10.gdb` or `BasinATLAS_v10_lev12.shp` if you already have one on disk. |
| `--era5-cache-dir` | No | Local cache | Folder where downloaded continental ERA5 climate tables are stored. |
| `--cache-dir` | No | `~/.cache/googlehydrology` | Base folder used for storing downloaded reference data. |
| `--auto-download` | No | Enabled | Automatically download missing reference files from Google Cloud Storage. |
| `--no-download` | No | Disabled | Turn off automatic downloads. Use this if you are offline and already have the reference data in `--gdb-path` and `--era5-cache-dir`. |
| `--clean-cache` | No | Disabled | Delete the local cache folder (`~/.cache/googlehydrology`) after the command finishes. |
| `--verbose`, `-v` | No | Disabled | Print detailed progress and debugging messages while running. |

---

### 2. Extract Attributes for Many Folders at Once (`extract-caravan-static-batch`)

Use `extract-caravan-static-batch` (or its alias `extract-static-attributes-batch`) when you have multiple dataset folders containing watershed boundary files and want to process all of them in a single run.

```bash
# Process all dataset folders inside a parent folder and also save one combined CSV
extract-caravan-static-batch \
    --parent-dir /path/to/watershed_folders/ \
    --output-dir /path/to/output_attributes/ \
    --era5-source hybas \
    --workers 16 \
    --combine

# Process a specific list of folders
extract-caravan-static-batch \
    --input-dirs /data/shapes/camels /data/shapes/camelsaus /data/shapes/lamah \
    --output-dir /path/to/output_attributes/ \
    --era5-source hybas \
    --workers 16
```

#### All Flags for `extract-caravan-static-batch`

*You must provide at least one of `--parent-dir`, `--input-dirs`, or `--input-files`.*

| Flag | Required? | Default | What It Does |
| :--- | :--- | :--- | :--- |
| `--parent-dir`, `-p` | One input flag required | — | One or more parent folders (local path or `gs://...`) that contain dataset subfolders of watershed files. |
| `--input-dirs`, `-d` | One input flag required | — | Space-separated list of specific dataset folders to process. |
| `--input-files`, `-f` | One input flag required | — | Space-separated list of specific polygon files (`.shp`, `.geojson`, `.gpkg`) to process. |
| `--output-dir`, `-o` | **Yes** | — | Local folder or Google Cloud Storage (`gs://...`) path where output files will be written. |
| `--era5-source` | **Yes** | — | How to calculate ERA5 climate numbers: `hybas` or `gridded`. |
| `--workers`, `-w` | No | `1` | Number of CPU processes to run in parallel. |
| `--combine` | No | Disabled | In addition to per-dataset files, save a single merged table (`attributes_caravan_combined.csv`) containing all watersheds across all processed folders. |
| `--partition-outputs`, `-P` / `--no-partition-outputs` | No | Auto | When enabled, writes separate HydroATLAS and Caravan climate tables (`attributes_hydroatlas_<dataset>.csv`, `attributes_caravan_<dataset>.csv`, and `attributes_<dataset>.parquet`) inside each dataset's output folder instead of a single flat CSV. |
| `--preserve-caravan-dirs` | No | Disabled | Organizes output files into `<collection>/attributes/<dataset>/` folders to match standard Caravan directory layouts. |
| `--gcs-output-uri` | No | `None` | Optional Google Cloud Storage (`gs://...`) destination where finished files should be uploaded after saving them locally in `--output-dir`. |
| `--no-resume` | No | Disabled | Re-run and overwrite datasets even if their output files already exist in `--output-dir` (by default, already finished datasets are skipped). |
| `--min-overlap-threshold` | No | `0.0` | Minimum overlap area in $\text{km}^2$ required for a sub-basin fragment to be included. |
| `--gridded-era5-uri` | No | Public GCS store | Custom Google Cloud Storage (`gs://...`) URI or local path for the daily ERA5-Land Zarr dataset. |
| `--gdb-path`, `-g` | No | Local cache | Path to an existing local copy of `BasinATLAS_v10.gdb` or shapefile. |
| `--era5-cache-dir` | No | Local cache | Folder where downloaded continental ERA5 climate tables are stored. |
| `--cache-dir` | No | `~/.cache/googlehydrology` | Base folder used for storing downloaded reference data. |
| `--no-download` | No | Disabled | Turn off automatic downloads from Google Cloud Storage. |
| `--clean-staging` | No | Disabled | Delete temporary downloaded input shapefiles after each dataset finishes, while keeping the HydroATLAS and ERA5 reference files cached. |
| `--clean-cache` | No | Disabled | Delete the entire local cache folder (`~/.cache/googlehydrology`) when the batch run finishes. |
| `--verbose`, `-v` | No | Disabled | Print detailed progress and debugging messages while running. |

---

## Using It in Python

### 1. Extract Attributes from a File to a DataFrame and CSV

```python
from static_extractor import StaticAttributesExtractor

extractor = StaticAttributesExtractor(era5_source="hybas")
df = extractor.extract_attributes_from_file(
    input_path="basins.geojson",
    output_csv_path="caravan_attributes.csv",
    id_column="gauge_id",
)
print(df.head())
```

### 2. Extract Attributes for a Single Polygon in Python

```python
import shapely.geometry
from static_extractor import StaticAttributesExtractor

extractor = StaticAttributesExtractor(era5_source="hybas")

# Polygon coordinates in (longitude, latitude)
polygon = shapely.geometry.Polygon([
    [-86.9, 40.4],
    [-86.8, 40.4],
    [-86.8, 40.5],
    [-86.9, 40.5],
    [-86.9, 40.4],
])

result = extractor.extract_attributes_for_polygon(
    polygon,
    catchment_id="my_basin_01",
    era5_source="hybas",  # or "gridded"
)

attrs = result["caravan_attributes"]
print("Drainage Area (km²):", result["total_area_km2"])
print("Mean Elevation (m):", attrs["ele_mt_sav"])
print("Mean Precipitation (mm/yr):", attrs["pre_mm_syr"])
```

### 3. Add Extracted Attributes to a Zarr Store

```python
extractor.append_attributes_to_zarr(
    master_zarr_path="/data/multimet/caravan.zarr",
    basin_id="my_basin_01",
    attributes=attrs,
)
```

---

## Checking Results Against Published Caravan Data (`benchmark-static-extractor`)

If you want to compare the numbers produced by this package against published Caravan values across 490 basins (spanning 7 Caravan datasets and 5 watershed size ranges), run `benchmark-static-extractor`:

```bash
# Compare all 490 basins
benchmark-static-extractor \
    --era5-source hybas \
    --workers 14 \
    -o ./benchmark_results/

# Quick check on a sample of 50 basins
benchmark-static-extractor \
    --era5-source hybas \
    --samples 50 \
    --workers 8 \
    -o ./benchmark_results/
```

This writes three files into `--output-dir`:
- `benchmark_report.md` — a summary report comparing extracted values to published Caravan values.
- `benchmark_attribute_metrics.csv` — one row per attribute showing correlation and error metrics.
- `benchmark_basin_metrics.csv` — one row per basin showing area differences and the attribute with the largest discrepancy.

#### All Flags for `benchmark-static-extractor`

| Flag | Required? | Default | What It Does |
| :--- | :--- | :--- | :--- |
| `--era5-source` | **Yes** | — | How to calculate ERA5 climate numbers during the benchmark: `hybas` or `gridded`. |
| `--output-dir`, `-o` | No | `./benchmark_results` | Folder where the benchmark report and CSV tables are saved. |
| `--samples` | No | All (`490`) | Number of basins to sample if you want a quick check instead of running all 490 basins. |
| `--regions`, `--datasets` | No | All datasets | Space-separated list of Caravan datasets to include (`camels`, `camelsaus`, `camelsbr`, `camelscl`, `camelsgb`, `hysets`, `lamah`). |
| `--size-tiers` | No | All sizes | Space-separated list of basin size groups to include (`1_micro`, `2_small`, `3_medium`, `4_large`, `5_macro`). |
| `--workers` | No | `8` | Number of parallel CPU processes to use. |
| `--dataset` | No | Downloaded automatically | Path to a custom `.parquet` reference dataset file if you are testing your own reference table. |
| `--gdb-path` | No | Local cache | Path to a local copy of `BasinATLAS_v10.gdb` or shapefile. |
| `--era5-cache-dir` | No | Local cache | Folder containing downloaded continental ERA5 climate tables. |
