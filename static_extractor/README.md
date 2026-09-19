# Caravan Static Attributes Extractor

The `static_extractor` package computes Caravan static attributes for user-supplied watershed polygons (GeoJSON, Shapefile, GeoPackage), following the published Caravan methodology. It covers the physiographic, hydro-climatic, soil, land-cover and anthropogenic attributes.

---

## 📌 Features & Capabilities

- **197+ Curated Catchment Attributes:** Extracts all standard HydroATLAS physiography, geology, land use, hydrology, and human impact metrics.
- **Strict Caravan Spatial Aggregation:**
  - **Area-weighted averaging** for continuous attributes (elevation, slope, precipitation, temperature, soil texture, etc.).
  - **Area-weighted majority voting** for discrete categorical classes (dominant land cover `glc_cl_smj`, soil class, wetland classification with class 13 no-wetland remapping).
  - **Downstream topological routing** (`NEXT_DOWN`) to extract pour-point outlet properties (e.g., natural discharge `dis_m3_pyr`).
- **40-Year ERA5-Land Climate Indices (1981–2020):**
  - Mean precipitation (`p_mean`) and FAO-56 Penman-Monteith potential evapotranspiration (`pet_mean_FAO_PM`).
  - Aridity index (`aridity_FAO_PM` and native ERA5-Land `aridity_ERA5_LAND`).
  - Snow fraction (`frac_snow`, $T < 0^\circ\text{C}$).
  - Knoben et al. (2018) annual moisture and seasonality indices.
  - Addor et al. (2017) extreme high precipitation frequency/duration and dry spell frequency/duration.
  - **Flexible Calculation Modes:** Choose between ultra-fast precalculated HydroSHEDS Level 12 subcatchment statistics (`hybas`, ~20 ms/basin) or recalculating directly from archived gridded ERA5 daily Zarr (`gridded`).
- **High-Performance Spatial Querying:**
  - Leverages native R-Tree spatial indexing in ESRI FileGDB (`pyogrio`) to read only overlapping Level 12 subcatchments in **10–15 ms per polygon**.
  - In-memory continental caching reduces per-basin lookup to **< 25 ms**, scaling to 50,000 polygons in ~20 minutes on a single core or ~1 minute with multi-core parallelism.
- **Direct Export Formats:** Outputs standard Caravan-formatted CSV and appends directly into MultiMet Zarr hierarchies.

---

## 🗄️ Data Sourcing & Paths

The extractor reads its input data from Google Cloud Storage. No local data copies are required prior to runtime.

### 1. GCS Data Stores
| Dataset | GCS URI | Description |
| :--- | :--- | :--- |
| **HydroATLAS Geodatabase** | `gs://open-multimet/ancillary-data/hydroatlas/BasinATLAS_v10.gdb/` | Full global ESRI FileGDB containing the `BasinATLAS_v10_lev12` layer (5.5 GiB, 1,034,083 Level 12 subcatchments). |
| **ERA5-Land Climate Tables** | `gs://open-multimet/ancillary-data/hydroatlas/era5_climate/` | 9 continental precomputed Level 12 climate tables (`af`, `ar`, `as`, `au`, `eu`, `gr`, `na`, `sa`, `si`; 1,034,027 basins). |
| **HydroATLAS Tabular Parquet** | `gs://open-multimet/ancillary-data/hydroatlas/hydro_atlas_lev12.parquet` | Complete pre-indexed tabular HydroATLAS Level 12 attributes (233 MiB). |
| **Gridded ERA5-Land Zarr** | `gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr` | Archived daily surface gridded ERA5-Land dataset used when `--era5-source gridded` is selected. |
| **Benchmark Catchment Dataset** | `gs://open-multimet/ancillary-data/benchmarks/benchmark_basins_500.parquet` | Reference benchmark dataset of 500 diverse global catchments for regression testing. |

### 2. Local Runtime Staging Cache
To enable fast random spatial reads by GDAL/`pyogrio`, the extractor stages data locally during runtime execution:
- **GDB Staging Path:** `~/.cache/googlehydrology/hydroatlas/BasinATLAS_v10.gdb`
- **ERA5 Staging Path:** `~/.cache/googlehydrology/era5_climate/`

> **Note on Storage Architecture:** Local directories are strictly used as temporary execution caches. If the local staging cache is empty, the extractor automatically downloads required files from `gs://open-multimet/ancillary-data/hydroatlas/`. No manual downloads or external local data stores are required.

---

## 🔬 Hydrological Methodology

### Subcatchments vs. Full Drainage Basins
1. **The Global Mesh:** HydroSHEDS / BasinATLAS Level 12 divides the world into **1,034,083 non-overlapping local subcatchments** (median area $\sim 100\text{ km}^2$), stored with their local drainage area in `SUB_AREA` and downstream topology in `NEXT_DOWN`.
2. **Catchment Decomposition:** When a user supplies a watershed polygon (e.g., the full contributing drainage area upstream of a river gauge), the extractor intersects the watershed boundary against the Level 12 mesh.
3. **Geodesic Area-Weighting:**
   Each intersecting fragment $i$ receives a fractional weight:
   $$w_i = \frac{\text{area}(\text{Catchment} \cap \text{Subcatchment}_i)}{\text{area}(\text{Catchment})}$$
4. **Continuous Aggregation:**
   $$\bar{X} = \sum_{i=1}^{N} w_i \cdot X_i$$
5. **Categorical Majority Voting:**
   $$C^* = \underset{c}{\operatorname{argmax}} \sum_{i: X_i = c} w_i$$
   *(Wetland values $\le 0$ or $-999$ are remapped to class 13, designating no wetland).*

---

## 💻 Command Line Interface (CLI)

The package installs console script `extract-caravan-static` (alias `extract-static-attributes`):

```bash
# Basic usage (default: fast precalculated HydroSHEDS Level 12 catchment statistics)
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv

# Using direct recalculation from archived daily gridded ERA5 Zarr
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv \
    --era5-source gridded

# With custom ID column, overlap threshold, and explicit GCS gridded Zarr URI
extract-caravan-static \
    --input /path/to/basins.shp \
    --output /path/to/attributes.csv \
    --id-column gauge_id \
    --era5-source gridded \
    --gridded-era5-uri gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr \
    --min-overlap-threshold 0.5
```

### CLI Arguments
- `--input`, `-i`: Path to vector polygon file (`.geojson`, `.shp`, `.gpkg`).
- `--output`, `-o`: Path to output CSV file for extracted Caravan attributes.
- `--workers`, `-w`: Number of parallel worker processes to use (default: 1).
- `--id-column`: Name of the property column containing the catchment/gauge identifier (defaults to auto-detection: `gauge_id`, `catchment_id`, `id`, `basin_id`).
- `--era5-source`: Choice of ERA5 climate attribute calculation method (`hybas` or `gridded`, default: `hybas`):
  - `hybas`: Fast area-weighted aggregation of precomputed HydroSHEDS Level 12 sub-basin statistics (~20 ms/basin).
  - `gridded`: Recalculates climate indices directly on the fly from 40-year daily surface gridded ERA5-Land data on GCS.
- `--gridded-era5-uri`: Custom GCS URI or local path for the daily surface ERA5 Zarr store (defaults to `gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr`).
- `--min-overlap-threshold`: Minimum sub-basin overlap area in $\text{km}^2$ to filter boundary slivers (default `0.0`).
- `--gdb-path`, `-g`: Optional override path to local `BasinATLAS_v10.gdb` (defaults to runtime cache).
- `--era5-cache-dir`: Optional override directory for ERA5 climate files (defaults to runtime cache).

---

## 🚀 Multi-Dataset Static Batch Runner (`extract-caravan-static-batch`)

For batch processing static attributes across multiple Caravan datasets in one command, the package provides `extract-caravan-static-batch` (alias `extract-static-attributes-batch`). It specifically extracts the static HydroATLAS physiographic attributes and ERA5 climate indices. It accepts parent directories, directory lists, or direct GCS URIs, auto-discovers watershed shapefiles, supports `--workers` parallelization, and outputs partitioned files matching the `caravan-new` schema (`attributes_hydroatlas_<ds>.csv`, `attributes_caravan_<ds>.csv`, `attributes_<ds>.parquet`):

```bash
# 1. Run all datasets within a parent directory (e.g. caravan/ containing camels/, hysets/, etc.)
extract-caravan-static-batch \
    --parent-dir /path/to/caravan_shapefiles/caravan/ \
    --output-dir /path/to/extracted_csvs/ \
    --workers 32 \
    --combine

# 2. Run directly across all Caravan collections in a single command using --preserve-caravan-dirs:
extract-caravan-static-batch \
    --parent-dir gs://open-multimet/caravan-new/ \
    --output-dir gs://open-multimet/caravan-new/ \
    --preserve-caravan-dirs \
    --workers 16

# 3. Run for an explicit list of dataset directories
extract-caravan-static-batch \
    --input-dirs /data/shapes/camels /data/shapes/camelsaus /data/shapes/lamah \
    --output-dir /path/to/extracted_csvs/ \
    --workers 32
```

### Batch Runner Arguments
- `--parent-dir`, `-p`: Parent directory containing dataset subdirectories (local path or `gs://...`). Can be passed multiple times or pointed at the root `gs://open-multimet/caravan-new/`.
- `--input-dirs`, `-d`: Explicit list of dataset directories.
- `--input-files`, `-f`: Explicit list of vector files (`.shp`, `.geojson`, `.gpkg`).
- `--output-dir`, `-o`: Output directory for generated files.
- `--preserve-caravan-dirs`: Automatically routes and partitions static attributes by Caravan collection and subdataset into `<collection>/attributes/<subdataset>/` matching the storage layout.
- `--partition-outputs`, `-P`: Explicitly enable partitioned subdataset outputs (`attributes_hydroatlas_<ds>.csv`, `attributes_caravan_<ds>.csv`, `attributes_<ds>.parquet`).
- `--workers`, `-w`: Number of parallel worker processes.
- `--era5-source`: `hybas` (default) or `gridded`.
- `--combine`: Generates aggregated combined tables (`attributes_caravan_combined.csv` and `attributes_combined.parquet`) merging all datasets.
- `--no-resume`: Disables resume (by default, already completed datasets are skipped).
- `--clean-cache`: Automatically cleans up the entire local runtime cache directory (`~/.cache/googlehydrology/`) after batch extraction completes.
- `--clean-staging`: Automatically cleans up only temporary staged shapefiles while preserving the HydroATLAS GDB and ERA5 climate tables.

---

## 🐍 Python API Usage

### 1. Extract from a Vector File
```python
from static_extractor import StaticAttributesExtractor

# Standard fast mode using precalculated HYBAS subcatchments
extractor = StaticAttributesExtractor(era5_source="hybas")
df = extractor.extract_attributes_from_file(
    input_path="basins.geojson",
    output_csv_path="caravan_attributes.csv",
    id_column="gauge_id",
)
print(df.head())

# Or recalculate climate indices directly from gridded ERA5-Land Zarr
extractor_gridded = StaticAttributesExtractor(era5_source="gridded")
df_gridded = extractor_gridded.extract_attributes_from_file(
    input_path="basins.geojson",
    output_csv_path="caravan_attributes_gridded.csv",
)
```

### 2. Extract for an Arbitrary Shapely Polygon / GeoJSON
```python
import shapely.geometry
from static_extractor import StaticAttributesExtractor

extractor = StaticAttributesExtractor()

# Shapely Polygon in EPSG:4326 (lon, lat)
polygon = shapely.geometry.Polygon([
    [-86.9, 40.4],
    [-86.8, 40.4],
    [-86.8, 40.5],
    [-86.9, 40.5],
    [-86.9, 40.4],
])

# Extract with default HYBAS or choose on-the-fly:
result = extractor.extract_attributes_for_polygon(
    polygon,
    catchment_id="my_basin_01",
    era5_source="hybas",  # or "gridded"
)

# Full Caravan dictionary (197+ properties)
caravan_attrs = result["caravan_attributes"]
print("Drainage Area (km²):", result["total_area_km2"])
print("Mean Elevation (m):", caravan_attrs["ele_mt_sav"])
print("Mean Precip (mm/yr):", caravan_attrs["pre_mm_syr"])
print("ERA5 Aridity:", caravan_attrs["aridity_FAO_PM"])
```

### 3. Append to MultiMet Zarr Store
```python
extractor.append_attributes_to_zarr(
    master_zarr_path="/data/multimet/caravan.zarr",
    basin_id="my_basin_01",
    attributes=caravan_attrs,
)
```

---

## ⚡ Performance & Scaling

| Mode | Scale | Single-Core Sequential | 16-Core Parallel | 32-Core Parallel |
| :--- | :--- | :--- | :--- | :--- |
| **`hybas` (Precalculated)** | **1 Basin** | 20 – 25 ms | — | — |
| **`hybas` (Precalculated)** | **1,000 Basins** | ~22 seconds | ~1.5 seconds | ~0.8 seconds |
| **`hybas` (Precalculated)** | **50,000 Basins** | **~18 to 20 minutes** | **~1.3 minutes** | **~45 seconds** |
| **`gridded` (Recalculated)** | **1 Basin** | 2 – 5 seconds | — | — |
| **`gridded` (Recalculated)** | **1,000 Basins** | ~40 minutes | ~3 minutes | ~1.5 minutes |

*Note: Initial run on an empty cache requires a one-time download of `BasinATLAS_v10.gdb` (5.5 GiB, ~15–30s on Google Cloud network).*

---

## 🧪 Testing & Verification

Run the dedicated test suite verifying geographic intersection, pour-point routing, Knoben/Addor climate math, and ground-truth comparison against reference Caravan basins:

```bash
pytest test/test_static_extractor.py -v
```

---

## 📊 Benchmarking

The benchmark compares the attributes this package produces against published
Caravan reference values for 490 basins, spread over 7 datasets and 5 catchment
size tiers.

It is **not** part of the test suite. It downloads real data and takes minutes
rather than seconds, so it is something you run deliberately when you want to
check the numbers yourself rather than take them on trust.

### What you need beforehand

Nothing. The reference dataset lives at

```
gs://open-multimet/ancillary-data/benchmarks/benchmark_basins_500.parquet
```

and is downloaded the first time you run the benchmark, then cached under
`static_extractor/data/`. The HydroATLAS store it compares against is fetched
the same way. You need read access to those buckets and roughly 12 MB of disk
for the reference file.

### Running it

```bash
# Everything: all 490 basins
benchmark-static-extractor --workers 14 -o ./benchmark_results/

# A quicker spot-check: 50 basins, stratified across datasets and size tiers
benchmark-static-extractor --samples 50 --workers 8 -o ./benchmark_results/

# Narrowed to particular datasets or catchment sizes
benchmark-static-extractor \
    --regions camels camelsaus lamah \
    --size-tiers 1_micro 5_macro \
    --workers 8 \
    -o ./benchmark_results/
```

### Reading the output

Three files land in the output directory:

- `benchmark_report.md` — a readable summary; start here.
- `benchmark_attribute_metrics.csv` — one row per attribute, showing how closely
  each one reproduces the reference values.
- `benchmark_basin_metrics.csv` — one row per basin, including the attribute
  that matched worst for that basin.

The two CSVs are there so you can find *which* attributes or basins disagree and
by how much, rather than relying on a single headline number.

