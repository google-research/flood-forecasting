# Catchment Delineation Integration Summary

**Author / Agent**: Jetski  
**Target Repository**: `google-research/flood-forecasting` (`googlehydrology`)  
**Local Workspace**: `/usr/local/google/home/gsnearing/Projects/catchment-delineation`  
**Git Branch**: `gsnearing-catchment-delineation`  
**Status**: Completed, verified, and committed  

---

## 1. Executive Summary

This task ported the pure DEM watershed delineation capabilities originally prototyped in `earthkit_hydro_web_plan` into a clean, standalone, modular Python package called `catchment_delineation` within the `flood-forecasting` (`googlehydrology`) repository.

Key objectives achieved:
1. **Isolated Clone**: Cloned a completely fresh working copy of `google-research/flood-forecasting.git` to avoid any conflict with other active agents working in `Projects/flood-forecasting` or `Projects/flood-forecasting-multimet`.
2. **Branch Created**: Created and checked out branch `gsnearing-catchment-delineation`.
3. **Parallel Package**: Placed `catchment_delineation/` directly parallel to `googlehydrology/`.
4. **Zero Heavy Dependencies**: The module requires only `numpy` and `shapely` (no server, UI, or google3 dependencies).
5. **Coordinate Activation**: Built both a Python API and a CLI (`delineate-catchment` / `python -m catchment_delineation`) allowing users to activate delineation by providing lat/lon coordinates (single points, lists, or CSV files).
6. **googlehydrology Package Integration**: Updated `setup.py` and `googlehydrology/__init__.py` to expose the delineator directly through `googlehydrology`.
7. **Comprehensive Testing**: Added 6 unit and integration tests in `test/test_catchment_delineation.py`, all passing.

---

## 2. Directory Structure & Changes

```text
/usr/local/google/home/gsnearing/Projects/catchment-delineation/
├── catchment_delineation/                    # [NEW PACKAGE]
│   ├── __init__.py                          # Public API exports (DemDelineator, delineate_dem, etc.)
│   ├── __main__.py                          # Entry point for `python -m catchment_delineation`
│   ├── cli.py                               # CLI for lat/lon coordinates, coordinate lists, and CSVs
│   ├── config.py                            # D8 inflow mapping, resolution constants, tile paths
│   ├── delineator.py                        # Core DemDelineator engine & Shapely vectorization
│   ├── tiles.py                             # 5x5 degree tile math, indexing, and coverage checks
│   └── README.md                            # Complete module documentation and examples
├── googlehydrology/
│   └── __init__.py                          # [MODIFIED] Added imports for DemDelineator & functions
├── test/
│   └── test_catchment_delineation.py         # [NEW TEST] 6 automated unit & integration tests
├── setup.py                                 # [MODIFIED] Added catchment_delineation & CLI console script
└── CATCHMENT_DELINEATION_SUMMARY.md         # [THIS SUMMARY]
```

---

## 3. DEM Data & Tile Specifications

The delineator uses the official **HydroSHEDS v1.4 3 arc-second (~90m)** conditioned digital elevation and flow-direction dataset for North America.

* **Resolution**: 3 arc-seconds ($\approx 90\text{ meters}$ at the equator, or exactly $\frac{1}{1200}^\circ$ per cell).
* **Flow Direction Format**: Standard ESRI D8 flow routing matrix where cell values encode downstream flow direction (`1`=E, `2`=SE, `4`=S, `8`=SW, `16`=W, `32`=NW, `64`=N, `128`=NE).
* **Tile Layout**: 5°×5° tiles stored as memory-mapped `.npy` files (`uint8`, shape `(6000, 6000)`, ~36 MB each), named following the convention `n{lat}w{lon}.npy` (e.g. `n40w090.npy`).
* **Tile Storage Location**:
  * Cloud storage (GCS): `gs://open-multimet/data/DEMs/tiles_5deg/` (119 tiles, 4.0 GB), `elevation_tiles_5deg/` (119 tiles, 8.0 GB), and raw GeoTIFFs (`na_dir_3s.tif`, `na_con_3s.tif`).
  * Primary local directory: `~/data/DEMs/tiles_5deg/` (or `~/data/DEMs/hydrosheds/tiles_5deg/`).
  * Active cache: `/usr/local/google/home/gsnearing/.cache/earthkit_hydro/data/hydrosheds_dem/tiles_5deg/` (119 pre-sliced tiles covering continental North America).
  * Raw full GeoTIFF: `/usr/local/google/home/gsnearing/.cache/earthkit_hydro/data/hydrosheds_dem/na_dir_3s.tif` (917 MB, 8.64B pixels).
  * Raw elevation GeoTIFF: `/usr/local/google/home/gsnearing/.cache/earthkit_hydro/data/hydrosheds_dem/na_con_3s.tif` (2.70 GB).
* **Automatic Discovery & GCS On-Demand Download**: In `catchment_delineation/config.py`, tiles are auto-resolved via:
  1. Explicit `--tiles-dir` argument or `DemDelineator(tiles_dir=...)`
  2. `DEM_TILES_DIR` environment variable
  3. `~/data/DEMs/tiles_5deg` (or `~/data/DEMs/hydrosheds/tiles_5deg`)
  4. `~/.cache/googlehydrology/hydrosheds_dem/tiles_5deg`
  5. `~/.cache/earthkit_hydro/data/hydrosheds_dem/tiles_5deg`
  6. `data/dem/tiles_5deg` relative to the repository root
  7. On-demand cloud download from `gs://open-multimet/data/DEMs/tiles_5deg/` when `--auto-download` or `auto_download=True` is enabled.


---

## 4. Delineation Algorithm Details

1. **Channel Snapping**: Given target coordinates `(lat, lon)`, computes the starting 5°×5° tile coordinates and converts to row/column grid indices. Executes a bounded local Breadth-First Search (BFS) within a configurable half-window (`snap_window_cells`, default 4 cells ~360m) to score upstream channel connectivity and snap to the channel outlet cell.
2. **Seamless Multi-Tile BFS Traversal**: Performs authentic reverse-flow tree traversal across the D8 flow matrix. If an upstream inflow crosses a 5°×5° tile boundary (`r < 0`, `r >= 6000`, `c < 0`, or `c >= 6000`), the neighbor tile is dynamically loaded via memory mapping and traversal continues seamlessly without boundary edge artifacts.
3. **Accurate Geodesic Area**: Sums all visited raster cells weighting by exact latitude ground scaling (`lat_scale * lon_scale`) to compute accurate basin area in $\text{km}^2$.
4. **Vectorization**: Uses row run-length interval fusion to construct rectangular bounding boxes for visited cells, followed by Shapely `unary_union` and adaptive simplification to output clean, valid GeoJSON Polygon / MultiPolygon geometries.

---

## 5. How to Activate via Lat/Lon Coordinates

### A. Command-Line Interface (`delineate-catchment` or `python -m catchment_delineation`)

1. **Single Lat/Lon Coordinate**:
   ```bash
   delineate-catchment --lat 39.6828 --lon -88.7729 --pretty -o catchment.geojson
   ```

2. **Multiple Coordinate Pairs**:
   ```bash
   delineate-catchment --coords "39.6828,-88.7729" "40.4172,-86.8858" --pretty -o multi_basins.geojson
   ```

3. **Batch Coordinates from CSV**:
   ```bash
   delineate-catchment --csv gauges.csv -o delineated_basins.geojson
   ```
   *(CSV must contain `lat`/`latitude` and `lon`/`longitude` columns, with optional `id` column).*

4. **Print GeoJSON to stdout**:
   ```bash
   delineate-catchment --lat 39.6828 --lon -88.7729 --pretty
   ```

5. **List Available Tiles**:
   ```bash
   delineate-catchment --list-tiles
   ```

### B. Python API

```python
from catchment_delineation import DemDelineator, delineate_dem

# 1. Using convenience function
basin = delineate_dem(lat=39.6828, lon=-88.7729)
print("Catchment ID:", basin["properties"]["catchment_id"])
print("Basin Area:", basin["properties"]["area_km2"], "km²")
print("Upstream Cells:", basin["properties"]["upstream_cells_count"])
print("Bounding Box:", basin["properties"]["bbox"])

# 2. Batch processing
delineator = DemDelineator()
coords = [(39.6828, -88.7729), (40.4172, -86.8858)]
feature_collection = delineator.delineate_batch(coords)
```

### C. Direct Access via `googlehydrology`

```python
import googlehydrology

basin = googlehydrology.delineate_dem(lat=39.6828, lon=-88.7729)
```

---

## 6. Testing & Quality Assurance

Test file: [`test/test_catchment_delineation.py`](test/test_catchment_delineation.py)  
Test execution:
```bash
pytest test/test_catchment_delineation.py -v
```

**Results**: 6/6 tests passed:
- `test_tile_key_and_filename`: Verifies 5°×5° tile grid math for northern, southern, eastern, and western hemispheres.
- `test_parse_coord_str`: Validates coordinate string parsing (`lat,lon`).
- `test_load_coords_from_csv`: Verifies CSV parsing with automatic header detection.
- `test_synthetic_dem_delineator`: Tests D8 BFS traversal, area calculation, and polygon creation on a synthetic grid.
- `test_live_tile_delineation`: Performs end-to-end delineation on real HydroSHEDS 90m raster tiles at Dalton City, IL.
- `test_cli_execution`: Tests CLI argument handling and GeoJSON file generation.

---

## 7. Git Status & Next Steps

* **Committed CL/Commit**: `8df497d`
* **Commit Message**: `feat: add DEM catchment delineation module parallel to googlehydrology`
* **Ready for Coordination**:
  - The branch `gsnearing-catchment-delineation` is ready for review, upstream pushing to GitHub (`origin/gsnearing-catchment-delineation`), or pull request creation.
  - Future tasks can add additional regional D8 DEM tiles (e.g. Europe, South America, Asia) or integrate streamflow routing models directly on delineated basin geometries.
