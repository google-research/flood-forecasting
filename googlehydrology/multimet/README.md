# MultiMet Meteorological Forcing Extractor

The `googlehydrology.multimet` module provides a high-performance, open-source meteorological data extraction and harmonization pipeline for hydrological modeling and flood forecasting.

It extracts catchment-averaged forcing time series from gridded meteorological products (reanalyses, satellite observations, NWP models, and machine learning weather models) and standardizes them to the **Caravan benchmark specification** ([Kratzert et al., 2023](https://nature.com/articles/s41597-023-01960-w); [Kratzert et al., 2024, arXiv:2411.09459](https://arxiv.org/abs/2411.09459)).

---

## 1. Supported Meteorological Products

In this initial release, the extractor supports local serial extraction for **5 core products**:

| Product | Type | Native Grid | Forecast Lead | Variables Extracted | Public Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ERA5-Land** | Hourly Reanalysis | $0.1^\circ$ (1801 $\times$ 3600) | N/A | 15 variables (Precipitation, Temp, Dewpoint, Pressure, Radiations, Wind, Soil Moisture layers 1–4, Snow, FAO-56 PET) | WeatherBench 2 on public GCS (`gs://weatherbench2/datasets/era5_daily/...`) or local hourly GRIB |
| **CPC Global Precip** | Daily Gauge | $0.5^\circ$ (360 $\times$ 720) | N/A | Daily precipitation ($\text{mm/day}$) | NOAA PSL NetCDF archive (`https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/`) or binary grids |
| **IMERG Early V07** | Daily / Half-Hourly Satellite | $0.1^\circ$ (1800 $\times$ 3600) | N/A | Daily precipitation ($\text{mm/day}$) | NASA GES DISC / Earthdata Login (`GPM_3IMERGDE.07`) |
| **ECMWF IFS HRES** | Operational NWP Forecast | $0.25^\circ$ (721 $\times$ 1440) | 10 days ($1 \dots 10$) | 5 variables (Incremental daily precipitation, Mean 2m temperature, Surface pressure, Radiation fluxes) | WeatherBench 2 public Zarr (`gs://weatherbench2/datasets/hres/...`) or ECMWF Open Data |
| **DeepMind GraphCast** | AI Weather Forecast | $0.25^\circ$ (721 $\times$ 1440) | 10 days ($1 \dots 10$) | 4 variables (Daily accumulated precipitation from 6h steps, Mean 2m temp, 10m U/V wind components) | WeatherBench 2 public Zarr (`gs://weatherbench2/datasets/graphcast/...`) |

---

## 2. Architecture & Capabilities

### A. Exact Fractional Zonal Averaging & BLAS Vectorization
- **Fractional Polygon Intersection**: Computes exact fractional overlap between basin boundaries (Shapely polygons from GeoJSON or Shapefiles) and raster grid cells, with latitude cosine weighting to account for spherical surface distortion.
- **Sparse BLAS Matrix (`ZonalWeightMatrix`)**: Formulates spatial averaging across $N$ basins and $(H \times W)$ raster cells into a SciPy CSR sparse matrix $W \in \mathbb{R}^{N \times (H \cdot W)}$. Zonal reduction for any 2D, 3D, or 4D meteorology grid is performed via sparse matrix multiplication ($Y = W \cdot X$), evaluating thousands of basins in milliseconds.
- **Weight Caching**: Weights can be exported to and imported from compressed `.npz` files (`weights_cache.npz`), skipping redundant polygon intersections on subsequent runs.

### B. Caravan Harmonization & Unit Standardization
- Converts cumulative energy fluxes ($\text{J/m}^2$) to mean rates ($\text{W/m}^2$).
- Converts Kelvin temperatures to Celsius ($^\circ\text{C}$).
- Converts surface pressure from Pascals to $\text{kPa}$.
- Implements the **FAO-56 Penman-Monteith** formulation for reference evapotranspiration (PET).
- Converts HRES continuous accumulations into daily increments ($P_d = P_{24d} - P_{24(d-1)}$).
- Sums 6-hour GraphCast intervals into 24-hour daily forecast totals.

### C. Consolidated Zarr Storage
- Outputs are stored in standardized **Zarr v2** hierarchies with consolidated metadata (`.zmetadata`).
- **Nowcast products** are indexed by `(basin, date)`.
- **Forecast products** are indexed by `(basin, date, lead_time)` with daily lead times ($1 \dots 10$ days).
- Compatible with OpenHydroNet's `Multimet` dataset loader (`googlehydrology.datasetzoo.multimet.Multimet`).

---

## 3. Quickstart: Local Serial Extraction

### Python API

```python
from googlehydrology.multimet import extract_multimet_serial

# Run serial extraction for all 5 products
output_stores = extract_multimet_serial(
    basins="test/test_data/shapefiles/us/us_basin_shapes.geojson",
    output_dir="/path/to/output_zarrs",
    products=["CPC", "ERA5_LAND", "IMERG", "HRES", "GRAPHCAST"],
    start_date="2020-01-01",
    end_date="2020-01-05",
    source="public",
)

for product, store_path in output_stores.items():
  print(f"Product {product} written to {store_path}")
```

### Command-Line Interface (CLI)

```bash
python -m googlehydrology.multimet.runner \
  --basins_path test/test_data/shapefiles/us/us_basin_shapes.geojson \
  --output_dir /tmp/multimet_extracted \
  --products CPC,ERA5_LAND,IMERG,HRES,GRAPHCAST \
  --start_date 2020-01-01 \
  --end_date 2020-01-02 \
  --source public
```

---

## 4. Relationship to Caravan MultiMet Paper (arXiv:2411.09459)

The Caravan MultiMet paper (*"Caravan MultiMet: Extending Caravan with Multiple Weather Nowcasts and Forecasts"*, [arXiv:2411.09459](https://arxiv.org/abs/2411.09459)) describes the creation of a large-scale, pre-computed benchmark dataset covering thousands of global watersheds and hosted as static NetCDF and Zarr archives on Zenodo and Google Cloud Platform.

### Key Differences Between the Paper and This Module:

1. **Static Benchmark vs. Active Extractor Engine**:
   - The paper published pre-computed time series for fixed Caravan basins up to late 2023.
   - This module is the **underlying reproducible extraction engine**, allowing researchers to generate forcing time series for **any custom basin geometries** (local watersheds, regional gauges) and **any arbitrary date intervals** (historical or near-real-time).
2. **Zero Proprietary Infrastructure**:
   - While original dataset production utilized distributed cloud batch jobs, this module is written entirely in portable Python (`xarray`, `zarr`, `geopandas`, `scipy`) and operates directly on public open-access endpoints without proprietary internal tools.
3. **End-to-End Integration with OpenHydroNet**:
   - Forcing data generated by this extractor can be consumed directly by `googlehydrology.datasetzoo.multimet.Multimet` to train and evaluate LSTM and Transformer flood forecasting models (e.g., `MeanEmbeddingForecastLSTM`).
