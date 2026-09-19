MultiMet Forcing Extractor
===========================

The **MultiMet Forcing Extractor** (``multimet``) is a high-performance meteorological data extraction and harmonization pipeline. It ingests raw gridded meteorological datasets (reanalyses, satellite observations, numerical weather predictions, and machine learning weather models) and extracts catchment-averaged forcing time series standardized according to the **Caravan benchmark specification** (`Kratzert et al., 2023 <https://nature.com/articles/s41597-023-01960-w>`_; `Kratzert et al., 2024, arXiv:2411.09459 <https://arxiv.org/abs/2411.09459>`_).

Supported Meteorological Products
---------------------------------

The extractor supports local serial extraction for **5 core products**:

1. **ERA5-Land (ECMWF)**:
   Hourly global reanalysis at 0.1° resolution. Extracts 15 harmonized variables including total precipitation (mm/day), 2m air temperature (°C), dewpoint (°C), surface pressure (kPa), net radiation fluxes (W/m²), 10m wind components (m/s), multi-layer volumetric soil water (m³/m³), snow depth water equivalent (mm), and potential evapotranspiration (PET) calculated using the **FAO-56 Penman-Monteith** formulation.
2. **CPC Global Unified Precipitation (NOAA PSL)**:
   Daily gauge-based precipitation analysis at 0.5° resolution, converted from tenths of millimeters to mm/day.
3. **IMERG Early V07 (NASA GPM)**:
   Global satellite-derived precipitation nowcast at 0.1° resolution.
4. **ECMWF IFS HRES**:
   Operational high-resolution numerical weather prediction (NWP) 10-day forecasts at 0.25° resolution. Extracts incremental daily forecast precipitation, daily mean temperature, surface pressure, and radiation fluxes.
5. **DeepMind GraphCast**:
   State-of-the-art machine learning weather forecasting model at 0.25° resolution. Extracts 10-day forecasts of daily precipitation (accumulated over 6-hourly steps), daily mean temperature, and 10m wind components.

Key Architecture
----------------

* **Exact Fractional Zonal Weighting**:
  Computes fractional overlap areas between catchment polygons and gridded raster cells, including cosine latitude weighting for spherical cell area distortion.
* **Vectorized Sparse BLAS Reduction (``ZonalWeightMatrix``)**:
  Represents spatial weights across :math:`N` basins and :math:`H \times W` grid cells as a SciPy CSR sparse matrix. Zonal reductions across thousands of basins take milliseconds via sparse BLAS matrix multiplication (:math:`Y = W \cdot X`).
* **Precomputed Weights Caching**:
  Weights can be computed once and persisted to a compressed ``.npz`` archive for fast startup in future runs.
* **Consolidated Zarr v2 Stores**:
  Extracted datasets are stored in chunked Zarr v2 stores with consolidated metadata (``.zmetadata``). Nowcast products use dimensions ``(basin, date)``, while forecast products use ``(basin, date, lead_time)`` with daily lead steps.
* **Open Science & Zero Internal Dependencies**:
  Runs purely on open-access public data streams (WeatherBench 2 on public Google Cloud Storage, NOAA PSL HTTP, NASA GES DISC, ECMWF Open Data) and standard scientific Python libraries.

Quickstart Examples
-------------------

Python API
^^^^^^^^^^

.. code-block:: python

    from multimet import extract_multimet_serial

    stores = extract_multimet_serial(
        basins="path/to/basins.geojson",
        output_dir="/path/to/zarr_archive",
        products=["CPC", "ERA5_LAND", "IMERG", "HRES", "GRAPHCAST"],
        start_date="2020-01-01",
        end_date="2020-01-05",
        source="public",
    )

    for product, path in stores.items():
        print(f"Saved {product} to {path}")

Command-Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    extract-multimet \
      --basins_path test/test_data/shapefiles/us/us_basin_shapes.geojson \
      --output_dir /tmp/multimet_extracted \
      --products CPC,ERA5_LAND,IMERG,HRES,GRAPHCAST \
      --start_date 2020-01-01 \
      --end_date 2020-01-02 \
      --source public

Caravan MultiMet Paper & Key Differences
----------------------------------------

This module implements the methodology introduced in:

    *Frederik Kratzert, Martin Gauch, Grey Nearing, et al.*, **"Caravan MultiMet: Extending Caravan with Multiple Weather Nowcasts and Forecasts"**, `arXiv:2411.09459 (2024) <https://arxiv.org/abs/2411.09459>`_.

**Differences Between the Paper and This Extractor Module:**

1. **Active Extraction Engine vs. Static Archive**:
   The arXiv paper published a static, pre-computed benchmark dataset for fixed Caravan catchments hosted on Zenodo and GCP. This module provides the **open-source extraction engine itself**, allowing hydrologists to extract MultiMet forcings for **any custom basin geometries** and **any time period**.
2. **Reproducibility Without Proprietary Infrastructure**:
   The extraction workflows operate directly on public open-access cloud endpoints without relying on Google-internal compute frameworks.
3. **Seamless Model Integration**:
   Generated Zarr archives can be ingested directly by ``googlehydrology.datasetzoo.multimet.Multimet`` to train models like ``MeanEmbeddingForecastLSTM``.
