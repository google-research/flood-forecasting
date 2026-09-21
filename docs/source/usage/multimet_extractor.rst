MultiMet Forcing Extractor
===========================

The **MultiMet Forcing Extractor** (``multimet``) is a high-performance meteorological data extraction and harmonization pipeline. It ingests either **Open-MultiMet Gridded Zarr Archives** (local or ``gs://`` URIs) or direct **third-party upstream agency feeds** and extracts catchment-averaged forcing time series standardized according to the **Caravan benchmark specification** (`Kratzert et al., 2023 <https://nature.com/articles/s41597-023-01960-w>`_; `Kratzert et al., 2024, arXiv:2411.09459 <https://arxiv.org/abs/2411.09459>`_).

Supported Meteorological Products
---------------------------------

The extractor supports **5 core products**, each including a companion ``<prefix>_missing_fraction`` audit variable recording the area-weighted fraction ``[0.0, 1.0]`` of missing (``NaN``) pixels per catchment and timestep:

1. **ERA5-Land (ECMWF)** (``ERA5_LAND``):
   Daily global reanalysis at 0.1° resolution (1801 × 3600). Extracted **exclusively from a user-supplied ERA5-Land gridded Zarr archive** (``source="archive"``) to prevent silent substitution with coarser 0.25° ERA5 products. Extracts **17 harmonized variables** including daily mean, daily minimum (``era5land_temperature_2m_min``), and daily maximum (``era5land_temperature_2m_max``) 2m air temperature (°C), dewpoint (°C), surface pressure (kPa), total precipitation (mm/day), net solar and thermal radiation fluxes (W/m²), 10m wind components (m/s), 4-layer volumetric soil water (m³/m³), snow depth water equivalent (mm), and **FAO-56 Penman-Monteith** potential evapotranspiration (PET), plus ``era5land_missing_fraction``.
2. **CPC Global Unified Precipitation (NOAA PSL)** (``CPC``):
   Daily gauge-based precipitation analysis at 0.5° resolution (360 × 720). Extracts both ``cpc_precipitation`` (mm/day) and ``cpc_num_stations`` (reporting rain-gauge station count per grid cell), plus ``cpc_missing_fraction``. Supports both user-supplied gridded Zarr archives (``--source archive``) and direct NOAA PSL NetCDF / CPC binary files (``--source public`` / ``--source local``).
3. **IMERG Early V07 (NASA GPM)** (``IMERG``):
   Global satellite-derived precipitation nowcast at 0.1° resolution (1800 × 3600). Extracts ``imerg_precipitation`` (mm/day) and ``imerg_missing_fraction``. Supports user-supplied gridded Zarr archives (``--source archive``), NASA GES DISC HTTP downloads (``--source public``), and dynamical.org Icechunk catalogs.
4. **ECMWF IFS HRES** (``HRES``):
   Operational high-resolution numerical weather prediction (NWP) 10-day forecasts at 0.25° resolution (721 × 1440). Extracts incremental daily forecast precipitation, daily mean temperature, surface pressure, and radiation fluxes, plus ``hres_missing_fraction``. Supports user-supplied gridded Zarr archives (``--source archive``) and explicit Zarr/GRIB stores.
5. **DeepMind GraphCast** (``GRAPHCAST``):
   Machine learning weather forecasting model at 0.25° resolution (721 × 1440). Extracts 10-day forecasts of daily precipitation, daily mean temperature, and 10m wind components, plus ``graphcast_missing_fraction``.

Data-Quality & Provenance Guarantees
------------------------------------

* **No Hardcoded Cloud Paths or Placeholder Dates**:
  All ``gs://`` archive URIs, ``start_date``, and ``end_date`` arguments must be explicitly supplied by the user.
* **Missing Data In = Missing Data Out**:
  Missing grid pixels or missing dates within a valid archive window propagate strictly as ``NaN`` without fallback substitution. Missing archive bands log at most one warning per store/product session.
* **No Out-of-Domain Nearest-Cell Snapping**:
  Sub-grid-scale polygons receive weight on a single grid cell only when the polygon genuinely lies inside that cell. Catchments outside the grid domain receive zero weight and evaluate to ``NaN`` (``missing_fraction = 1.0``).
* **Companion Coverage Variable (``<prefix>_missing_fraction``)**:
  Every extracted dataset records the area-weighted proportion of ``NaN`` grid pixels within each catchment at each timestep, matching ``CookieCutterResult.missing_values`` in Google's internal flood-forecasting ingestion pipeline.

Quickstart Examples
-------------------

Extracting from Gridded Zarr Archives (Python API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multimet import extract_multimet_serial

    stores = extract_multimet_serial(
        basins="path/to/basins.geojson",
        output_dir="/path/to/output_zarrs",
        products=["CPC", "ERA5_LAND", "IMERG", "HRES"],
        start_date="2022-01-01",
        end_date="2022-01-05",
        source="archive",
        archive_stores={
            "CPC": "gs://<your-bucket>/gridded-data-archives/CPC/daily_surface.zarr",
            "ERA5_LAND": "gs://<your-bucket>/data/era5_land/daily_surface.zarr",
            "IMERG": "gs://<your-bucket>/gridded-data-archives/IMERG/daily_surface.zarr",
            "HRES": "gs://<your-bucket>/gridded-data-archives/HRES/daily_surface.zarr",
        },
    )

Command-Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    extract-multimet \
      --basins_path multimet/test/test_data/shapefiles/us/us_basin_shapes.geojson \
      --output_dir /tmp/multimet_extracted \
      --products CPC,ERA5_LAND,IMERG,HRES \
      --start_date 2022-01-01 \
      --end_date 2022-01-05 \
      --source archive \
      --archive-store CPC=gs://<your-bucket>/gridded-data-archives/CPC/daily_surface.zarr \
      --archive-store ERA5_LAND=gs://<your-bucket>/data/era5_land/daily_surface.zarr \
      --archive-store IMERG=gs://<your-bucket>/gridded-data-archives/IMERG/daily_surface.zarr \
      --archive-store HRES=gs://<your-bucket>/gridded-data-archives/HRES/daily_surface.zarr
