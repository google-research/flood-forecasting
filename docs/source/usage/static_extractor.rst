==================================================
Extracting Static Attributes for Custom Watersheds
==================================================

To predict river flow in a watershed, OpenHydroNet needs a table of static watershed characteristics—such as area, elevation, slope, soil type, land cover, and long-term average climate.

If you are working with watersheds from the published `Caravan <https://www.nature.com/articles/s41597-023-01975-w>`_ dataset, those tables are already provided. If you want to run models on **your own watersheds**, the ``multimet.static_extractor`` package takes a map file of your watershed boundaries and creates a Caravan-compatible CSV table for you.

It calculates these attributes using the same community datasets and methods used by Caravan:

* `HydroATLAS <https://www.hydrosheds.org/hydroatlas>`_ (BasinATLAS v10, Level 12) for geography, elevation, slope, soil, land cover, lakes, and human footprint.
* `ERA5-Land <https://cds.climate.copernicus.eu/>`_ (1981–2020) for long-term climate averages (precipitation, temperature, evaporation, aridity, snow fraction, and wet/dry spell frequency).

-----------------
Before You Start
-----------------

Input File Requirements
^^^^^^^^^^^^^^^^^^^^^^^

* **File format:** A map file containing polygon boundaries for one or more watersheds in **GeoJSON** (``.geojson``), **Shapefile** (``.shp``), **GeoPackage** (``.gpkg``), or **GeoParquet** (``.parquet``) format.
* **Coordinates:** A defined coordinate reference system (standard latitude and longitude ``EPSG:4326`` / WGS84, or any projected coordinate system, which will be converted to ``EPSG:4326``).
* **Watershed ID column:** Each polygon must have a unique ID in the ``gauge_id`` column (or in the column you specify with ``--id-column <column_name>``).

Reference Datasets
^^^^^^^^^^^^^^^^^^

You provide the paths where the HydroATLAS and ERA5-Land reference datasets are stored:

* **HydroATLAS (** ``--gdb-path`` **):** Local path to ``BasinATLAS_v10.gdb`` (or ``BasinATLAS_v10_lev12.shp``). If you do not already have it on disk, pass ``--gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas/BasinATLAS_v10.gdb`` and the tool will download it into ``--gdb-path`` (~4.9 GB).
* **ERA5-Land Climate Data:**

  * When using ``--era5-source hybas``, pass ``--era5-cache-dir /path/to/era5_climate``. If you do not already have the continental climate tables on disk, also pass ``--gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate`` (~550 MB) to download them into ``--era5-cache-dir``.
  * When using ``--era5-source gridded``, pass ``--gridded-era5-uri gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr`` (or a local Zarr path). This streams the daily grid slices directly from the Zarr store without downloading the archive to your machine.

* If you want the tool to delete the local ``--gdb-path`` and ``--era5-cache-dir`` folders after the run finishes, add ``--clean-cache``.

Choosing ``--era5-source`` (``hybas`` vs. ``gridded``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every run requires the ``--era5-source`` flag to tell the tool how to calculate climate statistics:

.. list-table::
   :widths: 22 40 13 25
   :header-rows: 1

   * - Option
     - How It Works
     - Speed
     - When to Use It
   * - ``--era5-source hybas``
     - Combines pre-calculated climate summaries from the standard HydroATLAS sub-basins that overlap your watershed. *(If you also pass* ``--gridded-era5-uri`` *, the four* ``*_ERA5_LAND`` *evaporation columns are computed from the daily weather grid).*
     - Fast
     - **Recommended for most users.** Works well whenever your watersheds are roughly the size of standard river sub-basins (~100 km²) or larger.
   * - ``--era5-source gridded``
     - Reads 40 years (1981–2020) of daily ERA5-Land weather grids over your exact polygon boundary and calculates every climate number from scratch.
     - Slower (several seconds per basin)
     - Use this if your watersheds are very small, have custom boundaries that do not follow natural river sub-basins, or if you want to stream climate data directly from a Zarr store without downloading continental climate tables.

------------------
Command-Line Usage
------------------

1. Extract Attributes for One File (``extract-caravan-static``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``extract-caravan-static`` (or its alias ``extract-static-attributes``) when you have a single file of watershed polygons and want a single CSV table of attributes.

.. code-block:: bash

   # Using local HydroATLAS and pre-calculated sub-basin climate tables
   extract-caravan-static \
       --input /path/to/watershed_polygons.geojson \
       --output /path/to/extracted_caravan_attributes.csv \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --era5-source hybas \
       --era5-cache-dir /path/to/era5_climate

   # Downloading HydroATLAS and climate tables from Google Cloud Storage on first run
   extract-caravan-static \
       --input /path/to/watershed_polygons.geojson \
       --output /path/to/extracted_caravan_attributes.csv \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas/BasinATLAS_v10.gdb \
       --era5-source hybas \
       --era5-cache-dir /path/to/era5_climate \
       --gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate

   # Streaming climate numbers directly from daily ERA5-Land grids on Google Cloud Storage
   extract-caravan-static \
       --input /path/to/watershed_polygons.geojson \
       --output /path/to/extracted_caravan_attributes.csv \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --era5-source gridded \
       --gridded-era5-uri gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr

   # Specifying a custom ID column and running across 8 CPU cores
   extract-caravan-static \
       --input /path/to/basins.shp \
       --output /path/to/attributes.csv \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --era5-source hybas \
       --era5-cache-dir /path/to/era5_climate \
       --id-column station_id \
       --workers 8

All Flags for ``extract-caravan-static``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 12 48
   :header-rows: 1

   * - Flag
     - Required?
     - Default
     - What It Does
   * - ``--input``, ``-i``
     - **Yes**
     - —
     - Path to your input watershed boundary file (``.geojson``, ``.shp``, ``.gpkg``, or ``.parquet``).
   * - ``--output``, ``-o``
     - **Yes**
     - —
     - Path where the output CSV file will be saved.
   * - ``--gdb-path``, ``-g``
     - **Yes**
     - —
     - Local path to ``BasinATLAS_v10.gdb`` or ``BasinATLAS_v10_lev12.shp``.
   * - ``--era5-source``
     - **Yes**
     - —
     - How to calculate ERA5 climate numbers: ``hybas`` (from pre-calculated sub-basin tables) or ``gridded`` (from daily ERA5-Land grids).
   * - ``--era5-cache-dir``
     - Required if ``hybas``
     - ``None``
     - Local folder where continental ERA5 climate tables (``<continent>_climate_indices.txt``) are stored.
   * - ``--gridded-era5-uri``
     - Required if ``gridded``
     - ``None``
     - Google Cloud Storage (``gs://...``) URI or local folder path for the daily ERA5-Land Zarr dataset. Can also be passed with ``--era5-source hybas`` to compute the four ``*_ERA5_LAND`` columns.
   * - ``--gcs-gdb-uri``
     - No
     - ``None``
     - Google Cloud Storage (``gs://...``) URI from which to download ``BasinATLAS_v10.gdb`` into ``--gdb-path`` if it is not already on disk.
   * - ``--gcs-era5-climate-uri``
     - No
     - ``None``
     - Google Cloud Storage (``gs://...``) URI from which to download continental ERA5 climate tables into ``--era5-cache-dir`` if they are not already on disk.
   * - ``--id-column``
     - No
     - ``gauge_id``
     - Name of the column in your input file that holds the unique watershed ID.
   * - ``--workers``, ``-w``
     - No
     - ``1``
     - Number of CPU processes to run in parallel. Increase this (for example, ``-w 8``) when extracting many watersheds.
   * - ``--min-overlap-threshold``
     - No
     - ``0.0``
     - Minimum overlap area in km² required for a sub-basin fragment to be included (unless the watershed covers more than 50% of that sub-basin). Useful for ignoring tiny border slivers along the edge of a polygon.
   * - ``--clean-cache``
     - No
     - Disabled
     - Delete the local ``--gdb-path`` and ``--era5-cache-dir`` folders after the command finishes.
   * - ``--verbose``, ``-v``
     - No
     - Disabled
     - Print detailed progress and debugging messages while running.

2. Extract Attributes for Many Folders at Once (``extract-caravan-static-batch``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``extract-caravan-static-batch`` (or its alias ``extract-static-attributes-batch``) when you have multiple dataset folders containing watershed boundary files and want to process all of them in a single run.

.. code-block:: bash

   # Process all dataset folders inside a parent folder and also save one combined CSV
   extract-caravan-static-batch \
       --parent-dir /path/to/watershed_folders/ \
       --output-dir /path/to/output_attributes/ \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --era5-source hybas \
       --era5-cache-dir /path/to/era5_climate \
       --workers 16 \
       --combine

   # Process a specific list of folders
   extract-caravan-static-batch \
       --input-dirs /data/shapes/camels /data/shapes/camelsaus /data/shapes/lamah \
       --output-dir /path/to/output_attributes/ \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --era5-source hybas \
       --era5-cache-dir /path/to/era5_climate \
       --workers 16

All Flags for ``extract-caravan-static-batch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You must provide at least one of* ``--parent-dir`` *,* ``--input-dirs`` *, or* ``--input-files`` *.*

.. list-table::
   :widths: 26 18 12 44
   :header-rows: 1

   * - Flag
     - Required?
     - Default
     - What It Does
   * - ``--parent-dir``, ``-p``
     - One input flag required
     - —
     - One or more parent folders (local path or ``gs://...``) that contain dataset subfolders of watershed files.
   * - ``--input-dirs``, ``-d``
     - One input flag required
     - —
     - Space-separated list of specific dataset folders to process.
   * - ``--input-files``, ``-f``
     - One input flag required
     - —
     - Space-separated list of specific polygon files (``.shp``, ``.geojson``, ``.gpkg``, ``.parquet``) to process.
   * - ``--output-dir``, ``-o``
     - **Yes**
     - —
     - Local folder or Google Cloud Storage (``gs://...``) path where output files will be written.
   * - ``--gdb-path``, ``-g``
     - **Yes**
     - —
     - Local path to ``BasinATLAS_v10.gdb`` or ``BasinATLAS_v10_lev12.shp``.
   * - ``--era5-source``
     - **Yes**
     - —
     - How to calculate ERA5 climate numbers: ``hybas`` or ``gridded``.
   * - ``--era5-cache-dir``
     - Required if ``hybas``
     - ``None``
     - Local folder where continental ERA5 climate tables are stored.
   * - ``--gridded-era5-uri``
     - Required if ``gridded``
     - ``None``
     - Google Cloud Storage (``gs://...``) URI or local path for the daily ERA5-Land Zarr dataset.
   * - ``--staging-dir``
     - Required for ``gs://`` inputs/outputs
     - ``None``
     - Local folder used to stage downloaded input shapefiles or output CSVs when reading from or writing to ``gs://`` paths.
   * - ``--gcs-gdb-uri``
     - No
     - ``None``
     - Google Cloud Storage (``gs://...``) URI from which to download ``BasinATLAS_v10.gdb`` into ``--gdb-path`` if not already on disk.
   * - ``--gcs-era5-climate-uri``
     - No
     - ``None``
     - Google Cloud Storage (``gs://...``) URI from which to download continental ERA5 climate tables into ``--era5-cache-dir`` if not already on disk.
   * - ``--id-column``
     - No
     - ``gauge_id``
     - Name of the column in your input files that holds the unique watershed ID.
   * - ``--workers``, ``-w``
     - No
     - ``1``
     - Number of CPU processes to run in parallel.
   * - ``--combine``
     - No
     - Disabled
     - In addition to per-dataset files, save a single merged table (``attributes_caravan_combined.csv``) containing all watersheds across all processed folders.
   * - ``--partition-outputs``, ``-P``
     - No
     - Disabled
     - When enabled, writes separate HydroATLAS and Caravan climate tables (``attributes_hydroatlas_<dataset>.csv``, ``attributes_caravan_<dataset>.csv``, and ``attributes_<dataset>.parquet``) inside each dataset's output subfolder instead of a single flat CSV.
   * - ``--gcs-output-uri``
     - No
     - ``None``
     - Optional Google Cloud Storage (``gs://...``) destination where finished files should be uploaded after saving them locally in ``--output-dir``.
   * - ``--no-resume``
     - No
     - Disabled
     - Re-run and overwrite datasets even if their output files already exist in ``--output-dir`` (by default, already finished datasets are skipped).
   * - ``--min-overlap-threshold``
     - No
     - ``0.0``
     - Minimum overlap area in km² required for a sub-basin fragment to be included.
   * - ``--clean-staging``
     - No
     - Disabled
     - Delete ``--staging-dir`` after the batch run finishes.
   * - ``--clean-cache``
     - No
     - Disabled
     - Delete ``--gdb-path``, ``--era5-cache-dir``, and ``--staging-dir`` when the batch run finishes.
   * - ``--no-progress``
     - No
     - Disabled
     - Turn off interactive progress bars.
   * - ``--verbose``, ``-v``
     - No
     - Disabled
     - Print detailed progress and debugging messages while running.

------------------
Using It in Python
------------------

Extract Attributes from a File to a DataFrame and CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from multimet.static_extractor import StaticAttributesExtractor

   extractor = StaticAttributesExtractor(
       gdb_path="/path/to/BasinATLAS_v10.gdb",
       era5_source="hybas",
       era5_cache_dir="/path/to/era5_climate",
   )
   df = extractor.extract_attributes_from_file(
       input_path="basins.geojson",
       output_csv_path="caravan_attributes.csv",
       id_column="gauge_id",
   )
   print(df.head())

Extract Attributes for a Single Polygon in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import shapely.geometry
   from multimet.static_extractor import StaticAttributesExtractor

   extractor = StaticAttributesExtractor(
       gdb_path="/path/to/BasinATLAS_v10.gdb",
       era5_source="hybas",
       era5_cache_dir="/path/to/era5_climate",
   )

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
       era5_source="hybas",
   )

   attrs = result["caravan_attributes"]
   print("Drainage Area (km²):", result["total_area_km2"])
   print("Mean Elevation (m):", attrs["ele_mt_sav"])
   print("Mean Precipitation (mm/yr):", attrs["pre_mm_syr"])

Add Extracted Attributes to a Zarr Store
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   extractor.append_attributes_to_zarr(
       master_zarr_path="/data/multimet/caravan.zarr",
       basin_id="my_basin_01",
       attributes=attrs,
   )

--------------------------------------------------------------------------------
Checking Results Against Published Caravan Data (``benchmark-static-extractor``)
--------------------------------------------------------------------------------

If you want to compare the numbers produced by this package against published Caravan values across reference basins, run ``benchmark-static-extractor``:

.. code-block:: bash

   # Compare all basins in a reference dataset
   benchmark-static-extractor \
       --dataset /path/to/benchmark_basins_500.parquet \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --era5-source hybas \
       --era5-cache-dir /path/to/era5_climate \
       --workers 14 \
       -o /path/to/benchmark_results/

   # Quick check on a sample of 50 basins
   benchmark-static-extractor \
       --dataset /path/to/benchmark_basins_500.parquet \
       --gdb-path /path/to/BasinATLAS_v10.gdb \
       --era5-source hybas \
       --era5-cache-dir /path/to/era5_climate \
       --samples 50 \
       --workers 8 \
       -o /path/to/benchmark_results/

All Flags for ``benchmark-static-extractor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Flag
     - Required?
     - Default
     - What It Does
   * - ``--dataset``
     - **Yes**
     - —
     - Path to the ``.parquet`` reference dataset file.
   * - ``--gdb-path``
     - **Yes**
     - —
     - Local path to ``BasinATLAS_v10.gdb`` or shapefile.
   * - ``--era5-source``
     - **Yes**
     - —
     - How to calculate ERA5 climate numbers during the benchmark: ``hybas`` or ``gridded``.
   * - ``--output-dir``, ``-o``
     - **Yes**
     - —
     - Folder where the benchmark report and CSV tables are saved.
   * - ``--era5-cache-dir``
     - Required if ``hybas``
     - ``None``
     - Folder containing continental ERA5 climate tables.
   * - ``--gridded-era5-uri``
     - Required if ``gridded``
     - ``None``
     - Google Cloud Storage (``gs://...``) URI or local path for the daily ERA5-Land Zarr dataset.
   * - ``--samples``
     - No
     - All basins
     - Number of basins to sample if you want a quick check instead of running all basins in ``--dataset``.
   * - ``--regions``, ``--datasets``
     - No
     - All datasets
     - Space-separated list of Caravan datasets to include (``camels``, ``camelsaus``, ``camelsbr``, ``camelscl``, ``camelsgb``, ``hysets``, ``lamah``).
   * - ``--size-tiers``
     - No
     - All sizes
     - Space-separated list of basin size groups to include (``1_micro``, ``2_small``, ``3_medium``, ``4_large``, ``5_macro``).
   * - ``--workers``
     - No
     - ``8``
     - Number of parallel CPU processes to use.
