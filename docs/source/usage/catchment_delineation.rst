=====================
Catchment Delineation
=====================

This guide explains how to create watershed boundary polygons and calculate drainage areas (:math:`\text{km}^2`) for streamflow gauges using the ``delineate-catchment`` tool.

--------
Overview
--------

To extract weather forcing data (MultiMet) or static catchment attributes (Caravan) for a new streamflow gauge, you first need the **polygon boundary** of the land area that drains into that gauge.

Given the latitude and longitude of a river gauge, ``delineate-catchment`` produces that polygon in three steps:

1. **Snaps to the river channel:** Gauge coordinates recorded by water agencies are often a few hundred meters away from the center of the river on a digital map. The tool searches nearby 90-meter grid cells (about ``1.1 km`` by default) to place the point directly on the river channel. If you also know the approximate drainage area of the gauge (``--expected-area`` or ``--area-col``), the tool searches up to ``7.2 km`` around the point to find the specific river channel matching that area.
2. **Traces upstream water flow:** Each 90-meter pixel in a flow-direction map records which of its 8 neighboring pixels water flows into. Starting from the snapped river point, the tool follows water flow backward (upstream) through every pixel that drains into the gauge. If a large river crosses from one 5°×5° map tile into neighboring tiles, the tool loads those tiles automatically so rivers are never cut off at tile borders.
3. **Saves the polygon:** All upstream pixels are combined into a single boundary polygon and saved as a **GeoJSON**, **GeoParquet**, or **ESRI Shapefile** with the calculated drainage area in square kilometers (:math:`\text{km}^2`).

-----------------
Data Requirements
-----------------

Before running ``delineate-catchment``, you need a folder (local or on Google Cloud Storage ``gs://``) containing 90-meter (3-arc-second) D8 flow-direction map tiles saved as 5°×5° NumPy (``.npy``) files (such as ``n35w090_dir.npy``).

* **If you already have a folder or GCS bucket of 5°×5° ``.npy`` tiles:** Pass that folder directly with ``--tiles-dir /path/to/tiles_5deg`` (or ``--gcs-uri gs://... --cache-dir /tmp/tile_cache``).
* **If you are starting from raw HydroSHEDS or MERIT GeoTIFF files:** Use ``scripts/slice_continental_dems.py`` to slice the continental ``.tif`` files into 5°×5° ``.npy`` tiles:

.. code-block:: bash

   python scripts/slice_continental_dems.py \
     --source-dir /path/to/raw_hydrosheds_tifs \
     --output-dir /path/to/tiles_5deg

-----------
Quick Start
-----------

1. Delineate a Single Gauge
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run ``delineate-catchment`` with ``--lat``, ``--lon``, ``--tiles-dir``, and an output file ``-o``:

.. code-block:: bash

   delineate-catchment \
     --lat 39.6828 \
     --lon -88.7729 \
     --id USGS_05592500 \
     --expected-area 480.0 \
     --tiles-dir /path/to/tiles_5deg \
     -o basin.geojson \
     --pretty

.. tip::
   Whenever your water agency provides a published drainage area for a gauge, pass it with ``--expected-area`` (in :math:`\text{km}^2`). This guarantees the point snaps to the main river rather than a small nearby creek.

2. Delineate a Table of Gauges (CSV or Parquet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have many gauges, put them in a CSV or Parquet file (for example, ``gauges.csv``):

.. code-block:: text

   gauge_id,latitude,longitude,area_km2
   camels_01013500,47.2374,-68.5826,2252.7
   camels_03335500,40.4172,-86.8858,18821.0

Then run ``delineate-catchment`` with ``--csv`` and ``--workers`` to process multiple gauges at the same time:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --area-col area_km2 \
     --tiles-dir /path/to/tiles_5deg \
     --workers 8 \
     -o basins.geoparquet

3. Save in Standard Caravan Folder Layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your ``gauge_id`` values follow the Caravan naming style ``<subdataset>_<id>`` (such as ``camels_01013500`` or ``grdc_6340110``), add ``--preserve-caravan-dirs`` and ``--output-dir``. This saves the polygons into the folder structure expected by Caravan and MultiMet tools:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --area-col area_km2 \
     --tiles-dir /path/to/tiles_5deg \
     --output-dir /path/to/caravan_dataset \
     --preserve-caravan-dirs \
     --format all \
     --workers 8

This creates:

.. code-block:: text

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

4. Read Map Tiles from Google Cloud Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your 5°×5° ``.npy`` tiles are stored in a ``gs://`` bucket, provide both ``--gcs-uri`` and a local ``--cache-dir`` where downloaded tiles can be stored temporarily. Add ``--clean-cache`` if you want those downloaded files deleted automatically when the command finishes:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --area-col area_km2 \
     --gcs-uri gs://your-bucket/tiles_5deg \
     --cache-dir /tmp/dem_tile_cache \
     --clean-cache \
     --workers 8 \
     -o basins.geoparquet

5. Use from Python
^^^^^^^^^^^^^^^^^^

You can also run delineation directly inside a Python script or notebook:

.. code-block:: python

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

-----------------------
What to Watch Out For
-----------------------

Please keep these five rules in mind when running the tool:

1. **You must supply all file paths yourself (no hidden defaults)**
   The tool never guesses file locations and never falls back to hidden paths. You must always provide the path to your map tiles (``--tiles-dir`` or ``--gcs-uri`` + ``--cache-dir``) and your output path (``-o`` or ``--output-dir``).

2. **Gauges on wide rivers need an expected area hint (``--expected-area`` or ``--area-col``)**
   On a 90-meter grid, a wide river (like the Danube, Mississippi, or Amazon) is many pixels wide. A gauge coordinate near the riverbank can sit closer to a tiny creek on the bank than to the center of the main river.

   * Passing ``--expected-area`` (or ``--area-col`` for CSV tables) tells the tool to find the nearby channel whose drainage area is within ``±50%`` (configurable via ``--area-tolerance``) of the expected area.
   * **Loud failure if no river matches:** If you provide an expected area and no river within ``~7.2 km`` matches that area, the tool **will not guess or output a wrong polygon**. It logs a ``[AREA HINT FAILURE]`` error and refuses to produce a polygon for that gauge.

3. **Latitude limit (``-56°S`` to ``60°N``)**
   The 90-meter HydroSHEDS flow-direction maps cover latitudes from ``-56°S`` to ``60°N``.

   * If a gauge is north of ``60°N`` (such as in northern Norway, Sweden, Finland, Alaska, or northern Canada), it is outside map coverage.
   * Likewise, if a gauge sits south of ``60°N`` (for example at ``59.8°N``) but its upstream river reaches north across ``60°N``, the tool stops rather than cutting the river off at the ``60°N`` border. In batch runs, these basins are recorded with ``status: "out_of_coverage"`` and ``geometry: null``.

4. **All upstream map tiles must be present**
   Large rivers can start hundreds of kilometers away from the gauge and cross several 5°×5° map tiles. If your ``--tiles-dir`` has the tile for the gauge location but is missing an upstream tile that flows into that river, the tool stops and raises an error instead of returning a chopped-off polygon.

5. **Coordinate order and measurement units**

   * **Coordinates:** Always use standard decimal degrees (``EPSG:4326`` / WGS84) with **latitude first** (``-56`` to ``60``) and **longitude second** (``-180`` to ``180``).
   * **Drainage areas:** All area inputs (``--expected-area``, ``--area-col``) and outputs (``area_km2``, ``area``) are in **square kilometers** (:math:`\text{km}^2`).

------------------------------
Command-Line Arguments (CLI)
------------------------------

All arguments for ``delineate-catchment`` (also runnable as ``python -m catchment_delineation``) are listed below by group.

Coordinate Input Options
^^^^^^^^^^^^^^^^^^^^^^^^

-  ``--lat`` *(float, default: None)*: Latitude of a single river gauge or outlet point in decimal degrees (e.g., ``39.6828``). Must be used together with ``--lon``.
-  ``--lon`` *(float, default: None)*: Longitude of a single river gauge or outlet point in decimal degrees (e.g., ``-88.7729``). Must be used together with ``--lat``.
-  ``--id`` *(string, default: None)*: Custom name or gauge ID when running a single point with ``--lat`` and ``--lon`` (e.g., ``USGS_05592500``). If omitted, an ID is generated automatically from the coordinates.
-  ``--coords`` *(one or more strings, default: None)*: Space-separated ``"lat,lon"`` coordinate pairs for running a few points without creating a CSV file (e.g., ``--coords "39.68,-88.77" "40.42,-86.89"``).
-  ``--csv`` *(path or gs:// URI, default: None)*: Path to a CSV or Parquet file containing gauge coordinates, or a Caravan directory containing ``attributes/`` tables.
-  ``--lat-col`` *(string, default: auto-detected)*: Name of the latitude column in your ``--csv`` file. You only need this if your column is not named ``latitude``, ``lat``, ``gauge_lat``, ``caravan:gauge_lat``, ``outlet_lat``, or ``pour_point_lat``.
-  ``--lon-col`` *(string, default: auto-detected)*: Name of the longitude column in your ``--csv`` file. You only need this if your column is not named ``longitude``, ``lon``, ``long``, ``lng``, ``gauge_lon``, ``caravan:gauge_lon``, ``outlet_lon``, or ``pour_point_lon``.
-  ``--id-col`` *(string, default: auto-detected)*: Name of the gauge ID column in your ``--csv`` file. You only need this if your column is not named ``gauge_id``, ``catchment_id``, ``station_id``, ``hybas_id``, ``id``, or ``caravan:gauge_id``.
-  ``--workers``, ``-w`` *(int, default: 1)*: Number of CPU processes to run in parallel during batch runs. Set this to the number of CPU cores on your machine (for example, ``--workers 8``) to speed up CSV processing.

Map Tile & River Snapping Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``--tiles-dir`` *(path or gs:// URI, default: None)*: Folder containing the 5°×5° ``.npy`` flow-direction tiles. You must provide either ``--tiles-dir`` or ``--gcs-uri``. If you pass a ``gs://`` URI to ``--tiles-dir``, you must also specify ``--cache-dir``.
-  ``--gcs-uri`` *(gs:// URI, default: None)*: Google Cloud Storage folder containing the 5°×5° ``.npy`` flow-direction tiles. Must be paired with ``--cache-dir``.
-  ``--cache-dir`` *(local path, default: None)*: Local folder where tiles downloaded from Google Cloud Storage are saved. Required whenever ``--gcs-uri`` is used or when ``--tiles-dir`` points to a ``gs://`` path.
-  ``--snap-window`` *(int, default: 12)*: Half-width of the search box (in 90-meter grid cells) used to snap the gauge coordinate onto the nearest river channel. The default of ``12`` cells searches roughly ``1.1 km`` in each direction around the input coordinate.
-  ``--expected-area`` *(float, default: None)*: Approximate expected drainage area in :math:`\text{km}^2` for a single gauge (or applied to all points if ``--area-col`` is not set). When set, the tool searches up to ``80`` grid cells (``~7.2 km``) for a river channel whose upstream area matches ``--expected-area`` within ``--area-tolerance``. If no river matches, the tool logs an error and refuses to output a polygon.
-  ``--area-col`` *(string, default: None)*: Column name in your ``--csv`` or Parquet file containing the expected drainage area (:math:`\text{km}^2`) for each gauge. Rows with an empty/NaN value in this column fall back to regular snapping without an area hint.
-  ``--area-tolerance`` *(float, default: 0.50)*: Allowed relative difference between the delineated area and ``--expected-area`` / ``--area-col``. The default ``0.50`` accepts a river channel whose area is within ``±50%`` (from ``0.5×`` to ``1.5×``) of the expected area.
-  ``--max-cells`` *(int, default: None)*: Optional upper limit on the number of 90-meter grid cells traced upstream. By default this is ``None`` (no limit), so even continental rivers are traced all the way to their source. If you set a number and a river exceeds it, the tool aborts that basin rather than returning a cut-off polygon.

Output & Utility Options
^^^^^^^^^^^^^^^^^^^^^^^^

-  ``-o``, ``--output`` *(path or gs:// URI, default: stdout)*: File path where the delineated polygons are saved. Supported file extensions are ``.geojson`` (or ``.json``), ``.geoparquet`` (or ``.parquet``), and ``.shp`` (ESRI Shapefile). If neither ``-o`` nor ``--output-dir`` is given, GeoJSON is printed to the terminal.
-  ``--output-dir`` *(path or gs:// URI, default: None)*: Root folder for saving multi-file or Caravan-structured outputs.
-  ``--preserve-caravan-dirs`` *(flag, default: False)*: Groups gauges by the dataset prefix in ``gauge_id`` (the text before the first underscore, such as ``camels`` in ``camels_01013500``) and saves outputs into ``<output-dir>/shapefiles/<subdataset>/<subdataset>_basin_shapes.*``.
-  ``--format`` *(choice: all, geoparquet, parquet, geojson, shp; default: all)*: File format(s) to write when using ``--output-dir`` or ``--preserve-caravan-dirs``.
-  ``--pretty`` *(flag, default: False)*: Formats JSON/GeoJSON output with indentation and line breaks so it is easier for humans to read.
-  ``--clean-cache`` *(flag, default: False)*: Deletes only the ``.npy`` tile files that were downloaded into ``--cache-dir`` during this run, leaving any previously existing files untouched.
-  ``--list-tiles`` *(flag, default: False)*: Lists all ``.npy`` tile files found in ``--tiles-dir`` and exits immediately.

---------------------------------------
Benchmark CLI (``benchmark-catchment``)
---------------------------------------

The repository also provides ``benchmark-catchment`` to measure polygon accuracy (Intersection-over-Union, Dice score, and relative area error) against a reference dataset of published gauge polygons:

.. code-block:: bash

   benchmark-catchment \
     --dataset /path/to/benchmark_basins_1000.parquet \
     --tiles-dir /path/to/tiles_5deg \
     --workers 16 \
     --output benchmark_results.csv

Benchmark Arguments
^^^^^^^^^^^^^^^^^^^

-  ``--dataset`` *(path or gs:// URI, required)*: Path to the benchmark Parquet file containing reference gauge coordinates, reference areas (``reference_area_km2``), and reference polygons (``geometry_wkb``).
-  ``--tiles-dir`` *(path or gs:// URI, default: None)*: Local folder (or ``gs://`` URI) containing the 5°×5° ``.npy`` flow-direction tiles.
-  ``--gcs-uri`` *(gs:// URI, default: None)*: Google Cloud Storage URI containing the 5°×5° ``.npy`` tiles (requires ``--cache-dir``).
-  ``--cache-dir`` *(local path, default: None)*: Local folder used to store tiles downloaded from ``--gcs-uri``.
-  ``--output`` *(path, default: None)*: Optional output file path (``.csv`` or ``.parquet``) to save per-basin benchmark scores.
-  ``--workers`` *(int, default: 8)*: Number of parallel CPU worker processes.
-  ``--samples`` *(int, default: None)*: Optional number of basins to randomly sample (balanced across continents and basin sizes) for a faster test run.
-  ``--continents`` *(one or more strings, default: None)*: Run only on basins in the listed continents (e.g., ``--continents Europe "North America"``).
-  ``--size-tiers`` *(one or more strings, default: None)*: Run only on specific basin size buckets (``1_micro``, ``2_small``, ``3_medium``, ``4_large``, ``5_macro``).
-  ``--snap-window`` *(int, default: 12)*: Search window half-width in 90-meter grid cells around each gauge.
-  ``--no-area-hint`` *(flag, default: False)*: Turn off passing ``reference_area_km2`` as the ``--expected-area`` hint during benchmark runs (area hints are enabled by default).
-  ``--area-tolerance`` *(float, default: 0.50)*: Allowed relative tolerance around ``reference_area_km2`` when area hints are enabled (default ``0.50`` = ``±50%``).
-  ``--clean-cache`` *(flag, default: False)*: Delete only the ``.npy`` tile files downloaded into ``--cache-dir`` during the benchmark run.
