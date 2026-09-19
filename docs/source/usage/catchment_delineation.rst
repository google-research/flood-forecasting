=====================
Catchment Delineation
=====================

The ``catchment_delineation`` module provides high-performance, pure DEM flow-direction watershed delineation capabilities for ``googlehydrology``.
It performs authentic reverse-flow breadth-first search (BFS) graph traversal on high-resolution (90m / 3 arc-second) D8 flow-direction matrices with seamless cross-tile boundary routing.

--------------------
Methodology & Design
--------------------

* **D8 Flow Matrix Routing**: Traverses high-resolution ESRI D8 flow routing grids (HydroSHEDS v1.4 / MERIT Hydro) where each cell encodes downstream outflow direction.
* **Seamless Multi-Tile Traversal**: Automatically bridges across 5°×5° tile boundaries without edge truncation or perimeter artifacts, reconstructing the true natural watershed basin geometry regardless of river basin size.
* **Channel Outlet Snapping**: Given user-specified or gauge coordinates, evaluates upstream channel connectivity within a configurable window (default 12 cells ~1.1 km) to accurately snap coordinates to the physical stream outlet.
* **Geodesic Area Calculation**: Integrates cell ground surface footprints with exact latitude scaling (:math:`\text{lat\_scale} \times \text{lon\_scale}`) to calculate accurate drainage area in :math:`\text{km}^2`.
* **Run-Length Vectorization**: Combines row run-length raster interval fusion with Shapely ``unary_union`` and boundary simplification to output valid GeoJSON, GeoParquet, and ESRI Shapefile geometries.
* **Coverage Boundary Detection**: Detects when pour points or upstream watersheds exceed global DEM coverage bounds (-56° to 60° latitude) and cleanly aborts to prevent partial or truncated polygons.
* **Minimal Dependencies**: Pure Python implementation relying strictly on ``numpy``, ``shapely``, and ``geopandas`` (no GIS servers or GDAL runtime required).

-----------------------------
Data Sources & Cloud Storage
-----------------------------

In this open-source repository, DEM flow-direction grids and ancillary datasets are hosted canonically in Google Cloud Storage (GCS) under ``gs://open-multimet/``:

Where the Paths Are
^^^^^^^^^^^^^^^^^^^

.. list-table:: Canonical Storage Locations
   :widths: 25 35 40
   :header-rows: 1

   * - Resource
     - Path / URI
     - Description
   * - **Remote DEM Tiles**
     - ``gs://open-multimet/ancillary-data/dems/tiles_5deg/``
     - 763 pre-sliced 5°×5° D8 flow-direction tiles (``uint8``, 6000×6000 cells, ~34 MB each)
   * - **Master Continental DEMs**
     - ``gs://open-multimet/ancillary-data/dems/{na,sa,eu,af,as,au}_dir_3s.tif``
     - Full continental HydroSHEDS 3-arc-second flow-direction GeoTIFFs
   * - **Remote Benchmark Catalog**
     - ``gs://open-multimet/ancillary-data/benchmarks/benchmark_basins_1000.parquet``
     - Stratified global evaluation catalog of 1,200 validated reference catchments
   * - **Local Benchmark Catalog**
     - ``~/ancillary-data/benchmarks/benchmark_basins_1000.parquet``
     - Canonical local directory for benchmark evaluation datasets
   * - **Caravan Master Coordinates**
     - ``gs://open-multimet/caravan-new/all_caravan_coordinates.csv``
     - Master coordinate catalog of 26,708 Caravan pour points across 21 subdatasets
   * - **Rederived Catchments**
     - ``gs://open-multimet/caravan-new/<collection>/shapefiles-rederived/<subdataset>/``
     - Partitioned catchment polygons across the three Caravan provenance collections
   * - **Local Tile Cache**
     - ``~/.cache/googlehydrology/dem/``
     - Default local directory where required DEM tiles are cached automatically on first use
   * - **Custom User Path**
     - ``--tiles-dir <path>`` or ``tiles_dir="<path>"``
     - Optional user-supplied directory containing local ``.npy`` tiles

----------------
Python API Usage
----------------

Direct Access via ``googlehydrology``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import googlehydrology

   # Delineate a single catchment from coordinates
   watershed = googlehydrology.delineate_dem(lat=39.6828, lon=-88.7729)

   print("Catchment ID:", watershed["properties"]["catchment_id"])
   print("Basin Area:", watershed["properties"]["area_km2"], "km²")
   print("Upstream Cells:", watershed["properties"]["upstream_cells_count"])
   print("Bounding Box:", watershed["properties"]["bbox"])

Using ``DemDelineator`` Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from catchment_delineation import DemDelineator

   # Uses GCS bucket with local cache (~/.cache/googlehydrology/dem/)
   delineator = DemDelineator()

   # Delineate single catchment
   feature = delineator.delineate(
       lat=39.6828,
       lon=-88.7729,
       snap_window_cells=12,
       catchment_id="USGS_05592500",
   )

   # Output is a GeoJSON Feature dict with standardized properties
   print(feature["properties"])

Batch Processing Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from catchment_delineation import DemDelineator

   delineator = DemDelineator()

   coords = [(39.6828, -88.7729), (40.4172, -86.8858)]
   ids = ["DALTON_CITY", "LAFAYETTE"]

   feature_collection = delineator.delineate_batch(coords, ids=ids)

   for feat in feature_collection["features"]:
       props = feat["properties"]
       print(f"{props['catchment_id']}: {props['area_km2']} km²")

Using a Custom Local Tiles Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from catchment_delineation import DemDelineator

   # Strictly load from custom directory (no GCS download or searching)
   delineator = DemDelineator(tiles_dir="/path/to/custom/tiles")
   watershed = delineator.delineate(lat=39.6828, lon=-88.7729)

----------------------------
Command-Line Interface (CLI)
----------------------------

The package installs the ``delineate-catchment`` CLI command (also runnable via ``python -m catchment_delineation.cli``).

Single Coordinate Pair
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Output GeoJSON to stdout
   delineate-catchment --lat 39.6828 --lon -88.7729 --pretty

   # Save GeoJSON directly to file
   delineate-catchment --lat 39.6828 --lon -88.7729 -o dalton_city.geojson

Multiple Coordinate Pairs
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   delineate-catchment \
     --coords "39.6828,-88.7729" "40.4172,-86.8858" \
     --pretty -o multi_basins.geojson

Batch Processing from CSV
^^^^^^^^^^^^^^^^^^^^^^^^^

Given a CSV file ``gauges.csv`` with ``latitude`` and ``longitude`` headers:

.. code-block:: text

   gauge_id,latitude,longitude
   USGS_05592500,39.6828,-88.7729
   USGS_03335500,40.4172,-86.8858

Run parallel delineation with 8 worker processes:

.. code-block:: bash

   delineate-catchment --csv gauges.csv --workers 8 -o basins.geojson

Direct GCS Input and Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Delineate directly from GCS and write GeoParquet directly back to cloud storage:

.. code-block:: bash

   delineate-catchment \
     --csv gs://open-multimet/caravan-new/all_caravan_coordinates.csv \
     -o gs://open-multimet/caravan-new/all_delineated.geoparquet \
     --workers 16 \
     --clean-cache

Caravan Reorganization & Partitioned Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To delineate all 26,708 Caravan basins and automatically partition them into their canonical collections (``caravan-original``, ``caravan-extensions``, ``google-internal``) across all 21 subdatasets in all formats:

.. code-block:: bash

   delineate-catchment \
     --csv gs://open-multimet/caravan-new/all_caravan_coordinates.csv \
     --output-dir gs://open-multimet/caravan-new \
     --preserve-caravan-dirs \
     --format all \
     --workers 20 \
     --clean-cache

This automatically writes:

* GeoParquet: ``<subdataset>_basin_shapes.geoparquet``
* GeoJSON: ``<subdataset>_basin_shapes.geojson``
* Full Shapefile suite: ``<subdataset>_basin_shapes.shp``, ``.shx``, ``.dbf``, ``.prj``, ``.cpg``

---------------------------
Global Benchmarking Suite
---------------------------

The package includes a comprehensive global benchmarking runner (``benchmark-catchment`` or ``python -m catchment_delineation.benchmark``) to evaluate delineation accuracy against official reference catchment polygons:

* **1,200 Balanced Global Basins**: Dataset stratified equally across all 6 continents (200 each in Africa, Asia, Europe, North America, South America, Oceania), all 4 hemisphere quadrants (NW, NE, SW, SE), and 5 size tiers (micro to macro).
* **Core Spatial Metrics**: Computes Intersection-over-Union (IoU / Jaccard Index), Dice similarity coefficient, relative area bias (:math:`\Delta \text{Area} \%`), and stream snapping distances.
* **Hermetic Execution**: Slices required 5°×5° tiles on demand from ``gs://open-multimet/ancillary-data/dems/tiles_5deg/`` and benchmarks from ``gs://open-multimet/ancillary-data/benchmarks/benchmark_basins_1000.parquet`` (cached locally in ``~/ancillary-data/benchmarks/``).

Running Benchmarks
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run 200-basin benchmark using 20 workers with automatic cache cleanup
   benchmark-catchment --samples 200 --workers 20 --clean-cache -o benchmark_200.csv

   # Run specific continents (e.g. Europe and Africa)
   benchmark-catchment --continents Europe Africa --workers 8

   # Filter by basin size tiers
   benchmark-catchment --size-tiers 1_micro 2_small 3_medium 4_large 5_macro

--------------------
CLI Option Reference
--------------------

Delineation CLI (``delineate-catchment``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 20 40
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - ``--lat``
     - float
     - None
     - Latitude of pour point outlet
   * - ``--lon``
     - float
     - None
     - Longitude of pour point outlet
   * - ``--coords``
     - string(s)
     - None
     - One or more space-separated ``"lat,lon"`` pairs
   * - ``--csv``
     - path / URI
     - None
     - CSV/Parquet file path or ``gs://`` URI (shorthand ``caravan`` supported)
   * - ``--id``
     - string
     - None
     - Custom catchment ID for single-coordinate runs
   * - ``--workers``, ``-w``
     - int
     - 1
     - Number of parallel worker processes for batch processing
   * - ``--snap-window``
     - int
     - 12
     - Search window half-width in cells (~1.1 km at 90m resolution)
   * - ``--max-cells``
     - int
     - 50000000
     - Traversal safety limit for maximum upstream raster cells
   * - ``-o``, ``--output``
     - path / URI
     - stdout
     - Output file path (``.geojson``, ``.parquet``, ``.shp``) or ``gs://`` URI
   * - ``--output-dir``
     - path / URI
     - None
     - Directory or GCS bucket prefix for partitioned catchment outputs
   * - ``--preserve-caravan-dirs``
     - flag
     - False
     - Partition catchments into ``<collection>/shapefiles-rederived/<subdataset>/``
   * - ``--format``
     - choice
     - all
     - Format(s) to write: ``all``, ``geoparquet``, ``geojson``, ``shp``
   * - ``--clean-cache``
     - flag
     - False
     - Automatically purge local DEM tile cache after completion
   * - ``--pretty``
     - flag
     - False
     - Pretty-print output JSON with indentation
   * - ``--list-tiles``
     - flag
     - False
     - Print available ``.npy`` tile files and exit

Benchmark CLI (``benchmark-catchment``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 20 40
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - ``--samples``
     - int
     - 1000
     - Number of basins to evaluate (max 1200)
   * - ``--continents``
     - strings
     - None
     - Filter by continents (e.g. ``Africa``, ``Europe``, ``North America``)
   * - ``--size-tiers``
     - strings
     - None
     - Filter by size tiers (``1_micro``, ``2_small``, ``3_medium``, ``4_large``, ``5_macro``)
   * - ``--workers``
     - int
     - 8
     - Number of parallel worker processes
   * - ``--dataset``
     - path / URI
     - ``~/ancillary-data/benchmarks/...``
     - Custom benchmark dataset file path or ``gs://`` URI
   * - ``--tiles-dir``
     - path
     - None
     - Optional local DEM tile folder (defaults to GCS auto-download)
   * - ``--snap-window``
     - int
     - 12
     - Outlet snap window half-width in cells (~1.1 km)
   * - ``--clean-cache``
     - flag
     - False
     - Automatically purge local DEM tile cache after benchmark completes
   * - ``-o``, ``--output``
     - path
     - ``benchmark_results.csv``
     - Output file path (``.csv`` or ``.parquet``) for detailed per-basin metrics
