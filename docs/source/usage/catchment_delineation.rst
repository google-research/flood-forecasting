=====================
Catchment Delineation
=====================

The ``catchment_delineation`` module traces the boundary (polygon) and drainage area (:math:`\text{km}^2`) of the watershed upstream of any latitude and longitude point using 90-meter (3-arc-second) flow-direction maps.

-------------
How It Works
-------------

* **Upstream Flow Tracing**: Each 90-meter grid cell in a D8 flow-direction map records which of its 8 neighboring cells water flows into. Starting from your outlet coordinate, the tool traces upstream through every cell that drains into that point.
* **Crosses Tile Borders Seamlessly**: Elevation maps are stored in 5°×5° tiles (``6000 × 6000`` cells each). When a river basin crosses from one tile into another, the tool loads the neighboring tile automatically and stitches the basin into a single continuous polygon without cutting rivers off at tile edges.
* **Snapping to the River Channel**: Streamflow gauge coordinates are often slightly off the center of a river on a 90-meter grid. The tool searches nearby grid cells (default ``12`` cells ≈ ``1.1 km``) to snap the point onto the stream channel.
* **Optional Expected Area Hint**: On wide rivers (such as the Amazon or Danube) or when gauge coordinates are rounded, a small bankside creek may sit closer to the reported coordinate than the center of the main river. If you supply an approximate drainage area (``--expected-area`` or ``--area-col``), the tool searches up to ``80`` cells (``~7.2 km``) for the river channel matching that area (within ``±50%`` by default). **If no nearby channel matches the expected area, the tool logs an error and refuses to output a polygon.**
* **Explicit Paths & Loud Errors**: Nothing is hardcoded. You always provide the path to your tile folder (or ``gs://`` bucket) and your output path. If a required tile is missing or a watershed crosses outside the map's latitude bounds (``-56°S`` to ``60°N``), the tool stops and raises an error rather than returning a partial or made-up polygon.

----------------
Python API Usage
----------------

Single Point
^^^^^^^^^^^^

.. code-block:: python

   from catchment_delineation import DemDelineator

   delineator = DemDelineator(tiles_dir="/path/to/tiles_5deg")

   feature = delineator.delineate(
       lat=39.6828,
       lon=-88.7729,
       catchment_id="USGS_05592500",
       expected_area_km2=480.0,  # optional expected drainage area in km²
   )

   print("Catchment ID:", feature["properties"]["catchment_id"])
   print("Area (km²):", feature["properties"]["area_km2"])
   print("Upstream Cells:", feature["properties"]["upstream_cells_count"])

Batch Processing Multiple Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from catchment_delineation import DemDelineator

   delineator = DemDelineator(tiles_dir="/path/to/tiles_5deg")

   coords = [(39.6828, -88.7729), (40.4172, -86.8858)]
   ids = ["camels_05592500", "camels_03335500"]
   expected_areas = [480.0, 18821.0]  # optional

   feature_collection = delineator.delineate_batch(
       coords=coords,
       ids=ids,
       expected_areas_km2=expected_areas,
   )

   for feat in feature_collection["features"]:
       props = feat["properties"]
       print(props["catchment_id"], props["area_km2"], props["status"])

Reading Tiles from Google Cloud Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When reading tiles from a ``gs://`` bucket, provide both ``gcs_uri`` and a local ``cache_dir`` where downloaded tiles will be stored:

.. code-block:: python

   from catchment_delineation import DemDelineator

   delineator = DemDelineator(
       gcs_uri="gs://your-bucket/tiles_5deg",
       cache_dir="/tmp/dem_tile_cache",
   )
   feature = delineator.delineate(lat=39.6828, lon=-88.7729)
   delineator.clean_created_cache()  # removes only tiles downloaded in this run

----------------------------
Command-Line Interface (CLI)
----------------------------

The package installs the ``delineate-catchment`` command (also runnable via ``python -m catchment_delineation``).

Single Coordinate Pair
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Save GeoJSON to a local file using local DEM tiles
   delineate-catchment \
     --lat 39.6828 \
     --lon -88.7729 \
     --tiles-dir /path/to/tiles_5deg \
     -o dalton_city.geojson \
     --pretty

   # Supply an expected drainage area hint (in km²)
   delineate-catchment \
     --lat 48.25 \
     --lon 16.30 \
     --expected-area 101750 \
     --tiles-dir /path/to/tiles_5deg \
     -o danube_vienna.geojson

Batch Processing from a CSV or Parquet File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a CSV file ``gauges.csv``:

.. code-block:: text

   gauge_id,latitude,longitude,expected_area_km2
   camels_01013500,47.2374,-68.5826,2252.7
   camels_03335500,40.4172,-86.8858,18821.0

Run in parallel across 8 worker processes:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --area-col expected_area_km2 \
     --tiles-dir /path/to/tiles_5deg \
     --workers 8 \
     -o basins.geoparquet

Saving in Standard Caravan Folder Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When your ``gauge_id`` values use the Caravan ``<subdataset>_<id>`` convention (for example, ``camels_01013500`` or ``grdc_6340110``), ``--preserve-caravan-dirs`` saves outputs into ``<output-dir>/shapefiles/<subdataset>/<subdataset>_basin_shapes.*``:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --tiles-dir /path/to/tiles_5deg \
     --output-dir /path/to/caravan_output \
     --preserve-caravan-dirs \
     --format all \
     --workers 8

--------------------
CLI Option Reference
--------------------

Delineation CLI (``delineate-catchment``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - ``--lat``
     - float
     - None
     - Latitude of the outlet point
   * - ``--lon``
     - float
     - None
     - Longitude of the outlet point
   * - ``--coords``
     - string(s)
     - None
     - One or more space-separated ``"lat,lon"`` pairs
   * - ``--csv``
     - path / URI
     - None
     - Path to CSV/Parquet coordinate file (or Caravan directory)
   * - ``--id``
     - string
     - None
     - Custom catchment ID for single-point runs
   * - ``--expected-area``
     - float
     - None
     - Optional expected drainage area (:math:`\text{km}^2`) for single-point snapping
   * - ``--area-col``
     - string
     - None
     - Optional column in ``--csv`` with expected drainage area (:math:`\text{km}^2`)
   * - ``--area-tolerance``
     - float
     - 0.50
     - Allowed relative tolerance around expected area (``0.50`` = ``±50%``)
   * - ``--tiles-dir``
     - path / URI
     - None
     - Folder (or ``gs://`` URI) containing 5°×5° ``.npy`` flow-direction tiles
   * - ``--gcs-uri``
     - URI
     - None
     - GCS URI containing 5°×5° ``.npy`` tiles (requires ``--cache-dir``)
   * - ``--cache-dir``
     - path
     - None
     - Local folder used to store tiles downloaded from ``--gcs-uri``
   * - ``--snap-window``
     - int
     - 12
     - Search window half-width in cells (~1.1 km at 90m resolution)
   * - ``--max-cells``
     - int
     - None
     - Optional upstream cell limit (default ``None``: no river is cut off)
   * - ``--workers``, ``-w``
     - int
     - 1
     - Number of parallel worker processes for batch runs
   * - ``-o``, ``--output``
     - path / URI
     - stdout
     - Output file path (``.geojson``, ``.geoparquet``, ``.parquet``, ``.shp``)
   * - ``--output-dir``
     - path / URI
     - None
     - Output directory for partitioned outputs
   * - ``--preserve-caravan-dirs``
     - flag
     - False
     - Save outputs into ``<output-dir>/shapefiles/<subdataset>/<subdataset>_basin_shapes.*``
   * - ``--format``
     - choice
     - all
     - Format(s) to write: ``all``, ``geoparquet``, ``geojson``, ``shp``
   * - ``--clean-cache``
     - flag
     - False
     - Delete only the ``.npy`` tile files downloaded during this run
   * - ``--pretty``
     - flag
     - False
     - Pretty-print JSON output with indentation
   * - ``--list-tiles``
     - flag
     - False
     - List available ``.npy`` tile files in ``--tiles-dir`` and exit

Benchmark CLI (``benchmark-catchment``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   benchmark-catchment \
     --dataset /path/to/benchmark_basins_1000.parquet \
     --tiles-dir /path/to/tiles_5deg \
     --workers 16 \
     --output benchmark_results.csv
