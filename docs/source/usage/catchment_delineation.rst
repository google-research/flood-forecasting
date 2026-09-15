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
* **Channel Outlet Snapping**: Given user-specified or gauge coordinates, evaluates upstream channel connectivity within a configurable window (default 4 cells ~360m) to accurately snap coordinates to the physical stream outlet.
* **Geodesic Area Calculation**: Integrates cell ground surface footprints with exact latitude scaling (:math:`\text{lat\_scale} \times \text{lon\_scale}`) to calculate accurate drainage area in :math:`\text{km}^2`.
* **Run-Length Vectorization**: Combines row run-length raster interval fusion with Shapely ``unary_union`` and boundary simplification to output valid GeoJSON Polygon and MultiPolygon geometries.
* **Minimal Dependencies**: Pure Python implementation relying strictly on ``numpy`` and ``shapely`` (no GIS servers or GDAL runtime required).

-----------------------------
Data Sources & Cloud Storage
-----------------------------

In this open-source repository, DEM flow-direction data comes exclusively from the official Google Cloud Storage (GCS) bucket:

.. code-block:: text

   gs://open-multimet/data/DEMs/tiles_5deg/

Where the Paths Are
^^^^^^^^^^^^^^^^^^^

* **GCS Bucket (Default Remote Source)**: ``gs://open-multimet/data/DEMs/tiles_5deg/`` containing 119 pre-sliced 5°×5° D8 flow-direction tiles (``uint8``, 6000×6000 cells, ~34 MB each, named ``n{lat}w{lon}.npy``).
* **Local Cache Directory**: ``~/.cache/googlehydrology/dem/`` where required tiles are cached automatically on first use.
* **Custom User Path**: Users can optionally supply their own path to local tiles via the ``--tiles-dir`` command line flag or ``tiles_dir=...`` Python argument.

No Path Searching
^^^^^^^^^^^^^^^^^

There is no arbitrary candidate path scanning or fallback search. Tiles are retrieved strictly from the GCS bucket unless the user provides an explicit custom path.

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
       snap_window_cells=4,
       catchment_id="USGS_05592500",
   )

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

The package installs the ``delineate-catchment`` CLI command (also runnable via ``python -m catchment_delineation``).

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

   id,latitude,longitude
   USGS_05592500,39.6828,-88.7729
   USGS_03335500,40.4172,-86.8858

Run delineation:

.. code-block:: bash

   delineate-catchment --csv gauges.csv -o basins.geojson

Supplying a Custom Tiles Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   delineate-catchment \
     --lat 39.6828 --lon -88.7729 \
     --tiles-dir /path/to/my/tiles \
     -o custom_basin.geojson

Listing Available Cached Tiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   delineate-catchment --list-tiles
