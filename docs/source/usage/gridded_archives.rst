=========================
Gridded Archive Builders
=========================

The :mod:`multimet` package provides command-line tools to download public
weather datasets and store them as standardized, daily Zarr archives on your
local disk or in Google Cloud Storage.

.. note::

   **Do I need to run these tools?**
   If you are training or evaluating models with the published Caravan MultiMet
   dataset, **no**—you can point your training configuration directly to
   ``gs://caravan-multimet/v1.1``. You only need these tools if you want to
   build or update your own gridded weather archives from the original data
   providers (NOAA, ECMWF, or NASA).

.. list-table::
   :header-rows: 1
   :widths: 22 28 15 15 20

   * - Command
     - Weather Dataset
     - Resolution
     - Coverage
     - Output Store
   * - ``build-cpc-archive`` (:mod:`multimet.build_cpc_archive`)
     - NOAA CPC Global Unified daily precipitation
     - 0.5° (``360 × 720``)
     - 1979 – present
     - Required (``--target_zarr``)
   * - ``build-hres-archive`` (:mod:`multimet.build_hres_archive`)
     - ECMWF IFS HRES daily surface forecasts (lead days 1–10)
     - 0.25° (``721 × 1440``)
     - 2016 – present
     - Required (``--target_zarr``)
   * - ``build-imerg-archive`` (:mod:`multimet.build_imerg_archive`)
     - NASA GPM IMERG Early V07 daily precipitation
     - 0.1° (``1800 × 3600``)
     - 2000 – present
     - Required (``--target_zarr``)

------------------------
Setup and Requirements
------------------------

Install the repository in editable mode from the root folder:

.. code-block:: bash

   pip install -e .

   build-cpc-archive --help
   build-hres-archive --help
   build-imerg-archive --help

Extra Requirements for Specific Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **ECMWF HRES (for dates from July 13, 2023 onward):** Reading ECMWF's GRIB2
  forecast files requires the ``eccodes`` library:

  .. code-block:: bash

     conda install -c conda-forge eccodes python-eccodes

* **NASA GPM IMERG (when downloading from NASA GES DISC):** Requires a free
  NASA Earthdata Login account. Credentials can be supplied via ``~/.netrc``,
  environment variables (``EARTHDATA_TOKEN``, or ``EARTHDATA_USERNAME`` and
  ``EARTHDATA_PASSWORD``), or CLI flags (``--earthdata_token``, or
  ``--earthdata_username`` and ``--earthdata_password``).

------------------------------
1. NOAA CPC Daily Precipitation
------------------------------

``build-cpc-archive`` downloads yearly NetCDF files (``precip.{year}.nc``) from
the NOAA Physical Sciences Laboratory
(``https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/``) and converts
them into a daily Zarr store.

Data Transformations
^^^^^^^^^^^^^^^^^^^^

#. **Flips latitude to south-to-north order:** Converts NOAA's descending
   latitude axis (``+89.75 .. -89.75``) to ascending order
   (``-89.75 .. +89.75``).
#. **Shifts longitude to** ``[-180, 180)``: Rolls NOAA's ``0.25 .. 359.75``
   longitude axis to ``-179.75 .. +179.75``.
#. **Masks missing values to NaN:** Replaces negative fill values
   (``-9.96921e36``) with ``np.nan``.
#. **Strips unpublished future days:** Removes trailing pre-allocated ``NaN``
   days at the end of the current calendar year so future runs can cleanly
   append new days as they are published.

Output Schema
^^^^^^^^^^^^^

.. code-block:: text

   Dimensions:            (time, latitude, longitude)
   Coordinates:
     * time               datetime64[ns]     daily, midnight UTC
     * latitude           float32   360      -89.75 .. 89.75  (ascending)
     * longitude          float32   720      -179.75 .. 179.75
   Data variables:
       cpc_precipitation  float32   (time, latitude, longitude)   mm/day

Example Commands
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Build a local archive for 2020-2022 and clean up downloaded NetCDF files
   build-cpc-archive \
     --target_zarr ./data/cpc_daily.zarr \
     --start_year 2020 \
     --end_year 2022 \
     --cleanup_cache

   # Run again later to append any newly published days
   build-cpc-archive \
     --target_zarr ./data/cpc_daily.zarr \
     --cleanup_cache

------------------------------------
2. ECMWF HRES Daily Surface Forecasts
------------------------------------

``build-hres-archive`` builds a 10-day daily surface forecast archive (lead
days 1 through 10 initialized at 00:00 UTC each day) on a ``0.25°`` global grid
(``721 × 1440``):

* **2016-01-01 to 2023-01-10:** Read from the public WeatherBench 2 Zarr
  archive (``gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr``).
  Solar and thermal radiation are not archived in WeatherBench 2 and are stored
  as ``NaN`` for these dates.
* **2023-01-11 to 2023-07-12:** Filled with ``NaN`` slices when spanned by a
  multi-year build to keep the daily time coordinate contiguous; can be updated
  in-place using ``--in_place``.
* **2023-07-13 to present:** Read from the ``0.25°`` operational ECMWF Open
  Data GRIB2 archive (``gs://ecmwf-open-data/{YYYYMMDD}/00z/ifs/0p25/oper/``).

Output Schema
^^^^^^^^^^^^^

.. code-block:: text

   Dimensions:                       (time, lead_time, latitude, longitude)
   Coordinates:
     * time                          datetime64[ns]   forecast init date (00z)
     * lead_time                     int32     10     1 .. 10 (days ahead)
     * latitude                      float32   721    -90 .. 90
     * longitude                     float32   1440   0 .. 359.75
   Data variables:  (all float32, dims (time, lead_time, latitude, longitude))
       temperature_2m                        K       (24h mean)
       surface_pressure                      Pa      (24h mean)
       total_precipitation                   m       (24h accumulation)
       surface_net_solar_radiation           J/m^2   (24h accumulation)
       surface_net_thermal_radiation         J/m^2   (24h accumulation)

Example Commands
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Build a local HRES archive for a date range
   build-hres-archive \
     --target_zarr ./data/hres_daily.zarr \
     --start_date 2024-06-01 \
     --end_date 2024-06-10

   # Overwrite existing dates in-place
   build-hres-archive \
     --target_zarr ./data/hres_daily.zarr \
     --start_date 2024-06-05 \
     --end_date 2024-06-07 \
     --in_place

----------------------------------
3. NASA GPM IMERG Daily Precipitation
----------------------------------

``build-imerg-archive`` builds a daily ``0.1°`` global precipitation archive
(``1800 × 3600``) from NASA GPM IMERG Early Run V07:

* ``--source gesdisc`` (default): Downloads official daily V07 NetCDF-4 files
  (``3B-DAY-E.MS.MRG.3IMERG.*.V07*.nc4``) from NASA GES DISC.
* ``--source local --local_dir /path/to/files``: Reads pre-downloaded daily V07
  NetCDF-4 files (``.nc4`` / ``.nc``) or 48 half-hourly V07 HDF5 files
  (``.RT-H5`` / ``.HDF5``) per day from a local directory. When summing 48
  half-hourly HDF5 granules, all 48 half-hourly observations at a grid cell
  must be valid for the daily total to be finite; any cell with missing
  half-hours is masked to ``NaN``.

Output Schema
^^^^^^^^^^^^^

.. code-block:: text

   Dimensions:             (time, latitude, longitude)
   Coordinates:
     * time                datetime64[ns]     daily, midnight UTC
     * latitude            float32   1800     -89.95 .. 89.95  (ascending)
     * longitude           float32   3600     -179.95 .. 179.95
   Data variables:
       imerg_precipitation float32   (time, latitude, longitude)   mm/day

Example Commands
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Download from NASA GES DISC into a local Zarr store
   build-imerg-archive \
     --target_zarr ./data/imerg_daily.zarr \
     --start_date 2024-01-01 \
     --end_date 2024-01-10 \
     --cleanup_cache

   # Build from a local folder of pre-downloaded NetCDF-4 or HDF5 files
   build-imerg-archive \
     --target_zarr ./data/imerg_daily.zarr \
     --source local \
     --local_dir /path/to/local/imerg_files \
     --start_date 2024-01-01 \
     --end_date 2024-01-10

------------------------------------------
Resume, Missing Dates, and Target Paths
------------------------------------------

* **Automatic Resume and Backfill:** Running a builder on an existing Zarr
  store automatically retries any previously missing dates (recorded in Zarr
  root attributes ``missing_dates`` / ``failed_dates`` or trailing ``NaN``
  slices) in-place and appends new dates after the end of the store.
* **No Trailing NaN Lockout:** Unpublished dates at the end of a requested
  ``--end_date`` window are never written as ``NaN`` slices, so the store's time
  axis ends at the last available valid date.
* **Local vs. Cloud Paths:** Any ``--target_zarr`` path without a URI scheme
  (such as ``./out.zarr`` or ``output/cpc.zarr``) is written to the local
  filesystem. Pass an explicit ``gs://`` URI to write to Google Cloud Storage.
