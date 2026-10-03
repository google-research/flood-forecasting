# Gridded Weather Archive Builders (`multimet`)

This folder contains command-line tools to download public gridded weather data from NOAA, ECMWF, and NASA and save it into standardized daily Zarr archives.

> **Do you need these tools?**
> If you only want to train or evaluate flood-forecasting models using the published MultiMet dataset, **you do not need to run these tools**. Simply point `dynamics_data_dir` in your training configuration file to `gs://caravan-multimet/v1.1`.
>
> Use these tools only if you want to download raw weather grids directly from the upstream providers and build or update your own Zarr archives.

---

## Overview

Three command-line tools are installed when you run `pip install -e .` from the root of this repository:

| Command | Dataset | Spatial Grid | Available Dates |
| --- | --- | --- | --- |
| `build-cpc-archive` | NOAA CPC Global Unified Daily Precipitation | 0.5° (`360 × 720`) | 1979 to present |
| `build-hres-archive` | ECMWF IFS HRES Daily Surface Forecasts (Lead Days 1–10) | 0.25° (`721 × 1440`) | 2024-03-06 to present (see [section 2](#2-ecmwf-ifs-hres-daily-surface-forecasts-build-hres-archive) for earlier years) |
| `build-imerg-archive` | NASA GPM IMERG Early V07 Daily Precipitation | 0.1° (`1800 × 3600`) | 2000-06-01 to present |

Each tool downloads raw files from the weather agency, converts them onto a consistent daily grid, marks missing values as `NaN`, and saves the result to the `--target_zarr` path you specify.

---

## Prerequisites

Activate the `googlehydrology` Conda environment and install the repository in editable mode:

```bash
conda activate googlehydrology
pip install -e .
```

### Additional Requirements by Dataset

* **NOAA CPC (`build-cpc-archive`):** No extra packages or accounts are required.
* **ECMWF HRES (`build-hres-archive`):** No account is needed. The `eccodes` library is required to read ECMWF's GRIB2 files:
  ```bash
  conda install -c conda-forge eccodes python-eccodes
  ```
* **NASA GPM IMERG (`build-imerg-archive`):** Downloading from NASA GES DISC requires a free [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) account. You can provide your credentials in any of three ways:
  1. A `~/.netrc` file on your computer with entries for `urs.earthdata.nasa.gov` and `gpm1.gesdisc.eosdis.nasa.gov`.
  2. Environment variables: `EARTHDATA_TOKEN` (or `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`).
  3. Command-line arguments: `--earthdata_token` (or `--earthdata_username` and `--earthdata_password`).

---

## 1. NOAA CPC Daily Precipitation (`build-cpc-archive`)

Downloads yearly NetCDF files (`precip.{year}.nc`) from the NOAA Physical Sciences Laboratory, flips latitude so it runs south-to-north (`-89.75` to `+89.75`), shifts longitude to `-179.75` to `+179.75`, and writes the daily variable `cpc_precipitation` (`mm/day`, `float32`).

### Example Usage

```bash
# Build a local archive for 2020 to 2022 and delete temporary downloads
build-cpc-archive \
  --target_zarr ./data/cpc_daily.zarr \
  --start_year 2020 \
  --end_year 2022 \
  --cleanup_cache

# Run the same command later to append newly published days
build-cpc-archive \
  --target_zarr ./data/cpc_daily.zarr \
  --cleanup_cache
```

### Command-Line Arguments (`build-cpc-archive`)

* `--target_zarr` *(required)*: Path where the output Zarr archive is saved. Use a local folder path (e.g., `./data/cpc.zarr`) or a Google Cloud Storage URI starting with `gs://` (e.g., `gs://my-bucket/cpc.zarr`).
* `--start_year`: First year to download (integer, default: `1979`).
* `--end_year`: Last year to download, inclusive (integer, default: current calendar year).
* `--start_date`: Optional start date filter (`YYYY-MM-DD`) if you only want dates on or after a specific day inside `--start_year`.
* `--end_date`: Optional end date filter (`YYYY-MM-DD`) if you only want dates on or before a specific day inside `--end_year`.
* `--cache_dir`: Local folder used to store downloaded NOAA NetCDF files before processing. If omitted, a temporary folder is created and removed automatically when the command finishes.
* `--cleanup_cache`: Deletes each downloaded NetCDF file as soon as it is written to the Zarr archive. Recommended to save disk space.
* `--overwrite`: Deletes the existing Zarr archive at `--target_zarr` and rebuilds it from scratch. If not set, the tool resumes and appends only new dates.
* `--num_workers`: Number of years to download and process in parallel (integer, default: number of CPU cores up to `32`). Set to `1` to run one year at a time.
* `--project`: Google Cloud project ID used for billing and authentication when `--target_zarr` is a `gs://` bucket (default: `None`).
* `--source_url_template`: Custom download URL template containing `{year}` (default: official NOAA PSL URL).

---

## 2. ECMWF IFS HRES Daily Surface Forecasts (`build-hres-archive`)

Builds or updates a daily `0.25°` global surface forecast archive (`721` latitudes `-90.0 .. 90.0` by `1440` longitudes `0.0 .. 359.75`, lead days `1..10`) from **ECMWF Open Data** (`gs://ecmwf-open-data`, `ifs/0p25/oper`, 00:00 UTC run; no account or credentials required).

> **Historical coverage (2016 onward):** Public ECMWF Open Data at `0.25°` contains all required surface variables starting on **2024-03-06**. Earlier years (2016–2024) are built separately using the exact same schema and aggregation rules (`multimet/hres_schema.py`) and published as a pre-built archive (the public bucket path will be added here once released). Use `build-hres-archive` to append new forecast dates to a copy of that archive, or to build a standalone archive from `2024-03-06` onward.

### Dimensions and Coordinates

| Dimension | Values | Description |
| --- | --- | --- |
| `time` | Daily (`YYYY-MM-DD`) | Forecast issue date (00:00 UTC run). |
| `lead_time` | `1 .. 10` | Forecast lead day `d`, covering hours `24*(d-1)` to `24*d` after the 00:00 UTC initialization (e.g., lead day 1 covers hours `+0` to `+24` UTC on the issue date). |
| `latitude` | `721` values (`-90.0 .. 90.0`) | South-to-north in `0.25°` steps. |
| `longitude` | `1440` values (`0.0 .. 359.75`) | Eastward from the prime meridian in `0.25°` steps (`0° .. 360°` convention). |

### Variables

All variables are stored as `float32` in canonical MultiMet / Caravan units:

| Variable | Description | Units |
| --- | --- | --- |
| `temperature_2m_mean` | Daily mean 2 m air temperature | `°C` |
| `temperature_2m_min` | Daily minimum 2 m air temperature | `°C` |
| `temperature_2m_max` | Daily maximum 2 m air temperature | `°C` |
| `total_precipitation_sum` | Daily total precipitation (rain and melted snow) | `mm` |
| `surface_pressure_mean` | Daily mean surface pressure | `kPa` |
| `surface_net_solar_radiation_mean` | Daily mean surface net shortwave (solar) radiation (positive downward) | `W/m²` |
| `surface_net_thermal_radiation_mean` | Daily mean surface net longwave (thermal) radiation (positive downward; typically negative due to net radiative cooling) | `W/m²` |

**Daily aggregation:** Temperature and surface pressure statistics (`mean`, `min`, `max`) are computed over the instantaneous forecast steps in each 24-hour lead window (`24*(d-1) < step <= 24*d`): 8 steps at 3-hour intervals for lead days 1–6 (`+3h .. +144h`) and 4 steps at 6-hour intervals for lead days 7–10 (`+150h .. +240h`). Precipitation and radiation variables are computed by differencing ECMWF's cumulative run totals between steps `24*d` and `24*(d-1)` (with step `0` equal to `0`). See [`multimet/hres_schema.py`](hres_schema.py) for the full specification.

### Example Usage

```bash
# Build a local archive for January 2025
build-hres-archive \
  --target_zarr ./data/hres_daily.zarr \
  --start_date 2025-01-01 \
  --end_date 2025-01-31

# Run later without --start_date to append newly published forecasts up to today
# (also retries any dates previously recorded as missing)
build-hres-archive \
  --target_zarr ./data/hres_daily.zarr

# Re-download and overwrite a specific date range in-place
build-hres-archive \
  --target_zarr ./data/hres_daily.zarr \
  --start_date 2025-01-10 \
  --end_date 2025-01-12 \
  --in_place
```

Each forecast date downloads approximately **110 MB** via byte-range requests and takes roughly **1 minute** per worker. Requires the `eccodes` library (see [Prerequisites](#prerequisites)).

### Command-Line Arguments (`build-hres-archive`)

* `--target_zarr` *(required)*: Path where the output Zarr archive is saved (local folder path or `gs://` URI).
* `--start_date`: First forecast date to include (`YYYY-MM-DD`, must be `2024-03-06` or later).
  * If omitted when updating an existing archive, defaults to the day after the last date in the archive (or the earliest date `>= 2024-03-06` listed in `missing_dates` / `failed_dates`, if earlier).
  * Required when creating a new archive.
* `--end_date`: Last forecast date to include (`YYYY-MM-DD`, default: today). Dates that ECMWF has not finished publishing yet are skipped and can be appended on a subsequent run.
* `--project`: Google Cloud project ID used when writing to a `gs://` target archive (default: `None`). Not needed to read from `gs://ecmwf-open-data`.
* `--ecmwf_open_data_bucket`: Source GCS bucket containing ECMWF Open Data (default: `ecmwf-open-data`).
* `--batch_size`: Number of forecast dates accumulated before each Zarr write (integer, default: `10`).
* `--num_workers`: Number of forecast dates downloaded and decoded in parallel (integer, default: number of CPU cores up to `8`). Each worker uses up to ~1 GB of RAM. Set to `1` for sequential execution.
* `--overwrite`: Deletes the existing Zarr archive at `--target_zarr` and rebuilds it from scratch starting at `--start_date`. **Warning:** Do not use `--overwrite` on an archive containing pre-`2024-03-06` history, as those earlier dates cannot be rebuilt from ECMWF Open Data.
* `--in_place`: Re-downloads and overwrites the requested `--start_date` to `--end_date` dates in-place inside an existing Zarr archive. All requested dates must already exist in the archive.
* `--failure_log`: Optional path where a JSON report of any unpublished (`missing_dates`) dates will be saved.

---

## 3. NASA GPM IMERG Daily Precipitation (`build-imerg-archive`)

Builds a daily `0.1°` global precipitation archive (`1800` latitudes `-89.95 .. 89.95` by `3600` longitudes `-179.95 .. 179.95`) from NASA GPM IMERG Early Run Version 07 (V07), saving the variable `imerg_precipitation` (`mm/day`, `float32`).

### Example Usage

```bash
# Download daily V07 files from NASA GES DISC into a local Zarr archive
build-imerg-archive \
  --target_zarr ./data/imerg_daily.zarr \
  --start_date 2024-01-01 \
  --end_date 2024-01-10 \
  --cleanup_cache

# Build from a local directory of pre-downloaded V07 .nc4 or .RT-H5 files
build-imerg-archive \
  --target_zarr ./data/imerg_daily.zarr \
  --source local \
  --local_dir /path/to/local/imerg_files \
  --start_date 2024-01-01 \
  --end_date 2024-01-10
```

### Command-Line Arguments (`build-imerg-archive`)

* `--target_zarr` *(required)*: Path where the output Zarr archive is saved (local path or `gs://` URI).
* `--start_date`: First date to include in `YYYY-MM-DD` format (default: `2000-06-01`).
* `--end_date`: Last date to include in `YYYY-MM-DD` format (default: yesterday UTC).
* `--source`: Where to read IMERG data from (choices: `gesdisc` or `local`, default: `gesdisc`).
  * `gesdisc`: Downloads official daily V07 NetCDF-4 files from NASA GES DISC over HTTPS.
  * `local`: Reads pre-downloaded V07 files from `--local_dir`.
* `--local_dir`: Path to a local folder containing pre-downloaded IMERG V07 daily NetCDF-4 files (`.nc4` / `.nc`) or 48 half-hourly HDF5 granules (`.RT-H5` / `.HDF5`) per day. Required when `--source local` is used.
* `--earthdata_token`: NASA Earthdata Bearer token for `--source gesdisc` (can also be set via the `EARTHDATA_TOKEN` environment variable).
* `--earthdata_username`: NASA Earthdata username (can also be set via `EARTHDATA_USERNAME` or `~/.netrc`).
* `--earthdata_password`: NASA Earthdata password (can also be set via `EARTHDATA_PASSWORD` or `~/.netrc`).
* `--netrc_path`: Path to a custom `.netrc` file containing Earthdata credentials (default: `~/.netrc`).
* `--cache_dir` (or `--local_cache`): Local folder used to stage files downloaded from NASA GES DISC. If omitted, a temporary folder is created and cleaned up automatically on exit.
* `--cleanup_cache`: Deletes each downloaded NetCDF file immediately after it is processed and removes `--cache_dir` on exit.
* `--batch_size`: Number of daily grids accumulated before each Zarr write (integer, default: `30`).
* `--num_workers`: Number of dates downloaded or extracted in parallel (integer, default: `4`). Keep between `4` and `8` when downloading from NASA GES DISC to avoid server rate limits.
* `--granule_workers`: Number of parallel threads used to read the 48 half-hourly HDF5 files per day when using `--source local` (integer, default: `8`).
* `--overwrite`: Deletes the existing Zarr archive at `--target_zarr` and rebuilds it from scratch.
* `--in_place`: Overwrites the requested `--start_date` to `--end_date` dates in-place inside an existing Zarr archive.
* `--project`: Google Cloud project ID used when writing to a `gs://` bucket (default: `None`).
* `--gesdisc_url`: Custom base URL for NASA GES DISC IMERG V07 daily files.
* `--failure_log`: Optional file path where a JSON report of any missing or failed dates will be saved.

---

## What to Watch Out For (Important Things to Know)

1. **Local Paths vs. Cloud Paths (`gs://`)**
   Any `--target_zarr` path that does not start with a URI scheme (such as `./data/cpc.zarr` or `output/cpc.zarr`) is saved on your local computer. To write to Google Cloud Storage, always include `gs://` at the start of the path (for example, `gs://my-bucket/cpc.zarr`).

2. **Disk Space During Large Downloads**
   Multi-year weather downloads require tens of gigabytes of temporary space. Pass `--cleanup_cache` when running `build-cpc-archive` or `build-imerg-archive` so temporary NetCDF files are deleted as soon as each batch is written to the Zarr store.

3. **Safe Incremental Updates and Future End Dates**
   Running any builder against an existing Zarr archive (without `--overwrite`) automatically resumes from where the archive left off. If you pass an `--end_date` in the future (for example, the end of the current year), unpublished future days at the end of the range are **never** written as empty `NaN` slices. The archive stops at the last date that has valid data, so you can re-run the command at any time to append newly published days.

4. **ECMWF HRES Specifics**
   * **Earliest Open Data date (`2024-03-06`):** `build-hres-archive` raises an error if `--start_date` is earlier than `2024-03-06`. Earlier years come from the pre-built archive (see [Section 2](#2-ecmwf-ifs-hres-daily-surface-forecasts-build-hres-archive)).
   * **Publication delay (~8 hours):** ECMWF publishes the full `00:00 UTC` forecast (`+3h` to `+240h`) roughly 8 hours after initialization. A forecast date is ingested only when all required steps are present; if today's forecast is still being uploaded, it is skipped and picked up on the next run.
   * **Schema verification before updating:** Before writing to an existing archive, `build-hres-archive` verifies that the archive's variables, units, coordinates, and `schema_version` match [`multimet/hres_schema.py`](hres_schema.py), raising an error on any mismatch so incompatible archives are never mixed.
   * **Sub-daily sampling of `min` / `max` temperature:** `temperature_2m_min` and `temperature_2m_max` are the minimum and maximum across the 3-hourly (lead days 1–6) or 6-hourly (lead days 7–10) instantaneous forecast steps, rather than continuous 24-hour extremes.
   * **License:** ECMWF Open Data is published under the [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/). Please attribute ECMWF when redistributing data built with this tool.

5. **Strict Data Integrity (No Silent Fallbacks)**
   * **No version mixing in IMERG:** `build-imerg-archive` accepts only **IMERG Version 07 (V07)** files (variable `precipitation`). Legacy **Version 06 (V06)** files (`precipitationCal`) raise an error immediately so different calibration versions are never mixed.
   * **Complete 48-half-hour requirement for local IMERG HDF5 files:** When summing 48 half-hourly `.RT-H5` files for a day, all 48 half-hours must be present and valid at a grid cell. If any half-hour is missing at a grid cell, that cell is set to `NaN` for the day rather than summing an incomplete day.
   * **Network or file errors stop the run:** If a file is corrupted or a network error persists after retries, the builder stops with an error rather than writing fake or empty data.
