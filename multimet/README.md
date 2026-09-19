# MultiMet Gridded Weather Archive Builders

The `multimet` package provides command-line tools to download public weather datasets and store them as standardized, daily Zarr archives (either on your local disk or in a cloud bucket such as Google Cloud Storage).

> **Do I need to run these tools?**
> If you are training or evaluating models with the published Caravan MultiMet dataset, **no**—you can point your training configuration directly to `gs://caravan-multimet/v1.1`. You only need these tools if you want to build or update your own gridded weather archives from the original data providers (NOAA, ECMWF, or NASA).

## Overview of Available Builders

| Command | Python Module | Weather Dataset | Grid Resolution | Time Range |
| --- | --- | --- | --- | --- |
| `build-cpc-archive` | `multimet.build_cpc_archive` | NOAA CPC Global Unified Daily Precipitation | 0.5° (`360 × 720`) | 1979 – present |
| `build-hres-archive` | `multimet.build_hres_archive` | ECMWF IFS HRES Daily Surface Forecasts (Lead Days 1–10) | 0.25° (`721 × 1440`) | 2016 – present |
| `build-imerg-archive` | `multimet.build_imerg_archive` | NASA GPM IMERG Early V07 Daily Precipitation | 0.1° (`1800 × 3600`) | 2000 – present |

All three tools require you to specify the output path via `--target_zarr` (for example, `--target_zarr ./data/cpc.zarr` for a local directory, or `--target_zarr gs://my-bucket/cpc.zarr` for Google Cloud Storage).

---

## Setup and Requirements

After activating your Conda environment (`conda activate googlehydrology`), install the repository in editable mode from the root folder:

```bash
pip install -e .
```

This installs the `build-cpc-archive`, `build-hres-archive`, and `build-imerg-archive` commands.

### Extra Requirements for Specific Datasets

* **ECMWF HRES (for dates from July 13, 2023 onward):** Reading ECMWF's GRIB2 forecast files requires the `eccodes` C library and Python bindings:
  ```bash
  conda install -c conda-forge eccodes python-eccodes
  ```
* **NASA GPM IMERG (when downloading from NASA GES DISC):** Requires a free [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) account. You can supply your credentials in any of three ways:
  1. A `~/.netrc` file containing entries for `urs.earthdata.nasa.gov` and `gpm1.gesdisc.eosdis.nasa.gov`
  2. Environment variables (`EARTHDATA_TOKEN`, or `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`)
  3. Command-line flags (`--earthdata_token`, or `--earthdata_username` and `--earthdata_password`)

---

## 1. NOAA CPC Daily Precipitation (`build-cpc-archive`)

Downloads yearly NetCDF files (`precip.{year}.nc`) from the NOAA Physical Sciences Laboratory (`https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/`) and converts them into a single Zarr store.

### What It Does to the Data
1. **Flips latitude to south-to-north order:** Changes latitude from NOAA's north-to-south order (`+89.75` down to `-89.75`) to ascending order (`-89.75` to `+89.75`).
2. **Shifts longitude to `[-180, 180)`:** Shifts NOAA's `0.25 .. 359.75` longitude axis to `-179.75 .. +179.75`.
3. **Replaces missing values with `NaN`:** Masks negative sentinel values (`-9.96921e36`) as `np.nan`.
4. **Strips unpublished future days:** NOAA pre-allocates all 365 days in the current year's NetCDF file and fills future dates with missing values. The builder strips those trailing future `NaN` days so future updates can append new days as they are published.

### Output Zarr Structure
* **Dimensions:** `(time, latitude, longitude)` — `360` latitudes × `720` longitudes
* **Variable:** `cpc_precipitation` (`float32`, `mm/day`)

### Example Commands

```bash
# Build a local archive for 2020-2022 and delete temporary downloaded NetCDF files when done
build-cpc-archive \
  --target_zarr ./data/cpc_daily.zarr \
  --start_year 2020 \
  --end_year 2022 \
  --cleanup_cache

# Run again later to append any new days published since the last run
build-cpc-archive \
  --target_zarr ./data/cpc_daily.zarr \
  --cleanup_cache

# Rebuild an existing archive from scratch
build-cpc-archive \
  --target_zarr ./data/cpc_daily.zarr \
  --start_year 1979 \
  --overwrite \
  --cleanup_cache
```

---

## 2. ECMWF HRES Daily Surface Forecasts (`build-hres-archive`)

Builds a 10-day daily surface forecast archive (lead days 1 through 10 initialized at 00:00 UTC each day) on a `0.25°` global grid (`721 × 1440`):

* **2016-01-01 to 2023-01-10:** Read from the public WeatherBench 2 Zarr archive (`gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr`). Note that WeatherBench 2 does not include solar or thermal radiation (`surface_net_solar_radiation` and `surface_net_thermal_radiation` are `NaN` in this period).
* **2023-01-11 to 2023-07-12:** Neither WeatherBench 2 nor ECMWF Open Data covers this 6-month window. When building across this range, these dates are filled with `NaN` slices so the daily time index has no gaps, and can be overwritten later using `--in_place`.
* **2023-07-13 to present:** Read from the `0.25°` operational ECMWF Open Data GRIB2 archive (`gs://ecmwf-open-data/{YYYYMMDD}/00z/ifs/0p25/oper/`).

### What It Does to the Data
For each daily 00:00 UTC forecast run and each lead day `1..10`:
* **`temperature_2m` (K) and `surface_pressure` (Pa):** Averaged across the four 6-hourly steps of each lead day.
* **`total_precipitation` (m):** Converted from cumulative forecast totals into daily 24-hour totals (and clipped at `0.0` to remove tiny negative floating-point rounding artifacts).
* **`surface_net_solar_radiation` (J/m²) and `surface_net_thermal_radiation` (J/m²):** Converted from cumulative forecast totals into daily 24-hour totals (retaining negative values, since net thermal radiation is signed).

### Output Zarr Structure
* **Dimensions:** `(time, lead_time, latitude, longitude)` — `10` lead days × `721` latitudes (`-90.0 .. 90.0`) × `1440` longitudes (`0.0 .. 359.75`)
* **Variables:** `temperature_2m`, `surface_pressure`, `total_precipitation`, `surface_net_solar_radiation`, `surface_net_thermal_radiation` (`float32`)

### Example Commands

```bash
# Build a local HRES archive for a specific date range
build-hres-archive \
  --target_zarr ./data/hres_daily.zarr \
  --start_date 2024-06-01 \
  --end_date 2024-06-10

# Append newly available dates to an existing store
build-hres-archive \
  --target_zarr ./data/hres_daily.zarr \
  --end_date 2024-06-20

# Overwrite specific dates in-place within an existing store
build-hres-archive \
  --target_zarr ./data/hres_daily.zarr \
  --start_date 2024-06-05 \
  --end_date 2024-06-07 \
  --in_place
```

---

## 3. NASA GPM IMERG Daily Precipitation (`build-imerg-archive`)

Builds a daily `0.1°` global precipitation archive (`1800 × 3600`) from NASA GPM IMERG Early Run V07.

You can ingest data from two sources (`--source`):
1. **`--source gesdisc` (default):** Downloads official daily NetCDF-4 files (`3B-DAY-E.MS.MRG.3IMERG.*.V07*.nc4`) from NASA GES DISC.
2. **`--source local --local_dir /path/to/files`:** Reads pre-downloaded daily V07 NetCDF-4 files (`.nc4` / `.nc`) or 48 half-hourly V07 HDF5 files (`.RT-H5` / `.HDF5`) per day from a local directory.

### What It Does to the Data
1. **Transposes grid axes:** Raw IMERG files store data as `(lon, lat)` (`3600 × 1800`); the builder transposes each grid to `(latitude, longitude)` (`1800 × 3600`, ascending).
2. **Requires complete daily coverage for half-hourly HDF5 files:** When summing 48 half-hourly `.RT-H5` granules (`rate * 0.5 hr`), all 48 half-hourly values at a grid cell must be valid (`>= 0`). If a grid cell is missing data in any of the 48 half-hour slots, its daily total is set to `NaN` (never summed across partial days).
3. **Rejects legacy V06 files:** Only IMERG V07 (`precipitation`) is accepted; legacy V06 files (`precipitationCal`) raise an error.

### Output Zarr Structure
* **Dimensions:** `(time, latitude, longitude)` — `1800` latitudes (`-89.95 .. 89.95`) × `3600` longitudes (`-179.95 .. 179.95`)
* **Variable:** `imerg_precipitation` (`float32`, `mm/day`)

### Example Commands

```bash
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
```

---

## How Resume, Missing Dates, and Updates Work

* **Automatic Resume:** When you run any builder against an existing Zarr store (without `--overwrite`), it inspects the store, backfills any previously missing dates (`missing_dates` / `failed_dates` in Zarr root attributes or trailing `NaN` slices) in-place, and appends new dates after the last date in the store.
* **No Trailing `NaN` Lockout:** If you pass an `--end_date` in the future (for example, `--end_date 2026-12-31`), unpublished dates at the end of the range are **not** written as `NaN` slices. The Zarr store stops at the last date with valid upstream data, so future runs can cleanly append new days as time passes.
* **Local vs. Cloud Target Paths:** Any `--target_zarr` path without a URL scheme (such as `./out.zarr`, `output/cpc.zarr`, or `/tmp/cpc.zarr`) is written to the local filesystem. To write to Google Cloud Storage, always pass an explicit `gs://` URI (e.g., `--target_zarr gs://my-bucket/cpc.zarr`).

---

## Running the Tests

The unit and integration tests use synthetic local data and do not require network access or cloud credentials:

```bash
# Run all unit and integration tests
pytest multimet/test

# Run only fast unit tests
pytest multimet/test -m unit

# Run end-to-end local Zarr integration tests
pytest multimet/test -m integration
```

To run the optional live network checks against NOAA, WeatherBench 2, ECMWF, and NASA CMR:

```bash
pytest multimet/test -m canary --run-canary
```
