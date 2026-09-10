#!/usr/bin/env bash
# ==============================================================================
# MultiMet Canary Runner
# Quick local execution of forcing extraction.
# ==============================================================================

set -e

# Disable macOS fork safety restrictions which can cause crashes with GDAL/PROJ
# and multi-threaded Python urllib/aiohttp on macOS.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve basins file from candidates
if [ -f "${SCRIPT_DIR}/wabash_test_data/shapefiles/us/us_basin_shapes.geojson" ]; then
  DEFAULT_BASINS="${SCRIPT_DIR}/wabash_test_data/shapefiles/us/us_basin_shapes.geojson"
elif [ -f "${SCRIPT_DIR}/../test/test_data/shapefiles/us/us_basin_shapes.geojson" ]; then
  DEFAULT_BASINS="${SCRIPT_DIR}/../test/test_data/shapefiles/us/us_basin_shapes.geojson"
elif [ -f "${SCRIPT_DIR}/../../test/test_data/shapefiles/us/us_basin_shapes.geojson" ]; then
  DEFAULT_BASINS="${SCRIPT_DIR}/../../test/test_data/shapefiles/us/us_basin_shapes.geojson"
elif [ -f "${HOME}/multimet/canary/wabash_test_data/shapefiles/us/us_basin_shapes.geojson" ]; then
  DEFAULT_BASINS="${HOME}/multimet/canary/wabash_test_data/shapefiles/us/us_basin_shapes.geojson"
else
  DEFAULT_BASINS="${SCRIPT_DIR}/wabash_test_data/shapefiles/us/us_basin_shapes.geojson"
fi

DEFAULT_OUT="${SCRIPT_DIR}/output"
DEFAULT_PRODUCTS="CPC"
DEFAULT_START="2020-01-01"
DEFAULT_END="2020-01-02"

echo "======================================================================"
echo "🦅 MultiMet Local Canary Launcher"
echo "======================================================================"

REPO_DIR="$(cd "${SCRIPT_DIR}/../.." 2>/dev/null && pwd || echo "")"
if [ -d "${REPO_DIR}/multimet" ]; then
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
elif [ -d "${HOME}/Projects/flood-forecasting-multimet/multimet" ]; then
  export PYTHONPATH="${HOME}/Projects/flood-forecasting-multimet:${PYTHONPATH}"
fi

# Locate Python
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "❌ Error: Neither python3 nor python was found in PATH." >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "No arguments provided. Running default canary test case:"
  echo "  Catchment GeoJSON : ${DEFAULT_BASINS}"
  echo "  Output Directory  : ${DEFAULT_OUT}"
  echo "  Product           : ${DEFAULT_PRODUCTS}"
  echo "  Date Range        : ${DEFAULT_START} to ${DEFAULT_END}"
  echo "----------------------------------------------------------------------"
  "${PYTHON_CMD}" "${SCRIPT_DIR}/canary.py" \
    --basins_path "${DEFAULT_BASINS}" \
    --output_dir "${DEFAULT_OUT}" \
    --products "${DEFAULT_PRODUCTS}" \
    --start_date "${DEFAULT_START}" \
    --end_date "${DEFAULT_END}"
else
  echo "Passing arguments to canary runner: $@"
  echo "----------------------------------------------------------------------"
  "${PYTHON_CMD}" "${SCRIPT_DIR}/canary.py" "$@"
fi
