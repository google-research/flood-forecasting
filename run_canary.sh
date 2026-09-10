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
DEFAULT_BASINS="${SCRIPT_DIR}/test/test_data/shapefiles/us/us_basin_shapes.geojson"
DEFAULT_OUT="${TMPDIR:-/tmp}/multimet_canary"
DEFAULT_PRODUCTS="CPC"
DEFAULT_START="2020-01-01"
DEFAULT_END="2020-01-02"

echo "======================================================================"
echo "🦅 MultiMet Local Canary Launcher"
echo "======================================================================"

# Ensure the repository root is in PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

if [ "$#" -eq 0 ]; then
  echo "No arguments provided. Running default canary test case:"
  echo "  Catchment GeoJSON : ${DEFAULT_BASINS}"
  echo "  Output Directory  : ${DEFAULT_OUT}"
  echo "  Product           : ${DEFAULT_PRODUCTS}"
  echo "  Date Range        : ${DEFAULT_START} to ${DEFAULT_END}"
  echo "----------------------------------------------------------------------"
  python3 "${SCRIPT_DIR}/canary.py" \
    --basins_path "${DEFAULT_BASINS}" \
    --output_dir "${DEFAULT_OUT}" \
    --products "${DEFAULT_PRODUCTS}" \
    --start_date "${DEFAULT_START}" \
    --end_date "${DEFAULT_END}"
else
  echo "Passing arguments to canary runner: $@"
  echo "----------------------------------------------------------------------"
  python3 "${SCRIPT_DIR}/canary.py" "$@"
fi
