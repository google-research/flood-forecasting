@echo off
rem ==============================================================================
rem MultiMet Canary Runner (Windows Command Prompt)
rem Quick local execution of forcing extraction on Windows.
rem ==============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "DEFAULT_BASINS=%SCRIPT_DIR%test\test_data\shapefiles\us\us_basin_shapes.geojson"
if not defined TEMP (
  set "DEFAULT_OUT=%USERPROFILE%\AppData\Local\Temp\multimet_canary"
) else (
  set "DEFAULT_OUT=%TEMP%\multimet_canary"
)
set "DEFAULT_PRODUCTS=CPC"
set "DEFAULT_START=2020-01-01"
set "DEFAULT_END=2020-01-02"

echo ======================================================================
echo MultiMet Local Canary Launcher (Windows)
echo ======================================================================

rem Ensure repo root is in PYTHONPATH
if defined PYTHONPATH (
  set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%SCRIPT_DIR%"
)

rem Find Python executable
set "PY_CMD="
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
  set "PY_CMD=python"
) else (
  where py >nul 2>&1
  if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py"
  ) else (
    echo [ERROR] Neither 'python' nor 'py' was found in your PATH.
    echo Please install Python or activate your virtual/conda environment.
    exit /b 1
  )
)

if "%~1"=="" (
  echo No arguments provided. Running default canary test case:
  echo   Catchment GeoJSON : %DEFAULT_BASINS%
  echo   Output Directory  : %DEFAULT_OUT%
  echo   Product           : %DEFAULT_PRODUCTS%
  echo   Date Range        : %DEFAULT_START% to %DEFAULT_END%
  echo ----------------------------------------------------------------------
  %PY_CMD% "%SCRIPT_DIR%canary.py" ^
    --basins_path "%DEFAULT_BASINS%" ^
    --output_dir "%DEFAULT_OUT%" ^
    --products "%DEFAULT_PRODUCTS%" ^
    --start_date "%DEFAULT_START%" ^
    --end_date "%DEFAULT_END%"
) else (
  echo Passing arguments to canary runner: %*
  echo ----------------------------------------------------------------------
  %PY_CMD% "%SCRIPT_DIR%canary.py" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
