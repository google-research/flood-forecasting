<#
.SYNOPSIS
    MultiMet Canary Extraction Runner for Windows PowerShell.
.DESCRIPTION
    Runs local meteorological forcing extractions on Windows using PowerShell.
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

$DefaultBasins = if (Test-Path (Join-Path $ScriptDir "wabash_test_data\shapefiles\us\us_basin_shapes.geojson")) {
    Join-Path $ScriptDir "wabash_test_data\shapefiles\us\us_basin_shapes.geojson"
} else {
    Join-Path $ScriptDir "..\..\test\test_data\shapefiles\us\us_basin_shapes.geojson"
}

$DefaultOut = Join-Path $ScriptDir "output"
$DefaultProducts = "CPC"
$DefaultStart = "2020-01-01"
$DefaultEnd = "2020-01-02"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "MultiMet Local Canary Launcher (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$RepoDir = Join-Path $ScriptDir "..\.."
if (Test-Path (Join-Path $RepoDir "multimet")) {
    $env:PYTHONPATH = "$RepoDir;$ScriptDir;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH"
}

# Locate Python
$PythonExe = if (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} else {
    Write-Error "Neither 'python' nor 'py' executable was found in PATH. Please activate your environment."
    exit 1
}

$CanaryPy = Join-Path $ScriptDir "canary.py"

if ($ScriptArgs.Count -eq 0) {
    Write-Host "No arguments provided. Running default canary test case:"
    Write-Host "  Catchment GeoJSON : $DefaultBasins"
    Write-Host "  Output Directory  : $DefaultOut"
    Write-Host "  Product           : $DefaultProducts"
    Write-Host "  Date Range        : $DefaultStart to $DefaultEnd"
    Write-Host "----------------------------------------------------------------------"
    & $PythonExe $CanaryPy `
        --basins_path $DefaultBasins `
        --output_dir $DefaultOut `
        --products $DefaultProducts `
        --start_date $DefaultStart `
        --end_date $DefaultEnd
} else {
    Write-Host "Passing arguments to canary runner: $($ScriptArgs -join ' ')"
    Write-Host "----------------------------------------------------------------------"
    & $PythonExe $CanaryPy @ScriptArgs
}

exit $LASTEXITCODE
