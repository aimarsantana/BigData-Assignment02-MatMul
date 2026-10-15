# Reproduce the whole study (Windows PowerShell). Needs: gcc (MinGW-w64), uv (or Python 3.12+ with numpy pandas matplotlib).
# Inputs (~1.8 GB) are written OUTSIDE the project folder by default so that cloud-sync tools do not touch them while timing.
param(
  [string]$DataDir = "C:\bench_a2\data",
  [string]$Python = "uv run --with numpy --with pandas --with matplotlib python"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = $Python.Split(" ")
function Py { & $py[0] $py[1..($py.Length-1)] @args; if ($LASTEXITCODE -ne 0) { throw "python step failed" } }

.\build.ps1
.\scripts\collect_environment.ps1
Py scripts/generate_inputs.py --outdir $DataDir --check-a1 "..\Assignment01_AlejandroSantana_Materials\data\inputs\manifest.json"
Py tests/validate.py --data-dir $DataDir > results\validation.csv
if (Test-Path results\raw_samples.csv) { throw "results\ already contains measurements: move or delete it before a full re-run" }
# Phase 1: block-size sweep (the tile size of the 'blocked' method is fixed from this sweep, see config block_size)
Py scripts/run_benchmarks.py --data-dir $DataDir D2_block
# Phase 2: everything else with the configured tile size
Py scripts/run_benchmarks.py --data-dir $DataDir D1_size S1_sparsity S2_size_z99 S3_size_z99.9 S4_band_width S5_band_size
Py scripts/analyze.py
Py report/build_report.py
