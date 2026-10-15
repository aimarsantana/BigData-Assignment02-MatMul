# Build the C implementation with the exact flags used in the study (PowerShell, MinGW-w64 gcc).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force build | Out-Null
$cflags = "-O2 -std=c11 -Wall -Wextra"
# the flag string is passed to the program through a generated header (avoids shell quoting issues)
Set-Content -Encoding ascii build/cflags.h "#define CFLAGS_STR `"$cflags -lpsapi`""
& gcc -O2 -std=c11 -Wall -Wextra -include build/cflags.h -o build/matmul2.exe src/c/matmul2.c -lpsapi
if ($LASTEXITCODE -ne 0) { throw "gcc failed" }
Write-Host "built build/matmul2.exe with: gcc $cflags -lpsapi"
