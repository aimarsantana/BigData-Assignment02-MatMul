# Record the hardware / software environment of the measurements (writes results/environment.txt).
param([string]$Out = (Join-Path $PSScriptRoot "..\results\environment.txt"))
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$cpu = Get-CimInstance Win32_Processor
$cs  = Get-CimInstance Win32_ComputerSystem
$os  = Get-CimInstance Win32_OperatingSystem
$mem = Get-CimInstance Win32_PhysicalMemory
$caches = Get-CimInstance Win32_CacheMemory | Select-Object Purpose, InstalledSize
$lines = @()
$lines += "date_utc: " + (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$lines += "cpu_model: " + $cpu.Name.Trim()
$lines += "physical_cores: " + $cpu.NumberOfCores
$lines += "logical_processors: " + $cpu.NumberOfLogicalProcessors
$lines += "cpu_max_clock_mhz (reported by Windows): " + $cpu.MaxClockSpeed
$lines += "l2_cache_kb (Win32_Processor): " + $cpu.L2CacheSize
$lines += "l3_cache_kb (Win32_Processor): " + $cpu.L3CacheSize
$lines += "ram_total_gb: " + [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
$lines += "ram_modules: " + (($mem | ForEach-Object { "$([math]::Round($_.Capacity/1GB))GB@$($_.Speed)MT/s" }) -join ", ")
$lines += "os: " + $os.Caption + " " + $os.Version + " build " + $os.BuildNumber
$lines += "power_scheme: " + [regex]::Match((powercfg /getactivescheme | Out-String), '\(([^)]*)\)').Groups[1].Value
$bat = Get-CimInstance Win32_Battery
$lines += "battery_status (2 = AC power): " + $bat.BatteryStatus
$lines += "gcc: " + ((& gcc --version)[0])
$lines += "python (uv): " + ((& uv run --with numpy --with pandas --with matplotlib python -c "import sys,numpy,pandas,matplotlib;print(sys.version.split()[0],'numpy',numpy.__version__,'pandas',pandas.__version__,'matplotlib',matplotlib.__version__)"))
$lines += "build_flags: gcc -O2 -std=c11 -Wall -Wextra -lpsapi (see build.ps1)"
$lines | Set-Content -Encoding utf8 $Out
$lines
