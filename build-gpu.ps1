$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$src = "src\gpu_miner.c"
$out = "src\gpu_miner.exe"

$gcc = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gcc) {
  Write-Error @"
gcc was not found in PATH.

Install MSYS2 from https://www.msys2.org/
Open "MSYS2 MinGW x64" shell and run:
  pacman -S --needed mingw-w64-x86_64-gcc

Then add  C:\msys64\mingw64\bin  to your system PATH and re-run this script.
"@
  exit 1
}

Write-Host "Compiling $src -> $out ..."
gcc -O2 -o $out $src

if ($LASTEXITCODE -ne 0) {
  Write-Error "Compilation failed (exit $LASTEXITCODE)"
  exit 1
}

Write-Host "Done. Built: $out"
Write-Host ""
Write-Host "Usage:"
Write-Host "  npm run mine:gpu"
Write-Host "  node src/cli.js --gpu"
Write-Host "  node src/cli.js --gpu --list-gpu-devices"
