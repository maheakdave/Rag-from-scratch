# build.ps1  -  Configure and compile llm-server
#
# Usage:
#   .\scripts\build.ps1 [-Cuda] [-Metal] [-Vulkan] [-Static] [-Clean]
#
# Requires: CMake >= 3.18, a C++17 compiler (MSVC / GCC / Clang), Git

param(
    [switch]$Cuda,
    [switch]$Metal,
    [switch]$Vulkan,
    [switch]$Static,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$RootDir  = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RootDir "build"

# ── Optional clean ────────────────────────────────────────────────────────────

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "==> Cleaning $BuildDir"
    Remove-Item -Recurse -Force $BuildDir
}

# ── Collect extra CMake flags ─────────────────────────────────────────────────

$CMakeArgs = @("-DCMAKE_BUILD_TYPE=Release")

if ($Cuda)   { $CMakeArgs += "-DLLM_SERVER_CUBLAS=ON" }
if ($Metal)  { $CMakeArgs += "-DLLM_SERVER_METAL=ON"  }
if ($Vulkan) { $CMakeArgs += "-DLLM_SERVER_VULKAN=ON" }
if ($Static) { $CMakeArgs += "-DLLM_SERVER_STATIC=ON" }

# ── Configure ─────────────────────────────────────────────────────────────────

Write-Host "==> Configuring..."
cmake -S $RootDir -B $BuildDir @CMakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }

# ── Build ─────────────────────────────────────────────────────────────────────

$Cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if (-not $Cores) { $Cores = 4 }

Write-Host "==> Building with $Cores parallel jobs..."
cmake --build $BuildDir --config Release -j $Cores
if ($LASTEXITCODE -ne 0) { throw "CMake build failed (exit $LASTEXITCODE)" }

# ── Done ──────────────────────────────────────────────────────────────────────

$Exe = Join-Path $BuildDir "Release\llm-server.exe"
if (-not (Test-Path $Exe)) {
    # GCC/Clang on Windows puts the binary directly in build/
    $Exe = Join-Path $BuildDir "llm-server.exe"
}

Write-Host ""
Write-Host "Build complete: $Exe"
Write-Host ""
Write-Host "Run with:"
Write-Host "  $Exe --model C:\path\to\model.gguf"
