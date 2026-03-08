<#
.SYNOPSIS
    Build and verify the Instant-NGP SASS-level kernels.

.DESCRIPTION
    1. Compiles all .cu files into a validation executable
    2. Generates .cubin for SASS disassembly
    3. Runs validation (PTX vs reference)
    4. Disassembles cubins and dumps SASS to .sass files
    5. Counts key instructions per kernel

.PARAMETER Arch
    CUDA SM architecture (default: sm_89 for Ada Lovelace)

.PARAMETER CudaPath
    Path to CUDA bin directory (auto-detected if not specified)
#>

param(
    [string]$Arch = "sm_89",
    [string]$CudaPath = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ngpDir    = $scriptDir  # This script lives in instant_ngp/

# ── Auto-detect CUDA ──
if (-not $CudaPath) {
    $defaultPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $versions = Get-ChildItem -Path $defaultPath -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending
    if ($versions) {
        $CudaPath = Join-Path $versions[0].FullName "bin"
        Write-Host "[*] Auto-detected CUDA: $CudaPath" -ForegroundColor Cyan
    } else {
        Write-Error "No CUDA installation found. Set -CudaPath manually."
        exit 1
    }
}

$nvcc       = Join-Path $CudaPath "nvcc.exe"
$nvdisasm   = Join-Path $CudaPath "nvdisasm.exe"
$cuobjdump  = Join-Path $CudaPath "cuobjdump.exe"

if (-not (Test-Path $nvcc)) {
    Write-Error "nvcc not found at $nvcc"
    exit 1
}

# ── Detect MSVC ──
$vcvarsLocations = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
)

$vcvars = $vcvarsLocations | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) {
    Write-Error "vcvars64.bat not found. Install Visual Studio Build Tools."
    exit 1
}
Write-Host "[*] Using vcvars: $vcvars" -ForegroundColor Cyan

# ── Output directory ──
$outDir = Join-Path $ngpDir "build"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$sassDir = Join-Path $ngpDir "sass_output"
if (-not (Test-Path $sassDir)) { New-Item -ItemType Directory -Path $sassDir | Out-Null }

# ── Source files ──
$sources = @(
    "hashgrid_encode.cu",
    "mlp_forward.cu",
    "volume_render.cu",
    "ngp_validate.cu"
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║  Instant-NGP SASS Kernel Build & Verify Pipeline     ║" -ForegroundColor Yellow
Write-Host "║  Architecture: $Arch                                  ║" -ForegroundColor Yellow
Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Yellow
Write-Host ""

# ════════════════════════════════════════
# Step 1: Compile validation executable
# ════════════════════════════════════════
Write-Host "── Step 1: Compile validation executable ──" -ForegroundColor Green

$srcPaths = $sources | ForEach-Object { Join-Path $ngpDir $_ }
$exePath  = Join-Path $outDir "ngp_validate.exe"

$nvccArgs = @(
    "-arch=$Arch",
    "-O1",                     # Low optimization to preserve PTX structure
    "-allow-unsupported-compiler",
    "-lineinfo",               # Keep line info for SASS correlation
    "-o", $exePath
) + $srcPaths

$cmd = "cmd /c `"call `"$vcvars`" >nul 2>&1 && `"$nvcc`" $($nvccArgs -join ' ') 2>&1`""
Write-Host "  CMD: nvcc -arch=$Arch -O1 ..." -ForegroundColor DarkGray

$output = Invoke-Expression $cmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "  COMPILE FAILED:" -ForegroundColor Red
    $output | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    exit 1
}
Write-Host "  OK: $exePath" -ForegroundColor Green

# ════════════════════════════════════════
# Step 2: Generate .cubin for each kernel file
# ════════════════════════════════════════
Write-Host ""
Write-Host "── Step 2: Generate cubins for SASS inspection ──" -ForegroundColor Green

foreach ($src in @("hashgrid_encode.cu", "mlp_forward.cu", "volume_render.cu")) {
    $srcPath  = Join-Path $ngpDir $src
    $cubinOut = Join-Path $outDir ($src -replace '\.cu$', '.cubin')

    $cubinArgs = @(
        "-arch=$Arch",
        "-O1",
        "-allow-unsupported-compiler",
        "-cubin",
        "-o", $cubinOut,
        $srcPath
    )

    $cmd = "cmd /c `"call `"$vcvars`" >nul 2>&1 && `"$nvcc`" $($cubinArgs -join ' ') 2>&1`""
    $output = Invoke-Expression $cmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: $cubinOut" -ForegroundColor Green
    } else {
        Write-Host "  FAIL: $src" -ForegroundColor Red
        $output | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    }
}

# ════════════════════════════════════════
# Step 3: Disassemble SASS from cubins
# ════════════════════════════════════════
Write-Host ""
Write-Host "── Step 3: Disassemble SASS ──" -ForegroundColor Green

foreach ($src in @("hashgrid_encode", "mlp_forward", "volume_render")) {
    $cubinPath = Join-Path $outDir "$src.cubin"
    $sassPath  = Join-Path $sassDir "$src.sass"

    if (Test-Path $cubinPath) {
        $cmd = "cmd /c `"call `"$vcvars`" >nul 2>&1 && `"$nvdisasm`" -g -sf `"$cubinPath`" 2>&1`""
        $sassOutput = Invoke-Expression $cmd
        $sassOutput | Out-File -FilePath $sassPath -Encoding UTF8
        Write-Host "  OK: $sassPath" -ForegroundColor Green

        # Count key instructions
        $sassText = Get-Content $sassPath -Raw
        $ffmaCount  = ([regex]::Matches($sassText, "FFMA")).Count
        $imadCount  = ([regex]::Matches($sassText, "IMAD")).Count
        $lop3Count  = ([regex]::Matches($sassText, "LOP3")).Count
        $mufuCount  = ([regex]::Matches($sassText, "MUFU")).Count
        $ldgCount   = ([regex]::Matches($sassText, "LDG")).Count
        $stgCount   = ([regex]::Matches($sassText, "STG")).Count
        $ldsCount   = ([regex]::Matches($sassText, "LDS")).Count
        $shflCount  = ([regex]::Matches($sassText, "SHFL")).Count
        $totalLines = ($sassOutput | Measure-Object -Line).Lines

        Write-Host "    Instructions: FFMA=$ffmaCount IMAD=$imadCount LOP3=$lop3Count MUFU=$mufuCount" -ForegroundColor DarkCyan
        Write-Host "    Memory:       LDG=$ldgCount STG=$stgCount LDS=$ldsCount SHFL=$shflCount" -ForegroundColor DarkCyan
        Write-Host "    Total lines:  $totalLines" -ForegroundColor DarkCyan
    } else {
        Write-Host "  SKIP: $cubinPath not found" -ForegroundColor Yellow
    }
}

# ════════════════════════════════════════
# Step 4: Run validation
# ════════════════════════════════════════
Write-Host ""
Write-Host "── Step 4: Run validation ──" -ForegroundColor Green

if (Test-Path $exePath) {
    $cmd = "cmd /c `"call `"$vcvars`" >nul 2>&1 && `"$exePath`" 2>&1`""
    $output = Invoke-Expression $cmd
    $output | ForEach-Object { Write-Host "  $_" }

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "  ✓ ALL TESTS PASSED" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "  ✗ SOME TESTS FAILED" -ForegroundColor Red
    }
} else {
    Write-Host "  SKIP: executable not found" -ForegroundColor Yellow
}

# ════════════════════════════════════════
# Step 5: Generate PTX for inspection
# ════════════════════════════════════════
Write-Host ""
Write-Host "── Step 5: Generate PTX listings ──" -ForegroundColor Green

foreach ($src in @("hashgrid_encode.cu", "mlp_forward.cu", "volume_render.cu")) {
    $srcPath = Join-Path $ngpDir $src
    $ptxOut  = Join-Path $sassDir ($src -replace '\.cu$', '.ptx')

    $ptxArgs = @(
        "-arch=$Arch",
        "-O1",
        "-allow-unsupported-compiler",
        "-ptx",
        "-o", $ptxOut,
        $srcPath
    )

    $cmd = "cmd /c `"call `"$vcvars`" >nul 2>&1 && `"$nvcc`" $($ptxArgs -join ' ') 2>&1`""
    $output = Invoke-Expression $cmd
    if ($LASTEXITCODE -eq 0) {
        $ptxLines = (Get-Content $ptxOut | Measure-Object -Line).Lines
        Write-Host "  OK: $ptxOut ($ptxLines lines)" -ForegroundColor Green
    } else {
        Write-Host "  FAIL: $src" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║  Build & Verify Complete                             ║" -ForegroundColor Yellow
Write-Host "║  SASS output:   $sassDir" -ForegroundColor Yellow
Write-Host "║  Executables:   $outDir" -ForegroundColor Yellow
Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Yellow
