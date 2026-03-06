<#
.SYNOPSIS
  Complete SASS RE pipeline for GTX 1050 Ti (Pascal, SM 6.1).
  
.DESCRIPTION
  One-click script that:
    1. Compiles and disassembles all probes for SM 6.1
    2. Runs latency microbenchmark
    3. Runs throughput microbenchmark
    4. Runs encoding analysis
    5. Saves all results to results/Pascal_GTX1050Ti_<timestamp>/

.NOTES
  Requires: CUDA Toolkit, MSVC Build Tools, Python 3.x
  Run from src/sass_re/ directory with the GTX 1050 Ti installed and active.
#>

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SassReDir = Split-Path -Parent $ScriptDir
$Arch      = "sm_61"
$GpuTag    = "Pascal_GTX1050Ti"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir    = Join-Path $SassReDir "results\${GpuTag}_${Timestamp}"

# Tool paths — need CUDA 12.x for Pascal (SM 6.1 dropped from CUDA 13.x)
$Cuda12Paths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
)
$CudaBin = $Cuda12Paths | Where-Object { Test-Path (Join-Path $_ "nvcc.exe") } | Select-Object -First 1

if (-not $CudaBin) {
    Write-Host "ERROR: CUDA 12.x or 11.x required for Pascal (SM 6.1)!" -ForegroundColor Red
    Write-Host "CUDA 13.1 dropped SM < 7.5. Install CUDA 12.6 alongside 13.1:" -ForegroundColor Yellow
    Write-Host "  https://developer.nvidia.com/cuda-12-6-0-download-archive" -ForegroundColor White
    Write-Host "" 
    Write-Host "CUDA supports side-by-side installations." -ForegroundColor DarkGray
    Write-Host "After installing, re-run this script." -ForegroundColor DarkGray
    exit 1
}

$Nvcc    = Join-Path $CudaBin "nvcc.exe"
$VcVars  = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

Write-Host "Using CUDA: $CudaBin" -ForegroundColor Green

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SASS RE: Full Pipeline — GTX 1050 Ti" -ForegroundColor Cyan
Write-Host "  Architecture: SM 6.1 (Pascal)" -ForegroundColor Cyan
Write-Host "  Output: $RunDir" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Verify GPU
Write-Host "Checking GPU..." -ForegroundColor Yellow
$smiOutput = & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\..\extras\demo_suite\..\..\..\NVIDIA Corporation\NVSMI\nvidia-smi.exe" --query-gpu=name,compute_cap --format=csv,noheader 2>&1
if ($LASTEXITCODE -ne 0) {
    # Try standard location
    $smiOutput = & "nvidia-smi" --query-gpu=name,compute_cap --format=csv,noheader 2>&1
}
Write-Host "  GPU: $smiOutput" -ForegroundColor Green
Write-Host ""

# Step 1: Disassemble all probes
Write-Host "=== STEP 1: Disassemble probes ===" -ForegroundColor Cyan
& "$ScriptDir\disassemble_all.ps1" -Arch $Arch -GpuTag $GpuTag -CudaPath $CudaBin
Write-Host ""

# Find the output directory that was just created
$latestRun = Get-ChildItem (Join-Path $SassReDir "results") -Directory |
    Where-Object { $_.Name -like "${GpuTag}_*" } |
    Sort-Object Name -Descending |
    Select-Object -First 1
$RunDir = $latestRun.FullName
Write-Host "Results directory: $RunDir" -ForegroundColor Green

# Step 2: Latency benchmark
Write-Host ""
Write-Host "=== STEP 2: Latency Benchmark ===" -ForegroundColor Cyan
$latencyExe = Join-Path $RunDir "latency_bench.exe"
$latencySrc = Join-Path $SassReDir "microbench\microbench_latency.cu"

Write-Host "  Compiling latency benchmark for $Arch..." -NoNewline
$nvccCmd = """$VcVars"" >nul 2>&1 && ""$Nvcc"" -arch=$Arch -O1 -allow-unsupported-compiler -o ""$latencyExe"" ""$latencySrc"" 2>&1"
$result = cmd /c $nvccCmd 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED" -ForegroundColor Red
    $result | Out-File (Join-Path $RunDir "latency_compile_error.txt")
    Write-Host $result
} else {
    Write-Host " OK" -ForegroundColor Green
    Write-Host "  Running latency benchmark..."
    $latencyOutput = & $latencyExe 2>&1
    $latencyOutput | Out-String | Write-Host
    $latencyOutput | Out-File -Encoding utf8 (Join-Path $RunDir "latency_results.txt")
    Write-Host "  Saved to latency_results.txt" -ForegroundColor Green
}

# Step 3: Throughput benchmark  
Write-Host ""
Write-Host "=== STEP 3: Throughput Benchmark ===" -ForegroundColor Cyan
$throughputExe = Join-Path $RunDir "throughput_bench.exe"
$throughputSrc = Join-Path $SassReDir "microbench\microbench_throughput.cu"

Write-Host "  Compiling throughput benchmark for $Arch..." -NoNewline
$nvccCmd = """$VcVars"" >nul 2>&1 && ""$Nvcc"" -arch=$Arch -O1 -allow-unsupported-compiler -o ""$throughputExe"" ""$throughputSrc"" 2>&1"
$result = cmd /c $nvccCmd 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED" -ForegroundColor Red
    $result | Out-File (Join-Path $RunDir "throughput_compile_error.txt")
    Write-Host $result
} else {
    Write-Host " OK" -ForegroundColor Green
    Write-Host "  Running throughput benchmark..."
    $throughputOutput = & $throughputExe 2>&1
    $throughputOutput | Out-String | Write-Host
    $throughputOutput | Out-File -Encoding utf8 (Join-Path $RunDir "throughput_results.txt")
    Write-Host "  Saved to throughput_results.txt" -ForegroundColor Green
}

# Step 4: Encoding analysis
Write-Host ""
Write-Host "=== STEP 4: Encoding Analysis ===" -ForegroundColor Cyan
$analysisScript = Join-Path $ScriptDir "encoding_analysis.py"
if (Test-Path $analysisScript) {
    python $analysisScript $RunDir
    Write-Host "  Saved ENCODING_ANALYSIS.md to $RunDir" -ForegroundColor Green
} else {
    Write-Host "  encoding_analysis.py not found, skipping" -ForegroundColor DarkYellow
}

# Step 5: Generate comparison (if Ada results exist)
Write-Host ""
Write-Host "=== STEP 5: Cross-Architecture Comparison ===" -ForegroundColor Cyan
$adaRun = Get-ChildItem (Join-Path $SassReDir "results") -Directory |
    Where-Object { $_.Name -like "Ada_*" -or $_.Name -like "20*" } |
    Sort-Object Name -Descending |
    Select-Object -First 1

if ($adaRun) {
    Write-Host "  Found Ada results: $($adaRun.Name)" -ForegroundColor Green
    $compareScript = Join-Path $ScriptDir "compare_architectures.py"
    $compareOutput = Join-Path $SassReDir "COMPARISON.md"
    python $compareScript $adaRun.FullName $RunDir --output $compareOutput
    Write-Host "  Comparison report: $compareOutput" -ForegroundColor Green
} else {
    Write-Host "  No Ada Lovelace results found. Run disassemble_all.ps1 on 4070 Ti Super first." -ForegroundColor DarkYellow
    Write-Host "  You can generate the comparison later with:" -ForegroundColor DarkYellow
    Write-Host "    python scripts/compare_architectures.py results/<ada_dir> results/<pascal_dir>" -ForegroundColor White
}

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  COMPLETE: GTX 1050 Ti SASS RE Pipeline" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results: $RunDir" -ForegroundColor Green
Write-Host ""
Write-Host "Files generated:" -ForegroundColor Yellow
Get-ChildItem $RunDir | ForEach-Object {
    $size = if ($_.Length -gt 1024) { "{0:N0} KB" -f ($_.Length / 1024) } else { "$($_.Length) B" }
    Write-Host "  $($_.Name) ($size)"
}
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review COMPARISON.md for side-by-side analysis"
Write-Host "  2. Fill in PAPER_OUTLINE.md with measured data"
Write-Host "  3. Run 'python scripts/encoding_analysis.py $RunDir' for encoding deep-dive"
