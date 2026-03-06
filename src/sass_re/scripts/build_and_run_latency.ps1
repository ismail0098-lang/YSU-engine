<#
.SYNOPSIS
  Build and run the latency microbenchmark.

.DESCRIPTION
  Compiles microbench_latency.cu for the specified SM arch, then runs it.
  Also dumps the SASS of the benchmark itself so you can verify
  the instruction chains are what you expect.

.PARAMETER Arch
  SM architecture target (e.g. sm_89, sm_61). Default: sm_89
#>
param(
    [string]$Arch     = "sm_89",
    [string]$CudaPath = ""   # Override CUDA bin dir
)

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchDir   = Join-Path $ScriptDir "..\microbench"
$ResultDir  = Join-Path $ScriptDir "..\results"

# Tool paths — auto-detect or use override
$smNum = [int]($Arch -replace 'sm_', '')
if ($CudaPath) {
    $CudaBin = $CudaPath
} elseif ($smNum -lt 75) {
    # Try CUDA 12.x for older architectures
    $CudaBin = @("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
                 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin") |
        Where-Object { Test-Path (Join-Path $_ "nvcc.exe") } | Select-Object -First 1
    if (-not $CudaBin) { $CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin" }
} else {
    $CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
}
$Nvcc       = Join-Path $CudaBin "nvcc.exe"
$CuObjDump  = Join-Path $CudaBin "cuobjdump.exe"
$VcVars     = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

New-Item -ItemType Directory -Path $ResultDir -Force | Out-Null

$Src    = Join-Path $BenchDir "microbench_latency.cu"
$Exe    = Join-Path $ResultDir "latency_bench.exe"
$Cubin  = Join-Path $ResultDir "latency_bench.cubin"
$Sass   = Join-Path $ResultDir "latency_bench.sass"
$Output = Join-Path $ResultDir "latency_results.txt"

Write-Host "=== Build & Run Latency Benchmark ($Arch) ===" -ForegroundColor Cyan

# Step 1: Compile executable
Write-Host "Compiling $Src ..." -NoNewline
$nvccCmd = """$VcVars"" >nul 2>&1 && ""$Nvcc"" -arch=$Arch -O1 -lineinfo -allow-unsupported-compiler -o ""$Exe"" ""$Src"" 2>&1"
$compileOut = cmd /c $nvccCmd 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED" -ForegroundColor Red
    $compileOut
    exit 1
}
Write-Host " OK" -ForegroundColor Green

# Step 2: Also compile to cubin for SASS inspection
Write-Host "Compiling cubin for SASS dump..." -NoNewline
$nvccCmd = """$VcVars"" >nul 2>&1 && ""$Nvcc"" -arch=$Arch -O1 -cubin -lineinfo -allow-unsupported-compiler -o ""$Cubin"" ""$Src"" 2>&1"
$result = cmd /c $nvccCmd 2>&1
if ($LASTEXITCODE -eq 0) {
    & $CuObjDump -sass $Cubin 2>&1 | Out-File -Encoding utf8 $Sass
    Write-Host " OK (see latency_bench.sass)" -ForegroundColor Green
} else {
    Write-Host " SKIPPED" -ForegroundColor DarkYellow
}

# Step 3: Run
Write-Host ""
Write-Host "Running benchmark..." -ForegroundColor Yellow
& $Exe 2>&1 | Tee-Object -FilePath $Output
Write-Host ""
Write-Host "Results saved to: $Output" -ForegroundColor Cyan
Write-Host "SASS dump at:     $Sass" -ForegroundColor Cyan
