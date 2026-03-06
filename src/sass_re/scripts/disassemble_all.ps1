<#
.SYNOPSIS
  Compile all SASS probe kernels to cubins and dump disassembly.
  
.DESCRIPTION
  For each .cu file in probes/:
    1. nvcc -arch=<sm_XX> -cubin -> .cubin  (binary object)
    2. cuobjdump -sass          -> .sass    (human-readable SASS)
    3. nvdisasm -hex -raw        -> .raw    (raw encoding with hex)
    4. nvdisasm -cfg             -> .dot    (control flow graph)
  
  Results go to results/<gpu_tag>_<timestamp>/ directory.

.PARAMETER Arch
  SM architecture target (e.g. sm_89, sm_61). Default: sm_89

.PARAMETER GpuTag
  Short name for the GPU (e.g. "4070TiS", "1050Ti"). Used in output dir name.
  Default: auto-detected from Arch.

.NOTES
  Run from src/sass_re/ directory.
#>
param(
    [string]$Arch     = "sm_89",
    [string]$GpuTag   = "",
    [string]$CudaPath = ""   # Override CUDA bin dir (e.g. for older toolkit)
)

$ErrorActionPreference = "Continue"

# Auto-detect GPU tag from architecture if not provided
if (-not $GpuTag) {
    $archTags = @{
        "sm_50" = "Maxwell"
        "sm_61" = "Pascal_GTX1050Ti"
        "sm_75" = "Turing"
        "sm_80" = "Ampere_GA100"
        "sm_86" = "Ampere_GA10x"
        "sm_89" = "Ada_RTX4070TiS"
        "sm_90" = "Hopper"
    }
    $GpuTag = if ($archTags.ContainsKey($Arch)) { $archTags[$Arch] } else { $Arch }
}

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProbeDir   = Join-Path $ScriptDir "..\probes"
$ResultDir  = Join-Path $ScriptDir "..\results"
$Timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir     = Join-Path $ResultDir "${GpuTag}_${Timestamp}"

# Create output directory
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

# Tool paths — auto-detect or use override
if ($CudaPath) {
    $CudaBin = $CudaPath
} else {
    # Pick the right CUDA version: sm_61 needs CUDA 12.x, sm_75+ can use 13.x
    $smNum = [int]($Arch -replace 'sm_', '')
    $cuda12  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    $cuda121 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    $cuda13  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
    if ($smNum -lt 75 -and (Test-Path (Join-Path $cuda12 "nvcc.exe"))) {
        $CudaBin = $cuda12
        Write-Host "NOTE: Using CUDA 12.6 for $Arch (SM < 7.5 dropped from CUDA 13.x)" -ForegroundColor DarkYellow
    } elseif ($smNum -lt 75 -and (Test-Path (Join-Path $cuda121 "nvcc.exe"))) {
        $CudaBin = $cuda121
        Write-Host "NOTE: Using CUDA 12.8 for $Arch (SM < 7.5 dropped from CUDA 13.x)" -ForegroundColor DarkYellow
    } elseif (Test-Path (Join-Path $cuda13 "nvcc.exe")) {
        $CudaBin = $cuda13
    } else {
        # Fallback: try to find any CUDA
        $CudaBin = (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory | 
            Sort-Object Name -Descending | Select-Object -First 1).FullName + "\bin"
    }
}
$Nvcc       = Join-Path $CudaBin "nvcc.exe"
$CuObjDump  = Join-Path $CudaBin "cuobjdump.exe"
$NvDisasm   = Join-Path $CudaBin "nvdisasm.exe"
$VcVars     = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

Write-Host "=== SASS RE: Disassemble All Probes ===" -ForegroundColor Cyan
Write-Host "Architecture: $Arch"
Write-Host "Probe dir:    $ProbeDir"
Write-Host "Output dir:   $RunDir"
Write-Host ""

$probes = Get-ChildItem -Path $ProbeDir -Filter "*.cu"

# Skip tensor probe on architectures without tensor cores (< sm_70)
$smNum = [int]($Arch -replace 'sm_', '')
if ($smNum -lt 70) {
    $probes = $probes | Where-Object { $_.Name -ne "probe_tensor.cu" }
    Write-Host "NOTE: Skipping probe_tensor.cu (no tensor cores on $Arch)" -ForegroundColor DarkYellow
    Write-Host ""
}

$total = $probes.Count
$idx = 0

foreach ($cu in $probes) {
    $idx++
    $name = $cu.BaseName
    $cubin = Join-Path $RunDir "$name.cubin"
    $sass  = Join-Path $RunDir "$name.sass"
    $raw   = Join-Path $RunDir "$name.raw"
    $cfg   = Join-Path $RunDir "$name.cfg.dot"

    Write-Host "[$idx/$total] $($cu.Name)" -ForegroundColor Yellow

    # Step 1: Compile to cubin
    # -lineinfo: embed source correlation
    # -Xptxas -v: show register usage
    Write-Host "  Compiling..." -NoNewline
    # Use cmd /c to source vcvars then run nvcc
    $nvccCmd = """$VcVars"" >nul 2>&1 && ""$Nvcc"" -arch=$Arch -cubin -lineinfo -Xptxas -v -allow-unsupported-compiler -o ""$cubin"" ""$($cu.FullName)"" 2>&1"
    $compileResult = cmd /c $nvccCmd 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        $compileResult | Out-File (Join-Path $RunDir "$name.compile_error.txt")
        Write-Host "  Error saved to $name.compile_error.txt"
        continue
    }
    Write-Host " OK" -ForegroundColor Green

    # Show register usage from compile output
    $regInfo = $compileResult | Select-String "registers"
    if ($regInfo) {
        foreach ($line in $regInfo) {
            Write-Host "  $($line.Line.Trim())" -ForegroundColor DarkGray
        }
    }

    # Step 2: Dump readable SASS
    Write-Host "  Disassembling SASS..." -NoNewline
    & $CuObjDump -sass $cubin 2>&1 | Out-File -Encoding utf8 $sass
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
        $instrCount = (Get-Content $sass | Where-Object { $_ -match '^\s+/\*[0-9a-f]+\*/' }).Count
        Write-Host "  Instructions: $instrCount" -ForegroundColor DarkGray
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }

    # Step 3: Raw binary disassembly (shows hex encoding)
    Write-Host "  Raw encoding dump..." -NoNewline
    & $NvDisasm -hex -raw $cubin 2>&1 | Out-File -Encoding utf8 $raw
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }

    # Step 4: Control flow graph
    Write-Host "  CFG export..." -NoNewline
    & $NvDisasm -cfg $cubin 2>&1 | Out-File -Encoding utf8 $cfg
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " SKIPPED" -ForegroundColor DarkYellow
    }

    Write-Host ""
}

# Step 5: Generate summary report
$summary = Join-Path $RunDir "SUMMARY.md"
$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine("# SASS Disassembly Report")
[void]$sb.AppendLine("Date: $Timestamp")
[void]$sb.AppendLine("GPU: $GpuTag")
[void]$sb.AppendLine("Architecture: $Arch")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("## Instruction Frequency")
[void]$sb.AppendLine("")

# Parse all .sass files and count instruction mnemonics
$allInstrs = @{}
foreach ($sassFile in (Get-ChildItem $RunDir -Filter "*.sass")) {
    $content = Get-Content $sassFile.FullName
    foreach ($line in $content) {
        if ($line -match '^\s+/\*[0-9a-f]+\*/\s+(\w+[\.\w]*)') {
            $mnemonic = $Matches[1]
            if (-not $allInstrs.ContainsKey($mnemonic)) {
                $allInstrs[$mnemonic] = 0
            }
            $allInstrs[$mnemonic]++
        }
    }
}

[void]$sb.AppendLine("| Instruction | Count |")
[void]$sb.AppendLine("|---|---|")
foreach ($kv in ($allInstrs.GetEnumerator() | Sort-Object -Property Value -Descending)) {
    [void]$sb.AppendLine("| ``$($kv.Key)`` | $($kv.Value) |")
}

$sb.ToString() | Out-File -Encoding utf8 $summary

Write-Host "=== DONE ===" -ForegroundColor Cyan
Write-Host "Results in: $RunDir"
Write-Host "Summary:    $summary"
Write-Host ""
Write-Host "Quick inspection commands:" -ForegroundColor Yellow
Write-Host "  Get-Content $RunDir\probe_fp32_arith.sass | Select-String 'FADD|FMUL|FFMA'"
Write-Host "  Get-Content $RunDir\probe_mufu.sass | Select-String 'MUFU'"
Write-Host "  Get-Content $RunDir\probe_bitwise.sass | Select-String 'LOP3|SHF|PRMT'"
