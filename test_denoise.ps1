# Denoiser test with high noise (1 SPP)
# Compares: 1 SPP noisy vs 1 SPP denoised vs 16 SPP clean

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DENOISER TEST (1 SPP vs 16 SPP)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: 1 SPP WITHOUT denoiser
Write-Host "[1/3] Rendering 1 SPP WITHOUT denoiser (very noisy)..."
$env:YSU_GPU_WINDOW = 1
$env:YSU_GPU_W = 320
$env:YSU_GPU_H = 180
$env:YSU_GPU_SPP = 1
$env:YSU_GPU_DUMP_ONESHOT = 1
$env:YSU_NEURAL_DENOISE = 0
$env:YSU_GPU_SEED = 42

./gpu_demo.exe *>$null
if (Test-Path window_dump.ppm) {
    Move-Item -Path window_dump.ppm -Destination window_dump_1spp_noisy.ppm -Force
    Write-Host "      ✓ Generated: window_dump_1spp_noisy.ppm" -ForegroundColor Green
} else {
    Write-Host "      ✗ FAILED to generate" -ForegroundColor Red
    exit 1
}

# Test 2: 1 SPP WITH denoiser
Write-Host "[2/3] Rendering 1 SPP WITH denoiser..."
$env:YSU_NEURAL_DENOISE = 1

./gpu_demo.exe *>$null
if (Test-Path window_dump.ppm) {
    Move-Item -Path window_dump.ppm -Destination window_dump_1spp_denoised.ppm -Force
    Write-Host "      ✓ Generated: window_dump_1spp_denoised.ppm" -ForegroundColor Green
} else {
    Write-Host "      ✗ FAILED" -ForegroundColor Red
    exit 1
}

# Test 3: 16 SPP clean reference
Write-Host "[3/3] Rendering 16 SPP clean reference..."
$env:YSU_GPU_SPP = 16
$env:YSU_NEURAL_DENOISE = 0

./gpu_demo.exe *>$null
if (Test-Path window_dump.ppm) {
    Move-Item -Path window_dump.ppm -Destination window_dump_16spp_clean.ppm -Force
    Write-Host "      ✓ Generated: window_dump_16spp_clean.ppm" -ForegroundColor Green
} else {
    Write-Host "      ✗ FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Analyzing results with Python..."
Write-Host "============================================================"
Write-Host ""

python test_denoise_effectiveness.py

Write-Host ""
Write-Host "============================================================"
Write-Host "Test complete!" -ForegroundColor Green
Write-Host "============================================================"
Write-Host ""
Write-Host "Generated files:"
Write-Host "  - window_dump_1spp_noisy.ppm (very grainy)"
Write-Host "  - window_dump_1spp_denoised.ppm (denoised version)"
Write-Host "  - window_dump_16spp_clean.ppm (clean reference)"
Write-Host ""
Write-Host "Expected observations:"
Write-Host "  - Noisy: looks speckled/grainy (random noise)"
Write-Host "  - Denoised: should be much smoother"
Write-Host "  - Clean: smoothest (16 samples per pixel)"
Write-Host ""
