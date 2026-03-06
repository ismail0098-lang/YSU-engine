# SASS Reverse Engineering Toolkit

Hands-on tools for reverse-engineering NVIDIA SASS across multiple GPU architectures.
Currently supports **Ada Lovelace (SM 8.9, RTX 4070 Ti Super)** and
**Pascal (SM 6.1, GTX 1050 Ti)**. Real measurements, real disassembly, real encoding analysis.

See [RESULTS.md](RESULTS.md) for measured data and [PAPER_OUTLINE.md](PAPER_OUTLINE.md) for our comparison paper.

## Directory layout

```
src/sass_re/
  probes/           -- minimal CUDA kernels that isolate specific instructions
  microbench/       -- latency & throughput measurement harnesses
  scripts/          -- automation: compile, disassemble, compare, analyze
  results/          -- output from runs (GPU-tagged subdirectories)
  RESULTS.md        -- Ada Lovelace measurements and analysis
  PAPER_OUTLINE.md  -- research paper outline (Pascal vs Ada comparison)
  COMPARISON.md     -- auto-generated cross-architecture analysis
```

## Quick start

```powershell
# From repo root:
cd src/sass_re

# === RTX 4070 Ti Super (Ada Lovelace) ===
.\scripts\disassemble_all.ps1 -Arch sm_89 -GpuTag Ada_RTX4070TiS
.\scripts\build_and_run_latency.ps1 -Arch sm_89
.\scripts\build_and_run_throughput.ps1 -Arch sm_89

# === GTX 1050 Ti (Pascal) ===
# One-click pipeline (does everything):
.\scripts\run_pascal_pipeline.ps1

# Or step-by-step:
.\scripts\disassemble_all.ps1 -Arch sm_61 -GpuTag Pascal_GTX1050Ti
.\scripts\build_and_run_latency.ps1 -Arch sm_61
.\scripts\build_and_run_throughput.ps1 -Arch sm_61

# === Compare architectures ===
python scripts/compare_architectures.py results/Ada_RTX4070TiS_* results/Pascal_GTX1050Ti_*

# === Encoding analysis ===
python scripts/encoding_analysis.py results/<gpu_tag_timestamp>/
```

## Requirements

- **CUDA Toolkit 13.x** — for SM 7.5+ (Turing, Ampere, Ada, Hopper)
- **CUDA Toolkit 12.x** — required for SM 6.1 Pascal (CUDA 13.x dropped SM < 7.5).
  Install side-by-side: [CUDA 12.6 archive](https://developer.nvidia.com/cuda-12-6-0-download-archive)
- **MSVC Build Tools** (vcvars64.bat)
- **Python 3.x** (for comparison and encoding analysis)
- Scripts auto-detect the correct CUDA version based on target SM

## What each probe does

| Kernel file | Isolated instructions | SM 6.1 compat |
|---|---|---|
| probe_fp32_arith.cu | FADD, FMUL, FFMA, FMNMX | Yes |
| probe_int_arith.cu | IADD3/IADD, IMAD/XMAD, ISETP, LEA | Yes |
| probe_mufu.cu | MUFU (sin, cos, rsqrt, rcp, ex2, lg2) | Yes |
| probe_bitwise.cu | LOP3/LOP, SHF, PRMT, BFI, FLO, POPC | Yes |
| probe_memory.cu | LDG, STG, LDS, STS, atomics | Yes |
| probe_conversions.cu | F2I, I2F, F2F, FRND | Yes |
| probe_control_flow.cu | BRA, divergence, SHFL, VOTE, predication | Yes (MATCH.ALL guarded) |
| probe_special_regs.cu | S2R (tid, ctaid, clock, globaltimer) | Yes |
| probe_tensor.cu | HMMA (tensor core) via wmma intrinsics | **No** (auto-skipped on SM<7.0) |

## Supported GPUs

| GPU | Architecture | SM | Status |
|---|---|---|---|
| RTX 4070 Ti Super | Ada Lovelace | 8.9 | Measured |
| GTX 1050 Ti | Pascal | 6.1 | Ready (awaiting hardware) |

Any CUDA-capable GPU can be tested by passing the appropriate `-Arch sm_XX` parameter.
