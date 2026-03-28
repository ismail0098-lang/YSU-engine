#!/bin/sh
# Timeline profiling with Nsight Systems.
# Captures CUDA + NVTX + OS runtime traces.
#
# Usage: profile_nsys.sh [--grid <size>] [--output <dir>]

set -eu

GRID=128
OUTDIR="results"

while [ $# -gt 0 ]; do
    case "$1" in
        --grid)  GRID="$2"; shift 2 ;;
        --output) OUTDIR="$2"; shift 2 ;;
        *) echo "Usage: profile_nsys.sh [--grid <size>] [--output <dir>]"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="${SCRIPT_DIR}/../../../build/bin/lbm_bench"

if [ ! -x "$BENCH" ]; then
    echo "ERROR: lbm_bench not found at $BENCH" >&2
    exit 1
fi

command -v nsys >/dev/null 2>&1 || { echo "ERROR: nsys not in PATH" >&2; exit 1; }

mkdir -p "$OUTDIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REP_FILE="${OUTDIR}/lbm_bench_${GRID}_${TIMESTAMP}"

echo "=== Nsight Systems: Full benchmark at ${GRID}^3 ==="

nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --output "$REP_FILE" \
    "$BENCH" --all --grid "$GRID" --warmup 5 --steps 30

echo "Saved: ${REP_FILE}.nsys-rep"

# Post-process: extract kernel and API summaries
echo "Extracting statistics ..."
nsys stats --report cuda_gpu_kern_sum "${REP_FILE}.nsys-rep" \
    --output "${REP_FILE}_kernel_summary" --format csv 2>/dev/null || true
nsys stats --report cuda_api_sum "${REP_FILE}.nsys-rep" \
    --output "${REP_FILE}_api_summary" --format csv 2>/dev/null || true

echo "Done."
