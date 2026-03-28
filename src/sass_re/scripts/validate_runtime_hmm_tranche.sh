#!/bin/sh
# Build, run, and optionally profile the managed-memory / HMM runtime tranche.
# Focus:
#   - cold managed-memory GPU access after host residency
#   - explicit prefetch to GPU
#   - advice + prefetch to GPU
#   - host-side touch with and without prefetch back to CPU

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NVCC="${NVCC:-nvcc}"
NVCC_STD_FLAG="$("$SCRIPT_DIR/resolve_nvcc_std_flag.sh" "$NVCC")"
OUTDIR="${1:-$ROOT/results/runs/runtime_hmm_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_hmm_features_runner"
SRC="$ROOT/runners/runtime_hmm_features_runner.cu"

echo "Building runtime HMM tranche..."
$NVCC -arch=sm_89 $NVCC_STD_FLAG -O3 -lineinfo "$SRC" -o "$BIN"

echo "Running runtime HMM tranche..."
"$BIN" > "$OUTDIR/run.log" 2>&1

if command -v ncu >/dev/null 2>&1; then
    echo "Profiling runtime HMM tranche with Nsight Compute..."
    ncu --target-processes all \
        --metrics smsp__inst_executed.sum,smsp__warp_active.avg,l1tex__t_bytes.sum,dram__bytes.sum \
        --csv "$BIN" > "$OUTDIR/ncu.csv" 2>&1 || true
fi

{
    echo "runtime_hmm_tranche"
    echo "==================="
    echo ""
    cat "$OUTDIR/run.log"
    if [ -f "$OUTDIR/ncu.csv" ]; then
        echo ""
        echo "ncu_csv=$OUTDIR/ncu.csv"
    fi
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
