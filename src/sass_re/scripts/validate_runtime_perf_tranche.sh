#!/bin/sh
# Build, run, and optionally profile the runtime performance tranche for
# Ada-specific execution levers that are not primarily mnemonic-hunting flags:
#   - L2 persistence via cudaAccessPolicyWindow
#   - CUDA Graph launch replay vs plain launches
#   - cp.async-style overlap vs synchronous shared staging

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NVCC="${NVCC:-nvcc}"
NVCC_STD_FLAG="$("$SCRIPT_DIR/resolve_nvcc_std_flag.sh" "$NVCC")"
OUTDIR="${1:-$ROOT/results/runs/runtime_perf_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_perf_features_runner"
SRC="$ROOT/runners/runtime_perf_features_runner.cu"

echo "Building runtime performance tranche..."
$NVCC -arch=sm_89 $NVCC_STD_FLAG -O3 -lineinfo "$SRC" -o "$BIN"

echo "Running runtime performance tranche..."
"$BIN" > "$OUTDIR/run.log" 2>&1

if command -v ncu >/dev/null 2>&1; then
    echo "Profiling runtime performance tranche with Nsight Compute..."
    ncu --target-processes all \
        --metrics smsp__inst_executed.sum,smsp__warp_active.avg,l1tex__t_bytes.sum,dram__bytes.sum \
        --csv "$BIN" > "$OUTDIR/ncu.csv" 2>&1 || true
fi

{
    echo "runtime_perf_tranche"
    echo "===================="
    echo ""
    cat "$OUTDIR/run.log"
    if [ -f "$OUTDIR/ncu.csv" ]; then
        echo ""
        echo "ncu_csv=$OUTDIR/ncu.csv"
    fi
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
