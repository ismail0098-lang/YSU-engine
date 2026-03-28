#!/bin/sh
# Build and run direct Path A / Path B GreenBoost reproducer probes outside
# the LD_PRELOAD shim.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_greenboost_path_probe_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_greenboost_path_probe_runner"
SRC="$ROOT/runners/runtime_greenboost_path_probe_runner.cu"
BYTES="${GREENBOOST_TEST_BYTES:-536870912}"
SHIM="/usr/lib/libgreenboost_cuda.so"

cleanup() {
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building direct GreenBoost path probe..."
g++ -x c++ -O3 -std=c++23 -I/opt/cuda/include "$SRC" -lcuda -o "$BIN"

echo "Case path_b_direct..."
"$BIN" path_b_direct "$BYTES" > "$OUTDIR/path_b_direct.log" 2>&1 || true

echo "Case path_a_direct..."
sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=4 safety_reserve_gb=3 nvme_swap_gb=8 nvme_pool_gb=8
sudo "$BIN" path_a_direct "$BYTES" > "$OUTDIR/path_a_direct.log" 2>&1 || true
sudo sh -c 'cat /sys/class/greenboost/greenboost/pool_info' > "$OUTDIR/path_a_pool_info.txt" 2>&1 || true

echo "Case path_b_direct_with_shim..."
env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=0 \
    "$BIN" path_b_direct "$BYTES" > "$OUTDIR/path_b_direct_with_shim.log" 2>&1 || true

echo "Case path_a_direct_with_shim..."
sudo env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=1 \
    GREENBOOST_NO_HOSTREG=0 \
    "$BIN" path_a_direct "$BYTES" > "$OUTDIR/path_a_direct_with_shim.log" 2>&1 || true

{
    echo "runtime_greenboost_path_probe"
    echo "============================="
    echo ""
    for name in path_b_direct path_a_direct path_b_direct_with_shim path_a_direct_with_shim; do
        echo "[$name]"
        sed -n '1,160p' "$OUTDIR/$name.log"
        echo ""
    done
    if [ -f "$OUTDIR/path_a_pool_info.txt" ]; then
        echo "[path_a_pool_info]"
        sed -n '1,120p' "$OUTDIR/path_a_pool_info.txt"
    fi
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
