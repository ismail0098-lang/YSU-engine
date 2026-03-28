#!/bin/sh
# Driver-API-native GreenBoost path study.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_greenboost_driver_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_greenboost_driver_runner"
SRC="$ROOT/runners/runtime_greenboost_driver_runner.cu"
BYTES="${GREENBOOST_TEST_BYTES:-536870912}"
SHIM="/usr/lib/libgreenboost_cuda.so"

cleanup() {
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [ ! -f "$SRC" ]; then
    echo "warning: runner source not found, skipping: $SRC" >&2
    echo "Results: $OUTDIR (skipped -- runner missing)"
    exit 0
fi

echo "Building GreenBoost driver-api tranche..."
g++ -x c++ -O3 -std=c++23 -I/opt/cuda/include "$SRC" -lcuda -o "$BIN"

echo "Case baseline..."
"$BIN" "$BYTES" > "$OUTDIR/baseline.log" 2>&1 || true

echo "Case path_a (DMA-BUF + kernel module)..."
sudo modprobe -r greenboost >/dev/null 2>&1 || true
sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=4 safety_reserve_gb=3 nvme_swap_gb=8 nvme_pool_gb=8
sudo env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=1 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" > "$OUTDIR/path_a.log" 2>&1 || true
sudo sh -c 'cat /sys/class/greenboost/greenboost/pool_info' > "$OUTDIR/path_a_pool_info.txt" 2>&1 || true
sudo modprobe -r greenboost >/dev/null 2>&1 || true

echo "Case path_a_user_ctx (DMA-BUF + explicit user ctx)..."
sudo modprobe -r greenboost >/dev/null 2>&1 || true
sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=4 safety_reserve_gb=3 nvme_swap_gb=8 nvme_pool_gb=8
sudo env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=1 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_USE_USER_CTX=1 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" > "$OUTDIR/path_a_user_ctx.log" 2>&1 || true
sudo sh -c 'cat /sys/class/greenboost/greenboost/pool_info' > "$OUTDIR/path_a_user_ctx_pool_info.txt" 2>&1 || true
sudo modprobe -r greenboost >/dev/null 2>&1 || true

echo "Case path_b (HostReg fallback)..."
env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" > "$OUTDIR/path_b.log" 2>&1 || true

echo "Case path_c (managed/UVM fallback)..."
env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=1 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" > "$OUTDIR/path_c.log" 2>&1 || true

{
    echo "runtime_greenboost_driver_tranche"
    echo "================================"
    echo ""
    for name in baseline path_a path_a_user_ctx path_b path_c; do
        log="$OUTDIR/$name.log"
        echo "[$name]"
        if [ -f "$log" ]; then
            grep -E '^(device|bytes|context_mode|primary_ctx_active_before|primary_ctx_flags_before|ctx_flags|ctx_has_map_host|can_map_host_memory|unified_addressing|alloc_result|alloc_result_name|alloc_result_string|alloc_ms|sample_checksum|free_ms|free_before|free_after|env_)' "$log" || true
            if grep -q 'DMA-BUF import (pinned)' "$log"; then
                echo "resolved_path=DMA_BUF"
            elif grep -q 'cuMemHostGetDevicePointer FAILED' "$log"; then
                echo "resolved_path=DMA_BUF_FAILED"
            elif grep -q 'HostReg alloc (no kernel module)' "$log"; then
                echo "resolved_path=HOSTREG"
            elif grep -q 'Path B (HostReg) failed' "$log" && grep -q 'UVM alloc:' "$log"; then
                echo "resolved_path=HOSTREG_TO_UVM"
            elif grep -q 'UVM alloc:' "$log"; then
                echo "resolved_path=UVM"
            else
                echo "resolved_path=UNKNOWN"
            fi
        else
            echo "missing_log=1"
        fi
        echo ""
    done
    if [ -f "$OUTDIR/path_a_pool_info.txt" ]; then
        echo "[path_a_pool_info]"
        sed -n '1,120p' "$OUTDIR/path_a_pool_info.txt"
    fi
    if [ -f "$OUTDIR/path_a_user_ctx_pool_info.txt" ]; then
        echo ""
        echo "[path_a_user_ctx_pool_info]"
        sed -n '1,120p' "$OUTDIR/path_a_user_ctx_pool_info.txt"
    fi
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
