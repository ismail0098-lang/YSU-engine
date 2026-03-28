#!/bin/sh
# Build and run an oversubscription-focused GreenBoost tranche.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_greenboost_oversub_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_greenboost_oversub_runner"
SRC="$ROOT/runners/runtime_greenboost_oversub_runner.cu"
BYTES="${GREENBOOST_OVERSUB_BYTES:-15032385536}"
SHIM="/usr/lib/libgreenboost_cuda.so"
STD_FLAG="$("$ROOT/scripts/resolve_nvcc_std_flag.sh")"

cleanup() {
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building GreenBoost oversubscription tranche..."
nvcc -arch=sm_89 -O3 -lineinfo $STD_FLAG -cudart shared "$SRC" -o "$BIN"

run_case() {
    name="$1"
    shift
    echo "Case $name..."
    "$@" > "$OUTDIR/$name.log" 2>&1 || true
}

sudo modprobe -r greenboost >/dev/null 2>&1 || true
sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=4 safety_reserve_gb=4 nvme_swap_gb=8 nvme_pool_gb=8
run_case path_a sudo env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=1 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES"
sudo sh -c 'cat /sys/class/greenboost/greenboost/pool_info' > "$OUTDIR/path_a_pool_info.txt" 2>&1 || true
sudo modprobe -r greenboost >/dev/null 2>&1 || true

run_case path_b env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES"

run_case path_c env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=1 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES"

{
    echo "runtime_greenboost_oversub_tranche"
    echo "================================="
    echo ""
    for name in path_a path_b path_c; do
        log="$OUTDIR/$name.log"
        echo "[$name]"
        if [ -f "$log" ]; then
            grep -E '^(device|sm|bytes|page_size|page_count|hot_window_bytes|alloc_ms|first_touch_ms|second_touch_ms|hot_window_first_ms|hot_window_second_ms|free_ms|sample_checksum|free_before|free_after|env_)' "$log" || true
            if grep -q 'DMA-BUF import (pinned)' "$log"; then
                echo "resolved_path=DMA_BUF"
            elif grep -q 'HostReg alloc (no kernel module)' "$log"; then
                echo "resolved_path=HOSTREG"
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
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
