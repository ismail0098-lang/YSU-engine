#!/bin/sh
# Sweep GreenBoost Path A allocation sizes to find when DMA-BUF falls back.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_greenboost_dmabuf_size_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_greenboost_driver_runner"
SRC="$ROOT/runners/runtime_greenboost_driver_runner.cu"
SHIM="/usr/lib/libgreenboost_cuda.so"
SIZES_GIB="${GREENBOOST_DMABUF_SIZES_GIB:-4 8 10 12 14}"

cleanup() {
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building GreenBoost DMA-BUF size sweep..."
g++ -x c++ -O3 -std=c++23 -I/opt/cuda/include "$SRC" -lcuda -o "$BIN"

for gib in $SIZES_GIB; do
    bytes="$(awk -v g="$gib" 'BEGIN { printf "%.0f", g * 1024 * 1024 * 1024 }')"
    log="$OUTDIR/${gib}GiB.log"
    echo "Case ${gib}GiB..."
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
    sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=4 safety_reserve_gb=4 nvme_swap_gb=8 nvme_pool_gb=8
    sudo env \
        LD_PRELOAD="$SHIM" \
        GREENBOOST_ACTIVE=1 \
        GREENBOOST_DEBUG=1 \
        GREENBOOST_USE_DMA_BUF=1 \
        GREENBOOST_NO_HOSTREG=0 \
        GREENBOOST_VRAM_HEADROOM_MB=20000 \
        "$BIN" "$bytes" > "$log" 2>&1 || true
done

{
    echo "runtime_greenboost_dmabuf_size_sweep"
    echo "==================================="
    echo ""
    for gib in $SIZES_GIB; do
        log="$OUTDIR/${gib}GiB.log"
        echo "[${gib}GiB]"
        if [ -f "$log" ]; then
            grep -E '^(bytes|alloc_result|alloc_result_name|alloc_result_string|alloc_ms|free_ms|free_before|free_after|context_mode|ctx_flags|ctx_has_map_host|can_map_host_memory|unified_addressing)' "$log" || true
            if grep -q 'DMA-BUF import (pinned)' "$log"; then
                echo "resolved_path=DMA_BUF"
            elif grep -q 'HostReg alloc (no kernel module)' "$log"; then
                echo "resolved_path=HOSTREG"
            elif grep -q 'UVM alloc:' "$log"; then
                echo "resolved_path=UVM"
            else
                echo "resolved_path=UNKNOWN"
            fi
            if grep -q 'GB_IOCTL_PIN_USER_PTR failed' "$log"; then
                grep 'GB_IOCTL_PIN_USER_PTR failed' "$log" | tail -n1
            fi
        else
            echo "missing_log=1"
        fi
        echo ""
    done
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
