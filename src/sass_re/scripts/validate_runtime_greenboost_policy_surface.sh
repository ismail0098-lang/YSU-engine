#!/bin/sh
# Sweep GreenBoost Path A across virtual_vram_gb and request size.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_greenboost_policy_surface_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_greenboost_driver_runner"
SRC="$ROOT/runners/runtime_greenboost_driver_runner.cu"
SHIM="/usr/lib/libgreenboost_cuda.so"
VIRTUAL_GB_LIST="${GREENBOOST_SURFACE_VIRTUAL_GB_LIST:-4 6 8}"
SIZE_GIB_LIST="${GREENBOOST_SURFACE_SIZE_GIB_LIST:-4 4.125 6 8}"

cleanup() {
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building GreenBoost policy surface runner..."
g++ -x c++ -O3 -std=c++23 -I/opt/cuda/include "$SRC" -lcuda -o "$BIN"

for vgb in $VIRTUAL_GB_LIST; do
    for gib in $SIZE_GIB_LIST; do
        bytes="$(awk -v g="$gib" 'BEGIN { printf "%.0f", g * 1024 * 1024 * 1024 }')"
        tag="v${vgb}_s${gib}GiB"
        log="$OUTDIR/${tag}.log"
        pool="$OUTDIR/${tag}_pool_info.txt"
        echo "Case virtual_vram_gb=${vgb}, size=${gib}GiB..."
        sudo modprobe -r greenboost >/dev/null 2>&1 || true
        sudo modprobe greenboost \
            physical_vram_gb=12 \
            virtual_vram_gb="$vgb" \
            safety_reserve_gb=4 \
            nvme_swap_gb=8 \
            nvme_pool_gb=8
        sudo env \
            LD_PRELOAD="$SHIM" \
            GREENBOOST_ACTIVE=1 \
            GREENBOOST_DEBUG=1 \
            GREENBOOST_USE_DMA_BUF=1 \
            GREENBOOST_NO_HOSTREG=0 \
            GREENBOOST_VRAM_HEADROOM_MB=20000 \
            "$BIN" "$bytes" > "$log" 2>&1 || true
        sudo sh -c 'cat /sys/class/greenboost/greenboost/pool_info' > "$pool" 2>&1 || true
    done
done

{
    echo "runtime_greenboost_policy_surface"
    echo "================================"
    echo ""
    for vgb in $VIRTUAL_GB_LIST; do
        for gib in $SIZE_GIB_LIST; do
            tag="v${vgb}_s${gib}GiB"
            log="$OUTDIR/${tag}.log"
            echo "[virtual_vram_gb=${vgb} size_gib=${gib}]"
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
    done
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
