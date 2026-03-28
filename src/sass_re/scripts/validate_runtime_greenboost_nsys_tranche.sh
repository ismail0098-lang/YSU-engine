#!/bin/sh
# Capture Nsight Systems traces for GreenBoost DMA-BUF, HOSTREG, and UVM
# oversubscription paths across multiple access patterns.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_greenboost_nsys_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_greenboost_oversub_runner"
SRC="$ROOT/runners/runtime_greenboost_oversub_runner.cu"
BYTES="${GREENBOOST_NSYS_BYTES:-15032385536}"
SHIM="/usr/lib/libgreenboost_cuda.so"
STD_FLAG="$("$ROOT/scripts/resolve_nvcc_std_flag.sh")"

cleanup() {
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building GreenBoost Nsight Systems tranche..."
nvcc -arch=sm_89 -O3 -lineinfo $STD_FLAG -cudart shared "$SRC" -o "$BIN"

profile_case() {
    name="$1"
    shift
    rep="$OUTDIR/$name"
    "$@" >/dev/null 2>&1 || true
    if [ -f "$rep.nsys-rep" ]; then
        nsys stats --report cudaapisum,gpukernsum --format csv --output "$rep" "$rep.nsys-rep" >/dev/null 2>&1 || true
    fi
}

sudo modprobe -r greenboost >/dev/null 2>&1 || true
sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=16 safety_reserve_gb=4 nvme_swap_gb=8 nvme_pool_gb=8
profile_case path_a_dmabuf_14g \
    sudo nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt,nvtx \
    --output "$OUTDIR/path_a_dmabuf_14g" \
    env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=1 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES"
sudo modprobe -r greenboost >/dev/null 2>&1 || true

profile_case path_b_hostreg_14g \
    nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt,nvtx \
    --output "$OUTDIR/path_b_hostreg_14g" \
    env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES"

sudo modprobe -r greenboost >/dev/null 2>&1 || true
sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=16 safety_reserve_gb=4 nvme_swap_gb=8 nvme_pool_gb=8
profile_case path_a_dmabuf_hot_14g \
    sudo nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt,nvtx \
    --output "$OUTDIR/path_a_dmabuf_hot_14g" \
    env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=1 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" hot_only
sudo modprobe -r greenboost >/dev/null 2>&1 || true

profile_case path_b_hostreg_hot_14g \
    nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt,nvtx \
    --output "$OUTDIR/path_b_hostreg_hot_14g" \
    env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" hot_only

sudo modprobe -r greenboost >/dev/null 2>&1 || true
sudo modprobe greenboost physical_vram_gb=12 virtual_vram_gb=16 safety_reserve_gb=4 nvme_swap_gb=8 nvme_pool_gb=8
profile_case path_a_dmabuf_hop_14g \
    sudo nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt,nvtx \
    --output "$OUTDIR/path_a_dmabuf_hop_14g" \
    env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=1 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" hop
sudo modprobe -r greenboost >/dev/null 2>&1 || true

profile_case path_b_hostreg_hop_14g \
    nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt,nvtx \
    --output "$OUTDIR/path_b_hostreg_hop_14g" \
    env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=0 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES" hop

profile_case path_c_uvm_14g \
    nsys profile --force-overwrite=true --sample=none --trace=cuda,osrt,nvtx \
    --output "$OUTDIR/path_c_uvm_14g" \
    env \
    LD_PRELOAD="$SHIM" \
    GREENBOOST_ACTIVE=1 \
    GREENBOOST_DEBUG=1 \
    GREENBOOST_USE_DMA_BUF=0 \
    GREENBOOST_NO_HOSTREG=1 \
    GREENBOOST_VRAM_HEADROOM_MB=20000 \
    "$BIN" "$BYTES"

{
    echo "runtime_greenboost_nsys_tranche"
    echo "==============================="
    echo ""
    for name in \
        path_a_dmabuf_14g \
        path_b_hostreg_14g \
        path_a_dmabuf_hot_14g \
        path_b_hostreg_hot_14g \
        path_a_dmabuf_hop_14g \
        path_b_hostreg_hop_14g \
        path_c_uvm_14g; do
        echo "[$name]"
        if [ -f "$OUTDIR/$name.nsys-rep" ]; then
            echo "nsys_rep=$OUTDIR/$name.nsys-rep"
        else
            echo "missing_nsys_rep=1"
        fi
        if [ -f "$OUTDIR/${name}_cudaapisum.csv" ]; then
            echo "cudaapisum_csv=$OUTDIR/${name}_cudaapisum.csv"
        fi
        if [ -f "$OUTDIR/${name}_gpukernsum.csv" ]; then
            echo "gpukernsum_csv=$OUTDIR/${name}_gpukernsum.csv"
        fi
        echo ""
    done
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
