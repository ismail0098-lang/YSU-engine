#!/bin/sh
# Build and run a multi-pattern GreenBoost performance tranche.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_greenboost_perf_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_greenboost_perf_runner"
SRC="$ROOT/runners/runtime_greenboost_perf_runner.cu"
BYTES="${GREENBOOST_TEST_BYTES:-268435456}"
SHIM="/usr/lib/libgreenboost_cuda.so"
STD_FLAG="$("$ROOT/scripts/resolve_nvcc_std_flag.sh")"

cleanup() {
    sudo modprobe -r greenboost >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building runtime GreenBoost perf tranche..."
nvcc -arch=sm_89 -O3 -lineinfo $STD_FLAG -cudart shared "$SRC" -o "$BIN"

run_case() {
    name="$1"
    shift
    echo "Case $name..."
    "$@" > "$OUTDIR/$name.log" 2>&1 || true
}

run_case baseline "$BIN" "$BYTES"

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

summarize_case() {
    name="$1"
    log="$OUTDIR/$name.log"
    echo "[$name]"
    if [ ! -f "$log" ]; then
        echo "missing_log=1"
        echo ""
        return
    fi
    grep -E '^(device|sm|bytes|elems|rounds|stride_elems|alloc_ms|free_ms|sample_checksum|free_before|free_after|env_)' "$log" || true
    grep '^pattern=' "$log" || true
    if grep -q 'DMA-BUF import (pinned)' "$log"; then
        echo "resolved_path=DMA_BUF"
    elif grep -q 'HostReg alloc (no kernel module)' "$log"; then
        echo "resolved_path=HOSTREG"
    elif grep -q 'UVM alloc:' "$log"; then
        echo "resolved_path=UVM"
    else
        echo "resolved_path=UNKNOWN"
    fi
    echo ""
}

extract_second_ms() {
    log="$1"
    pattern="$2"
    sed -n "s/^pattern=$pattern first_ms=[0-9.]* second_ms=\\([0-9.]*\\) .*/\\1/p" "$log" | head -n1
}

{
    echo "runtime_greenboost_perf_tranche"
    echo "=============================="
    echo ""
    summarize_case baseline
    summarize_case path_a
    summarize_case path_b
    summarize_case path_c

    echo "[relative_slowdowns_vs_baseline]"
    for pattern in stream_rw read_reduce stride_rw compute_heavy compute_very_heavy; do
        base="$(extract_second_ms "$OUTDIR/baseline.log" "$pattern")"
        a="$(extract_second_ms "$OUTDIR/path_a.log" "$pattern")"
        b="$(extract_second_ms "$OUTDIR/path_b.log" "$pattern")"
        c="$(extract_second_ms "$OUTDIR/path_c.log" "$pattern")"
        if [ -n "$base" ]; then
            awk -v p="$pattern" -v base="$base" -v a="$a" -v b="$b" -v c="$c" '
                function ratio(x, y) { return (x == "" || y == "" || y == 0.0) ? "NA" : sprintf("%.6f", x / y); }
                BEGIN {
                    printf("%s path_a_vs_baseline=%.6f path_b_vs_baseline=%.6f path_c_vs_baseline=%.6f\n",
                           p, (a + 0.0) / (base + 0.0), (b + 0.0) / (base + 0.0), (c + 0.0) / (base + 0.0));
                }'
        fi
    done
    if [ -f "$OUTDIR/path_a_pool_info.txt" ]; then
        echo ""
        echo "[path_a_pool_info]"
        sed -n '1,120p' "$OUTDIR/path_a_pool_info.txt"
    fi
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
