#!/bin/sh
# Build and run a raw host-registration probe outside the GreenBoost shim.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_hostreg_probe_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_hostreg_probe_runner"
SRC="$ROOT/runners/runtime_hostreg_probe_runner.cu"

echo "Building raw hostreg probe..."
g++ -x c++ -O3 -std=c++23 -I/opt/cuda/include "$SRC" -lcuda -o "$BIN"

echo "Case mmap_anon..."
"$BIN" > "$OUTDIR/mmap_anon.log" 2>&1 || true

echo "Case malloc..."
HOSTREG_USE_MALLOC=1 "$BIN" > "$OUTDIR/malloc.log" 2>&1 || true

echo "Case mmap_anon_mlock..."
HOSTREG_TRY_MLOCK=1 "$BIN" > "$OUTDIR/mmap_anon_mlock.log" 2>&1 || true

{
    echo "runtime_hostreg_probe"
    echo "====================="
    echo ""
    for name in mmap_anon malloc mmap_anon_mlock; do
        echo "[$name]"
        sed -n '1,120p' "$OUTDIR/$name.log"
        echo ""
    done
} > "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
