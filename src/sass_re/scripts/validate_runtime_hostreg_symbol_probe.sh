#!/bin/sh
# Compare unsuffixed vs _v2 libcuda host-registration symbols on this host.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$ROOT/results/runs/runtime_hostreg_symbol_probe_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

BIN="$OUTDIR/runtime_hostreg_symbol_probe_runner"
SRC="$ROOT/runners/runtime_hostreg_symbol_probe_runner.cu"
BYTES="${GREENBOOST_TEST_BYTES:-536870912}"

echo "Building hostreg symbol probe..."
g++ -x c++ -O3 -std=c++23 -I/opt/cuda/include "$SRC" -lcuda -ldl -o "$BIN"

"$BIN" "$BYTES" > "$OUTDIR/run.log" 2>&1 || true
cp "$OUTDIR/run.log" "$OUTDIR/summary.txt"

echo "Results: $OUTDIR"
echo "Summary: $OUTDIR/summary.txt"
