#!/bin/sh
# Single-kernel Nsight Compute profiling.
# Usage: profile_ncu.sh <kernel_name> <grid_size> [output_dir]
#
# Produces:
#   1. Full .ncu-rep file (--set full)
#   2. CSV with key metrics for analysis
#
# Requires: ncu (Nsight Compute CLI) in PATH.

set -eu

KERNEL="${1:?Usage: profile_ncu.sh <kernel_name> <grid_size> [output_dir]}"
GRID="${2:?Usage: profile_ncu.sh <kernel_name> <grid_size> [output_dir]}"
OUTDIR="${3:-results}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="${SCRIPT_DIR}/../../../build/bin/lbm_bench"

if [ ! -x "$BENCH" ]; then
    echo "ERROR: lbm_bench not found at $BENCH" >&2
    echo "Build with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build" >&2
    exit 1
fi

command -v ncu >/dev/null 2>&1 || { echo "ERROR: ncu not in PATH" >&2; exit 1; }

mkdir -p "$OUTDIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REP_FILE="${OUTDIR}/${KERNEL}_${GRID}_${TIMESTAMP}.ncu-rep"
CSV_FILE="${OUTDIR}/${KERNEL}_${GRID}_${TIMESTAMP}_metrics.csv"

echo "=== Nsight Compute: ${KERNEL} at ${GRID}^3 ==="

# Pass 1: Full profiling (warmed-up iterations)
echo "Pass 1: Full .ncu-rep ..."
ncu \
    --set full \
    --launch-skip 5 \
    --launch-count 3 \
    --export "$REP_FILE" \
    "$BENCH" --kernel "$KERNEL" --grid "$GRID" --warmup 5 --steps 10

echo "Saved: $REP_FILE"

# Pass 2: Extract key metrics to CSV
echo "Pass 2: Metrics CSV ..."
ncu \
    --launch-skip 5 \
    --launch-count 3 \
    --metrics \
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        lts__throughput.avg.pct_of_peak_sustained_elapsed,\
        l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__warps_active.avg.pct_of_peak_sustained_active,\
        launch__registers_per_thread,\
        launch__shared_mem_per_block_dynamic,\
        launch__shared_mem_per_block_static,\
        smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active,\
        smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
        smsp__warp_issue_stalled_wait.avg.pct_of_peak_sustained_active,\
        smsp__warp_issue_stalled_not_selected.avg.pct_of_peak_sustained_active,\
        lts__t_sector_hit_rate.pct,\
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    --csv \
    "$BENCH" --kernel "$KERNEL" --grid "$GRID" --warmup 5 --steps 10 \
    > "$CSV_FILE"

echo "Saved: $CSV_FILE"
echo "Done."
