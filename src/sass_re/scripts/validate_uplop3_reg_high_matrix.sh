#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )) || (( ($# % 2) == 0 )); then
    echo "usage: $0 <outdir> <baseline_cubin> <kernel_name> <label1> <cubin1> [label2 cubin2 ...]" >&2
    exit 2
fi

OUTDIR="$1"
BASELINE="$2"
KERNEL="$3"
shift 3
SPECS=("$@")

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$OUTDIR/uplop3_reg_high_cubin_driver_runner"
mkdir -p "$OUTDIR"

g++ -std=c++20 -O2 \
    "$ROOT/runners/uplop3_reg_high_cubin_driver_runner.cpp" \
    -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -ldl -o "$RUNNER"

SUMMARY="$OUTDIR/summary.txt"
{
    echo "uplop3_reg_high_cubin_pattern_matrix"
    echo "===================================="
    echo
    echo "kernel=$KERNEL"
    echo
} > "$SUMMARY"

run_case() {
    local cubin="$1"
    local pattern="$2"
    timeout 20s "$RUNNER" "$cubin" "$KERNEL" "$pattern"
}

for pattern in 0 1 2 3; do
    echo "pattern=$pattern" >> "$SUMMARY"
    baseline_log="$OUTDIR/pattern_${pattern}_baseline.txt"
    if run_case "$BASELINE" "$pattern" >"$baseline_log" 2>&1; then
        baseline_rc=0
    else
        baseline_rc=$?
    fi
    echo "- baseline:" >> "$SUMMARY"
    echo "  rc=$baseline_rc" >> "$SUMMARY"
    sed 's/^/  /' "$baseline_log" >> "$SUMMARY"
    echo >> "$SUMMARY"

    for ((i = 0; i < ${#SPECS[@]}; i += 2)); do
        label="${SPECS[i]}"
        cubin="${SPECS[i + 1]}"
        log="$OUTDIR/pattern_${pattern}_${label}.txt"
        if run_case "$cubin" "$pattern" >"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
        echo "- $label:" >> "$SUMMARY"
        echo "  rc=$rc" >> "$SUMMARY"
        if (( rc == 0 )) && diff -u "$baseline_log" "$log" >/dev/null; then
            echo "  relation=same_as_baseline" >> "$SUMMARY"
        elif (( rc == 0 )); then
            echo "  relation=diff_from_baseline" >> "$SUMMARY"
        else
            echo "  relation=n/a" >> "$SUMMARY"
        fi
        sed 's/^/  /' "$log" >> "$SUMMARY"
        echo >> "$SUMMARY"
    done
done

echo "$OUTDIR"
