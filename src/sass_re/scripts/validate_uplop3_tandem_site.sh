#!/usr/bin/env bash
set -euo pipefail

if (( $# < 7 )) || (( ($# % 2) == 0 )); then
    echo "usage: $0 <outdir> <runner_cpp> <baseline_cubin> <kernel> <pattern> <label1> <cubin1> [label2 cubin2 ...]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$1"
RUNNER_CPP="$2"
BASELINE="$3"
KERNEL="$4"
PATTERN="$5"
shift 5
SPECS=("$@")
mkdir -p "$OUTDIR"

RUNNER="$OUTDIR/$(basename "${RUNNER_CPP%.cpp}")"
g++ -std=c++20 -O2 "$RUNNER_CPP" -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -ldl -o "$RUNNER"

run_one() {
    local label="$1"
    local cubin="$2"
    "$RUNNER" "$cubin" "$KERNEL" "$PATTERN" > "$OUTDIR/${label}_run.log"
}

run_one baseline "$BASELINE"

for ((i = 0; i < ${#SPECS[@]}; i += 2)); do
    run_one "${SPECS[i]}" "${SPECS[i + 1]}"
done

if command -v ncu >/dev/null 2>&1; then
    BASE_METRICS="smsp__cycles_elapsed.avg,smsp__inst_executed.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__shared_mem_per_block_static"
    STALL_METRICS="smsp__warp_issue_stalled_barrier_per_warp_active,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_membar_per_warp_active,smsp__warp_issue_stalled_wait_per_warp_active"
    for ((i = -2; i < ${#SPECS[@]}; i += 2)); do
        if (( i < 0 )); then
            label="baseline"
            cubin="$BASELINE"
        else
            label="${SPECS[i]}"
            cubin="${SPECS[i + 1]}"
        fi
        ncu --target-processes all --kernel-name regex:"$KERNEL" \
          --metrics "$BASE_METRICS" --csv --page raw \
          "$RUNNER" "$cubin" "$KERNEL" "$PATTERN" \
          > "$OUTDIR/${label}_ncu.csv" 2> "$OUTDIR/${label}_ncu.stderr" || true
        ncu --target-processes all --kernel-name regex:"$KERNEL" \
          --metrics "$STALL_METRICS" --csv --page raw \
          "$RUNNER" "$cubin" "$KERNEL" "$PATTERN" \
          > "$OUTDIR/${label}_ncu_stalls.csv" 2> "$OUTDIR/${label}_ncu_stalls.stderr" || true
    done
fi

if command -v compute-sanitizer >/dev/null 2>&1; then
    for ((i = -2; i < ${#SPECS[@]}; i += 2)); do
        if (( i < 0 )); then
            label="baseline"
            cubin="$BASELINE"
        else
            label="${SPECS[i]}"
            cubin="${SPECS[i + 1]}"
        fi
        /opt/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
          "$RUNNER" "$cubin" "$KERNEL" "$PATTERN" \
          > "$OUTDIR/${label}_memcheck.txt" 2>&1 || true
    done
fi

if command -v nsys >/dev/null 2>&1; then
    for ((i = -2; i < ${#SPECS[@]}; i += 2)); do
        if (( i < 0 )); then
            label="baseline"
            cubin="$BASELINE"
        else
            label="${SPECS[i]}"
            cubin="${SPECS[i + 1]}"
        fi
        nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
          --output "$OUTDIR/${label}_timeline" --force-overwrite=true \
          -- "$RUNNER" "$cubin" "$KERNEL" "$PATTERN" \
          > "$OUTDIR/${label}_nsys.stdout" 2> "$OUTDIR/${label}_nsys.stderr" || true
        nsys stats "$OUTDIR/${label}_timeline.nsys-rep" \
          --report gputrace --format csv \
          --output "$OUTDIR/${label}_gpu_trace" >/dev/null 2>&1 || true
    done
fi

python3 "$ROOT/scripts/uplop3_diff_fuzz.py" \
  --runner "$RUNNER" \
  --baseline "$BASELINE" \
  --kernel "$KERNEL" \
  --outdir "$OUTDIR/fuzz" \
  --pattern-start 100 \
  --count 24 \
  "${SPECS[@]}"

cat > "$OUTDIR/summary.txt" <<EOF
uplop3_tandem_site
==================

runner_cpp=$RUNNER_CPP
kernel=$KERNEL
pattern=$PATTERN
baseline=$BASELINE
EOF

echo "$OUTDIR"
