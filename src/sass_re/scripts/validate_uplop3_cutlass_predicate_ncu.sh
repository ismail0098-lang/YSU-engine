#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/results/runs/uplop3_cutlass_predicate_ncu_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

BASE_DIR="${2:-$ROOT/results/runs/uplop3_cutlass_predicate_combo_matrix_latest}"
PATTERN="${3:-2}"

RUNNER="$RUN_DIR/uplop3_cutlass_predicate_cubin_driver_runner"
g++ -std=c++20 -O2 \
  "$ROOT/runners/uplop3_cutlass_predicate_cubin_driver_runner.cpp" \
  -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -ldl -o "$RUNNER"

declare -A CASES=(
  [baseline]="$BASE_DIR/probe_cutlass_predicate_pipeline_O3.cubin"
  [occ4]="$BASE_DIR/occ4.cubin"
  [occ5]="$BASE_DIR/occ5.cubin"
  [occ2_occ4_occ5]="$BASE_DIR/occ2_occ4_occ5.cubin"
)

BASELINE="${CASES[baseline]}"
if [[ ! -f "$BASELINE" ]]; then
  echo "missing baseline cubin: $BASELINE" >&2
  exit 2
fi

if [[ ! -f "${CASES[occ4]}" ]]; then
  python3 "$ROOT/scripts/patch_uplop3_cubin.py" \
    --input "$BASELINE" --output "${CASES[occ4]}" --occurrence 4 \
    > "$RUN_DIR/occ4.patch.txt"
fi
if [[ ! -f "${CASES[occ5]}" ]]; then
  python3 "$ROOT/scripts/patch_uplop3_cubin.py" \
    --input "$BASELINE" --output "${CASES[occ5]}" --occurrence 5 \
    > "$RUN_DIR/occ5.patch.txt"
fi

BASE_METRICS="smsp__cycles_elapsed.avg,smsp__inst_executed.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__shared_mem_per_block_static"
STALL_METRICS="smsp__warp_issue_stalled_barrier_per_warp_active,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_membar_per_warp_active,smsp__warp_issue_stalled_wait_per_warp_active"

for label in baseline occ4 occ5 occ2_occ4_occ5; do
  cubin="${CASES[$label]}"
  if [[ ! -f "$cubin" ]]; then
    echo "missing cubin for $label: $cubin" >&2
    exit 2
  fi
  "$RUNNER" "$cubin" probe_cutlass_predicate_pipeline "$PATTERN" > "$RUN_DIR/${label}_run.log"
  ncu --target-processes all --kernel-name regex:probe_cutlass_predicate_pipeline \
    --metrics "$BASE_METRICS" \
    --csv --page raw "$RUNNER" "$cubin" probe_cutlass_predicate_pipeline "$PATTERN" \
    > "$RUN_DIR/${label}_ncu.csv" 2> "$RUN_DIR/${label}_ncu.stderr" || true
  ncu --target-processes all --kernel-name regex:probe_cutlass_predicate_pipeline \
    --metrics "$STALL_METRICS" \
    --csv --page raw "$RUNNER" "$cubin" probe_cutlass_predicate_pipeline "$PATTERN" \
    > "$RUN_DIR/${label}_ncu_stalls.csv" 2> "$RUN_DIR/${label}_ncu_stalls.stderr" || true
done

cat > "$RUN_DIR/summary.txt" <<EOF
uplop3_cutlass_predicate_ncu
============================

base_dir=$BASE_DIR
pattern=$PATTERN

cases:
- baseline
- occ4
- occ5
- occ2_occ4_occ5
EOF

echo "$RUN_DIR"
