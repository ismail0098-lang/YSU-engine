#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/results/runs/combo_uniform_divergent_atomg64sys_profile_depth_safe_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

NVCC_STD="$("$ROOT/scripts/resolve_nvcc_std_flag.sh")"
BIN="$RUN_DIR/combo_uniform_divergent_atomg64sys_profile_depth_safe_runner"
SASS="$RUN_DIR/combo_uniform_divergent_atomg64sys_profile_depth_safe_runner.sass"
LOG="$RUN_DIR/run.log"
SUMMARY="$RUN_DIR/summary.txt"
PTXAS_EXTRA="${PTXAS_EXTRA:-}"
PTXAS_FLAGS="-v"
if [[ -n "$PTXAS_EXTRA" ]]; then
  PTXAS_FLAGS="$PTXAS_FLAGS,$PTXAS_EXTRA"
fi

nvcc -arch=sm_89 -O3 "$NVCC_STD" -Xptxas "$PTXAS_FLAGS" \
  "$ROOT/runners/combo_uniform_divergent_atomg64sys_profile_depth_safe_runner.cu" \
  "$ROOT/probes/probe_combo_uniform_divergent_atomg64sys_profile_depth_safe.cu" \
  -o "$BIN"

cuobjdump --dump-sass "$BIN" > "$SASS"
"$BIN" 180 | tee "$LOG"

ncu --target-processes all --kernel-name regex:probe_combo_uniform_divergent_atomg64sys_profile_depth_safe \
  --csv --metrics smsp__cycles_elapsed.avg,smsp__inst_executed.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__shared_mem_per_block_static \
  "$BIN" 180 > "$RUN_DIR/ncu.csv"

ncu --target-processes all --kernel-name regex:probe_combo_uniform_divergent_atomg64sys_profile_depth_safe \
  --csv --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct \
  "$BIN" 180 > "$RUN_DIR/ncu_stalls.csv"

cat > "$SUMMARY" <<EOF
combo_uniform_divergent_atomg64sys_profile_depth_safe_tranche
============================================================

Probe:
- src/sass_re/probes/probe_combo_uniform_divergent_atomg64sys_profile_depth_safe.cu

Runner:
- src/sass_re/runners/combo_uniform_divergent_atomg64sys_profile_depth_safe_runner.cu

Artifacts:
- $SASS
- $LOG
- $RUN_DIR/ncu.csv
- $RUN_DIR/ncu_stalls.csv

Goal:
- Deepen the lighter uniform + divergent + SYS64 midpoint branch without
  adding block-red or direct SYS store, so the explorer can compare a
  genuinely unseen runtime-safe branch instead of another conceptual alias.
EOF

echo "$RUN_DIR"
