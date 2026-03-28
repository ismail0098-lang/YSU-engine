#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/results/runs/combo_blockred_warp_atomg64sys_ops_store_tranche_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

STD_FLAG="$("$ROOT/scripts/resolve_nvcc_std_flag.sh")"
BIN="$RUN_DIR/combo_blockred_warp_atomg64sys_ops_store_runner"

/opt/cuda/bin/nvcc -arch=sm_89 ${STD_FLAG} -O3 -Xptxas -O3 -lineinfo \
  "$ROOT/runners/combo_blockred_warp_atomg64sys_ops_store_runner.cu" \
  "$ROOT/probes/probe_combo_blockred_warp_atomg64sys_ops_store_cache_wombo.cu" \
  -o "$BIN"

"$BIN" 200 > "$RUN_DIR/run.log"

if command -v ncu >/dev/null 2>&1; then
  ncu --target-processes all --kernel-name regex:probe_combo_blockred_warp_atomg64sys_ops_store_cache_wombo \
    --metrics \
smsp__cycles_elapsed.avg,smsp__inst_executed.sum,smsp__warps_issue_stalled_barrier_per_warp_active.avg,smsp__warps_issue_stalled_short_scoreboard_per_warp_active.avg,smsp__warps_issue_stalled_long_scoreboard_per_warp_active.avg,smsp__warps_issue_stalled_membar_per_warp_active.avg,smsp__warps_issue_stalled_wait_per_warp_active.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv --page raw "$BIN" 200 > "$RUN_DIR/ncu.csv" 2> "$RUN_DIR/ncu.stderr" || true
fi

cat > "$RUN_DIR/summary.txt" <<EOF
combo_blockred_warp_atomg64sys_ops_store_tranche
===============================================

runner:
- src/sass_re/runners/combo_blockred_warp_atomg64sys_ops_store_runner.cu

probe:
- src/sass_re/probes/probe_combo_blockred_warp_atomg64sys_ops_store_cache_wombo.cu

artifacts:
- $RUN_DIR/run.log
- $RUN_DIR/ncu.csv
- $RUN_DIR/ncu.stderr
EOF

echo "$RUN_DIR"
