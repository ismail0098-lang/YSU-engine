#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH_DIR="${1:-$ROOT/results/runs/combo_family_safe_anchor_batch_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BATCH_DIR"

SAFE_DEFAULT="$BATCH_DIR/profile_safe_default"
SAFE_DLCM="$BATCH_DIR/profile_safe_dlcm_cg"
DEPTH_DEFAULT="$BATCH_DIR/profile_depth_safe_default"
DEPTH_DLCM="$BATCH_DIR/profile_depth_safe_dlcm_cg"

bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_safe_tranche.sh" "$SAFE_DEFAULT" > "$BATCH_DIR/path_safe_default.txt"
PTXAS_EXTRA=-dlcm=cg \
  bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_safe_tranche.sh" "$SAFE_DLCM" > "$BATCH_DIR/path_safe_dlcm_cg.txt"
bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_depth_safe_tranche.sh" "$DEPTH_DEFAULT" > "$BATCH_DIR/path_depth_default.txt"
PTXAS_EXTRA=-dlcm=cg \
  bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_depth_safe_tranche.sh" "$DEPTH_DLCM" > "$BATCH_DIR/path_depth_dlcm_cg.txt"

python3 - "$SAFE_DEFAULT" "$SAFE_DLCM" "$DEPTH_DEFAULT" "$DEPTH_DLCM" <<'PY' > "$BATCH_DIR/summary.txt"
import csv
import statistics
import sys
from pathlib import Path

def load_metric_summary(run_dir: Path):
    ncu = run_dir / "ncu.csv"
    stalls = run_dir / "ncu_stalls.csv"
    out = {}
    if ncu.exists():
        header = None
        rows = []
        with ncu.open() as f:
            for row in csv.reader(f):
                if row and row[0] == "ID":
                    header = row
                elif row and row[0].isdigit():
                    rows.append(row)
        fields = [
            "smsp__cycles_elapsed.avg",
            "smsp__inst_executed.sum",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__t_sector_hit_rate.pct",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
        ]
        idx = {f: header.index(f) for f in fields}
        for f in fields:
            vals = [float(r[idx[f]]) for r in rows]
            out[f] = {
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
            }
    if stalls.exists():
        header = None
        rows = []
        with stalls.open() as f:
            for row in csv.reader(f):
                if row and row[0] == "ID":
                    header = row
                elif row and row[0].isdigit():
                    rows.append(row)
        fields = [
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_membar_per_warp_active.pct",
            "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        ]
        idx = {f: header.index(f) for f in fields}
        for f in fields:
            vals = [float(r[idx[f]]) for r in rows]
            out[f] = {
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
            }
    return out

labels = [
    ("profile_safe_default", Path(sys.argv[1])),
    ("profile_safe_dlcm_cg", Path(sys.argv[2])),
    ("profile_depth_default", Path(sys.argv[3])),
    ("profile_depth_dlcm_cg", Path(sys.argv[4])),
]

print("combo_family_safe_anchor_batch")
print("==============================")
print()
for label, path in labels:
    stats = load_metric_summary(path)
    print(label)
    print("-" * len(label))
    print(f"run_dir: {path}")
    for key in [
        "smsp__cycles_elapsed.avg",
        "smsp__inst_executed.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sector_hit_rate.pct",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_static",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_membar_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    ]:
        if key in stats:
            print(f"{key}: mean={stats[key]['mean']:.4f} median={stats[key]['median']:.4f}")
    print()
PY

echo "$BATCH_DIR"
