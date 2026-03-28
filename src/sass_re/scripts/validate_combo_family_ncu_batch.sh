#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/results/runs/combo_family_ncu_batch_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

run_step() {
  local name="$1"
  shift
  echo "=== $name ==="
  "$@" "$RUN_DIR/$name"
}

run_step small_safe_default bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_safe_tranche.sh"
PTXAS_EXTRA=-dlcm=cg run_step small_safe_dlcm_cg bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_safe_tranche.sh"
run_step small_depth_default bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_depth_safe_tranche.sh"
PTXAS_EXTRA=-dlcm=cg run_step small_depth_dlcm_cg bash "$ROOT/scripts/validate_combo_warp_atomic_cache_profile_depth_safe_tranche.sh"
run_step sys_shallow_default bash "$ROOT/scripts/validate_combo_blockred_warp_atomg64sys_ops_profile_safe_tranche.sh"
PTXAS_EXTRA=-dlcm=cg run_step sys_shallow_dlcm_cg bash "$ROOT/scripts/validate_combo_blockred_warp_atomg64sys_ops_profile_safe_tranche.sh"
run_step sys_deep_default bash "$ROOT/scripts/validate_combo_blockred_warp_atomg64sys_ops_profile_depth_safe_tranche.sh"
run_step sys_store_default bash "$ROOT/scripts/validate_combo_blockred_warp_atomg64sys_ops_store_profile_safe_tranche.sh"
PTXAS_EXTRA=-dlcm=cg run_step sys_store_dlcm_cg bash "$ROOT/scripts/validate_combo_blockred_warp_atomg64sys_ops_store_profile_safe_tranche.sh"

python "$ROOT/scripts/summarize_combo_family_ncu_batch.py" "$RUN_DIR"
echo "$RUN_DIR"
