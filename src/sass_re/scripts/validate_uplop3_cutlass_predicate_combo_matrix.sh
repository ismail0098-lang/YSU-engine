#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/results/runs/uplop3_cutlass_predicate_combo_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

STD_FLAG="$("$ROOT/scripts/resolve_nvcc_std_flag.sh")"
BASELINE="$RUN_DIR/probe_cutlass_predicate_pipeline_O3.cubin"
BASELINE_SASS="$RUN_DIR/probe_cutlass_predicate_pipeline_O3.sass"

/opt/cuda/bin/nvcc -arch=sm_89 ${STD_FLAG} -O3 -cubin \
  "$ROOT/probes/probe_cutlass_predicate_pipeline.cu" \
  -o "$BASELINE"
nvdisasm "$BASELINE" > "$BASELINE_SASS"

python3 "$ROOT/scripts/patch_uplop3_cubin.py" \
  --input "$BASELINE" --output "$RUN_DIR/occ2_occ4.cubin" \
  --occurrence 2 --occurrence 4 > "$RUN_DIR/occ2_occ4.patch.txt"
python3 "$ROOT/scripts/patch_uplop3_cubin.py" \
  --input "$BASELINE" --output "$RUN_DIR/occ4_occ5.cubin" \
  --occurrence 4 --occurrence 5 > "$RUN_DIR/occ4_occ5.patch.txt"
python3 "$ROOT/scripts/patch_uplop3_cubin.py" \
  --input "$BASELINE" --output "$RUN_DIR/occ2_occ4_occ5.cubin" \
  --occurrence 2 --occurrence 4 --occurrence 5 > "$RUN_DIR/occ2_occ4_occ5.patch.txt"

for cubin in "$RUN_DIR"/occ2_occ4.cubin "$RUN_DIR"/occ4_occ5.cubin "$RUN_DIR"/occ2_occ4_occ5.cubin; do
  nvdisasm "$cubin" > "${cubin%.cubin}.sass"
done

bash "$ROOT/scripts/validate_uplop3_cutlass_predicate_matrix.sh" \
  "$RUN_DIR" \
  "$BASELINE" \
  probe_cutlass_predicate_pipeline \
  occ2_occ4 "$RUN_DIR/occ2_occ4.cubin" \
  occ4_occ5 "$RUN_DIR/occ4_occ5.cubin" \
  occ2_occ4_occ5 "$RUN_DIR/occ2_occ4_occ5.cubin"

echo "$RUN_DIR"
