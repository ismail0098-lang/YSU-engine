#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$ROOT/results/runs/p2r_symbolic_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

STD_FLAG="$("$ROOT/scripts/resolve_nvcc_std_flag.sh")"
BASE="probe_p2r_two_stage_bank_exact"

/opt/cuda/bin/nvcc -arch=sm_89 ${STD_FLAG} -O3 -Xptxas -O3 \
  -cubin "$ROOT/probes/${BASE}.cu" -o "$RUN_DIR/${BASE}_O3.cubin"
cuobjdump --dump-sass "$RUN_DIR/${BASE}_O3.cubin" > "$RUN_DIR/${BASE}_O3.sass"

/opt/cuda/bin/nvcc -arch=sm_89 ${STD_FLAG} -O3 -Xptxas -O3,--maxrregcount=32 \
  -cubin "$ROOT/probes/${BASE}.cu" -o "$RUN_DIR/${BASE}_O3_maxreg32.cubin"
cuobjdump --dump-sass "$RUN_DIR/${BASE}_O3_maxreg32.cubin" > "$RUN_DIR/${BASE}_O3_maxreg32.sass"

python "$ROOT/scripts/score_p2r_symbolic_boundary.py" \
  "$RUN_DIR/${BASE}_O3.sass" \
  "$ROOT/results/runs/cudnn_library_sm86_mining_20260320_103900/libcudnn_engines_precompiled_sm86.sass" \
  --outdir "$RUN_DIR/score_o3"

python "$ROOT/scripts/score_p2r_symbolic_boundary.py" \
  "$RUN_DIR/${BASE}_O3_maxreg32.sass" \
  "$ROOT/results/runs/cudnn_library_sm86_mining_20260320_103900/libcudnn_engines_precompiled_sm86.sass" \
  --outdir "$RUN_DIR/score_o3_maxreg32"

cat > "$RUN_DIR/summary.txt" <<EOF
p2r_symbolic_matrix
===================

source:
- src/sass_re/probes/${BASE}.cu

lanes:
- O3
- O3 + maxrregcount=32

scoring:
- src/sass_re/scripts/score_p2r_symbolic_boundary.py
- reference: src/sass_re/results/runs/cudnn_library_sm86_mining_20260320_103900/libcudnn_engines_precompiled_sm86.sass

artifacts:
- $RUN_DIR/${BASE}_O3.sass
- $RUN_DIR/${BASE}_O3_maxreg32.sass
- $RUN_DIR/score_o3/summary.txt
- $RUN_DIR/score_o3_maxreg32/summary.txt
EOF

echo "$RUN_DIR"
