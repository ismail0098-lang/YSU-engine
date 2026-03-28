#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${ROOT}/results/runs/p2r_cubin_runtime_trial_${STAMP}"
mkdir -p "${OUTDIR}"

RUNNER="${OUTDIR}/p2r_cubin_driver_runner"
SRC="${ROOT}/runners/p2r_cubin_driver_runner.cpp"

/opt/cuda/bin/nvcc -std=c++20 -O2 -arch=sm_89 "${SRC}" -lcuda -o "${RUNNER}"

BASE_CUBIN="${ROOT}/results/runs/p2r_symbolic_matrix_20260322_194108/probe_p2r_two_stage_bank_exact_O3.cubin"
PATCH_RUN="${ROOT}/results/runs/p2r_cubin_patch_trial_20260322_233700"

{
  echo "p2r_cubin_runtime_trial"
  echo "======================="
  echo
} > "${OUTDIR}/summary.txt"

for spec in \
  "baseline ${BASE_CUBIN}" \
  "b2 ${PATCH_RUN}/probe_p2r_two_stage_bank_exact_O3_b2.cubin" \
  "b3 ${PATCH_RUN}/probe_p2r_two_stage_bank_exact_O3_b3.cubin"
do
  set -- ${spec}
  label="$1"
  cubin="$2"
  log="${OUTDIR}/${label}.log"
  if "${RUNNER}" "${cubin}" probe_p2r_two_stage_bank_exact > "${log}" 2>&1; then
    echo "- ${label}: ok" >> "${OUTDIR}/summary.txt"
    sed 's/^/  /' "${log}" >> "${OUTDIR}/summary.txt"
  else
    echo "- ${label}: FAIL" >> "${OUTDIR}/summary.txt"
    sed 's/^/  /' "${log}" >> "${OUTDIR}/summary.txt"
  fi
  echo >> "${OUTDIR}/summary.txt"
done

echo "${OUTDIR}"
