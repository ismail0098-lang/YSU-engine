#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )) || (( ($# % 2) == 0 )); then
  echo "usage: $0 <kernel_name> <baseline_label> <baseline_cubin> [<label> <cubin> ...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${ROOT}/results/runs/p2r_cubin_pattern_matrix_${STAMP}"
mkdir -p "${OUTDIR}"

RUNNER="${OUTDIR}/p2r_cubin_driver_runner"
SRC="${ROOT}/runners/p2r_cubin_driver_runner.cpp"
STD_FLAG="$("${ROOT}/scripts/resolve_nvcc_std_flag.sh")"

/opt/cuda/bin/nvcc ${STD_FLAG} -O2 -arch=sm_89 "${SRC}" -lcuda -o "${RUNNER}"

KERNEL_NAME="$1"
shift
SPECS=("$@")

{
  echo "p2r_cubin_pattern_matrix"
  echo "========================"
  echo
  echo "kernel=${KERNEL_NAME}"
  echo
} > "${OUTDIR}/summary.txt"

baseline_ref=""
for pattern in 0 1 2 3; do
  baseline_ref=""
  echo "pattern=${pattern}" >> "${OUTDIR}/summary.txt"
  set -- "${SPECS[@]}"
  while (( $# > 0 )); do
    label="$1"
    cubin="$2"
    shift 2
    log="${OUTDIR}/${label}_pattern${pattern}.log"
    rc=0
    if timeout 10s "${RUNNER}" "${cubin}" "${KERNEL_NAME}" "${pattern}" > "${log}" 2>&1; then
      :
    else
      rc=$?
    fi
    sum="$(sed -n 's/^sum=//p' "${log}" | head -n1)"
    first8="$(sed -n 's/^out\\[[0-7]\\]=//p' "${log}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    if [[ "${label}" == "baseline" && "${rc}" -eq 0 ]]; then
      baseline_ref="${sum}|${first8}"
    fi
    relation="n/a"
    if [[ "${label}" != "baseline" && -n "${baseline_ref}" && "${rc}" -eq 0 ]]; then
      current="${sum}|${first8}"
      if [[ "${current}" == "${baseline_ref}" ]]; then
        relation="same_as_baseline"
      else
        relation="diff_from_baseline"
      fi
    fi
    echo "- ${label}:" >> "${OUTDIR}/summary.txt"
    echo "  rc=${rc}" >> "${OUTDIR}/summary.txt"
    [[ -n "${sum}" ]] && echo "  sum=${sum}" >> "${OUTDIR}/summary.txt"
    [[ -n "${first8}" ]] && echo "  first8=${first8}" >> "${OUTDIR}/summary.txt"
    echo "  relation=${relation}" >> "${OUTDIR}/summary.txt"
    sed 's/^/  /' "${log}" >> "${OUTDIR}/summary.txt"
    echo >> "${OUTDIR}/summary.txt"
  done
done

echo "${OUTDIR}"
