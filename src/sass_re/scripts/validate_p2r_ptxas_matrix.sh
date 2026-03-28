#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${ROOT}/results/runs/p2r_ptxas_matrix_${STAMP}"
mkdir -p "${OUTDIR}"

SEARCH="${ROOT}/scripts/search_p2r_ptx.py"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"

declare -a LABELS=()
declare -a PTXASES=()

LABELS+=("cuda13_1")
PTXASES+=("${PTXAS_13_1:-/opt/cuda/bin/ptxas}")

PTXAS_11_8_DEFAULT="/opt/nvidia/hpc_sdk_sidecar/Linux_x86_64/24.11/cuda/11.8/bin/ptxas"
PTXAS_12_6_DEFAULT="/opt/nvidia/hpc_sdk_sidecar/Linux_x86_64/24.11/cuda/12.6/bin/ptxas"
PTXAS_11_8_PATH="${PTXAS_11_8:-${PTXAS_11_8_DEFAULT}}"
PTXAS_12_6_PATH="${PTXAS_12_6:-${PTXAS_12_6_DEFAULT}}"

if [[ -x "${PTXAS_11_8_PATH}" ]]; then
  LABELS+=("sidecar11_8")
  PTXASES+=("${PTXAS_11_8_PATH}")
fi

if [[ -x "${PTXAS_12_6_PATH}" ]]; then
  LABELS+=("sidecar12_6")
  PTXASES+=("${PTXAS_12_6_PATH}")
fi

{
  echo "p2r_ptxas_matrix"
  echo "================"
  echo
  echo "- outdir: ${OUTDIR}"
  echo
} > "${OUTDIR}/summary.txt"

for idx in "${!LABELS[@]}"; do
  label="${LABELS[$idx]}"
  ptxas="${PTXASES[$idx]}"
  run_dir="${OUTDIR}/${label}"
  mkdir -p "${run_dir}"
  version_text="$("$ptxas" --version 2>/dev/null || true)"
  ptx_version="8.8"
  case "${version_text}" in
    *"release 11.8"*) ptx_version="7.8" ;;
    *"release 12.6"*) ptx_version="8.5" ;;
    *"release 13.1"*) ptx_version="8.8" ;;
  esac
  echo "==> ${label}: ${ptxas}"
  python "${SEARCH}" \
    --ptxas "${ptxas}" \
    --cuobjdump "${CUOBJDUMP}" \
    --ptx-version "${ptx_version}" \
    --outdir "${run_dir}"
  {
    echo "- ${label}: ${ptxas} (ptx ${ptx_version})"
    sed 's/^/  /' "${run_dir}/summary.txt"
    echo
  } >> "${OUTDIR}/summary.txt"
done

echo "${OUTDIR}"
