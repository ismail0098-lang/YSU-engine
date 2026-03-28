#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${ROOT}/results/runs/p2r_nvcc_debug_matrix_${STAMP}"
mkdir -p "${OUTDIR}"

CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
CCBIN_13_1="${CCBIN_13_1:-}"
CCBIN_12_6="${CCBIN_12_6:-}"
CCBIN_11_8="${CCBIN_11_8:-}"

declare -a LABELS=()
declare -a NVCCS=()

LABELS+=("cuda13_1")
NVCCS+=("${NVCC_13_1:-/opt/cuda/bin/nvcc}")

if [[ -x "${NVCC_11_8:-/opt/nvidia/hpc_sdk_sidecar/Linux_x86_64/24.11/cuda/11.8/bin/nvcc}" ]]; then
  LABELS+=("sidecar11_8")
  NVCCS+=("${NVCC_11_8:-/opt/nvidia/hpc_sdk_sidecar/Linux_x86_64/24.11/cuda/11.8/bin/nvcc}")
fi

if [[ -x "${NVCC_12_6:-/opt/nvidia/hpc_sdk_sidecar/Linux_x86_64/24.11/cuda/12.6/bin/nvcc}" ]]; then
  LABELS+=("sidecar12_6")
  NVCCS+=("${NVCC_12_6:-/opt/nvidia/hpc_sdk_sidecar/Linux_x86_64/24.11/cuda/12.6/bin/nvcc}")
fi

PROBES=(
  "probe_p2r_banked_reload.cu"
  "probe_p2r_bytepack.cu"
  "probe_p2r_bytepack_strict_inline_ptx.cu"
  "probe_p2r_byteview_carrier_exact.cu"
  "probe_p2r_mov_pack_inline_ptx.cu"
  "probe_p2r_plop3_source_exact.cu"
  "probe_p2r_two_stage_bank_exact.cu"
  "probe_p2r_vector_pack_inline_ptx.cu"
)

{
  echo "p2r_nvcc_debug_matrix"
  echo "====================="
  echo
  echo "- outdir: ${OUTDIR}"
  echo
} > "${OUTDIR}/summary.txt"

for idx in "${!LABELS[@]}"; do
  label="${LABELS[$idx]}"
  nvcc="${NVCCS[$idx]}"
  run_dir="${OUTDIR}/${label}"
  mkdir -p "${run_dir}"
  version_text="$("$nvcc" --version 2>/dev/null || true)"
  std_flag="-std=c++20"
  extra_flags=()
  ccbin=""
  case "${version_text}" in
    *"release 11.8"*)
      std_flag="-std=c++17"
      extra_flags+=("-allow-unsupported-compiler")
      ccbin="${CCBIN_11_8:-$(command -v g++-13 || command -v g++-14 || true)}"
      ;;
    *"release 12.6"*)
      std_flag="-std=c++20"
      extra_flags+=("-allow-unsupported-compiler")
      ccbin="${CCBIN_12_6:-$(command -v g++-13 || command -v g++-14 || true)}"
      ;;
    *"release 13.1"*)
      std_flag="-std=c++20"
      ccbin="${CCBIN_13_1:-}"
      ;;
  esac
  echo "==> ${label}: ${nvcc}"
  {
    echo "- ${label}: ${nvcc} (${std_flag}${extra_flags:+ ${extra_flags[*]}}${ccbin:+ -ccbin ${ccbin}})"
  } >> "${OUTDIR}/summary.txt"
  for probe in "${PROBES[@]}"; do
    src="${ROOT}/probes/${probe}"
    stem="${probe%.cu}"
    cubin="${run_dir}/${stem}.cubin"
    sass="${run_dir}/${stem}.sass"
    log="${run_dir}/${stem}.log"
    status="ok"
    cmd=("$nvcc" -arch=sm_89 -O3 -G "${std_flag}")
    if [[ -n "${ccbin}" ]]; then
      cmd+=(-ccbin "${ccbin}")
    fi
    cmd+=("${extra_flags[@]}" -cubin "${src}" -o "${cubin}")
    "${cmd[@]}" >"${log}" 2>&1 || status="FAIL"
    if [[ "${status}" == "ok" ]]; then
      "${CUOBJDUMP}" -sass "${cubin}" > "${sass}"
      byte_count="$(rg -c 'P2R\\.B[123]' "${sass}" || true)"
      plain_count="$(rg -c 'P2R ' "${sass}" || true)"
      r2p_count="$(rg -c 'R2P' "${sass}" || true)"
      bar_count="$(rg -c 'BAR\\.SYNC|BSSY|BSYNC|WARPSYNC' "${sass}" || true)"
      uldc_count="$(rg -c 'ULDC' "${sass}" || true)"
      echo "  - ${stem}: byte_p2r=${byte_count} plain_p2r=${plain_count} r2p=${r2p_count} bar=${bar_count} uldc=${uldc_count}" >> "${OUTDIR}/summary.txt"
    else
      echo "  - ${stem}: FAIL" >> "${OUTDIR}/summary.txt"
    fi
  done
  echo >> "${OUTDIR}/summary.txt"
done

echo "${OUTDIR}"
