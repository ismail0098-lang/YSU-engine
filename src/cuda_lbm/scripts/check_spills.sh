#!/bin/sh
# Parse ptxas verbose output for register spills and LMEM usage.
# Exits 1 if any production kernel has LMEM > 0 bytes.
#
# Usage: check_spills.sh [build_log]
#
# If no build_log is provided, builds the project and captures ptxas output.
# ptxas writes to stderr, so the build must be captured with 2>&1.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."

if [ $# -ge 1 ] && [ -f "$1" ]; then
    LOG="$1"
else
    echo "Building with ptxas verbose output ..."
    LOG="$(mktemp)"
    trap 'rm -f "$LOG"' EXIT
    # ptxas writes register info to stderr
    cmake --build "${PROJECT_ROOT}/build" --target ysu_cuda_lbm 2>&1 | tee "$LOG"
fi

echo ""
echo "=== Register and LMEM Report ==="
echo ""
printf "%-50s  %5s  %8s  %8s  %8s\n" "Kernel" "Regs" "SMEM(B)" "LMEM(B)" "Spills"
printf "%-50s  %5s  %8s  %8s  %8s\n" "------" "----" "-------" "-------" "------"

# ptxas output format:
# ptxas info    : Compiling entry function 'kernel_name' for 'sm_89'
# ptxas info    : Function properties for kernel_name
#     X bytes stack frame, Y bytes spill stores, Z bytes spill loads
# ptxas info    : Used N registers, M bytes smem, K bytes lmem

HAS_LMEM_PRODUCTION=0

# Extract register/smem/lmem lines
# Format: "Used N registers, M bytes smem" or "Used N registers, M bytes smem, K bytes lmem"
# Preceded by "Compiling entry function 'name'"

# We parse pairs of lines: function name + register info
awk '
/Compiling entry function/ {
    match($0, /\047([^\047]+)\047/, arr)
    current_fn = arr[1]
}
/Used [0-9]+ registers/ {
    regs = 0; smem = 0; lmem = 0; spills = ""
    match($0, /Used ([0-9]+) registers/, arr)
    regs = arr[1]
    if (match($0, /([0-9]+) bytes smem/, arr)) smem = arr[1]
    if (match($0, /([0-9]+) bytes lmem/, arr)) lmem = arr[1]
    printf "%-50s  %5d  %8d  %8d\n", current_fn, regs, smem, lmem
}
/spill (stores|loads)/ {
    match($0, /([0-9]+) bytes spill stores/, ss)
    match($0, /([0-9]+) bytes spill loads/, sl)
    printf "  -> spill stores: %s, spill loads: %s\n", ss[1], sl[1]
}
' "$LOG"

echo ""

# Check for LMEM in production kernels
# Production kernels: all lbm_step_* and initialize_* that are physics_valid
# (We check any kernel with "lbm_step" in the name, excluding int4 and fp4)

LMEM_LINES=$(grep -E 'lmem' "$LOG" || true)
if [ -n "$LMEM_LINES" ]; then
    # Check if any non-INT4/FP4 kernel has lmem
    PROD_LMEM=$(echo "$LMEM_LINES" | grep -v 'int4\|fp4' || true)
    if [ -n "$PROD_LMEM" ]; then
        echo "FATAL: Production kernels have LMEM usage (register spills to local memory)."
        echo "LMEM on Ada routes through L1 -> DRAM with ~92 cycle latency."
        echo "For MRT kernels (~128 regs/thread), LMEM spills severely degrade performance."
        echo ""
        echo "$PROD_LMEM"
        HAS_LMEM_PRODUCTION=1
    fi

    # Warn about non-production kernels
    NONPROD_LMEM=$(echo "$LMEM_LINES" | grep -E 'int4|fp4' || true)
    if [ -n "$NONPROD_LMEM" ]; then
        echo "WARNING: Non-production kernels have LMEM usage (not fatal):"
        echo "$NONPROD_LMEM"
    fi
fi

if [ "$HAS_LMEM_PRODUCTION" -eq 0 ]; then
    echo "OK: No LMEM usage in production kernels."
    exit 0
else
    exit 1
fi
