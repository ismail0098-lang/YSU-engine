# CUDA LBM kernel library, benchmark, and test targets.
# Requires: CMake 3.18+ (CUDA language support), CUDA Toolkit 11.8+.
# Target architectures: SM 89 (Ada Lovelace), SM 80 (Ampere fallback).

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target architectures
set(CMAKE_CUDA_ARCHITECTURES "89;80")

# Compiler flags
#   --use_fast_math: ftz, prec-div=false, prec-sqrt=false
#   -lineinfo: source correlation for Nsight Compute
#   -Xptxas -v: register/smem reporting (used by check_spills.sh)
#   -dlto: device link-time optimization
set(CUDA_COMMON_FLAGS
    --use_fast_math
    -lineinfo
    -Xptxas=-v
    -dlto
)

# Release-specific flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# ============================================================================
# Kernel source files
# ============================================================================

set(LBM_KERNEL_SOURCES
    kernels_soa.cu
    kernels_fp16.cu
    kernels_fp16_soa.cu
    kernels_fp16_soa_half2.cu
    kernels_fp8.cu
    kernels_fp8_soa.cu
    kernels_fp8_e5m2.cu
    kernels_fp8_e5m2_soa.cu
    kernels_int8.cu
    kernels_int8_soa.cu
    kernels_int16.cu
    kernels_int16_soa.cu
    kernels_bf16.cu
    kernels_bf16_soa.cu
    kernels_fp32_soa_cs.cu
    kernels_fp64.cu
    kernels_fp64_soa.cu
    kernels_dd.cu
    kernels_int4.cu
    kernels_fp4.cu
    kernels_tensor_core.cu
    kernels_box_counting.cu
    sparse/kernels_sparse_lbm.cu
    sparse/kernels_sparse_map.cu
)

# ============================================================================
# Static library: ysu_cuda_lbm
# ============================================================================

add_library(ysu_cuda_lbm STATIC
    ${LBM_KERNEL_SOURCES}
    host_wrappers.cu
)

target_include_directories(ysu_cuda_lbm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(ysu_cuda_lbm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>
)

set_target_properties(ysu_cuda_lbm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================================
# Benchmark executable: lbm_bench
# ============================================================================

add_executable(lbm_bench
    bench/bench_main.cu
    bench/bench_harness.cu
    bench/bench_validate.cu
)

target_link_libraries(lbm_bench PRIVATE ysu_cuda_lbm)

target_include_directories(lbm_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
)

target_compile_options(lbm_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>
)

# Link NVTX for range annotations
find_library(NVTX_LIB nvToolsExt HINTS ${CUDAToolkit_LIBRARY_DIR})
if(NVTX_LIB)
    target_link_libraries(lbm_bench PRIVATE ${NVTX_LIB})
    target_compile_definitions(lbm_bench PRIVATE LBM_HAVE_NVTX=1)
endif()

# ============================================================================
# Test executable: lbm_test
# ============================================================================

add_executable(lbm_test
    test/test_main.cu
    test/test_conservation.cu
    test/test_regression.cu
    test/test_mrt_stability.cu
)

target_link_libraries(lbm_test PRIVATE ysu_cuda_lbm)

target_include_directories(lbm_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
)

target_compile_options(lbm_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>
)
