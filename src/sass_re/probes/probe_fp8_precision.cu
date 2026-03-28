/*
 * SASS RE Probe: FP8 Precision Conversions (E4M3 and E5M2)
 * Isolates: FP8 encode/decode intrinsics on Ada Lovelace SM 8.9
 *
 * FP8 is a storage format, not an arithmetic format. The GPU converts
 * FP8 <-> FP16/FP32 via hardware conversion paths:
 *   E4M3: 4-bit exponent, 3-bit mantissa, range +-448, ~12.5% rel error
 *   E5M2: 5-bit exponent, 2-bit mantissa, range +-57344, ~25% rel error
 *
 * Key SASS to look for:
 *   F2FP.E4M3  -- float32/float16 to FP8 E4M3 conversion
 *   F2FP.E5M2  -- float32/float16 to FP8 E5M2 conversion
 *   Possibly inlined as shift/mask sequences if hardware path not available
 *
 * These conversions are the hot path in kernels_fp8.cu and kernels_fp8_soa.cu
 * where 19 distributions per cell are decoded from FP8 -> FP32 for collision
 * and re-encoded FP32 -> FP8 for storage.
 */

#include <cuda_fp8.h>
#include <cuda_fp16.h>

// E4M3 conversions: FP32 -> FP8 -> FP32 round-trip
extern "C" __global__ void __launch_bounds__(32)
probe_fp8_e4m3_convert(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];

    // FP32 -> FP8 E4M3 (quantize)
    __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);

    // FP8 E4M3 -> FP16 (hardware decode path)
    __half_raw hraw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    __half h;
    memcpy(&h, &hraw, sizeof(h));

    // FP16 -> FP32
    float result = __half2float(h);

    out[i] = result;
}

// E5M2 conversions: FP32 -> FP8 -> FP32 round-trip
extern "C" __global__ void __launch_bounds__(32)
probe_fp8_e5m2_convert(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];

    // FP32 -> FP8 E5M2 (quantize)
    __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E5M2);

    // FP8 E5M2 -> FP16
    __half_raw hraw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E5M2);
    __half h;
    memcpy(&h, &hraw, sizeof(h));

    float result = __half2float(h);
    out[i] = result;
}

// Batch conversion chain: 19 FP8 values decoded (simulates D3Q19 distribution load)
extern "C" __global__ void __launch_bounds__(32)
probe_fp8_batch_decode(float *out, const __nv_fp8_storage_t *in) {
    int i = threadIdx.x;
    float sum = 0.0f;

    // Decode 19 FP8 E4M3 values in sequence (D3Q19 pattern)
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        __nv_fp8_storage_t fp8 = in[d * 32 + i];
        __half_raw hraw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
        __half h;
        memcpy(&h, &hraw, sizeof(h));
        sum += __half2float(h);
    }
    out[i] = sum;
}

// Batch encode chain: 19 FP32 values encoded to FP8 (simulates distribution store)
extern "C" __global__ void __launch_bounds__(32)
probe_fp8_batch_encode(__nv_fp8_storage_t *out, const float *in) {
    int i = threadIdx.x;

    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float val = in[d * 32 + i];
        __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
        out[d * 32 + i] = fp8;
    }
}

// Vectorized FP8 load: 4 bytes = 4 FP8 values via uchar4
extern "C" __global__ void __launch_bounds__(32)
probe_fp8_vectorized_load(float *out, const unsigned char *in) {
    int i = threadIdx.x;

    // Load 4 FP8 values as a single 32-bit read (uchar4 pattern from kernels_fp8.cu)
    const uchar4 *vec = reinterpret_cast<const uchar4*>(in);
    uchar4 packed = vec[i];

    // Decode each byte as FP8 E4M3
    __half_raw h0 = __nv_cvt_fp8_to_halfraw(packed.x, __NV_E4M3);
    __half_raw h1 = __nv_cvt_fp8_to_halfraw(packed.y, __NV_E4M3);
    __half_raw h2 = __nv_cvt_fp8_to_halfraw(packed.z, __NV_E4M3);
    __half_raw h3 = __nv_cvt_fp8_to_halfraw(packed.w, __NV_E4M3);

    __half hh0, hh1, hh2, hh3;
    memcpy(&hh0, &h0, sizeof(hh0));
    memcpy(&hh1, &h1, sizeof(hh1));
    memcpy(&hh2, &h2, sizeof(hh2));
    memcpy(&hh3, &h3, sizeof(hh3));

    out[i] = __half2float(hh0) + __half2float(hh1)
           + __half2float(hh2) + __half2float(hh3);
}
