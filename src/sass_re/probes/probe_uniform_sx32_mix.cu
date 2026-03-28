/*
 * SASS RE Probe: mixed thread-index + signed-uniform offset addressing
 *
 * Existing local ULEA/ULEA.HI hits come from mixed address patterns rather than
 * purely uniform loads. This probe injects a signed warp-uniform offset into a
 * scaled per-thread address calculation to try to surface ULEA.HI.X.SX32.
 */

#include <cuda_runtime.h>

extern "C" __global__ void __launch_bounds__(128)
probe_sx32_offset_copy(float *out, const float *in, int n, int signed_offset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = i + signed_offset;

    if (j >= 0 && j < n) {
        out[j] = in[j];
    }
}

extern "C" __global__ void __launch_bounds__(128)
probe_sx32_offset_stride2(float *out, const float *in, int n, int signed_offset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = i * 2 + signed_offset;

    if (j >= 0 && j < n) {
        out[j] = in[j];
    }
}

extern "C" __global__ void __launch_bounds__(128)
probe_sx32_offset_stride4(float *out, const float *in, int n, int signed_offset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = i * 4 + signed_offset;

    if (j >= 0 && j < n) {
        out[j] = in[j];
    }
}
