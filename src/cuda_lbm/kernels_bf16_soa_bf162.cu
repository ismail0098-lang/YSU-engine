// BF16 i-major SoA D3Q19 LBM -- bfloat162 ILP variant (2 cells per thread).
//
// WHY: SASS RE measurements show BF16 packed FMA (HFMA2.BF16_V2) is the
// FASTEST FMA instruction on Ada Lovelace SM 8.9:
//   HFMA2.BF16:  4.01 cy latency (12% faster than FP16 HFMA2 at 4.54 cy)
//   Throughput:   312.1 ops/clk/SM (22% higher than FP16 at 260.1)
//   Conversion:   8.54 cy round-trip (19% faster than FP16 at 10.54 cy)
//
// This kernel ports the kernels_fp16_soa_half2.cu pattern to BF16:
//   Thread k handles cells idx0=2k and idx1=2k+1.
//   __nv_bfloat162 for velocity moment accumulation (2x throughput).
//   FP32 for collision operator (precision required for stability).
//
// Expected: +20-25% MLUPS over kernels_bf16_soa.cu based on measured
// throughput advantage of HFMA2.BF16 over scalar BF16 operations.
//
// BF16 precision: 7-bit mantissa (~0.78% relative error).
//   Risk: tau < 0.55 instability at shear boundaries.
//   Same dynamic range as FP32 (8-bit exponent): no overflow risk.
//
// Minimum architecture: SM 8.0+ (Ampere). Ada SM 8.9 has native BF16.
// VRAM at 128^3: 19 * 2,097,152 * 2 * 2 (ping+pong) = ~152 MB.

#include <cuda_bf16.h>

__constant__ int CX_B162[19] = {
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};
__constant__ int CY_B162[19] = {
    0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1
};
__constant__ int CZ_B162[19] = {
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1
};

__constant__ float W_B162[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__device__ __forceinline__ bool finite_b162(float x) {
    return (x == x) && (x <= 3.402823466e38f) && (x >= -3.402823466e38f);
}

// BF16 SoA bfloat162 ILP variant.
// Thread k: cell 0 = 2k, cell 1 = 2k+1.
// Uses __nv_bfloat162 for velocity moment accumulation; FP32 for collision.
extern "C" __launch_bounds__(128, 4) __global__ void lbm_step_bf16_soa_bf162_kernel(
    const __nv_bfloat16* __restrict__ f_in,
    __nv_bfloat16* __restrict__ f_out,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    const float* __restrict__ tau,
    const float* __restrict__ force,
    int nx, int ny, int nz
) {
    int n_cells = nx * ny * nz;
    int half_cells = (n_cells + 1) / 2;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= half_cells) return;

    int idx0 = 2 * k;
    int idx1 = 2 * k + 1;
    bool valid1 = (idx1 < n_cells);

    int x0 = idx0 % nx, y0 = (idx0 / nx) % ny, z0 = idx0 / (nx * ny);
    int x1 = 0, y1 = 0, z1 = 0;
    if (valid1) { x1 = idx1 % nx; y1 = (idx1 / nx) % ny; z1 = idx1 / (nx * ny); }

    // FP32 buffers for BGK collision
    float fa[19], fb[19];
    // __nv_bfloat162 accumulators for velocity moments (HFMA2.BF16_V2 path)
    __nv_bfloat162 rho2 = __float2bfloat162_rn(0.0f);
    __nv_bfloat162 mx2  = __float2bfloat162_rn(0.0f);
    __nv_bfloat162 my2  = __float2bfloat162_rn(0.0f);
    __nv_bfloat162 mz2  = __float2bfloat162_rn(0.0f);

    #pragma unroll
    for (int d = 0; d < 19; d++) {
        // Pull-scheme: read from backward neighbor
        int bx0 = (x0 - CX_B162[d] + nx) % nx;
        int by0 = (y0 - CY_B162[d] + ny) % ny;
        int bz0 = (z0 - CZ_B162[d] + nz) % nz;
        int back0 = bx0 + by0 * nx + bz0 * nx * ny;

        float f0 = __bfloat162float(f_in[d * n_cells + back0]);
        fa[d] = f0;

        float f1 = 0.0f;
        if (valid1) {
            int bx1 = (x1 - CX_B162[d] + nx) % nx;
            int by1 = (y1 - CY_B162[d] + ny) % ny;
            int bz1 = (z1 - CZ_B162[d] + nz) % nz;
            int back1 = bx1 + by1 * nx + bz1 * nx * ny;
            f1 = __bfloat162float(f_in[d * n_cells + back1]);
        }
        fb[d] = f1;

        // Pack into bfloat162 for vectorized moment accumulation
        __nv_bfloat162 f2 = __floats2bfloat162_rn(f0, f1);
        rho2 = __hadd2(rho2, f2);

        // Velocity moments: e_i * f_i (packed)
        __nv_bfloat162 cx2 = __float2bfloat162_rn((float)CX_B162[d]);
        __nv_bfloat162 cy2 = __float2bfloat162_rn((float)CY_B162[d]);
        __nv_bfloat162 cz2 = __float2bfloat162_rn((float)CZ_B162[d]);
        mx2 = __hfma2(cx2, f2, mx2);  // HFMA2.BF16_V2 -- fastest FMA on Ada!
        my2 = __hfma2(cy2, f2, my2);
        mz2 = __hfma2(cz2, f2, mz2);
    }

    // Unpack bfloat162 moments to FP32 for collision (precision required)
    float rho0 = __low2float(rho2), rho1 = __high2float(rho2);
    float ux0 = __low2float(mx2), ux1 = __high2float(mx2);
    float uy0 = __low2float(my2), uy1 = __high2float(my2);
    float uz0 = __low2float(mz2), uz1 = __high2float(mz2);

    if (!finite_b162(rho0) || rho0 <= 1.0e-20f) rho0 = 1.0f;
    float inv_rho0 = 1.0f / rho0;
    ux0 *= inv_rho0; uy0 *= inv_rho0; uz0 *= inv_rho0;

    float inv_rho1 = 1.0f;
    if (valid1) {
        if (!finite_b162(rho1) || rho1 <= 1.0e-20f) rho1 = 1.0f;
        inv_rho1 = 1.0f / rho1;
        ux1 *= inv_rho1; uy1 *= inv_rho1; uz1 *= inv_rho1;
    }

    // BGK collision (FP32 precision)
    float tau0 = tau[idx0];
    float tau1 = valid1 ? tau[idx1] : 0.6f;
    float inv_tau0 = 1.0f / tau0;
    float inv_tau1 = 1.0f / tau1;
    float usq0 = ux0*ux0 + uy0*uy0 + uz0*uz0;
    float usq1 = ux1*ux1 + uy1*uy1 + uz1*uz1;

    // Guo forcing
    float fx0 = 0.0f, fy0 = 0.0f, fz0 = 0.0f;
    float fx1 = 0.0f, fy1 = 0.0f, fz1 = 0.0f;
    if (force) {
        fx0 = force[idx0]; fy0 = force[n_cells + idx0]; fz0 = force[2*n_cells + idx0];
        if (valid1) {
            fx1 = force[idx1]; fy1 = force[n_cells + idx1]; fz1 = force[2*n_cells + idx1];
        }
    }
    float pref0 = 1.0f - 0.5f * inv_tau0;
    float pref1 = 1.0f - 0.5f * inv_tau1;

    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float w = W_B162[d];
        float cx = (float)CX_B162[d], cy = (float)CY_B162[d], cz = (float)CZ_B162[d];

        // Equilibrium (Horner form)
        float eu0 = cx*ux0 + cy*uy0 + cz*uz0;
        float feq0 = w * rho0 * fmaf(fmaf(eu0, 4.5f, 3.0f), eu0, 1.0f - 1.5f*usq0);
        fa[d] -= (fa[d] - feq0) * inv_tau0;

        // Guo forcing
        if (fx0*fx0 + fy0*fy0 + fz0*fz0 > 1e-40f) {
            float eu_f0 = (cx - ux0)*fx0 + (cy - uy0)*fy0 + (cz - uz0)*fz0;
            float eu_uf0 = eu0 * (cx*fx0 + cy*fy0 + cz*fz0);
            fa[d] += pref0 * w * (3.0f * eu_f0 + 9.0f * eu_uf0);
        }

        if (valid1) {
            float eu1 = cx*ux1 + cy*uy1 + cz*uz1;
            float feq1 = w * rho1 * fmaf(fmaf(eu1, 4.5f, 3.0f), eu1, 1.0f - 1.5f*usq1);
            fb[d] -= (fb[d] - feq1) * inv_tau1;

            if (fx1*fx1 + fy1*fy1 + fz1*fz1 > 1e-40f) {
                float eu_f1 = (cx - ux1)*fx1 + (cy - uy1)*fy1 + (cz - uz1)*fz1;
                float eu_uf1 = eu1 * (cx*fx1 + cy*fy1 + cz*fz1);
                fb[d] += pref1 * w * (3.0f * eu_f1 + 9.0f * eu_uf1);
            }
        }

        // Store as BF16
        f_out[d * n_cells + idx0] = __float2bfloat16(fa[d]);
        if (valid1) f_out[d * n_cells + idx1] = __float2bfloat16(fb[d]);
    }

    // Store macroscopic
    rho_out[idx0] = rho0;
    u_out[idx0] = ux0; u_out[n_cells + idx0] = uy0; u_out[2*n_cells + idx0] = uz0;
    if (valid1) {
        rho_out[idx1] = rho1;
        u_out[idx1] = ux1; u_out[n_cells + idx1] = uy1; u_out[2*n_cells + idx1] = uz1;
    }
}

// Init kernel (1 thread per cell, not 2-cell)
extern "C" __launch_bounds__(128) __global__ void initialize_uniform_bf16_soa_bf162_kernel(
    __nv_bfloat16* __restrict__ f_a,
    __nv_bfloat16* __restrict__ f_b,
    float* __restrict__ rho_out,
    float* __restrict__ u_out,
    float* __restrict__ tau_arr,
    float* __restrict__ force_arr,
    float rho, float ux, float uy, float uz,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;
    if (idx >= n_cells) return;

    float feq[19];
    float usq = ux*ux + uy*uy + uz*uz;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float cx = (float)CX_B162[d], cy = (float)CY_B162[d], cz = (float)CZ_B162[d];
        float eu = cx*ux + cy*uy + cz*uz;
        feq[d] = W_B162[d] * rho * fmaf(fmaf(eu, 4.5f, 3.0f), eu, 1.0f - 1.5f*usq);
        __nv_bfloat16 bf = __float2bfloat16(feq[d]);
        f_a[d * n_cells + idx] = bf;
        f_b[d * n_cells + idx] = bf;
    }

    rho_out[idx] = rho;
    u_out[idx] = ux; u_out[n_cells + idx] = uy; u_out[2*n_cells + idx] = uz;
    tau_arr[idx] = 0.6f;
    if (force_arr) {
        force_arr[idx] = 0.0f;
        force_arr[n_cells + idx] = 0.0f;
        force_arr[2*n_cells + idx] = 0.0f;
    }
}
