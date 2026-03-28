/*
 * SASS RE: Fill All Remaining N/A Latencies
 *
 * Measures every value still listed as N/A in the latency table:
 *   FP8 E5M2 round-trip, DD mul, Posit decode, NF4 encode/decode,
 *   INT128 add chain (asm), FP4 E2M1 round-trip
 *
 * Build: nvcc -arch=sm_89 -O1 -I../probes -o fill_na microbench_fill_all_na.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define N 512

// FP8 E5M2 round-trip chain
__global__ void __launch_bounds__(32)
k_fp8_e5m2_roundtrip(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2);
        __half_raw hraw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E5M2);
        __half h; __builtin_memcpy(&h, &hraw, sizeof(h));
        x = __half2float(h);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// FP4 E2M1 round-trip chain (LUT decode + min-distance encode)
__constant__ float FP4_LUT_NA[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
__device__ __forceinline__ unsigned char fp4_encode_na(float val) {
    float aval = fabsf(val);
    unsigned char n;
    if (aval < 0.25f) n=0; else if (aval < 0.75f) n=1;
    else if (aval < 1.25f) n=2; else if (aval < 1.75f) n=3;
    else if (aval < 2.5f) n=4; else if (aval < 3.5f) n=5;
    else if (aval < 5.0f) n=6; else n=7;
    if (val < 0.0f) n |= 0x8;
    return n;
}
__global__ void __launch_bounds__(32)
k_fp4_e2m1_roundtrip(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        unsigned char fp4 = fp4_encode_na(x);
        x = FP4_LUT_NA[fp4];
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// NF4 round-trip chain
__constant__ float NF4_LUT_NA[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.09105f, 0.0f,
     0.07959f, 0.1609f, 0.2461f, 0.3379f, 0.4407f, 0.5626f, 0.7230f, 1.0f
};
__constant__ float NF4_BOUNDS_NA[15] = {
    -0.8481f,-0.6106f,-0.4600f,-0.3397f,-0.2346f,-0.1380f,-0.04553f,
     0.03979f,0.1203f,0.2035f,0.2920f,0.3893f,0.5017f,0.6428f,0.8615f
};
__device__ __forceinline__ unsigned char nf4_enc_na(float val) {
    val = fmaxf(-1.0f, fminf(1.0f, val));
    unsigned char idx = 0;
    for (int b = 0; b < 15; b++) if (val > NF4_BOUNDS_NA[b]) idx = b + 1;
    return idx;
}
__global__ void __launch_bounds__(32)
k_nf4_roundtrip(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        unsigned char nf4 = nf4_enc_na(x);
        x = NF4_LUT_NA[nf4];
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// DD multiply chain
__device__ __forceinline__
void two_prod_na(double a, double b, double &p, double &e) {
    p = a * b;
    e = fma(a, b, -p);
}
__device__ __forceinline__
void dd_mul_na(double ah, double al, double bh, double bl,
               double &ph, double &pl) {
    double p, e;
    two_prod_na(ah, bh, p, e);
    e += ah * bl + al * bh;
    double s, e2;
    ph = p + e;
    pl = e - (ph - p);
}
__global__ void __launch_bounds__(32)
k_dd_mul(volatile double *vals, volatile long long *out) {
    double ah = vals[0], al = vals[1];
    double bh = 0.999, bl = 1e-18;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++)
        dd_mul_na(ah, al, bh, bl, ah, al);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = ah; vals[3] = al;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

// Posit<8,0> decode chain
__device__ __forceinline__ float posit8_dec_na(unsigned char p) {
    if (p == 0) return 0.0f;
    if (p == 0x80) return 1.0f / 0.0f;
    int sign = (p >> 7) & 1;
    unsigned char abs_p = sign ? (~p + 1) : p;
    int regime_sign = (abs_p >> 6) & 1;
    int regime_len = 0;
    for (int b = 6; b >= 0; b--) {
        if (((abs_p >> b) & 1) == regime_sign) regime_len++; else break;
    }
    int k = regime_sign ? (regime_len - 1) : (-regime_len);
    int frac_bits = 7 - regime_len - 1;
    if (frac_bits < 0) frac_bits = 0;
    unsigned char frac = abs_p & ((1 << frac_bits) - 1);
    float value = exp2f((float)k) * (1.0f + (float)frac / (float)(1 << frac_bits));
    return sign ? -value : value;
}
__global__ void __launch_bounds__(32)
k_posit8_decode(volatile unsigned char *vals, volatile long long *out) {
    unsigned char p = vals[threadIdx.x % 8];
    float acc = 0.0f;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        acc += posit8_dec_na(p);
        p = (unsigned char)((int)(acc * 127.0f) & 0x7F);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[8 + threadIdx.x % 8] = p;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

// INT128 add via inline asm (pure carry chain, no loop overhead)
__global__ void __launch_bounds__(32)
k_int128_add_asm(volatile long long *vals, volatile long long *out) {
    unsigned long long lo = (unsigned long long)vals[0];
    unsigned long long hi = (unsigned long long)vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        // 128-bit add of 1: lo += 1, hi += carry
        asm volatile(
            "{"
            ".reg .u64 tmp;"
            ".reg .pred p;"
            "add.u64 %0, %0, 1;"
            "setp.eq.u64 p, %0, 0;"    // carry if lo wrapped to 0
            "@p add.u64 %1, %1, 1;"
            "}"
            : "+l"(lo), "+l"(hi)
        );
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = (long long)lo; vals[3] = (long long)hi;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

typedef void (*kfp_t)(volatile float*, volatile long long*);
typedef void (*kdp_t)(volatile double*, volatile long long*);
typedef void (*kllp_t)(volatile long long*, volatile long long*);
typedef void (*kubp_t)(volatile unsigned char*, volatile long long*);

static double mfp(kfp_t k,float*d,long long*o,long long*h){
    k<<<1,32>>>(d,o);cudaDeviceSynchronize();double t=0;
    for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;}
static double mdp(kdp_t k,double*d,long long*o,long long*h){
    k<<<1,32>>>(d,o);cudaDeviceSynchronize();double t=0;
    for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;}
static double mll(kllp_t k,long long*d,long long*o,long long*h){
    k<<<1,32>>>(d,o);cudaDeviceSynchronize();double t=0;
    for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;}
static double mub(kubp_t k,unsigned char*d,long long*o,long long*h){
    k<<<1,32>>>(d,o);cudaDeviceSynchronize();double t=0;
    for(int r=0;r<10;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/10;}

int main() {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Fill All N/A Latencies ===\n");
    printf("SM %d.%d | %s\n\n", prop.major, prop.minor, prop.name);

    long long *d_o, h[4]; CHECK(cudaMalloc(&d_o, 32));
    float hf[4]={0.5f,0.3f,0,0}; float *d_f; CHECK(cudaMalloc(&d_f,16));
    CHECK(cudaMemcpy(d_f,hf,16,cudaMemcpyHostToDevice));
    double hd[4]={1.0,1e-16,0,0}; double *d_d; CHECK(cudaMalloc(&d_d,32));
    CHECK(cudaMemcpy(d_d,hd,32,cudaMemcpyHostToDevice));
    long long hll[4]={42,0,0,0}; long long *d_ll; CHECK(cudaMalloc(&d_ll,32));
    CHECK(cudaMemcpy(d_ll,hll,32,cudaMemcpyHostToDevice));
    unsigned char hub[16]={42,85,100,200,10,20,30,40,0,0,0,0,0,0,0,0};
    unsigned char *d_ub; CHECK(cudaMalloc(&d_ub,16));
    CHECK(cudaMemcpy(d_ub,hub,16,cudaMemcpyHostToDevice));

    printf("%-28s %12s %12s\n", "Format", "cy/op", "Previous");
    printf("%-28s %12s %12s\n", "----------------------------", "------------", "------------");

    printf("%-28s %12.2f %12s\n", "FP8 E5M2 round-trip", mfp(k_fp8_e5m2_roundtrip, d_f, d_o, h), "N/A");
    printf("%-28s %12.2f %12s\n", "FP4 E2M1 round-trip", mfp(k_fp4_e2m1_roundtrip, d_f, d_o, h), "N/A");
    printf("%-28s %12.2f %12s\n", "NF4 round-trip", mfp(k_nf4_roundtrip, d_f, d_o, h), "N/A");
    printf("%-28s %12.2f %12s\n", "DD FP128 MUL", mdp(k_dd_mul, d_d, d_o, h), "N/A");
    printf("%-28s %12.2f %12s\n", "Posit<8,0> decode", mub(k_posit8_decode, d_ub, d_o, h), "N/A");
    printf("%-28s %12.2f %12s\n", "INT128 ADD (asm chain)", mll(k_int128_add_asm, d_ll, d_o, h), "N/A");

    printf("\nReference: FP8 E4M3 r/t=18.54cy, DD ADD=500.28cy, INT64 ADD=2.59cy\n");

    cudaFree(d_o); cudaFree(d_f); cudaFree(d_d); cudaFree(d_ll); cudaFree(d_ub);
    return 0;
}
