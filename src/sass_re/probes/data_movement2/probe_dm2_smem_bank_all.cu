
// Shared memory access at EVERY bank (32 banks, stride patterns)
extern "C" __global__ void __launch_bounds__(32)
dm2_smem_all_banks(float *out) {
    __shared__ float s[1024]; // 32 banks x 32 entries
    int tid=threadIdx.x;
    // Write to every bank (stride 1 = sequential banks)
    #pragma unroll
    for(int b=0;b<32;b++) s[tid+b*32]=float(tid*32+b);
    __syncwarp();
    // Read from same bank as another thread (stride 32 = bank 0 only)
    float v=0;
    #pragma unroll
    for(int b=0;b<32;b++) v+=s[b*32]; // All threads hit bank 0
    // Read with stride 1 (no conflict)
    #pragma unroll
    for(int b=0;b<32;b++) v+=s[tid*32+b]; // Each thread hits different bank
    out[tid]=v;
}
