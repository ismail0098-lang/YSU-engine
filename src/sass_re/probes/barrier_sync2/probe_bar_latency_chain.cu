
// BAR.SYNC latency measurement: single-thread repeated barrier
extern "C" __global__ void __launch_bounds__(32)
bar_sync_latency(volatile long long *out) {
    long long t0,t1;
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<512;i++) __syncthreads(); // 512 barriers
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t1));
    if(threadIdx.x==0){out[0]=t1-t0;out[1]=512;}
}

extern "C" __global__ void __launch_bounds__(32)
syncwarp_latency(volatile long long *out) {
    long long t0,t1;
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<512;i++) __syncwarp(0xFFFFFFFF);
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t1));
    if(threadIdx.x==0){out[0]=t1-t0;out[1]=512;}
}
