
// Volatile memory access patterns (bypasses cache, generates different SASS)
extern "C" __global__ void __launch_bounds__(128)
volatile_load_store(volatile float *out, volatile const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i];    // Volatile load (may bypass L1)
    out[i]=v*2.0f;    // Volatile store
}

extern "C" __global__ void __launch_bounds__(128)
volatile_rmw(volatile int *data, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    data[i]=data[i]+1; // Volatile read-modify-write (2 accesses)
}
