
// Constant memory with dynamic (non-broadcast) indexing
__constant__ float CONST_TAB[1024];
extern "C" __global__ void __launch_bounds__(128)
dm2_const_dynamic(float *out, const int *indices, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Each thread reads different constant address (NOT broadcast)
    out[i]=CONST_TAB[indices[i]%1024]; // LDC with divergent index
}
