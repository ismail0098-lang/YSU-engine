
// Indirect function call via function pointer
__device__ float fn_add(float a, float b){return a+b;}
__device__ float fn_mul(float a, float b){return a*b;}
__device__ float fn_max(float a, float b){return fmaxf(a,b);}
__device__ float fn_min(float a, float b){return fminf(a,b);}

typedef float (*binop_t)(float,float);
__device__ binop_t ops[]={fn_add,fn_mul,fn_max,fn_min};

extern "C" __global__ void __launch_bounds__(128)
cf_indirect_call(float *out, const float *a, const float *b, int op_idx, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Indirect call via function pointer (may generate CALL.ABS or BRX)
    out[i]=ops[op_idx&3](a[i],b[i]);
}
