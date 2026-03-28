
// Type punning: reinterpret same memory as different types
extern "C" __global__ void __launch_bounds__(128)
dm2_type_pun(void *out, const void *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Load as int, store as float (type pun via union)
    union { int i; float f; } u;
    u.i=((const int*)in)[i];
    ((float*)out)[i]=u.f+1.0f;
}
