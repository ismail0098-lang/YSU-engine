
#include <string.h>

// INT8 warp-level reduction via dp4a
extern "C" __global__ void __launch_bounds__(32)
int8_warp_reduce(int *out, const signed char *in, int n) {
    int lane=threadIdx.x, gi=blockIdx.x*128+lane*4;
    int packed=0;
    if(gi+3<n) memcpy(&packed, &in[gi], 4);
    int sum=__dp4a(packed, 0x01010101, 0); // Sum 4 bytes (weights=1)
    sum=__reduce_add_sync(0xFFFFFFFF, sum);
    if(lane==0) out[blockIdx.x]=sum;
}
