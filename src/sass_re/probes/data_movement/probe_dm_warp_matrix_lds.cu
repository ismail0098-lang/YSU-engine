
#include <mma.h>
using namespace nvcuda;
// LDSM: shared memory matrix load for tensor cores (all fragment types)
extern "C" __global__ void __launch_bounds__(32)
dm_ldsm_fp16(half *D, const half *gA, const half *gB) {
    __shared__ half sA[16*16], sB[16*16];
    for(int j=threadIdx.x;j<256;j+=32){sA[j]=gA[j];sB[j]=gB[j];}
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> fB;
    wmma::fragment<wmma::accumulator,16,16,16,float> fC;
    wmma::load_matrix_sync(fA,sA,16); // LDSM from shared
    wmma::load_matrix_sync(fB,sB,16); // LDSM from shared
    wmma::fill_fragment(fC,0.0f);
    wmma::mma_sync(fC,fA,fB,fC);
    wmma::fragment<wmma::accumulator,16,16,16,half> fD;
    for(int j=0;j<fC.num_elements;j++) fD.x[j]=__float2half(fC.x[j]);
    wmma::store_matrix_sync(D,fD,16,wmma::mem_row_major);
}
