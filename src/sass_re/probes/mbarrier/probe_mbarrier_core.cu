#include <cuda_awbarrier_primitives.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(128)
probe_mbarrier_init_arrive_wait(uint64_t *out_tokens, uint32_t *out_pending) {
    __shared__ __align__(8) __mbarrier_t barrier;
    if (threadIdx.x == 0) {
        __mbarrier_init(&barrier, blockDim.x);
    }
    __syncthreads();

    __mbarrier_token_t token = __mbarrier_arrive(&barrier);
    while (!__mbarrier_test_wait(&barrier, token)) {
    }

    if (threadIdx.x == 0) {
        out_tokens[0] = token;
        out_pending[0] = __mbarrier_token_pending_count(token);
        __mbarrier_inval(&barrier);
    }
}

extern "C" __global__ void __launch_bounds__(128)
probe_mbarrier_arrive_drop(uint64_t *out_tokens, uint32_t *out_pending) {
    __shared__ __align__(8) __mbarrier_t barrier;
    if (threadIdx.x == 0) {
        __mbarrier_init(&barrier, blockDim.x);
    }
    __syncthreads();

    __mbarrier_token_t token = __mbarrier_arrive_and_drop(&barrier);
    if (threadIdx.x == 0) {
        out_tokens[0] = token;
        out_pending[0] = __mbarrier_token_pending_count(token);
        __mbarrier_inval(&barrier);
    }
}

extern "C" __global__ void __launch_bounds__(128)
probe_mbarrier_try_wait(uint32_t *out_ready, uint64_t *out_tokens, uint32_t spin_ns) {
    __shared__ __align__(8) __mbarrier_t barrier;
    if (threadIdx.x == 0) {
        __mbarrier_init(&barrier, 1);
    }
    __syncthreads();

    __mbarrier_token_t token = 0;
    if (threadIdx.x == 0) {
        token = __mbarrier_arrive(&barrier);
        out_ready[0] = __mbarrier_try_wait(&barrier, token, spin_ns) ? 1u : 0u;
        out_tokens[0] = token;
        __mbarrier_inval(&barrier);
    }
}
