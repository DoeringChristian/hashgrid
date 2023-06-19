
#include "vec.h"
#include <cstdint>

extern "C" __global__ void scatter_atomic_add_uint(uint32_t *target,
                                                   uint32_t *idx, uint32_t *dst,
                                                   uint32_t n_values) {
  int N = blockIdx.x * blockDim.x + threadIdx.x;
  if ((uint32_t)N < n_values) {
    dst[N] = atomicAdd(&target[idx[N]], 1);
  }
}

extern "C" __global__ void uint_prefix_sum(uint32_t *input, uint32_t *output) {}
