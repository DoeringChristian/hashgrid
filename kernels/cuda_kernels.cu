
#include "vec.h"
#include <cstdint>

extern "C" __global__ void scatter_atomic_add_uint(uint32_t *target,
                                                   uint32_t *idx, uint32_t *dst,
                                                   uint32_t n_values) {
  int N = blockIdx.x * blockDim.x + threadIdx.x;
  if ((uint32_t)N < n_values) {
    dst[N] = atomicInc(&target[idx[N]], 1);
  }
}
