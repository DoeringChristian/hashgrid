
#include "vec.h"
#include <cstdint>

extern "C" __global__ void compute_index_in_cell(uint32_t *cell,
                                                 uint32_t *cell_size,
                                                 uint32_t *index_in_cell,
                                                 uint32_t n_samples) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ((uint32_t)idx < n_samples) {
    index_in_cell[idx] = atomicInc(&cell_size[cell[idx]], 1);
  }
}
