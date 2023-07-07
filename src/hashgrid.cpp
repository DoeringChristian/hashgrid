#include "hashgrid.h"
#include "../kernels/vec.h"
#include "cuda_helpers.h"
#include <cstdint>
#include <drjit-core/containers.h>
#include <drjit/array.h>
#include <drjit/util.h>
#include <iostream>

namespace dr = drjit;

int int_div_up(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

std::pair<Vec3i, Vec3i> get_launch_parameters(int n_threads) {
  int total_num_threads = n_threads;
  Vec3i block_size(16, 1, 1);
  Vec3i grid_size(int_div_up(n_threads, block_size.y), 1, 1);
  return {grid_size, block_size};
}

void scatter_atomic_inc_uint_cuda(uint64_t target, uint64_t idx, uint64_t dst,
                                  int n_values) {

  cuda_load_kernels();

  auto [grid_size, block_size] = get_launch_parameters(n_values);

  void *args[] = {&target, &idx, &dst, &n_values};

  CUcontext ctx = CUcontext(jit_cuda_context());
  scoped_set_context guard(ctx);

  cuda_check(cuLaunchKernel(cuda_scatter_atomic_add_uint, grid_size.x,
                            grid_size.y, grid_size.z, block_size.x,
                            block_size.y, block_size.z, 0, 0, args, 0));
  cuda_check(cuCtxSynchronize());
}
