#include "hashgrid.h"
#include "../kernels/vec.h"
#include "cuda_helpers.h"
#include <cstdint>
#include <drjit-core/containers.h>
#include <drjit/array.h>
#include <drjit/util.h>

namespace dr = drjit;

int int_div_up(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

std::pair<Vec3i, Vec3i> get_launch_parameters(int n_threads) {
  int total_num_threads = n_threads;
  Vec3i block_size(16, 1, 1);
  Vec3i grid_size(int_div_up(n_threads, block_size.y), 1, 1);
  return {grid_size, block_size};
}

template <typename UInt32>
UInt32 scatter_atomic_add_uint(UInt32 &target, const UInt32 &value,
                               const UInt32 &idx) {

  int n_values = idx.size();
  int n_target = target.size();
  UInt32 dst = dr::zeros<UInt32>(n_values);

  dr::eval(target, value, idx, dst);

  assert(dr::is_cuda_v<Float>());
  cuda_load_kernels();

  auto [grid_size, block_size] = get_launch_parameters(n_values);

  const uint32_t *value_ptr = value.data();
  const uint32_t *idx_ptr = idx.data();
  uint32_t *target_ptr = target.data();
  uint32_t *dst_ptr = dst.data();

  void *args[] = {&target_ptr, &value_ptr, &idx_ptr, &dst_ptr, &n_values};

  CUcontext ctx = CUcontext(jit_cuda_context());
  scoped_set_context guard(ctx);

  cuda_check(cuLaunchKernel(compute_index_in_cell, grid_size.x, grid_size.y,
                            grid_size.z, block_size.x, block_size.y,
                            block_size.z, 0, 0, args, 0));
  cuda_check(cuCtxSynchronize());
}

template dr::CUDAArray<uint32_t>
scatter_atomic_add_uint(dr::CUDAArray<uint32_t> &,
                        const dr::CUDAArray<uint32_t> &,
                        const dr::CUDAArray<uint32_t> &);
