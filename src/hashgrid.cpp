#include "hashgrid.h"
#include "../kernels/vec.h"
#include "cuda_helpers.h"
#include <cstdint>
#include <drjit/util.h>

namespace dr = drjit;

int int_div_up(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

std::pair<Vec3i, Vec3i> get_launch_parameters(int n_threads) {
  int total_num_threads = n_threads;
  Vec3i block_size(16, 1, 1);
  Vec3i grid_size(int_div_up(n_threads, block_size.y), 1, 1);
  return {grid_size, block_size};
}

template <typename Float>
HashGrid<Float>::HashGrid(Point3f &sample, int resolution, int n_cells) {
  n_samples = sample.shape()[1];
  this->n_cells = n_cells;
  this->resolution = resolution;
  this->bbmin = dr::minimum(dr::min(sample.x),
                            dr::minimum(dr::min(sample.y), dr::min(sample.z)));
  this->bbmax = dr::maximum(dr::max(sample.x),
                            dr::maximum(dr::max(sample.y), dr::max(sample.z)));

  UInt32 cell = cell_idx(sample);
  dr::eval(cell);

  // this->cell_offset = dr::zeros<UInt32>(n_cells);
  this->sample_index = dr::zeros<UInt32>(n_samples);
  auto index_in_cell = dr::zeros<UInt32>(n_samples);
  this->cell_size = dr::zeros<UInt32>(n_cells);

  assert(dr::is_cuda_v<Float>());
  cuda_load_kernels();

  auto [grid_size, block_size] = get_launch_parameters(n_samples);

  uint32_t *cell_ptr = cell.data();
  uint32_t *cell_size_ptr = cell_size.data();
  uint32_t *index_in_cell_ptr = index_in_cell.data();

  void *args[] = {&cell_ptr, &cell_size_ptr, &index_in_cell_ptr, &n_samples};

  CUcontext ctx = CUcontext(jit_cuda_context());
  scoped_set_context guard(ctx);

  cuda_check(cuLaunchKernel(compute_index_in_cell, grid_size.x, grid_size.y,
                            grid_size.z, block_size.x, block_size.y,
                            block_size.z, 0, 0, args, 0));
  cuda_check(cuCtxSynchronize());
}

template <typename Float>
typename HashGrid<Float>::UInt HashGrid<Float>::hash(Point3u &p) {
  return ((p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791)) %
         this->n_cells;
}

template <typename Float>
typename HashGrid<Float>::UInt32 HashGrid<Float>::cell_idx(Point3f &p) {
  return hash(cell_pos(p));
}

template <typename Float>
typename HashGrid<Float>::Point3u HashGrid<Float>::cell_pos(Point3f &p) {

  auto p_normalized =
      (p - this->bbmin) / (this->bbmax - this->bbmin) * this->resolution;
  auto p_normalized_uint = Point3u(
      UInt32(p_normalized.x), UInt32(p_normalized.y), UInt32(p_normalized.z));
  return p_normalized_uint;
}
