#pragma once

#include <cstdint>
#include <drjit/array.h>
#include <drjit/jit.h>
#include <drjit/tensor.h>

namespace dr = drjit;

/**
 * @brief Scatter cell indices into an array while keeping the previous value.
 *
 * @tparam UInt32 UInt32
 * @param cell cell indices to scatter
 * @return A tuple of the cell_size and index_in_cell
 */
// template <typename UInt32>
// UInt32 scatter_atomic_inc_uint(UInt32 &target, const UInt32 &idx);

void scatter_atomic_inc_uint_cuda(uint64_t target, uint64_t idx, uint64_t dst,
                                  int n_values);

// template <typename Float> struct HashGrid {
//   using UInt32 = dr::uint32_array_t<Float>;
//   using UInt = UInt32;
//   using Point3f = dr::Array<Float, 3>;
//   using Point3u = dr::Array<UInt32, 3>;
//
//   int n_cells;
//   int n_samples;
//   int resolution;
//   float bbmin, bbmax;
//
//   UInt32 cell_size;
//   UInt32 sample_index;
//
//   HashGrid(Point3f sample, int resolution, int n_cells);
//   UInt32 cell_idx(Point3f &p);
//   Point3u cell_pos(Point3f &p);
//
//   UInt32 hash(Point3u &p);
// };
