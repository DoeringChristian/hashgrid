#pragma once

#include <drjit/util.h>

namespace dr = drjit;

template <typename Float> struct HashGrid {
  using UInt32 = dr::uint32_array_t<Float>;
  using UInt = UInt32;
  using Point3f = dr::Array<Float, 3>;
  using Point3u = dr::Array<UInt32, 3>;

  int n_cells;
  int n_samples;
  int resolution;
  float bbmin, bbmax;

  UInt32 cell_size;
  UInt32 sample_index;

  HashGrid(Point3f &sample, int resolution, int n_cells);
  UInt32 cell_idx(Point3f &p);
  Point3u cell_pos(Point3f &p);

  UInt32 hash(Point3u &p);
};
