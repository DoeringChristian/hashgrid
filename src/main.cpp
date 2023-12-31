#include <cstdint>

#include <pybind11/pybind11.h>

#include "hashgrid.h"

#include <drjit/jit.h>
#include <drjit/tensor.h>
#include <string>

namespace py = pybind11;
namespace dr = drjit;

PYBIND11_MODULE(_hashgrid_core, m) {

  // using Float = dr::CUDAArray<float>;
  // using Point3f = dr::Array<Float, 3>;
  py::module drjit = py::module::import("drjit");

  m.def("scatter_atomic_inc_uint_cuda", &scatter_atomic_inc_uint_cuda);
}
