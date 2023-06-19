#include <cstdint>

#include <pybind11/pybind11.h>

#include "hashgrid.h"

#include <drjit/jit.h>
#include <drjit/tensor.h>
#include <string>

namespace py = pybind11;
namespace dr = drjit;

PYBIND11_MODULE(_hashgrid_core, m) {

  using Float = dr::CUDAArray<float>;
  using Point3f = dr::Array<Float, 3>;
  py::module drjit = py::module::import("drjit");

  m.def("test", [](dr::CUDAArray<float> &test) { return std::string("test"); });
  m.def("float", []() { return dr::CUDAArray<float>(0.); });
  m.def("scatter_atomic",
        &scatter_atomic_inc_uint<dr::CUDAArray<unsigned int>>);
}
