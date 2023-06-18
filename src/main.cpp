#include "hashgrid.h"
#include <cstdint>

#include <pybind11/pybind11.h>

#include <drjit/jit.h>
#include <drjit/tensor.h>
#include <string>

namespace py = pybind11;
namespace dr = drjit;

PYBIND11_MODULE(_hashgrid_core, m) {

  using Float = dr::CUDAArray<float>;
  using Point3f = dr::Array<Float, 3>;
  py::module drjit = py::module::import("drjit");

  m.def("test", []() { return std::string("test"); });
  py::class_<HashGrid<Float>>(m, "HashGrid")
      // .def(py::init<Point3f &, int, int>());
      .def("cell_pos", &HashGrid<Float>::cell_pos);
  // .def_readonly("cell_size", &HashGrid<Float>::cell_size);
  // .def("cell_idx", &HashGrid<Float>::cell_idx);
}
