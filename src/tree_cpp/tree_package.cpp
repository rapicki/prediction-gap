#include "TreeParser.h"
#include <pybind11/pybind11.h>
#include <string>
using namespace std;
namespace py = pybind11;
// namespace py = pybind11;

PYBIND11_MODULE(tree_package, m) {
  m.doc() = "pybind11 tree plugin"; // optional module docstring

  py::class_<TreeParser>(m, "TreeParser")
      .def(py::init<string, float>())
      .def("eval", &TreeParser::eval_numpy);
}
