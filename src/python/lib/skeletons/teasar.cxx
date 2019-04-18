#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/skeletons/teasar.hxx"
#include "xtensor-python/pytensor.hpp"


namespace py = pybind11;

namespace nifty {
namespace skeletons {

    void exportTeasar(py::module & module) {

        module.def("skeletonize", [](const xt::pytensor<uint8_t, 3> & mask,
                                     const std::vector<double> & pixel_pitch,
                                     const double boundary_weight){
            {
                py::gil_scoped_release liftGil;
                skeletonize(mask, pixel_pitch, boundary_weight);
            }
        }, py::arg("mask"), py::arg("pixel_pitch"), py::arg("boundary_weight"));

    }


}
}
