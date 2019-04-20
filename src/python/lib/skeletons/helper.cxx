#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/skeletons/helper.hxx"


namespace py = pybind11;

namespace nifty {
namespace skeletons {

    void exportHelper(py::module & module) {

        //
        module.def("euclidean_distance", [](const xt::pytensor<bool, 3> & mask,
                                            const Coord & root,
                                            const std::array<float, 3> & voxel_size){
            xt::pytensor<float, 3> distances = xt::zeros<float>(mask.shape());
            {
                py::gil_scoped_release lift_gil;
                euclidean_distance(mask, root, voxel_size, distances);
            }
            return distances;
        }, py::arg("mask"), py::arg("root"), py::arg("voxel_size"));


        //
        module.def("boundary_voxel", [](const xt::pytensor<bool, 3> & mask){
            Coord coord;
            {
                py::gil_scoped_release lift_gil;
                boundary_voxel(mask, coord);

            }
            return coord;
        }, py::arg("mask"));

    }

}
}
