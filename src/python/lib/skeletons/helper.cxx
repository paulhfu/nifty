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
        module.def("dijkstra", [](const xt::pytensor<float, 3> & field,
                                  const Coord & src, const Coord & target){
            std::vector<Coord> path;
            {
                py::gil_scoped_release lift_gil;
                dijkstra(field, src, target, path);
            }
            return path;

        }, py::arg("field"), py::arg("src"), py::arg("target"));



        //
        module.def("boundary_voxel", [](const xt::pytensor<bool, 3> & mask){
            Coord coord;
            {
                py::gil_scoped_release lift_gil;
                boundary_voxel(mask, coord);

            }
            return coord;
        }, py::arg("mask"));


        //
        module.def("compute_path_mask", [](const xt::pytensor<float, 3> & distance,
                                           const std::vector<Coord> & path,
                                           const double mask_scale,
                                           const double mask_min_radius,
                                           const std::array<float, 3> & voxel_size){
            const std::size_t n_voxels = distance.size();
            // NOTE we compute flat output mask, needs to be rehaped in pytgon
            xt::pytensor<bool, 1> out = xt::zeros<bool>({n_voxels});
            {
                py::gil_scoped_release lift_gil;
                compute_path_mask(distance, path, mask_scale, mask_min_radius, voxel_size, out);
            }
            return out;
        }, py::arg("distance"), py::arg("path"),
           py::arg("mask_scale"), py::arg("mask_min_radius"),
           py::arg("voxel_size"));


        //
        module.def("pathlength", [](const Coord & shape,
                                    const std::vector<Coord> & path,
                                    const std::array<float, 3> & voxel_size){
            double len;
            {
                py::gil_scoped_release lift_gil;
                len = pathlength(shape, path, voxel_size);
            }
            return len;
        }, py::arg("shape"), py::arg("path"), py::arg("voxel_size"));
    }

}
}
