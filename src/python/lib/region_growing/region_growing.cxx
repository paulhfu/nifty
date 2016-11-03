#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

namespace nifty {
namespace region_growing {

    void exportEdgeBasedWatershed(py::module &);
    void exportZWatershed(py::module &);

    void initSubmoduleRegionGrowing(py::module & niftyModule) {
        auto regionGrowingModule = niftyModule.def_submodule("region_growing","region growing submodule");
        exportEdgeBasedWatershed(regionGrowingModule);
        exportZWatershed(regionGrowingModule);
    }

} // namespace region_growing
} // namespace nifty
