#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace region_growing{
    void exportEdgeBasedWatershed(py::module &);
}
}

    
PYBIND11_PLUGIN(_region_growing) {
    py::module regionGrowingModule("_region_growing","region growing submodule");
    
    using namespace nifty::region_growing;

    exportEdgeBasedWatershed(regionGrowingModule);

    return regionGrowingModule.ptr();
}
