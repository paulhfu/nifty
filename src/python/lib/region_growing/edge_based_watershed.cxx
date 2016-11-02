#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/region_growing/edge_based_watershed.hxx"


namespace py = pybind11;

namespace nifty {
namespace region_growing {

    template<unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
    void exportEdgeBasedWatershedT(py::module & regionGrowingModule) {
        regionGrowingModule.def("edgeBasedWatershed",[](
            const marray::PyView<DATA_TYPE,DIM+1> affinityMap,
            const DATA_TYPE low,
            const DATA_TYPE high){

                size_t shape[DIM];
                for(int d = 0; d < DIM; d++)
                    shape[d] = affinityMap.shape(d);
                marray::PyView<LABEL_TYPE,DIM> segmentationOut(shape,shape+DIM);
                {
                    py::gil_scoped_release allowThreads;
                    auto counts = edgeBasedWatershed<DIM>(affinityMap,low,high,segmentationOut);
                }
                return segmentationOut;
            },
            py::arg("affinityMap"),
            py::arg("low"),
            py::arg("high"));
    }


    void exportEdgeBasedWatershed(py::module & regionGrowingModule) {
        // TODO make c++ implementation work for other template options
        exportEdgeBasedWatershedT<2,float,uint32_t>(regionGrowingModule);
        exportEdgeBasedWatershedT<2,double,uint32_t>(regionGrowingModule);
    }

} // namespace region_growing
} // namepace nifty

