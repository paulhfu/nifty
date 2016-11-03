#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/region_growing/zwatershed.hxx"


namespace py = pybind11;

namespace nifty {
namespace region_growing {
    
    template<unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
    void exportZWatershedT(py::module & regionGrowingModule) {
        regionGrowingModule.def("zWatershed",[](
            const marray::PyView<DATA_TYPE,DIM+1> affinityMap,
            const DATA_TYPE mergeThreshold,
            const size_t    sizeThreshold,
            const DATA_TYPE lowThreshold,
            const DATA_TYPE highThreshold){
                
                size_t shape[DIM];
                for(int d = 0; d < DIM; d++)
                    shape[d] = affinityMap.shape(d);

                marray::PyView<LABEL_TYPE,DIM> segmentationOut(shape,shape+DIM);
                {
                    py::gil_scoped_release allowThreads;
                    zWatershed<DIM>(affinityMap, mergeThreshold, sizeThreshold, lowThreshold, highThreshold, segmentationOut);
                }
                return segmentationOut;
            },
            py::arg("affinityMap"),
            py::arg("mergeThreshold"),
            py::arg("sizeThreshold") = 10,
            py::arg("lowThreshod") = 0.0001,
            py::arg("highThreshold") = .9999); // default values for low and high from zwatershed repo
    }

    void exportZWatershed(py::module & regionGrowingModule) {
        // TODO make c++ implementation work for other template uint64
        exportZWatershedT<2,float,uint32_t>(regionGrowingModule);
        exportZWatershedT<3,float,uint32_t>(regionGrowingModule);
    }

} // namespace region_growing
} // namepace nifty
