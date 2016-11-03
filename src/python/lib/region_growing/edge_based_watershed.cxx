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
                    // run the initial watershed
                    auto counts = edgeBasedWatershed<DIM>(affinityMap, lowThreshold, highThreshold, segmentationOut);
                    // build the region graph
                    std::vector<std::tuple<DATA_TYPE,LABEL_TYPE,LABEL_TYPE>> regionGraph;
                    LABEL_TYPE maxId = counts.size() - 1;
                    getRegionGraph<DIM>(affinityMap, segmentationOut, maxId, regionGraph);
                    // merge the segments
                    mergeSegments(segmentationOut, regionGraph, counts, mergeThreshold, lowThreshold, sizeThreshold);
                }
                return segmentationOut;
            },
            py::arg("affinityMap"),
            py::arg("mergeThreshold"),
            py::arg("sizeThreshold") = 10,
            py::arg("lowThreshod") = 0.0001,
            py::arg("highThreshold") = .9999); // default values for low and high from zwatershed repo
    }


    void exportEdgeBasedWatershed(py::module & regionGrowingModule) {
        // TODO make c++ implementation work for other template uint64
        exportEdgeBasedWatershedT<2,float,uint32_t>(regionGrowingModule);
        exportEdgeBasedWatershedT<3,float,uint32_t>(regionGrowingModule);
        //exportEdgeBasedWatershedT<2,double,uint32_t>(regionGrowingModule);
        exportZWatershedT<2,float,uint32_t>(regionGrowingModule);
        exportZWatershedT<3,float,uint32_t>(regionGrowingModule);
    }

} // namespace region_growing
} // namepace nifty

