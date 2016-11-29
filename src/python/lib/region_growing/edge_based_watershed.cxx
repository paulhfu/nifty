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
            marray::PyView<DATA_TYPE,DIM+1> affinityMap,
            const DATA_TYPE lowerThreshold,
            const DATA_TYPE upperThreshold
        ){

                size_t shape[DIM];
                for(int d = 0; d < DIM; ++d)
                    shape[d] = affinityMap.shape(d);
                
                marray::PyView<LABEL_TYPE,DIM> segmentationOut(shape,shape+DIM);
                {
                    py::gil_scoped_release allowThreads;
                    edgeBasedWatershed<DIM>(affinityMap,lowerThreshold,upperThreshold,segmentationOut);
                }
                return segmentationOut;
            },
            py::arg("affinityMap"),
            py::arg("lowerThreshold"),
            py::arg("upperThreshold"));
    }
    
    
    template<unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
    void exportZWatershedT(py::module & regionGrowingModule) {
        regionGrowingModule.def("zWatershed",[](
            marray::PyView<DATA_TYPE,DIM+1> affinityMap,
            const DATA_TYPE lowerThreshold,
            const DATA_TYPE upperThreshold,
            const size_t sizeThreshold,
            const DATA_TYPE weightThreshold
        ){

                size_t shape[DIM];
                for(int d = 0; d < DIM; ++d)
                    shape[d] = affinityMap.shape(d);
                
                marray::PyView<LABEL_TYPE,DIM> segmentationOut(shape,shape+DIM);
                {
                    py::gil_scoped_release allowThreads;
                    zWatershed<DIM>(affinityMap,lowerThreshold,upperThreshold,sizeThreshold,weightThreshold,segmentationOut);
                }
                return segmentationOut;
            },
            py::arg("affinityMap"),
            py::arg("lowerThreshold"),
            py::arg("upperThreshold"),
            py::arg("sizeThreshold"),
            py::arg("weightThreshold"));
    }


    void exportEdgeBasedWatershed(py::module & regionGrowingModule) {
        
        exportEdgeBasedWatershedT<2,float,uint32_t>(regionGrowingModule);
        exportEdgeBasedWatershedT<3,float,uint32_t>(regionGrowingModule);
        
        exportZWatershedT<2,float,uint32_t>(regionGrowingModule);
        exportZWatershedT<3,float,uint32_t>(regionGrowingModule);
    }

} // namespace region_growing
} // namepace nifty
