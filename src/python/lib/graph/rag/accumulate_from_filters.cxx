#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#ifdef WITH_FASTFILTERS
#include "nifty/graph/rag/grid_rag_accumulate_filters.hxx"
#endif

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#endif

namespace py = pybind11;


namespace nifty{
namespace graph{
    
    using namespace py;
    
    #ifdef WITH_FASTFILTERS
    template<size_t DIM, class RAG, class DATA>
    void exportAccumulateEdgeStandartFeaturesFromFilters(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStandartFeaturesFromFilters",
        [](
            const RAG & rag,
            const DATA & data,
            std::vector<std::vector<int64_t>> startCoordinates,
            std::vector<int64_t> blockShape,
            const bool onePass,
            const int numberOfThreads
        ){
            typedef typename DATA::DataType DataType;
            typedef nifty::marray::PyView<DataType> NumpyArrayType;
            size_t nStats = onePass ? 9 : 11;
            size_t nChannels = (DIM==2) ? 12 : 15;
            auto nFeatures = nStats * nChannels;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(nFeatures)});
            if(onePass) {
                std::cout << "OnePass Features from filters" << std::endl;
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeaturesOnePass(rag, data, startCoordinates, blockShape, edgeOut, numberOfThreads);
            }
            else {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeaturesTwoPass(rag, data, startCoordinates, blockShape, edgeOut, numberOfThreads);
            }
            return edgeOut;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("startCoordinates"),
        py::arg("blockShape"),
        py::arg("onePass")=true,
        py::arg("numberOfThreads")= -1
        );
    }

    void exportAccumulateFilters(py::module & ragModule) {

        {
            #ifdef WITH_HDF5
            typedef Hdf5Labels<3, uint32_t> Hdf5Labels3dUInt32; 
            typedef Hdf5Labels<3, uint64_t> Hdf5Labels3dUInt64; 

            typedef GridRag<3, Hdf5Labels3dUInt32> Hdf5Rag3dUInt32;
            typedef GridRag<3, Hdf5Labels3dUInt64> Hdf5Rag3dUInt64;

            typedef hdf5::Hdf5Array<float> FloatArray;
            typedef hdf5::Hdf5Array<uint8_t> UInt8Array;
            
            exportAccumulateEdgeStandartFeaturesFromFilters<3, Hdf5Rag3dUInt32, FloatArray>(ragModule);
            //exportAccumulateEdgeStandartFeaturesFromFilters<3, Hdf5Rag3dUInt32, UInt8Array>(ragModule);
            exportAccumulateEdgeStandartFeaturesFromFilters<3, Hdf5Rag3dUInt64, FloatArray>(ragModule);
            //exportAccumulateEdgeStandartFeaturesFromFilters<3, Hdf5Rag3dUInt64, UInt8Array>(ragModule);
            #endif
        }
    }
    #endif


} // end namespace graph
} // end namespace nifty
