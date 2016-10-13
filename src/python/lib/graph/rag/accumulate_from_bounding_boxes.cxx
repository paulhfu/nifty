#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/grid_rag_accumulate_from_bounding_boxes.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#endif

namespace py = pybind11;


namespace nifty{
namespace graph{
    
    using namespace py;
    
    template<size_t DIM, class RAG, class DATA>
    void exportAccumulateNodeStandartFeaturesFromBoundingBoxes(
        py::module & ragModule
    ){
        ragModule.def("accumulateNodeStandartFeaturesFromBoundingBoxes",
        [](
            const RAG & rag,
            const DATA & data,
            std::vector<std::vector<int64_t>> startCoordinates,
            std::vector<int64_t> blockShape,
            const double minVal,
            const double maxVal,
            const bool onePass,
            const int numberOfThreads
        ){
            typedef typename DATA::DataType DataType;
            typedef nifty::marray::PyView<DataType> NumpyArrayType;
            auto nFeatures = onePass ? 9 : 11;
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(nFeatures)});
            if(onePass) {
                py::gil_scoped_release allowThreads;
                accumulateNodeStandartFeaturesOnePass(rag, data, startCoordinates, blockShape, minVal, maxVal, nodeOut, numberOfThreads);
            }
            else {
                py::gil_scoped_release allowThreads;
                accumulateNodeStandartFeaturesTwoPass(rag, data, startCoordinates, blockShape, minVal, maxVal, nodeOut, numberOfThreads);
            }
            return nodeOut;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("startCoordinates"),
        py::arg("blockShape"),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("onePass")=true,
        py::arg("numberOfThreads")= -1
        );
    }
    
    template<size_t DIM, class RAG, class DATA>
    void exportAccumulateEdgeStandartFeaturesFromBoundingBoxes(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStandartFeaturesFromBoundingBoxes",
        [](
            const RAG & rag,
            const DATA & data,
            std::vector<std::vector<int64_t>> startCoordinates,
            std::vector<int64_t> blockShape,
            const double minVal,
            const double maxVal,
            const bool onePass,
            const int numberOfThreads
        ){
            typedef typename DATA::DataType DataType;
            typedef nifty::marray::PyView<DataType> NumpyArrayType;
            auto nFeatures = onePass ? 9 : 11;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(nFeatures)});
            if(onePass) {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeaturesOnePass(rag, data, startCoordinates, blockShape, minVal, maxVal, edgeOut, numberOfThreads);
            }
            else {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeaturesTwoPass(rag, data, startCoordinates, blockShape, minVal, maxVal, edgeOut, numberOfThreads);
            }
            return edgeOut;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("startCoordinates"),
        py::arg("blockShape"),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("onePass")=true,
        py::arg("numberOfThreads")= -1
        );
    }

    void exportAccumulateFromBoundingBoxes(py::module & ragModule) {

        {
            #ifdef WITH_HDF5
            //typedef Hdf5Labels<2, uint32_t> Hdf5Labels2dUInt32; 
            typedef Hdf5Labels<3, uint32_t> Hdf5Labels3dUInt32; 
            //typedef Hdf5Labels<2, uint64_t> Hdf5Labels2dUInt64; 
            typedef Hdf5Labels<3, uint64_t> Hdf5Labels3dUInt64; 

            //typedef GridRag<2, Hdf5Labels2dUInt32> Hdf5Rag2dUInt32;
            typedef GridRag<3, Hdf5Labels3dUInt32> Hdf5Rag3dUInt32;
            //typedef GridRag<2, Hdf5Labels2dUInt64> Hdf5Rag2dUInt64;
            typedef GridRag<3, Hdf5Labels3dUInt64> Hdf5Rag3dUInt64;

            typedef hdf5::Hdf5Array<float> FloatArray;
            typedef hdf5::Hdf5Array<uint8_t> UInt8Array;
            
            exportAccumulateNodeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt32, FloatArray>(ragModule);
            exportAccumulateNodeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt32, UInt8Array>(ragModule);
            exportAccumulateNodeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt64, FloatArray>(ragModule);
            exportAccumulateNodeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt64, UInt8Array>(ragModule);
            
            exportAccumulateEdgeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt32, FloatArray>(ragModule);
            exportAccumulateEdgeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt32, UInt8Array>(ragModule);
            exportAccumulateEdgeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt64, FloatArray>(ragModule);
            exportAccumulateEdgeStandartFeaturesFromBoundingBoxes<3, Hdf5Rag3dUInt64, UInt8Array>(ragModule);
            #endif
        }
    }



} // end namespace graph
} // end namespace nifty
