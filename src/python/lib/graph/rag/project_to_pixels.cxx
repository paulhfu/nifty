#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/project_to_pixels.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_chunked.hxx"
#include "nifty/graph/rag/project_to_pixels_chunked.hxx"
#endif


namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class RAG,class T,size_t DATA_DIM>
    void exportProjectScalarNodeDataToPixelsT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
               const RAG & rag,
                nifty::marray::PyView<T, 1> nodeData,
               const int numberOfThreads
           ){  
                const auto labelsProxy = rag.labelsProxy();
                const auto & shape = labelsProxy.shape();
                const auto labels = labelsProxy.labels(); 

                nifty::marray::PyView<T, DATA_DIM> pixelData(shape.begin(),shape.end());
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
                return pixelData;
           },
           py::arg("graph"),py::arg("nodeData"),py::arg("numberOfThreads")
        );
    }

    
    #ifdef WITH_HDF5
    template<class RAG,class T>
    void exportProjectScalarNodeDataToPixelsSlicedT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixelsSliced",
           [](
                const RAG & rag,
                nifty::marray::PyView<T, 1> nodeData,
                const std::string & dataPath,
                const std::string & dataKey,
                const int numberOfThreads
           ){  
                const auto labelsProxy = rag.labelsProxy();
                const auto & labels = labelsProxy.labels(); 
                const auto shape = labels.shape();

                // create the pixel data
                hid_t dataFile = nifty::hdf5::createFile(dataPath);
                nifty::hdf5::Hdf5Array<T> pixelData(dataFile, dataKey, shape.begin(), shape.end(), labels.chunkShape().begin());

                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, numberOfThreads);
                }
                return pixelData;
           
            },
           py::arg("graph"),py::arg("nodeData"),py::arg("dataPath"),py::arg("dataKey"),py::arg_t< int >("numberOfThreads", -1)
        );
    }
    #endif




    void exportProjectToPixels(py::module & ragModule) {


        typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
        typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;



        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint64_t, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint64_t, 3>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, float, 2>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, float, 3>(ragModule);

        
        // export sliced rag (only if we have hdf5 support)
        #ifdef WITH_HDF5
        typedef ChunkedLabelsGridRagSliced<uint32_t> ChunkedLabelsGridRag;
        exportProjectScalarNodeDataToPixelsSlicedT<ChunkedLabelsGridRag, float>(ragModule);
        exportProjectScalarNodeDataToPixelsSlicedT<ChunkedLabelsGridRag, uint32_t>(ragModule);
        exportProjectScalarNodeDataToPixelsSlicedT<ChunkedLabelsGridRag, uint64_t>(ragModule);
        #endif

    }

} // end namespace graph
} // end namespace nifty
    
