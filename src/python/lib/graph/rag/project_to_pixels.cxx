#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/project_to_pixels.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#endif



namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class RAG,class T,size_t DATA_DIM, bool AUTO_CONVERT>
    void exportProjectScalarNodeDataToPixelsT(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
               const RAG & rag,
                nifty::marray::PyView<T, 1, AUTO_CONVERT> nodeData,
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
           py::arg("graph"),py::arg("nodeData"),py::arg("numberOfThreads")=-1
        );
    }
    

    #ifdef WITH_HDF5
    template<class RAG,class T,size_t DATA_DIM, bool AUTO_CONVERT>
    void exportProjectScalarNodeDataToPixelsHdf5T(py::module & ragModule){

        ragModule.def("projectScalarNodeDataToPixels",
           [](
                const RAG & rag,
                nifty::marray::PyView<T, 1, AUTO_CONVERT> nodeData,
                std::vector<int64_t> start,
                std::vector<int64_t> stop,
                const int numberOfThreads
           ){  
                std::vector<int64_t> shape(3);
                for(int d =0; d < DATA_DIM; d++) {
                    shape[d] = stop[d] - start[d];
                }
                nifty::marray::PyView<T, DATA_DIM> pixelData(shape.begin(),shape.end());
                {
                    py::gil_scoped_release allowThreads;
                    projectScalarNodeDataToPixels(rag, nodeData, pixelData, start, numberOfThreads);
                }
                return pixelData;
           },
           py::arg("graph"),
           py::arg("nodeData"),
           py::arg("start"),
           py::arg("stop"),
           py::arg("numberOfThreads")=-1
        );
    }
    #endif



    void exportProjectToPixels(py::module & ragModule) {

        typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
        typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;



        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint32_t, 2, false>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint32_t, 3, false>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, uint64_t, 2, false>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, uint64_t, 3, false>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, float, 2, false>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, float, 3, false>(ragModule);

        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag2D, double, 2, true>(ragModule);
        exportProjectScalarNodeDataToPixelsT<ExplicitLabelsGridRag3D, double, 3, true>(ragModule);

        #ifdef WITH_HDF5
        typedef Hdf5LabelsGridRag<3, uint64_t> Hdf5Rag3dUInt64;
        typedef Hdf5LabelsGridRag<3, uint32_t> Hdf5Rag3dUInt32;

        exportProjectScalarNodeDataToPixelsHdf5T<Hdf5Rag3dUInt64,uint64_t,3,false>(ragModule);
        exportProjectScalarNodeDataToPixelsHdf5T<Hdf5Rag3dUInt32,uint32_t,3,false>(ragModule);
        #endif

    }

} // end namespace graph
} // end namespace nifty
    
