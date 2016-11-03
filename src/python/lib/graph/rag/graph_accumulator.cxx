#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag_features.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
    
#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif


namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;


    template<class RAG,class T,size_t DATA_DIM>
    void exportGridRagAccumulateLabelsT(py::module & ragModule){

        ragModule.def("gridRagAccumulateLabels",
            [](
                const RAG & rag,
                nifty::marray::PyView<T, DATA_DIM> labels
            ){  
                nifty::marray::PyView<T> nodeLabels({rag.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateLabels(rag, labels, nodeLabels);
                }
                return nodeLabels;

            },
            py::arg("graph"),py::arg("labels")
        );
    }
    

    // export only if we have HDF5 support
    /*
    template<class RAG,class T, class EDGE_MAP, class NODE_MAP>
    void exportGridRagAccumulateFeaturesT(py::module & ragModule){
        ragModule.def("gridRagSlicedAccumulateFeatures",
            [](
                const RAG & rag,
                const hdf5::Hdf5Array<T> data,
                EDGE_MAP & edgeMap
            ){  
                {
                    //py::gil_scoped_release allowThreads;
                    gridRagAccumulateFeatures(rag, data, edgeMap);
                }
            },
            py::arg("graph"),py::arg("data"),py::arg("edgeMap")
        );
    }
    */

    template<class RAG,class LABELS>
    void exportGridRagStackedAccumulateLabels(py::module & ragModule){

        ragModule.def("gridRagAccumulateLabels",
            [](
                const RAG & rag,
                const LABELS & labels, // TODO this needs to be call by reference, otherwise this yields strange hdf5 behaviout, but is this ok, if we give it a PyView ?
                const int numberOfThreads
            ){  
                typedef typename LABELS::DataType LabelsType;
                nifty::marray::PyView<LabelsType> nodeLabels({rag.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    gridRagAccumulateLabels(rag, labels, nodeLabels, numberOfThreads);
                }
                return nodeLabels;

            },
            py::arg("graph"),py::arg("labels"),py::arg_t< int >("numberOfThreads",-1)
        );
    }



    void exportGraphAccumulator(py::module & ragModule) {

        // exportGridRagAccumulateLabels
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
            typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;
            // accumulate labels
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag2D, uint32_t, 2>(ragModule);
            exportGridRagAccumulateLabelsT<ExplicitLabelsGridRag3D, uint32_t, 3>(ragModule);

            // ***********************
            // Export Stacked Grid Rag
            // ***********************    

            typedef GridRagStacked2D<ExplicitLabels<3,uint32_t>> ExplicitGridRagStacked2D;
            exportGridRagStackedAccumulateLabels<ExplicitGridRagStacked2D, marray::PyView<uint32_t>>(ragModule);
            
            #ifdef WITH_HDF5
            typedef GridRagStacked2D<Hdf5Labels<3,uint32_t>> Hdf5GridRagStacked2D;
            exportGridRagStackedAccumulateLabels<Hdf5GridRagStacked2D, hdf5::Hdf5Array<uint32_t>>(ragModule);
            #endif
        }
    }

} // end namespace graph
} // end namespace nifty
    
