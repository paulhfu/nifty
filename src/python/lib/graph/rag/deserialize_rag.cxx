#include <pybind11/pybind11.h>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/python/converter.hxx"
    
#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#endif


namespace py = pybind11;


namespace nifty{
namespace graph{
    
    using namespace py;
    
    template<class RAG>
    void exportDeserializeExplicitRagT(py::module & ragModule, const std::string & exportName ){

        typedef RAG GridRagType;
        typedef typename GridRagType::LabelsProxy LabelsProxyType;
        typedef typename GridRagType::LabelsProxy::ViewType LabelsViewType;
        typedef typename GridRagType::DontComputeRag DontCompute;
        typedef typename GridRagType::Settings Settings;

        ragModule.def(exportName.c_str(),
           [](
                LabelsViewType & labels,
                nifty::marray::PyView<uint64_t,1> serialization
           ){  
                auto ptr = new GridRagType(labels, Settings(), DontCompute());
                ptr->deserialize( &serialization(0) );
                return ptr;
           },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("rag"),py::arg("serialization")
        );
    }
    
    
    #ifdef WITH_HDF5
    template<class RAG>
    void exportDeserializeHdf5RagT(py::module & ragModule, const std::string & exportName ){

        typedef RAG GridRagType;
        typedef typename GridRagType::LabelsProxy LabelsProxyType;
        typedef typename GridRagType::DontComputeRag DontCompute;
        typedef typename GridRagType::Settings Settings;

        ragModule.def(exportName.c_str(),
           [](
                LabelsProxyType & labels,
                nifty::marray::PyView<uint64_t,1> serialization
           ){  
                auto ptr = new GridRagType(labels, Settings(), DontCompute());
                ptr->deserialize( &serialization(0) );
                return ptr;
           },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("rag"),py::arg("serialization")
        );
    }
    #endif
    
    
    void exportDeserializeRag(py::module & ragModule) {
        
        // ***************
        // Export Grid Rag
        // ***************            
        typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;
        typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;

        exportDeserializeExplicitRagT<ExplicitLabelsGridRag2D>(ragModule, "deserializeExplicitLabelsGridRag2D");
        exportDeserializeExplicitRagT<ExplicitLabelsGridRag3D>(ragModule, "deserializeExplicitLabelsGridRag3D");

        #ifdef WITH_HDF5
        typedef GridRag<2, Hdf5Labels<2, uint32_t>>  Hdf5LabelsGridRag2D;
        typedef GridRag<3, Hdf5Labels<3, uint32_t>>  Hdf5LabelsGridRag3D;
        exportDeserializeHdf5RagT<Hdf5LabelsGridRag2D>(ragModule, "deserializeGridRag2DHdf5");
        exportDeserializeHdf5RagT<Hdf5LabelsGridRag3D>(ragModule, "deserializeGridRag3DHdf5");
        #endif
        
        
        // ***********************
        // Export Stacked Grid Rag
        // ***********************    
        
        // TODO export StackedGridRag explicit
        
        #ifdef WITH_HDF5
        typedef GridRagStacked2D<Hdf5Labels<3,uint32_t>> Hdf5GridRagStacked2D;
        exportDeserializeHdf5RagT<Hdf5GridRagStacked2D>(ragModule, "deserializeGridRagStacked2DHdf5Impl");
        #endif

    }

    



} // end namespace graph
} // end namespace nifty
