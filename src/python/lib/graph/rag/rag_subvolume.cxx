#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_extract_subvolume.hxx"
    
#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif



namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    // This has gotten really ugly - I don't really know how to best handle nparrrays / marrays of prior unknown size
    template<class RAG>
    void exportExtractNodesAndEdgesFromNodeListT(py::module & ragModule) {

        typedef typename RAG::LabelType LabelType;
        typedef typename RAG::EdgeType  EdgeType;
        
        // TODO this should be exported only once for multiple exports
        // this introduces some data copys... maybe can be rid of this, but I don't know if we won't run into mem errors then...
        struct ExtractNodesReturnType {
            ExtractNodesReturnType(const std::vector<EdgeType> & innerEdges, const std::vector<EdgeType> & outerEdges, const std::vector<std::pair<LabelType,LabelType>> & uvIds)
                : innerEdges_(innerEdges), outerEdges_(outerEdges), uvIds_(uvIds) {
        }

            const std::vector<EdgeType> & getInnerEdges() const {
                return innerEdges_;
            }
            
            const std::vector<EdgeType> & getOuterEdges() const {
                return outerEdges_;
            }
            
            const std::vector<std::pair<LabelType,LabelType>> & getUvIds() const {
                return uvIds_;
            }

        private:
            std::vector<EdgeType> innerEdges_;
            std::vector<EdgeType> outerEdges_;
            std::vector<std::pair<LabelType,LabelType>> uvIds_;
        };
        
        py::class_<ExtractNodesReturnType>(ragModule, "ExtractNodesReturnType")
            // need to def init ?
            .def("innerEdges",[](const ExtractNodesReturnType & self) {
                const auto & in = self.getInnerEdges();
                marray::PyView<EdgeType,1> out({in.size()});
                for(size_t i = 0; i < in.size(); ++i)
                    out(i) = in[i];
                return out;
            })
            .def("outerEdges",[](const ExtractNodesReturnType & self) {
                const auto & in = self.getOuterEdges();
                marray::PyView<EdgeType,1> out({in.size()});
                for(size_t i = 0; i < in.size(); ++i)
                    out(i) = in[i];
                return out;
            })
            .def("uvIds",[](const ExtractNodesReturnType & self) {
                const auto & in = self.getUvIds();
                marray::PyView<LabelType,2> out({in.size(),uint64_t(2)});
                for(size_t i = 0; i < in.size(); ++i) {
                    out(i,0) = in[i].first;
                    out(i,1) = in[i].second;
                }
                return out;
            })
        ;


        ragModule.def("extractNodesAndEdgesFromNodeListImpl",
           [](  const RAG & rag,
                const marray::PyView<LabelType,1> nodeList
                ) {  
                
                std::vector<EdgeType> innerEdgesVec;  
                std::vector<EdgeType> outerEdgesVec;  
                std::vector<std::pair<LabelType,LabelType>> uvIdsVec;  
                {
                    py::gil_scoped_release allowThreads;
                    extractNodesAndEdgesFromNodeList(rag, nodeList, innerEdgesVec, outerEdgesVec, uvIdsVec);
                }
                
                return ExtractNodesReturnType(innerEdgesVec, outerEdgesVec, uvIdsVec);
           },
        py::arg("rag"),py::arg("nodeList")
        );
    }


    void exportExtractNodesAndEdgesFromNodeList(py::module & ragModule) {

        // TODO extract for more rags        

        #ifdef WITH_HDF5
        typedef GridRagStacked2D<Hdf5Labels<3,uint32_t>> Hdf5GridRagStacked2D;
        exportExtractNodesAndEdgesFromNodeListT<Hdf5GridRagStacked2D>(ragModule);
        #endif

    }




} // end namespace graph
} // end namespace nifty
