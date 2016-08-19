#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/graph/undirected_list_graph.hxx"

#include "export_undirected_graph_class_api.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportUndirectedListGraph(py::module & graphModule) {

        typedef UndirectedGraph<> Graph;
        const auto clsName = std::string("UndirectedGraph");
        
        // this introduces some data copys... maybe can be rid of this, but I don't know if we won't run into mem errors then...
        struct SubgraphReturnType {
            SubgraphReturnType(const std::vector<int64_t> & innerEdges, const std::vector<int64_t> & outerEdges, const std::vector<std::pair<int64_t,int64_t>> & uvIds)
                : innerEdges_(innerEdges), outerEdges_(outerEdges), uvIds_(uvIds) {
        }

            const std::vector<int64_t> & getInnerEdges() const {
                return innerEdges_;
            }
            
            const std::vector<int64_t> & getOuterEdges() const {
                return outerEdges_;
            }
            
            const std::vector<std::pair<int64_t,int64_t>> & getUvIds() const {
                return uvIds_;
            }

        private:
            std::vector<int64_t> innerEdges_;
            std::vector<int64_t> outerEdges_;
            std::vector<std::pair<int64_t,int64_t>> uvIds_;
        };
        
        
        py::class_<SubgraphReturnType>(graphModule, "SubgraphReturnType")
            // need to def init ?
            .def("innerEdges",[](const SubgraphReturnType & self) {
                const auto & in = self.getInnerEdges();
                marray::PyView<int64_t,1> out({in.size()});
                for(size_t i = 0; i < in.size(); ++i)
                    out(i) = in[i];
                return out;
            })
            .def("outerEdges",[](const SubgraphReturnType & self) {
                const auto & in = self.getOuterEdges();
                marray::PyView<int64_t,1> out({in.size()});
                for(size_t i = 0; i < in.size(); ++i)
                    out(i) = in[i];
                return out;
            })
            .def("uvIds",[](const SubgraphReturnType & self) {
                const auto & in = self.getUvIds();
                marray::PyView<int64_t,2> out({in.size(),uint64_t(2)});
                for(size_t i = 0; i < in.size(); ++i) {
                    out(i,0) = in[i].first;
                    out(i,1) = in[i].second;
                }
                return out;
            })
        ;
        
        auto undirectedGraphCls = py::class_<Graph>(graphModule, clsName.c_str());

        undirectedGraphCls
            .def(py::init<const uint64_t,const uint64_t>(),
               py::arg_t<uint64_t>("numberOfNodes",0),
               py::arg_t<uint64_t>("reserveEdges",0)
            )
            .def("insertEdge",&Graph::insertEdge)
            .def("insertEdges",
                [](Graph & g, nifty::marray::PyView<uint64_t> array) {
                    NIFTY_CHECK_OP(array.dimension(),==,2,"wrong dimensions");
                    NIFTY_CHECK_OP(array.shape(1),==,2,"wrong shape");
                    for(size_t i=0; i<array.shape(0); ++i){
                        g.insertEdge(array(i,0),array(i,1));
                    }
                }
            )
            .def("uvIds",
                [](Graph & g) {
                    nifty::marray::PyView<uint64_t> out({uint64_t(g.numberOfEdges()), uint64_t(2)});
                    for(const auto edge : g.edges()){
                        const auto uv = g.uv(edge); 
                        out(edge,0) = uv.first;
                        out(edge,1) = uv.second;
                    }
                    return out;
                }
            )
            .def("serialize",
                [](const Graph & g) {
                    nifty::marray::PyView<uint64_t> out({g.serializationSize()});
                    auto ptr = &out(0);
                    g.serialize(ptr);
                    return out;
                }
            )
            .def("deserialize",
                [](Graph & g, nifty::marray::PyView<uint64_t,1> serialization) {
                    auto ptr = &serialization(0);
                    g.deserialize(ptr);
                }
            )
            .def("extractSubgraphFromNodesImpl",
                []( Graph & g, const marray::PyView<int64_t,1> nodeList) {
                    
                    std::vector<int64_t> innerEdgesVec;  
                    std::vector<int64_t> outerEdgesVec;  
                    std::vector<std::pair<int64_t,int64_t>> uvIdsVec;  
                    {
                        py::gil_scoped_release allowThreads;
                        g.extractSubgraphFromNodes(nodeList, innerEdgesVec, outerEdgesVec, uvIdsVec);
                    }
                    
                    return SubgraphReturnType(innerEdgesVec, outerEdgesVec, uvIdsVec);

                }
            )
        ;

        // export the base graph API (others might derive)
        exportUndirectedGraphClassAPI<Graph>(graphModule, undirectedGraphCls,clsName);


    }

}
}
