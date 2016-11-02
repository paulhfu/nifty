#include <pybind11/numpy.h>

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    typedef UndirectedGraph<> Graph;
    
    // this introduces some data copys... maybe can be rid of this, but I don't know if we won't run into mem errors then...
    struct SubgraphReturnType {
        SubgraphReturnType(const std::vector<int64_t> & innerEdges, const std::vector<int64_t> & outerEdges, const Graph & graph)
            : innerEdges_(innerEdges), outerEdges_(outerEdges), graph_(graph) {
    }
    
        const std::vector<int64_t> & getInnerEdges() const {
            return innerEdges_;
        }
        
        const std::vector<int64_t> & getOuterEdges() const {
            return outerEdges_;
        }
        
        const Graph & getSubgraph() const {
            return graph_;
        }
    
    private:
        std::vector<int64_t> innerEdges_;
        std::vector<int64_t> outerEdges_;
        Graph graph_;
    };
    
    
    void exportSubgraphReturnType(py::module & graphModule) {
        py::class_<SubgraphReturnType>(graphModule, "SubgraphReturnType")
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
            .def("subgraph",[](const SubgraphReturnType & self) {
                return self.getSubgraph();
            })
        ;
    }

}
}
