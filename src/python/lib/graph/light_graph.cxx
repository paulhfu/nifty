
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/graph/light_graph.hxx"


namespace py = pybind11;

namespace nifty {
namespace graph {

    void exportLightGraph(py::module & module) {

        py::class_<LightGraph>(module, "LightGraph")
            .def_property_readonly("numberOfNodes", &LightGraph::numberOfNodes)
            .def_property_readonly("numberOfEdges", &LightGraph::numberOfEdges)

            .def("findEdge", &LightGraph::findEdge)   // TODO lift gil

            .def("findEdges", [](const LightGraph & self,
                                 const xt::pytensor<NodeType, 2> uvs){
                typedef xt::pytensor<EdgeIndexType, 1> OutType;
                typedef typename OutType::shape_type OutShape;
                OutShape shape = {uvs.shape()[0]};
                OutType out(shape);
                {
                    py::gil_scoped_release allowThreads;
                    for(size_t i = 0; i < shape[0]; ++i) {
                        // FIXME: why not a second index here...???
                        out[i] = self.findEdge(uvs(i,0), uvs(i,1));
                    }
                }
                return out;
            })

            .def("uvIds", [](const LightGraph & self){
                typedef xt::pytensor<NodeType, 2> OutType;
                typedef typename OutType::shape_type OutShape;
                OutShape shape = {static_cast<int64_t>(self.numberOfEdges()), 2L};
                OutType out(shape);
                {
                    py::gil_scoped_release allowThreads;
                    const auto & edges = self.edges();
                    size_t edgeId = 0;
                    for(const auto edge : edges) {
                        out(edgeId, 0) = edge.first;
                        out(edgeId, 1) = edge.second;
                        ++edgeId;
                    }
                }
                return out;
            })

            .def("nodes", [](const LightGraph & self){
                typedef xt::pytensor<NodeType, 1> OutType;
                xt::pytensor<NodeType, 1> nodes = xt::zeros<NodeType>({self.numberOfNodes()});
                {
                    py::gil_scoped_release allowThreads;
                    std::set<NodeType> nodesTmp;
                    self.nodes(nodesTmp);

                    size_t nodeId = 0;
                    for(const auto node : nodesTmp) {
                        nodes(nodeId) = node;
                        ++nodeId;
                    }
                }
                return nodes;
            })

            ;

           module.def("lightGraph",
               [](
               const size_t number_of_labels,
               const xt::pytensor<NodeType, 2> uvs
            ){
               {
                   py::gil_scoped_release allowThreads;
                   auto ptr = new LightGraph(number_of_labels, uvs);
                   return ptr;
               }
            },
//            py::return_value_policy::take_ownership,
//            py::keep_alive<0, 1>(),
            py::arg("number_of_labels"), py::arg("uvs")
            );
    }


}
}

