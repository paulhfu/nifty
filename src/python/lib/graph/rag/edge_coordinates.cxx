#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag_edge_coordinates.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    using namespace py;

    template<size_t DIM, class RAG>
    void exportEdgeCoordinates(
        py::module & ragModule
    ){
        ragModule.def("edgeCoordinates",
        [](
            const RAG & rag,
            const int numberOfThreads
        ){

            typedef array::StaticArray<int64_t, DIM> Coord;
            std::vector<std::vector<Coord>> coordinatesOut(rag.edgeIdUpperBound()+1);
            {
                py::gil_scoped_release allowThreads;
                getEdgeCoordinates(rag, coordinatesOut, numberOfThreads);
            }
            return coordinatesOut;
        },
        py::arg("rag"),
        py::arg("numberOfThreads")= -1
        );
    }
    
    template<size_t DIM, class RAG, class T>
    void exportRenderEdges(
        py::module & ragModule
    ){
        ragModule.def("renderEdges",
        [](
            const RAG & rag,
            nifty::marray::PyView<T, 1> edgeData,
            const int numberOfThreads
        ){

            const auto & shape = rag.labelsProxy().shape();
            
            nifty::marray::PyView<T, DIM> volOut(shape.begin(),shape.end());
            {
                py::gil_scoped_release allowThreads;
                renderEdges(rag, edgeData, volOut, numberOfThreads);
            }
            return volOut;
        },
        py::arg("rag"),
        py::arg("edgeData"),
        py::arg("numberOfThreads")= -1
        );
    }
    
    
    void exportEdgeCoordinates(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d32;
            //typedef ExplicitLabelsGridRag<3, uint64_t> Rag3d64;

            exportEdgeCoordinates<3, Rag3d32>(ragModule);
            exportRenderEdges<3, Rag3d32, uint32_t>(ragModule);
            //exportEdgeCoordinates<3, Rag3d64>(ragModule);
        }
    }

} // end namespace graph
} // end namespace nifty
