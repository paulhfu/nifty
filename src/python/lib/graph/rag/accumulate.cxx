#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"

#ifdef WITH_FASTFILTERS
#include "nifty/graph/rag/grid_rag_accumulate_filters.hxx"
#endif

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif

namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<size_t DIM, class RAG, class DATA_T>
    void exportAccumulateEdgeMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeMeanAndLength",
        [](
            const RAG & rag,
            nifty::marray::PyView<DATA_T, DIM> data,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){

            nifty::marray::PyView<DATA_T> out({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(2)});
            array::StaticArray<int64_t, DIM> blocKShape_;
            accumulateEdgeMeanAndLength(rag, data, blocKShape, out, numberOfThreads);
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }


    template<size_t DIM, class RAG, class DATA_T>
    void exportAccumulateMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateMeanAndLength",
        [](
            const RAG & rag,
            nifty::marray::PyView<DATA_T, DIM> data,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;

            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(2)});
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(2)});
            array::StaticArray<int64_t, DIM> blocKShape_;
            accumulateMeanAndLength(rag, data, blocKShape, edgeOut, nodeOut, numberOfThreads);

            return OutType(edgeOut, nodeOut);;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }
    
    #ifdef WITH_FASTFILTERS
    template<size_t DIM, class RAG, class DATA>
    void exportAccumulateEdgeStatisticsFromFilters(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStatisticsFromFilters",
        [](
            const RAG & rag,
            DATA & data,
            const int numberOfThreads
        ){
            typedef typename DATA::DataType DataType;
            
            // it is: NSigma (3) * NFilter (4/5) * NStatistics (10) = 120 / 150
            // TODO get this magic number from somewhere reliably
            // or resize in function
            // TODO condition on type of rag and dimension
            uint64_t numberOfChannels = 120; // 150 for 3d normal gridrag
            //uint64_t numberOfChannels = 144; // 150 for 3d normal gridrag
            nifty::marray::PyView<DataType> out({uint64_t(rag.edgeIdUpperBound()+1),numberOfChannels});
            //accumulateEdgeStatisticsFromFiltersTwoPass<DIM>(rag, data, out, numberOfThreads);
            accumulateEdgeStatisticsFromFilters<DIM>(rag, data, out, numberOfThreads);
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("numberOfThreads")= -1
        );
    }
    #endif



    void exportAccumulate(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> Rag2d;
            typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d;

            exportAccumulateEdgeMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateEdgeMeanAndLength<3, Rag3d, float>(ragModule);
            exportAccumulateMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateMeanAndLength<3, Rag3d, float>(ragModule);
        }
            
        // stacked hdf5 rag
        #ifdef WITH_HDF5
        {
            typedef GridRagStacked2D<Hdf5Labels<3,uint32_t>> Hdf5GridRagStacked2D;
            #ifdef WITH_FASTFILTERS
            exportAccumulateEdgeStatisticsFromFilters<3, Hdf5GridRagStacked2D, hdf5::Hdf5Array<float>>(ragModule);
            #endif
        }
        #endif
    }

} // end namespace graph
} // end namespace nifty
    
