#pragma once
#ifndef NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX
#define NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/array/arithmetic_array.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#endif

//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{



template<
    size_t DIM, 
    class LABELS_TYPE, 
    class PIXEL_ARRAY, 
    class NODE_MAP
>
void projectScalarNodeDataToPixels(
    const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
    NODE_MAP & nodeData,
    PIXEL_ARRAY & pixelData,
    const int numberOfThreads = -1
){
    typedef array::StaticArray<int64_t, DIM> Coord;

    const auto labelsProxy = graph.labelsProxy();
    const auto & shape = labelsProxy.shape();
    const auto labels = labelsProxy.labels(); 
    

    // if scalar 
    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpt);
    nifty::tools::parallelForEachCoordinate(threadpool, shape,
    [&](int tid, const Coord & coord){
        const auto node = labels(coord.asStdArray());
        pixelData(coord.asStdArray()) = nodeData[node];
    });

}

#ifdef WITH_HDF5
template<
    size_t DIM, 
    class LABELS_TYPE, 
    class PIXEL_ARRAY, 
    class NODE_MAP,
    class COORD
>
void projectScalarNodeDataToPixels(
    const Hdf5LabelsGridRag<DIM, LABELS_TYPE> & graph,
    NODE_MAP & nodeData,
    PIXEL_ARRAY & pixelData,
    const COORD & start,
    const int numberOfThreads = -1
){
    typedef array::StaticArray<int64_t, DIM> Coord;

    const auto & labelsProxy = graph.labelsProxy();
    
    Coord shape;
    COORD stop(3);
    for(int d = 0; d < DIM; d++) {
        shape[d] = pixelData.shape(d);
        stop[d] = start[d] + shape[d];
    }

    marray::Marray<LABELS_TYPE> labels(shape.begin(), shape.end());
    labelsProxy.readSubarray(start, stop, labels);
    
    // if scalar 
    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpt);
    nifty::tools::parallelForEachCoordinate(threadpool, shape,
    [&](int tid, const Coord & coord){
        const auto node = labels(coord.asStdArray());
        pixelData(coord.asStdArray()) = nodeData[node];
    });

}
#endif


} // namespace graph
} // namespace nifty



#endif /* NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX */
