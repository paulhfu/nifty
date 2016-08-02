#pragma once
#ifndef NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_CHUNKED_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"
//
#include "nifty/hdf5/hdf5_array.hxx"


namespace nifty{
namespace graph{

template<
    class LABELS_TYPE, 
    class NODE_MAP,
    class SCALAR_TYPE
>
void projectScalarNodeDataToPixels(
    const ChunkedLabelsGridRagSliced<LABELS_TYPE> & graph,
    NODE_MAP & nodeData,
    nifty::hdf5::Hdf5Array<SCALAR_TYPE> & pixelData,
    const int numberOfThreads = -1
){
    typedef std::array<int64_t, 2> Coord;

    const auto & labelsProxy = graph.labelsProxy();
    const auto & shape = labelsProxy.shape();
    const auto & labels = labelsProxy.labels(); 
        
    size_t sliceShape[] = { size_t(shape[0]), size_t(shape[1]), 1};

    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpt);
    
    // TODO better to parallelize inside slice or over the slices ?!
    //if(pOpts.getActualNumThreads()<=1){
    
    marray::Marray<LABELS_TYPE> currentLabels(sliceShape, sliceShape+3);
    marray::Marray<SCALAR_TYPE> currentData(sliceShape, sliceShape+3);
    
    for(size_t z = 0; z < shape[2]; z++) {
        
        // checkout this slice
        size_t sliceStart[] = {0,0,z};
        labels.readSubarray(sliceStart, currentLabels);

        nifty::tools::parallelForEachCoordinate(threadpool,std::array<int64_t,2>({ int64_t(shape[0]), int64_t(shape[1]) }) ,
        [&](int tid, const Coord & coord){
            const auto x = coord[0];
            const auto y = coord[1];
            const auto node = currentLabels(x,y,0);
            currentData(x,y,0) = nodeData[node];
        });
        pixelData.writeSubarray(sliceStart,currentData);
    }
    //else {
    //    
    //    nifty::parallel::parallel_foreach(threadpool, shape[2], [&](int tid, int z) {
    //    
    //        marray::Marray<LABELS_TYPE> currentLabels(sliceShape, sliceShape+3);
    //        marray::Marray<SCALAR_TYPE> currentData(sliceShape, sliceShape+3);
    //        
    //        // checkout this slice
    //        size_t sliceStart[] = {0,0,z};
    //        labels.readSubarray(sliceStart, currentLabels);
    //        
    //        for(size_t x = 0; x < shape[0]; x++ ) {
    //            for(size_t y = 0; y < shape[1]; y++ ) {
    //                const auto node = currentLabels(x,y,0);
    //                currentData(x,y,0) = nodeData[node];
    //            }
    //        }
    //    
    //        pixelData.writeSubarray(sliceStart,currentData);
    //    });
    //}

}




} // namespace graph
} // namespace nifty


#endif /* NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_CHUNKED_HXX */
