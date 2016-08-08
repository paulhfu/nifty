#pragma once
#ifndef NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HDF5_HXX
#define NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HDF5_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"
//


namespace nifty{
namespace graph{

    template<
        class LABELS_TYPE, 
        class NODE_MAP,
        class SCALAR_TYPE
    >
    void projectScalarNodeDataToPixels(
        const GridRagStacked2d<LABELS_TYPE> & graph,
        NODE_MAP & nodeData,
        nifty::hdf5::Hdf5Array<SCALAR_TYPE> & pixelData,
        const int numberOfThreads = -1
    ){
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});
        
        { 
            // allocate / create data for each thread
            struct PerThreadData{
                marray::Marray<LABEL_TYPE> sliceLabels;
                marray::Marray<LABELS> sliceData;
            };
            
            std::vector<PerThreadData> perThreadDataVec(nThreads);
            parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
                perThreadDataVec[i].sliceLabels.resize(sliceShape3.begin(), sliceShape3.end());
                perThreadDataVec[i].sliceData.resize(sliceShape3.begin(), sliceShape3.end());
            });
            
            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

                // fetch the data for the slice
                auto & sliceLabelsFlat3D = perThreadDataVec[tid].sliceLabels;
                auto & sliceDataFlat3D   = perThreadDataVec[tid].sliceData;
                
                const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3D);
                auto sliceLabels = sliceLabelsFlat3D.squeezedView();
                
                auto sliceData = sliceDataFlat3D.squeezedView();
                
                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
            
                    const auto node = sliceLabels(coord.asStdArray());
                    sliceData(coord.asStdArray()) = nodeData[node];
                
                });
                
                pixelData.writeSubarray(blockBegin,sliceDataFlat3d);

            });
        }
}




} // namespace graph
} // namespace nifty


#endif /* NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HDF5_HXX */
