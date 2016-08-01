#pragma once
#ifndef NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_CHUNKED_HXX


#include <random>
#include <functional>
#include <ctime>
#include <stack>
#include <algorithm>
#include <unordered_set>

// for strange reason travis does not find the boost flat set
#ifdef WITHIN_TRAVIS
#include <set>
#define __setimpl std::set
#else
#include <boost/container/flat_set.hpp>
#define __setimpl boost::container::flat_set
#endif


#include "nifty/graph/rag/grid_rag_labels_chunked.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{


template<class LABELS_PROXY>
class GridRagSliced;


namespace detail_rag{

template<class LABEL_TYPE>
struct ComputeRag< GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> > {
    
    static void computeRag(
        GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> & rag,
        const typename GridRagSliced<ChunkedLabels<3, LABEL_TYPE>>::Settings & settings ){
        
        typedef GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> Graph;
        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);

        const auto & labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto & labels = labelsProxy.labels(); 
        const auto & shape = labels.shape();

        rag.assign(numberOfLabels);

        size_t sliceShape[] = {shape[0], shape[1], 1};

        marray::Marray<LABEL_TYPE> currentSlice(sliceShape, sliceShape + 3);
        marray::Marray<LABEL_TYPE> nextSlice(sliceShape, sliceShape + 3);

        // TODO parallel versions of the code

        for(size_t z = 0; z < shape[2]; z++) {

            // checkout this slice
            size_t sliceStart[] = {0, 0, z};
            labels.readSubarray(sliceStart, currentSlice);

            if(z < shape[2] - 1) {
                // checkout next slice
                size_t nextStart[] = {0, 0, z+1};
                labels.readSubarray(nextStart, nextSlice);
            }

            // single core
            if(pOpts.getActualNumThreads()<=1){
            
                for(size_t x = 0; x < shape[0]; x++) {
                    for(size_t y = 0; y < shape[1]; y++) {
                        
                        const auto lu = currentSlice(x,y,0);
                        
                        if(x < shape[0]-1) {
                            const auto lv = currentSlice(x+1,y,0);
                            if(lu != lv)
                                rag.insertEdge(lu,lv);
                        }
                        
                        if(y < shape[1]-1) {
                            const auto lv = currentSlice(x,y+1,0);
                            if(lu != lv)
                                rag.insertEdge(lu,lv);
                        }

                        if(z < shape[2]-1) {
                        // we don't need to check if the labels are different, due to the sliced labels
                            const auto lv = nextSlice(x,y,0);
                            rag.insertEdge(lu,lv);
                        }
                    }
                }
            }
        }
    }


};


} // end namespace detail_rag
} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_CHUNKED_HXX */
