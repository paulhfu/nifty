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

        if(pOpts.getActualNumThreads()<=1){

            marray::Marray<LABEL_TYPE> currentSlice(sliceShape, sliceShape + 3);
            marray::Marray<LABEL_TYPE> nextSlice(sliceShape, sliceShape + 3);

            // get the inner slice edges
            for(size_t z = 0; z < shape[2]; z++) {

                // checkout this slice
                size_t sliceStart[] = {0, 0, z};
                labels.readSubarray(sliceStart, currentSlice);

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

                    }
                }
            }

            rag.transitionEdge_ = rag.numberOfEdges();

            // get the edges in between slices
            for(size_t z = 0; z < shape[2]-1; z++) {
                
                // checkout this slice
                size_t sliceStart[] = {0, 0, z};
                labels.readSubarray(sliceStart, currentSlice);

                // checkout next slice
                size_t nextStart[] = {0, 0, z+1};
                labels.readSubarray(nextStart, nextSlice);
                
                for(size_t x = 0; x < shape[0]; x++) {
                    for(size_t y = 0; y < shape[1]; y++) {
                        
                        // we don't need to check if the labels are different, due to the sliced labels
                        const auto lu = currentSlice(x,y,0);
                        const auto lv = nextSlice(x,y,0);
                        rag.insertEdge(lu,lv);
                    }
                }
            }
        }
        else {

            typedef typename Graph::NodeAdjacency NodeAdjacency;
            typedef typename Graph::EdgeStorage EdgeStorage;
            typedef typename Graph::NodeStorage NodeStorage;
            
            nifty::parallel::ThreadPool threadpool(pOpts);
            struct PerThread{
                std::vector< __setimpl<uint64_t> > adjacency;
            };

            std::vector<PerThread> perThreadDataVec(pOpts.getActualNumThreads());
            for(size_t i=0; i<perThreadDataVec.size(); ++i)
                perThreadDataVec[i].adjacency.resize(numberOfLabels);
            
            // go over the slices in parallel to find inner slice edges
            nifty::parallel::parallel_foreach(threadpool, shape[2], [&](int tid, int z){
            
                // checkout this slice
                marray::Marray<LABEL_TYPE> currentSlice(sliceShape, sliceShape + 3);
                size_t sliceStart[] = {0, 0, size_t(z)};
                labels.readSubarray(sliceStart, currentSlice);
                
                auto & adjacency = perThreadDataVec[tid].adjacency;
                
                for(size_t x = 0; x < shape[0]; x++) {
                    for(size_t y = 0; y < shape[1]; y++) {
                        
                        const auto lu = currentSlice(x,y,0);
                    
                        if(x < shape[0]-1) {
                            const auto lv = currentSlice(x+1,y,0);
                            if(lu != lv) {
                                adjacency[lu].insert(lv);
                                adjacency[lv].insert(lu);
                            }
                        }
                        
                        if(y < shape[1]-1) {
                            const auto lv = currentSlice(x,y+1,0);
                            if(lu != lv) {
                                adjacency[lv].insert(lu);
                                adjacency[lu].insert(lv);
                            }
                        }
                    }
                }
            });
            
            // TODO this is not elegant...
            // get number of inner slice edges and set transition edge
            // copy set 0, make a node storage and merge the node adjacency sets 
            std::vector<NodeStorage> tempNodes;
            tempNodes.resize(numberOfLabels);
            std::vector< __setimpl<uint64_t> > tempAdjacency = perThreadDataVec[0].adjacency;
            
            nifty::parallel::parallel_foreach(threadpool, numberOfLabels, [&](int tid, int label){
                auto & set0 = tempAdjacency[label];
                for(size_t i=1; i<perThreadDataVec.size(); ++i){
                    const auto & setI = perThreadDataVec[i].adjacency[label];
                    set0.insert(setI.begin(), setI.end());
                }
                for(auto otherNode : set0)
                     tempNodes[label].insert(NodeAdjacency(otherNode));
            });

            // count the in-between slice edges
            uint64_t numInnerEdges = 0;
            for(uint64_t u = 0; u< numberOfLabels; ++u){
                for(auto & vAdj :  tempNodes[u]){
                    const auto v = vAdj.node();
                    if(u < v){
                        ++numInnerEdges;
                    }
                }
            }
            
            rag.transitionEdge_ = numInnerEdges;
            
            // go over two consecutive slices in parallel to find between slice edges
            nifty::parallel::parallel_foreach(threadpool, shape[2]-1, [&](int tid, int z){
                
                // checkout this slice
                marray::Marray<LABEL_TYPE> currentSlice(sliceShape, sliceShape + 3);
                size_t sliceStart[] = {0, 0, size_t(z)};
                labels.readSubarray(sliceStart, currentSlice);

                marray::Marray<LABEL_TYPE> nextSlice(sliceShape, sliceShape + 3);
                // checkout next slice
                size_t nextStart[] = {0, 0, size_t(z+1)};
                labels.readSubarray(nextStart, nextSlice);
                
                auto & adjacency = perThreadDataVec[tid].adjacency;
                
                for(size_t x = 0; x < shape[0]; x++) {
                    for(size_t y = 0; y < shape[1]; y++) {
                        
                        // we don't need to check if the labels are different, due to the sliced labels
                        const auto lu = currentSlice(x,y,0);
                        const auto lv = nextSlice(x,y,0);
                        adjacency[lv].insert(lu);
                        adjacency[lu].insert(lv);
                    }
                }
            });
            
            // merge the node adjacency sets for each in-between slice node
            nifty::parallel::parallel_foreach(threadpool, numberOfLabels, [&](int tid, int label){
                auto & set0 = perThreadDataVec[0].adjacency[label];
                for(size_t i=1; i<perThreadDataVec.size(); ++i){
                    const auto & setI = perThreadDataVec[i].adjacency[label];
                    set0.insert(setI.begin(), setI.end());
                }
                for(auto otherNode : set0)
                     rag.nodes_[label].insert(NodeAdjacency(otherNode));
            });

            // insert the  edge index for each in-between slice edge
            uint64_t edgeIndex = 0;
            auto & edges = rag.edges_;
            for(uint64_t u = 0; u< numberOfLabels; ++u){
                for(auto & vAdj :  rag.nodes_[u]){
                    const auto v = vAdj.node();
                    if(u < v){
                        edges.push_back(EdgeStorage(u, v));
                        vAdj.changeEdgeIndex(edgeIndex);
                        auto fres =  rag.nodes_[v].find(NodeAdjacency(u));
                        fres->changeEdgeIndex(edgeIndex);
                        ++edgeIndex;
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
