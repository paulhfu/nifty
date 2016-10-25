#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_EDGE_COORDINATES_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_EDGE_COORDINATES_HXX

#include <vector>
#include <cmath>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace graph{
    
    template<size_t DIM, class LABELS_PROXY>
    void getEdgeCoordinates(
        const GridRag<DIM, LABELS_PROXY> & rag,
        std::vector<std::vector<array::StaticArray<int64_t, DIM>>> & coordinatesOut,
        const int numberOfThreads = -1
    ){
        typedef array::StaticArray<int64_t, DIM> Coord;
        
        const auto & shape = rag.shape();
        const auto & labels = rag.labelsProxy().labels();
        
        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();
                
        nifty::tools::parallelForEachCoordinate(
            threadpool,
            shape
            ,[&](int tid, const Coord & coordU){
                
                const auto lU = labels(coordU.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    auto coordV = makeCoord2(coordU, axis);
                    if(coordV[axis] < shape[axis]){
                        const auto lV = labels(coordV.asStdArray());
                        if(lU != lV){
                            const auto edge = rag.findEdge(lU,lV);
                            coordinatesOut[edge].push_back(coordU);
                            coordinatesOut[edge].push_back(coordV);
                        }
                    }
                }
        });
    }
    
    
    template<size_t DIM, class LABELS_PROXY, class DATA_TYPE>
    void renderEdges(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const marray::View<DATA_TYPE> & edgeData,
        marray::View<DATA_TYPE> & volOut,
        const int numberOfThreads = -1
    ){
        typedef array::StaticArray<int64_t, DIM> Coord;
        
        const auto & shape = rag.shape();
        const auto & labels = rag.labelsProxy().labels();
        
        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();
                
        nifty::tools::parallelForEachCoordinate(
            threadpool,
            shape
            ,[&](int tid, const Coord & coordU){
                
                const auto lU = labels(coordU.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    auto coordV = makeCoord2(coordU, axis);
                    if(coordV[axis] < shape[axis]){
                        const auto lV = labels(coordV.asStdArray());
                        if(lU != lV){
                            const auto edge = rag.findEdge(lU,lV);
                            const auto edgeVal = edgeData(edge);
                            volOut(coordU.asStdArray()) = edgeVal; 
                            volOut(coordV.asStdArray()) = edgeVal;
                        }
                    }
                }
        });
    }


} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX */
