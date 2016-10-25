#pragma once
#ifndef NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX
#define NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX

#include <vector>



#include "nifty/container/boost_flat_set.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/tools/for_each_block.hxx"

namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class Hdf5Labels;

template<size_t DIM, class LABELS_PROXY>
class GridRag;

template<class LABEL_TYPE>
class GridRagStacked2D;


// \cond SUPPRESS_DOXYGEN
namespace detail_rag{

template< class GRID_RAG>
struct ComputeRag;


template<size_t DIM, class LABEL_TYPE>
struct ComputeRag< GridRag<DIM,  Hdf5Labels<DIM, LABEL_TYPE> > > {

    template<class S>
    static void computeRag(
        GridRag<DIM,  Hdf5Labels<DIM, LABEL_TYPE> > & rag,
        const S & settings
    ){


        typedef array::StaticArray<int64_t, DIM> Coord;


        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        

        const auto numberOfLabels = labelsProxy.numberOfLabels();
        rag.assign(numberOfLabels);


        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();


        // allocate / create data for each thread
        Coord blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(settings.blockShape[d]+1, shape[d]);
        }
        struct PerThreadData{
            marray::Marray<LABEL_TYPE> blockLabels;
            std::vector< container::BoostFlatSet<uint64_t> > adjacency;
        };
        std::vector<PerThreadData> perThreadDataVec(nThreads);
        parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
            perThreadDataVec[i].blockLabels.resize(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
            perThreadDataVec[i].adjacency.resize(numberOfLabels);
        });
        
        
        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };


        const Coord overlapBegin(0), overlapEnd(1);
        const Coord zeroCoord(0);
        tools::parallelForEachBlockWithOverlap(threadpool,shape, settings.blockShape, overlapBegin, overlapEnd,
        [&](
            const int tid,
            const Coord & blockCoreBegin, const Coord & blockCoreEnd,
            const Coord & blockBegin, const Coord & blockEnd
        ){
            const Coord actualBlockShape = blockEnd - blockBegin;
            auto blockLabels = perThreadDataVec[tid].blockLabels.view(zeroCoord.begin(), actualBlockShape.begin());

            Coord marrayShape;
            Coord viewShape;

            for(auto d=0; d<DIM; ++d){
                marrayShape[d] = perThreadDataVec[tid].blockLabels.shape(d);
                viewShape[d] = blockLabels.shape(d);
            }


            labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);

            auto & adjacency = perThreadDataVec[tid].adjacency;

            nifty::tools::forEachCoordinate(actualBlockShape,[&](const Coord & coord){
                const auto lU = blockLabels(coord.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    const auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < actualBlockShape[axis]){
                        const auto lV = blockLabels(coord2.asStdArray());
                        //std::cout<<"lU "<<lU<<" lV"<<lV<<"\n";
                        if(lU != lV){
                            //std::cout<<"HUTUHUT\n";
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });
        });

        rag.mergeAdjacencies(perThreadDataVec, threadpool);

    }

    // TODO need to figure out how to best parallelize this
    // FIXME unify COORD and Coord!
    template<class S, class COORD>
    static void computeRag(
        GridRag<DIM,  Hdf5Labels<DIM, LABEL_TYPE> > & rag,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        const S & settings
    ){

        typedef array::StaticArray<int64_t, DIM> Coord;

        typedef LABEL_TYPE LabelType;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        

        const auto numberOfLabels = labelsProxy.numberOfLabels();
        rag.assign(numberOfLabels);


        // single thread for debugging
        //nifty::parallel::ParallelOptions pOpts(1);
        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        // allocate / create data for each thread
        Coord blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(blockShape[d]+1, shape[d]);
        }
        
        struct PerThreadData{
            std::vector< container::BoostFlatSet<uint64_t> > adjacency;
        };
        std::vector<PerThreadData> perThreadDataVec(nThreads);
        
        parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
            perThreadDataVec[i].adjacency.resize(numberOfLabels);
        });
        
        
        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        const size_t nBlocks = startCoordinates.size();

        marray::Marray<LabelType> blockLabels(blockShapeWithBorder.begin(),blockShapeWithBorder.end());
        
        for(size_t blockId = 0; blockId < nBlocks; blockId++ ) {
            std::cout << "Block: " << blockId << std::endl;

            Coord blockBegin;
            Coord blockEnd;

            for(auto d=0; d<DIM; ++d){
                blockBegin[d] = startCoordinates[blockId][d];
                blockEnd[d]   = startCoordinates[blockId][d] + blockShapeWithBorder[d];
            }

            // need to try catch in case of violating overlaps
            bool overlapViolates = false;
            try { 
                labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);
            }
            catch (const std::runtime_error & e) {
                std::cout << "Overlap violating and reduced" << std::endl;
                overlapViolates = true;
                for(auto d=0; d<DIM; ++d) {
                    blockEnd[d] -= 1;
                    blockShapeWithBorder[d] -= 1;
                }
                blockLabels = marray::Marray<LabelType>(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);
            }

            nifty::tools::parallelForEachCoordinate(
                threadpool,
                blockShapeWithBorder,
                [&](int tid, const Coord & coord) {

                    auto & adjacency = perThreadDataVec[tid].adjacency;
                    
                    const auto lU = blockLabels(coord.asStdArray());
                    
                    for(size_t axis=0; axis<DIM; ++axis){
                        const auto coord2 = makeCoord2(coord, axis);
                        if(coord2[axis] < blockShapeWithBorder[axis]){
                            const auto lV = blockLabels(coord2.asStdArray());
                            if(lU != lV){
                                // FIXME this crashes for non-contiguous labels!
                                adjacency[lV].insert(lU);
                                adjacency[lU].insert(lV);
                            }
                        }
                    }
            });
            
            // reset shape and marray if overlap was violating
            if(overlapViolates) {
                for(auto d=0; d<DIM; ++d)
                    blockShapeWithBorder[d] += 1;
                blockLabels = marray::Marray<LabelType>(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
            }
        }

        rag.mergeAdjacencies(perThreadDataVec, threadpool);

    }


};



} // end namespace detail_rag
// \endcond

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX */
