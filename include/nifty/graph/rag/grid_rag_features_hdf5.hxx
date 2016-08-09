#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HDF5_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HDF5_HXX

#include "nifty/graph/rag/grid_rag_hdf5.hxx"
//#include "nifty/graph/rag/feature_functors.hxx"


namespace nifty{
namespace graph{
    
    /*
    // TODO clear API definition + implementing the functor
    template< size_t DIM, class LABELS_TYPE, class T, class EDGE_MAP, class FEATURE_FUNCTOR>
    void gridRagAccumulateFeatures(
        const GridRagStacked2d<DIM, LABELS_TYPE> & graph, // the stacked rag
        const nifty::hdf5::Hdf5Array<T> & data,  // the raw data
        EDGE_MAP & edgeMap,         // the edge map TODO change API?!
        FEATURE_FUNCTOR calculateFeature,     // functor to caclulate the feature on the fly
        const int numberOfThreads = -1
    ){
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();

        // TODO do we need this
        std::vector< std::vector<T> > edgeValues;
        edgeValues.resize(rag.numberOfEdges(),std::vector<T>());
        
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});
        
        /////////////////////////////////////////////////////
        // Phase 1 : Accumulate in-slice edge features
        /////////////////////////////////////////////////////
        
        std::cout<<"phase 1\n";
        { 
            // allocate / create data for each thread
            struct PerThreadData{
               marray::Marray<LABEL_TYPE> sliceLabels;
               marray::Marray<T> sliceData;
               marray::Marray<T> sliceFeatures;
            };
            
            std::vector<PerThreadData> perThreadDataVec(nThreads);
            parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
                perThreadDataVec[i].sliceLabels.resize(sliceShape3.begin(), sliceShape3.end());
                perThreadDataVec[i].sliceData.resize(sliceShape3.begin(), sliceShape3.end());
                // TODO need to change this for multichannel features
                perThreadDataVec[i].sliceFeatures.resize(sliceShape3.begin(), sliceShape3.end());
            });

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

                // fetch the data for the slice
                auto & sliceLabelsFlat3D = perThreadDataVec[tid].sliceLabels;
                auto & sliceDataFlat3D   = perThreadDataVec[tid].sliceData;
                
                const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3D);
                auto sliceLabels = sliceLabelsFlat3D.squeezedView();
                
                data.readSubarray(blockBegin, blockEnd, sliceDataFlat3d);
                auto sliceData = sliceDataFlat3D.squeezedView();
                
                auto & sliceFeatures = perThreadDataVec[tid].sliceFeatures;
                calculateFeature(sliceFeatures);

                // do the thing 
                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                    const auto lU = sliceLabels(coord.asStdArray());
                    const auto dU = sliceFeatures(coord.asStdArray());
                    for(size_t axis=0; axis<2; ++axis){
                        Coord2 coord2 = coord;
                        ++coord2[axis];
                        if(coord2[axis] < sliceShape2[axis]){
                            const auto lV = sliceLabels(coord2.asStdArray());
                            const auto dV = sliceFeatures(coord2.asStdArray());
                            const auto e = graph.findEdge(lU, lV);
                            edgeMap.accumulate(e, dU);
                            edgeMap.accumulate(e, dV);
                        }
                    }
                });
            });
        }

    }
    */

    
    template<class LABELS_TYPE, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const GridRagStacked2D<LABELS_TYPE> & graph, // the stacked rag
        const nifty::hdf5::Hdf5Array<LABELS> & data,  // the raw data
        NODE_MAP &  nodeMap,
        const int numberOfThreads = -1
    ){
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});

        /*
        // check that the data covers a whole slice in xy
        NIFTY_CHECK_OP(data.shape(0),==,shape[0], "Shape along x does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(data.shape(2),==,shape[2], "Shape along z does not agree")
        */

        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());
        
        { 
            // allocate / create data for each thread
            struct PerThreadData{
                marray::Marray<LABELS_TYPE> sliceLabels;
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
                
                data.readSubarray(blockBegin, blockEnd, sliceDataFlat3D);
                auto sliceData = sliceDataFlat3D.squeezedView();
                
                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                    const auto node = sliceLabels( coord.asStdArray() );            
                    const auto l    = sliceData( coord.asStdArray() );
                    overlaps[node][l] += 1;
                });

            });
        }
        
        {
        parallel::parallel_foreach(threadpool, graph.numberOfNodes(), [&](const int tid, const int64_t nodeId){
                const auto & ol = overlaps[nodeId];
                // find max ol 
                uint64_t maxOl = 0 ;
                uint64_t maxOlLabel = 0;
                for(auto kv : ol){
                    if(kv.second > maxOl){
                        maxOl = kv.second;
                        maxOlLabel = kv.first;
                    }
                }
                nodeMap[nodeId] = maxOlLabel;
            });
        }

    }



} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HDF5_HXX */
