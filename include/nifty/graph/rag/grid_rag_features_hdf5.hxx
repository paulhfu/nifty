#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HDF5_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HDF5_HXX

#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#include "nifty/graph/rag/feature_functors.hxx"


namespace nifty{
namespace graph{
    
    // TODO clear API definition + implementing the functor
    template< class LABELS_PROXY, class T, class ACC_CHAIN>
    std::vector<ACC_CHAIN> gridRagAccumulateFeatures(
        const GridRagStacked2D<nifty::hdf5::Hdf5Array<LABELS_PROXY>> & graph, // the stacked rag
        const nifty::hdf5::Hdf5Array<T> & data,  // the raw data
        const int numberOfThreads = -1
    ){
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LABELS_PROXY::LabelType LabelType;
        
        typedef typename LabelsProxyType::BlockStorageType LabelsBlockStorageType;
        typedef tools::BlockStorage<T> DataBlockStorageType;
        
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;
        
        typedef ACC_CHAIN AccChainType;
        typedef std::vector<AccChainType> AccChainVectorType; 

        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();

        // TODO do we need this
        std::vector< std::vector<T> > edgeValues;
        edgeValues.resize(graph.numberOfEdges(),std::vector<T>());
        
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});

        std::vector<double> sigmas({1.6,4.2,8.4});
        GaussianSmoothing gs;
        LaplacianOfGaussian log;
        HessianOfGaussianEigenvalues hog;
        std::vector<FilterBase*> filts(&gs, &log, &hog);
        ApplyFilters<2> filters(sigmas, filts);
        
        Coord sliceShape3({int64_t(filters.numberOfChannels()), shape[1], shape[2]});

        // TODO is this thread safe ??
        std::vector<AccChainVectorType> accChains(filters.numberOfChannels,
                AccChainType( graph.edgeIdUpperBound() ) );
        
        /////////////////////////////////////////////////////
        // Phase 1 : Accumulate in-slice edge features
        /////////////////////////////////////////////////////
        
        { 
            
            LabelsBlockStorageType sliceLabelsStorage(threadpool, sliceShape3, nThreads);
            DataBlockStorageType sliceDataStorage(threadpool, sliceShape3, nThreads);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

                // fetch the data for the slice
                auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);
                auto sliceDataFlat3DView = sliceDataStorage.getView(tid);
                
                const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3DView);
                auto sliceLabels = sliceLabelsFlat3DView.squeezedView();
                
                data.readSubarray(blockBegin.begin(), sliceDataFlat3DView);
                auto sliceData = sliceDataFlat3DView.squeezedView();
                
                // TODO should this also be a block storage ?
                marray::Marray<T> sliceFeatures(sliceShape3, sliceShape3 + 3);
                calculateFeature(sliceFeatures);

                // accumulate filter responses
                for(size_t chan = 0; chan < filters.numberOfChannels(); chan++) {
                    nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                        const auto lU = sliceLabels(coord.asStdArray());
                        for(size_t axis=0; axis<2; ++axis){
                            Coord2 coord2 = coord;
                            ++coord2[axis];
                            if(coord2[axis] < sliceShape2[axis]){
                                const auto lV = sliceLabels(coord2.asStdArray());
                                const auto e = graph.findEdge(lU, lV);
                                    const auto dV = sliceFeatures(chan, coord[0], coord[1]);
                                    const auto dU = sliceFeatures(chan, coord2[0], coord2[1]);
                                    // TODO properly loop over the passes
                                    accChains[chan][e].updatePassN(dU, 0);
                                    accChains[chan][e].updatePassN(dV, 0);
                                }
                            }
                    });
                }
            });
        }
        
        /////////////////////////////////////////////////////
        // Phase 2 : Accumulate between-slice edge features
        /////////////////////////////////////////////////////
        
        { 
            // TODO
        }

    }

    
    template<class LABELS_PROXY, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const GridRagStacked2D<LABELS_PROXY> & graph, // the stacked rag
        const nifty::hdf5::Hdf5Array<LABELS> & data,  // the raw data
        NODE_MAP &  nodeMap,
        const int numberOfThreads = -1
    ){
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LABELS_PROXY::LabelType LabelType;
        typedef typename LabelsProxyType::BlockStorageType LabelsBlockStorageType;
        typedef tools::BlockStorage<LABELS> DataBlockStorageType;
        
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
            
            LabelsBlockStorageType sliceLabelsStorage(threadpool, sliceShape3, nThreads);
            DataBlockStorageType sliceDataStorage(threadpool, sliceShape3, nThreads);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

                // fetch the data for the slice
                auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);
                auto sliceDataFlat3DView = sliceDataStorage.getView(tid);
                
                const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3DView);
                auto sliceLabels = sliceLabelsFlat3DView.squeezedView();
                
                data.readSubarray(blockBegin.begin(), sliceDataFlat3DView);
                auto sliceData = sliceDataFlat3DView.squeezedView();
                
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
