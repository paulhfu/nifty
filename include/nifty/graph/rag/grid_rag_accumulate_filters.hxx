#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_HXX

#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#endif

namespace nifty{
namespace graph{
    
    // accumulator with data
    template<class EDGE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F, class COORD>
    void accumulateEdgeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f,
        const AccOptions & accOptions = AccOptions()
    ){

        typedef typename DATA::DataType DataType;

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef array::StaticArray<int64_t, DIM+1> Coord4;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> ChannelAccChainVectorType; 
        typedef std::vector<ChannelAccChainVectorType> EdgeAccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        const auto & shape = rag.shape();
        
        // filters and sigmas
        features::GaussianSmoothing gs;
        features::LaplacianOfGaussian log;
        features::HessianOfGaussianEigenvalues hog;

        std::vector<features::FilterBase*> filters({&gs, &log, &hog});
        std::vector<double> sigmas({1.6,4.2,8.2});

        features::ApplyFilters<DIM> applyFilters(sigmas, filters);

        std::vector< EdgeAccChainVectorType * > perThreadEdgeAccChainVector(actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadEdgeAccChainVector[i] = new EdgeAccChainVectorType(rag.edgeIdUpperBound()+1,
                ChannelAccChainVectorType(applyFilters.numberOfChannels()));
        });

        const auto passesRequired = (*perThreadEdgeAccChainVector.front()).front().front().passesRequired();

        Coord blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(blockShape[d]+1, shape[d]);
        }
        
        Coord4 filterShape;
        filterShape[0] = applyFilters.numberOfChannels();
        for(auto d=0; d<DIM; ++d){
            filterShape[d+1] = blockShapeWithBorder[d];
        }
            
        
        // calc filters for the first block and set minmax
        if(accOptions.setMinMax){
            // get min and max for the first block (using block id 1 here due to FIB pecularities)
            size_t blockId = 1;
            
            Coord blockBegin;
            Coord blockEnd;
            for(auto d=0; d<DIM; ++d){
                blockBegin[d] = startCoordinates[blockId][d];
                blockEnd[d]   = startCoordinates[blockId][d] + blockShapeWithBorder[d];
            }
                
            marray::Marray<DataType> dataBlockView(blockShapeWithBorder.begin(),blockShapeWithBorder.end());
            tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);

            typedef typename std::conditional<DIM==2,
                fastfilters_array2d_t,
                fastfilters_array3d_t>::type
            FastfiltersArrayType;

            FastfiltersArrayType dataFF;
            features::detail_fastfilters::convertMarray2ff(dataBlockView, dataFF);

            std::vector<std::pair<features::FilterBase*,double>> filtersAndSigmas;
            for(auto filter : filters) {
                for(auto sigma :  sigmas)
                    filtersAndSigmas.push_back(std::make_pair(filter,sigma));
            }
            std::vector<vigra::HistogramOptions> channelHistogramOptions(filtersAndSigmas.size());

            Coord shapeSingleChannel = blockShapeWithBorder;
            Coord4 shapeMultiChannel;
            shapeMultiChannel[0] = DIM;
            for(auto d=0; d<DIM; ++d){
                shapeMultiChannel[d+1] = blockShapeWithBorder[d];
            }
            

            parallel::parallel_foreach(threadpool, filtersAndSigmas.size(), [&](
                const int tid, const int fid
            ) {
                auto filter = filtersAndSigmas[fid].first;
                auto sigma = filtersAndSigmas[fid].second;

                marray::Marray<DataType> filterResponse = filter->isMultiChannel() ? marray::Marray<DataType>(shapeMultiChannel.begin(), shapeMultiChannel.end()) :marray::Marray<DataType>(shapeSingleChannel.begin(), shapeSingleChannel.end());

                // FIXME we neglect different min / max for multichanne here...
                (*filter)(dataFF, filterResponse, sigma);
                
                auto minMax = std::minmax_element(filterResponse.begin(), filterResponse.end());
                channelHistogramOptions[fid].setMinMax( *(minMax.first), *(minMax.second));
            });

            parallel::parallel_foreach(threadpool, actualNumberOfThreads,
            [&](int tid, int i){
                auto & edgeAccVec = *(perThreadEdgeAccChainVector[i]);
                for(auto & edgeAcc : edgeAccVec){
                    size_t offset = 0;
                    for(size_t fid = 0; fid < channelHistogramOptions.size(); fid++) {
                        size_t nChans = filtersAndSigmas[fid].first->isMultiChannel() ? DIM : 1;
                        const auto & histogramOpts = channelHistogramOptions[fid];
                        for(size_t c = 0; c < nChans; c++) {
                            edgeAcc[c+offset].setHistogramOptions(histogramOpts);
                        }
                        offset += nChans;
                    }
                }
            });
        }
        
        const size_t nBlocks = startCoordinates.size();

        // do N passes of accumulator
        for(auto pass=1; pass <= passesRequired; ++pass){
            
            LabelBlockStorage labelsBlockStorage(threadpool, blockShapeWithBorder, actualNumberOfThreads);
            DataBlocKStorage dataBlockStorage(threadpool, blockShapeWithBorder, actualNumberOfThreads);
            DataBlocKStorage filterBlockStorage(threadpool, filterShape, actualNumberOfThreads);
            
            parallel::parallel_foreach(threadpool, nBlocks,
            [&](
                const int tid, const int blockId
            ){
                // get the accumulator vector for this thread
                auto & accVec = *(perThreadEdgeAccChainVector[tid]);

                // get the accumulator vector for this thread
                // read the labels block and the data block
                auto labelsBlockView = labelsBlockStorage.getView(blockShapeWithBorder, tid);
                auto dataBlockView = dataBlockStorage.getView(blockShapeWithBorder, tid);
                auto filterBlockView = filterBlockStorage.getView(filterShape, tid);
                
                Coord blockBegin;
                Coord blockEnd;

                for(auto d=0; d<DIM; ++d){
                    blockBegin[d] = startCoordinates[blockId][d];
                    blockEnd[d]   = startCoordinates[blockId][d] + blockShapeWithBorder[d];
                }

                tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);

                applyFilters(dataBlockView, filterBlockView);

                nifty::tools::forEachCoordinate(blockShapeWithBorder,[&](const Coord & coordU){
                    const auto lU = labelsBlockView(coordU.asStdArray());
                    for(size_t axis=0; axis<DIM; ++axis){
                        auto coordV = makeCoord2(coordU, axis);
                        if(coordV[axis] < blockShapeWithBorder[axis]){
                            const auto lV = labelsBlockView(coordV.asStdArray());
                            if(lU != lV){
                                const auto edge = rag.findEdge(lU,lV);

                                VigraCoord vigraCoordU;
                                VigraCoord vigraCoordV;

                                Coord4 filterCoordU;
                                Coord4 filterCoordV;

                                for(size_t d=0; d<DIM; ++d){
                                    vigraCoordU[d] = coordU[d];
                                    vigraCoordV[d] = coordV[d];
                                    filterCoordU[d+1] = coordU[d];
                                    filterCoordV[d+1] = coordV[d];
                                }

                                for(size_t c = 0; c < applyFilters.numberOfChannels(); c++) {
                                    filterCoordU[0] = c;
                                    filterCoordV[0] = c;
                                    const auto dataU = filterBlockView(filterCoordU.asStdArray());
                                    const auto dataV = filterBlockView(filterCoordV.asStdArray());
                                    accVec[edge][c].updatePassN(dataU, vigraCoordU, pass);
                                    accVec[edge][c].updatePassN(dataV, vigraCoordV, pass);
                                }
                            }
                        }
                    }
                });
            });
        }

        auto & resultAccVec = *(perThreadEdgeAccChainVector.front());


        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, resultAccVec.size(), 
        [&](const int tid, const int64_t edge){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                for(size_t c = 0; c < applyFilters.numberOfChannels(); c++) {
                    resultAccVec[edge][c].merge((*(perThreadEdgeAccChainVector[t]))[edge][c]);
                }
            }            
        });
        
        // call functor with finished acc chain
        f(resultAccVec);

        // delete 
        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            delete perThreadEdgeAccChainVector[i];
        });
    }
    
    // 9 features per channel
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateEdgeStandartFeaturesOnePass(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;
        typedef FEATURE_TYPE DataType;


        typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
        typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

        typedef acc::Select<
            acc::DataArg<1>,
            acc::Mean,        //1
            acc::Variance,    //1
            Quantiles         //7
        > SelectType;
        
        typedef SelectType SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        accumulateEdgeFeaturesWithAccChain<AccChainType>(
            rag,
            data,
            startCoordinates,
            blockShape,
            pOpts,
            threadpool,
            [&](
                const std::vector<std::vector<AccChainType>> & edgeAccChainVec
            ){
                using namespace vigra::acc;
                size_t numberOfChannels = edgeAccChainVec.front().size(); 
                size_t numberOfStats = 9;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    size_t offset = 0;
                    for(size_t c = 0; c < numberOfChannels; c++) {
                        offset = c * numberOfStats;
                        const auto & chainChannel = chain[c];
                        const auto mean = get<acc::Mean>(chainChannel);
                        const auto quantiles = get<Quantiles>(chainChannel);
                        edgeFeaturesOut(edge, offset) = replaceIfNotFinite(mean,     0.0);
                        edgeFeaturesOut(edge, offset+1) = replaceIfNotFinite(get<acc::Variance>(chainChannel), 0.0);
                        for(auto qi=0; qi<7; ++qi)
                            edgeFeaturesOut(edge, offset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                }); 
            },
            AccOptions(0., 1.) // set dummy min and max to activate setMinMax
        );
    }
    
    
    // 11 features per channel
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateEdgeStandartFeaturesTwoPass(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;
        typedef FEATURE_TYPE DataType;

        typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
        typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

        typedef acc::Select<
            acc::DataArg<1>,
            acc::Mean,        //1
            acc::Variance,    //1
            acc::Skewness,    //1
            acc::Kurtosis,    //1
            Quantiles         //7
        > SelectType;
        
        typedef SelectType SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        accumulateEdgeFeaturesWithAccChain<AccChainType>(
            rag,
            data,
            startCoordinates,
            blockShape,
            pOpts,
            threadpool,
            [&](
                const std::vector<std::vector<AccChainType>> & edgeAccChainVec
            ){
                using namespace vigra::acc;
                size_t numberOfChannels = edgeAccChainVec.front().size(); 
                size_t numberOfStats = 11;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    size_t offset = 0;
                    for(size_t c = 0; c < numberOfChannels; c++) {
                        offset = c * numberOfStats;
                        const auto & chainChannel = chain[c];
                        const auto mean = get<acc::Mean>(chainChannel);
                        const auto quantiles = get<Quantiles>(chainChannel);
                        edgeFeaturesOut(edge, offset) = replaceIfNotFinite(mean,     0.0);
                        edgeFeaturesOut(edge, offset+1) = replaceIfNotFinite(get<acc::Variance>(chainChannel), 0.0);
                        edgeFeaturesOut(edge, offset+1) = replaceIfNotFinite(get<acc::Skewness>(chainChannel), 0.0);
                        edgeFeaturesOut(edge, offset+1) = replaceIfNotFinite(get<acc::Kurtosis>(chainChannel), 0.0);
                        for(auto qi=0; qi<7; ++qi)
                            edgeFeaturesOut(edge, offset+4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                }); 
            },
            AccOptions()
        );
    }

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_HXX */
