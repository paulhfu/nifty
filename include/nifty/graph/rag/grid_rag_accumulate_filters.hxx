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
    // for old implementation with blocks in parallel see feature_acc_old_impl
    template<class EDGE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F, class COORD>
    void accumulateEdgeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        features::FilterBase * filter,
        const double sigma,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f,
        const AccOptions & accOptions = AccOptions()
    ){

        typedef typename DATA::DataType DataType;

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LabelsProxyType::LabelType LabelType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef array::StaticArray<int64_t, DIM+1> Coord4;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> ChannelAccChainVectorType; 
        typedef std::vector<ChannelAccChainVectorType> EdgeAccChainVectorType; 

        typedef typename std::conditional<DIM==2,
            fastfilters_array2d_t,
            fastfilters_array3d_t>::type
        FastfiltersArrayType;

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();
        size_t numberOfChannels = filter->isMultiChannel() ? DIM : 1; 

        const auto & shape = rag.shape();

        std::vector< EdgeAccChainVectorType * > perThreadEdgeAccChainVector(actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadEdgeAccChainVector[i] = new EdgeAccChainVectorType(rag.edgeIdUpperBound()+1,
                ChannelAccChainVectorType(numberOfChannels) );
        });

        Coord blockShapeWithBorder;
        Coord base;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(blockShape[d]+1, shape[d]);
            base[d] = 0;
        }
        
        Coord4 filterShape;
        filterShape[0] = numberOfChannels;
        for(auto d=0; d<DIM; ++d){
            filterShape[d+1] = blockShapeWithBorder[d];
        }
            
        marray::Marray<LabelType> labelsBlockView(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
        marray::Marray<DataType>  dataBlock(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
        marray::Marray<float> filtersBlockView(filterShape.begin(), filterShape.end());
        
        // calc filters for the first block and set minmax
        if(accOptions.setMinMax){
            std::cout << "Setting Minmax" << std::endl;
            // get min and max for the first block (using block id 1 here due to FIB pecularities)
            size_t blockId = 1;
            
            Coord blockBegin;
            Coord blockEnd;
            for(auto d=0; d<DIM; ++d){
                blockBegin[d] = startCoordinates[blockId][d];
                blockEnd[d]   = startCoordinates[blockId][d] + blockShapeWithBorder[d];
            }
                
            tools::readSubarray(data, blockBegin, blockEnd, dataBlock);
            FastfiltersArrayType dataFF;
            
            // FIXME we neglect different min / max for multichanne here...
            // need to convert the input data for non-float data
            marray::Marray<float> dataCopy;
            marray::View<float> dataBlockView;
            if( typeid(DataType) == typeid(float) ) {
                dataBlockView = dataBlock.view(base.begin(), blockShapeWithBorder.begin());
            }
            else {
                // need to copy the data
                dataCopy.resize(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
                std::copy(dataBlock.begin(),dataBlock.end(),dataCopy.begin());
                dataBlockView = dataCopy.view(base.begin(), blockShapeWithBorder.begin());
            }
            
            features::detail_fastfilters::convertMarray2ff(dataBlockView, dataFF);
            filtersBlockView.squeeze();
            (*filter)(dataFF, filtersBlockView, sigma);
                
            vigra::HistogramOptions histoOpts;
            auto minMax = std::minmax_element(filtersBlockView.begin(), filtersBlockView.end());
            std::cout << "Min: " <<*(minMax.first) << " Max: " << *(minMax.second) << std::endl;
            histoOpts.setMinMax( *(minMax.first), *(minMax.second));

            parallel::parallel_foreach(threadpool, actualNumberOfThreads,
            [&](int tid, int i){
                auto & edgeAccVec = *(perThreadEdgeAccChainVector[i]);
                for(auto & edgeAcc : edgeAccVec){
                    for(size_t c = 0; c < numberOfChannels ; c++) {
                        edgeAcc[c].setHistogramOptions(histoOpts);
                    }
                }
            });
        }

        const auto passesRequired = (*perThreadEdgeAccChainVector.front()).front().front().passesRequired();
        
        const size_t nBlocks = startCoordinates.size();

        // do N passes of accumulator
        for(auto pass=1; pass <= passesRequired; ++pass){

            std::cout << "Pass: " << pass << std::endl;
            
            for(size_t blockId = 0; blockId < nBlocks; blockId++) {
                
                std::cout << blockId << " / " << nBlocks  << std::endl;

                Coord blockBegin;
                Coord blockEnd;

                for(auto d=0; d<DIM; ++d){
                    blockBegin[d] = startCoordinates[blockId][d];
                    blockEnd[d]   = startCoordinates[blockId][d] + blockShapeWithBorder[d];
                }

                // need to try catch in case of violating overlaps
                bool overlapViolates = false;
                try { 
                    tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                    tools::readSubarray(data, blockBegin, blockEnd, dataBlock);
                }
                catch (const std::runtime_error & e) {
                    std::cout << "Overlap violating and reduced" << std::endl;
                    overlapViolates = true;
                    for(auto d=0; d<DIM; ++d) {
                        blockEnd[d] -= 1;
                        blockShapeWithBorder[d] -= 1;
                        filterShape[d+1] -= 1; 
                    }
                    labelsBlockView.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    dataBlock.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    filtersBlockView.resize(filterShape.begin(), filterShape.end());
                    tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                    tools::readSubarray(data, blockBegin, blockEnd, dataBlock);
                }

                FastfiltersArrayType dataFF; 
                // need to convert the input data for non-float data
                marray::Marray<float> dataCopy;
                marray::View<float> dataBlockView;
                if( typeid(DataType) == typeid(float) ) {
                    dataBlockView = dataBlock.view(base.begin(),blockShapeWithBorder.begin());
                }
                else {
                    // need to copy the data
                    dataCopy.resize(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
                    std::copy(dataBlock.begin(),dataBlock.end(),dataCopy.begin());
                    dataBlockView = dataCopy.view(base.begin(), blockShapeWithBorder.begin());
                }
                
                filtersBlockView.squeeze();
                features::detail_fastfilters::convertMarray2ff(dataBlockView, dataFF);
                (*filter)(dataFF, filtersBlockView, sigma);
                
                // for debugging purposes only
                //auto minMax = std::minmax_element(filtersBlockView.begin(), filtersBlockView.end());
                //std::cout << "Min: " <<*(minMax.first) << " Max: " << *(minMax.second) << std::endl;

                nifty::tools::parallelForEachCoordinate(
                    threadpool,
                    blockShapeWithBorder
                    ,[&](int tid, const Coord & coordU){
                    
                        auto & accVec = *(perThreadEdgeAccChainVector[tid]);
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

                                    if(filter->isMultiChannel()) {
                                        for(size_t c = 0; c < numberOfChannels; c++) {
                                            filterCoordU[0] = c;
                                            filterCoordV[0] = c;
                                            const auto dataU = filtersBlockView(filterCoordU.asStdArray());
                                            const auto dataV = filtersBlockView(filterCoordV.asStdArray());
                                            accVec[edge][c].updatePassN(dataU, vigraCoordU, pass);
                                            accVec[edge][c].updatePassN(dataV, vigraCoordV, pass);
                                        }
                                    }
                                    else {
                                        const auto dataU = filtersBlockView(coordU.asStdArray());
                                        const auto dataV = filtersBlockView(coordV.asStdArray());
                                        accVec[edge][0].updatePassN(dataU, vigraCoordU, pass);
                                        accVec[edge][0].updatePassN(dataV, vigraCoordV, pass);
                                    }
                                }
                            }
                        }
                });

                if( !filter->isMultiChannel() )
                    filtersBlockView.resize(filterShape.begin(), filterShape.end());
                
                // reset shape and marray if overlap was violating
                if(overlapViolates) {
                    for(auto d=0; d<DIM; ++d) {
                        blockShapeWithBorder[d] += 1;
                        filterShape[d+1] += 1; 
                    }
                    labelsBlockView.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    dataBlock.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    filtersBlockView.resize(filterShape.begin(), filterShape.end());
                }
            }
        }

        auto & resultAccVec = *(perThreadEdgeAccChainVector.front());

        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, resultAccVec.size(), 
        [&](const int tid, const int64_t edge){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                for(size_t c = 0; c < numberOfChannels; c++) {
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



    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateEdgeFeaturesOverFilters(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        const int numberOfThreads = -1,
        const bool onePass = true
    )
    {
        // filters and sigmas
        features::GaussianSmoothing gs;
        features::LaplacianOfGaussian log;
        features::HessianOfGaussianEigenvalues hog;

        std::vector<features::FilterBase*> filters({&gs, &log, &hog});
        std::vector<double> sigmas({1.6,4.2,8.4});
        
        //std::vector<features::FilterBase*> filters({&hog});
        //std::vector<double> sigmas({1.6,4.2});
        
        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        size_t numberOfStats = 11;
        if(onePass)
            numberOfStats = 9;

        size_t offset = 0;
        for( auto filter : filters ) {
            for( auto sigma : sigmas ) {
                std::cout << "Offset: " << offset << std::endl;
                if(onePass)
                    accumulateEdgeStandartFeaturesOnePass(rag, data, startCoordinates, blockShape, edgeFeaturesOut, filter, sigma, offset, threadpool, pOpts);
                else
                    accumulateEdgeStandartFeaturesTwoPass(rag, data, startCoordinates, blockShape, edgeFeaturesOut, filter, sigma, offset, threadpool, pOpts);
                offset += (filter->isMultiChannel() ? DIM : 1) * numberOfStats;
            }
        }
    }

    
    // 9 features per channel
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateEdgeStandartFeaturesOnePass(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        features::FilterBase * filter,
        const double sigma,
        const size_t offset,
        parallel::ThreadPool & threadpool,
        const parallel::ParallelOptions & pOpts
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

        accumulateEdgeFeaturesWithAccChain<AccChainType>(
            rag,
            data,
            startCoordinates,
            blockShape,
            filter,
            sigma,
            pOpts,
            threadpool,
            [&](
                const std::vector<std::vector<AccChainType>> & edgeAccChainVec
            ){
                using namespace vigra::acc;
                size_t numberOfChannels = filter->isMultiChannel() ? DIM : 1; 
                size_t numberOfStats = 9;
                
                // debugging
                //for(size_t edge = 0; edge < edgeAccChainVec.size(); edge++) {
                //    if(edge > 10)
                //        continue;
                //    size_t channelOffset = offset;
                //    std::cout << "Edge: " << edge << " / " << edgeAccChainVec.size() << std::endl;
                //    const auto & chain = edgeAccChainVec[edge];
                //    for(size_t c = 0; c < numberOfChannels; c++) {
                //        std::cout << "channel: " << c << std::endl;
                //        std::cout << "channelOffset: " << channelOffset << std::endl; 
                //        const auto & chainChannel = chain[c];
                //        const auto mean = get<acc::Mean>(chainChannel);
                //        std::cout << "Mean: " << mean << std::endl;
                //        const auto quantiles = get<Quantiles>(chainChannel);
                //        edgeFeaturesOut(edge, channelOffset) = replaceIfNotFinite(mean,     0.0);
                //        edgeFeaturesOut(edge, channelOffset+1) = replaceIfNotFinite(get<acc::Variance>(chainChannel), 0.0);
                //        for(auto qi=0; qi<7; ++qi) {
                //            edgeFeaturesOut(edge, channelOffset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                //        channelOffset += numberOfStats;
                //        }
                //    }
                //}
                
                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    size_t channelOffset = offset;
                    for(size_t c = 0; c < numberOfChannels; c++) {
                        const auto & chainChannel = chain[c];
                        const auto mean = get<acc::Mean>(chainChannel);
                        const auto quantiles = get<Quantiles>(chainChannel);
                        edgeFeaturesOut(edge, channelOffset) = replaceIfNotFinite(mean,     0.0);
                        edgeFeaturesOut(edge, channelOffset+1) = replaceIfNotFinite(get<acc::Variance>(chainChannel), 0.0);
                        for(auto qi=0; qi<7; ++qi)
                            edgeFeaturesOut(edge, channelOffset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                        channelOffset += numberOfStats;
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
        features::FilterBase * filter,
        const double sigma,
        const size_t offset,
        parallel::ThreadPool & threadpool,
        const parallel::ParallelOptions & pOpts
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

        accumulateEdgeFeaturesWithAccChain<AccChainType>(
            rag,
            data,
            startCoordinates,
            blockShape,
            filter,
            sigma,
            pOpts,
            threadpool,
            [&](
                const std::vector<std::vector<AccChainType>> & edgeAccChainVec
            ){
                using namespace vigra::acc;
                size_t numberOfChannels = filter->isMultiChannel() ? DIM : 1; 
                size_t numberOfStats = 11;
                size_t channelOffset = offset;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    size_t offset = 0;
                    for(size_t c = 0; c < numberOfChannels; c++) {
                        const auto & chainChannel = chain[c];
                        const auto mean = get<acc::Mean>(chainChannel);
                        const auto quantiles = get<Quantiles>(chainChannel);
                        edgeFeaturesOut(edge, channelOffset) = replaceIfNotFinite(mean,     0.0);
                        edgeFeaturesOut(edge, channelOffset+1) = replaceIfNotFinite(get<acc::Variance>(chainChannel), 0.0);
                        edgeFeaturesOut(edge, channelOffset+2) = replaceIfNotFinite(get<acc::Skewness>(chainChannel), 0.0);
                        edgeFeaturesOut(edge, channelOffset+3) = replaceIfNotFinite(get<acc::Kurtosis>(chainChannel), 0.0);
                        for(auto qi=0; qi<7; ++qi)
                            edgeFeaturesOut(edge, channelOffset+4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                        channelOffset += numberOfStats;
                    }
                }); 
            },
            AccOptions()
        );
    }

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_HXX */
