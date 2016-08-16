#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_HXX

#include <vector>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "vigra/accumulator.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

namespace nifty{
namespace graph{
    
    // TODO implement
    template<class EDGE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F>
    void accumulateEdgeFeaturesFromFiltersWithAccChain(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        const bool userRangeHistogram,
        F && f
    ){}
    
    
    template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F>
    void accumulateEdgeFeaturesFromFiltersWithAccChain(
        const GridRagStacked2D<LABELS_PROXY> & rag,
        const DATA & data,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        const bool userRangeHistogram,
        F && f
    ){
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<3>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlockStorage;
        typedef typename DATA::DataType DataType;

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;
        
        
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> AccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        const auto & shape = rag.shape();
        const auto & labelsProxy = rag.labelsProxy();
        
        // the filters we use
        features::GaussianSmoothing gs;
        features::LaplacianOfGaussian log;
        features::HessianOfGaussianEigenvalues hog;

        std::vector<features::FilterBase*> filters({&gs, &log, &hog});
        std::vector<double> sigmas({1.6,4.2,8.2});

        features::ApplyFilters<2> applyFilters(sigmas, filters);

        //std::vector< AccChainVectorType > perThreadAccChainVector(actualNumberOfThreads, 
        //    AccChainVectorType(rag.edgeIdUpperBound()+1));
        
        // I think this should be threadsafe, because we never will never have the same edge in different threads!
        std::vector< AccChainVectorType > accChainVector( rag.edgeIdUpperBound()+1, 
            AccChainVectorType(applyFilters.numberOfChannels()) );

        uint64_t numberOfSlices = shape[0];
        
        Coord2 sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});
        Coord sliceABShape({int64_t(2),shape[1], shape[2]});
        Coord filtersShape({int64_t(applyFilters.numberOfChannels()), shape[1], shape[2]});

        // set minmax for the histogram if necessary -> implement in filters
        // checkout the first slice instead... TODO this is still pretty hacky... 
        // we could also condition on the number of passes insted of an extra bool...
        if(userRangeHistogram) {
            Coord slice0Begin({int64_t(0),int64_t(0),int64_t(0)});
            Coord slice0End({int64_t(1),shape[1],shape[2]});
            marray::Marray<DataType> data0(sliceShape3.begin(), sliceShape3.end());
            tools::readSubarray(data,slice0Begin,slice0End,data0);
            auto data0View = data0.squeezedView();
            fastfilters_array2d_t data0ff;
            features::detail_fastfilters::convertMarray2ff(data0View, data0ff);
            Coord2 filtShapeSingle({  shape[1], shape[2] } );
            Coord filtShapeMulti({ int64_t(2), shape[1], shape[2] } );
            parallel::parallel_foreach(threadpool, accChainVector.size(),[&](
                const int tid, const int64_t edge
            ){
                size_t offset = 0;
                for(auto f : filters) {
                    //auto min = f->getMin();
                    //auto max = f->getMax();
                    // calc filter only for the first sigma for now, could also do it for all ... 
                    
                    marray::Marray<DataType> filt0 = f->isMultiChannel() ? marray::Marray<DataType>( filtShapeMulti.begin(), filtShapeMulti.end() ) : marray::Marray<DataType>( filtShapeSingle.begin(), filtShapeSingle.end() ) ;
                    (*f)(data0ff, filt0, sigmas[0]);
                    auto minMax = std::minmax_element( filt0.begin(), filt0.end()  );
                    auto min = *(minMax.first);
                    auto max = *(minMax.second);
                    size_t nChannels = sigmas.size() * (f->isMultiChannel() ? 2 : 1);
                    for(size_t c = 0; c < nChannels; ++c) {
                        accChainVector[edge][c+offset].setHistogramOptions(vigra::HistogramOptions().setMinMax(min,max));
                    }
                    offset += nChannels;
                }
            });
        }

        // do N passes of accumulator
        for(auto pass=1; pass <= accChainVector.front().front().passesRequired(); ++pass){
            // TODO is there any more efficient way where we don't apply the filters multiple times?
            
            LabelBlockStorage sliceABStorage(threadpool, sliceABShape, actualNumberOfThreads);
            DataBlockStorage sliceADataStorage(threadpool, sliceShape3, actualNumberOfThreads);
            DataBlockStorage sliceBDataStorage(threadpool, sliceShape3, actualNumberOfThreads);

            for(auto startIndex : {0,1}){
            
                parallel::parallel_foreach(threadpool, numberOfSlices-1, [&](const int tid, const int64_t sliceAIndex){
                    
                    const auto oddIndex = bool(sliceAIndex%2);
                    if((startIndex==0 && !oddIndex) || (startIndex==1 && oddIndex )){
                    
                        const auto sliceBIndex = sliceAIndex + 1;
                        
                        // fetch the data for the slice
                        auto sliceAB = sliceABStorage.getView(tid);
                        
                        auto sliceDataFlat3DViewA  = sliceADataStorage.getView(tid);
                        auto sliceDataFlat3DViewB  = sliceBDataStorage.getView(tid);
                        
                        // labels
                        const Coord blockABBegin({sliceAIndex,int64_t(0),int64_t(0)});
                        const Coord blockABEnd({sliceAIndex+2, sliceShape2[0], sliceShape2[1]});
                        
                        labelsProxy.readSubarray(blockABBegin, blockABEnd, sliceAB);
                        const Coord coordAOffset{0L,0L,0L};
                        const Coord coordBOffset{1L,0L,0L};
                        auto sliceLabelsA = sliceAB.view(coordAOffset.begin(), sliceShape3.begin()).squeezedView();
                        auto sliceLabelsB = sliceAB.view(coordBOffset.begin(), sliceShape3.begin()).squeezedView();
                        
                        // data 
                        const Coord blockABegin = blockABBegin;
                        const Coord blockAEnd({sliceAIndex + 1, sliceShape2[0], sliceShape2[1]});
                        const Coord blockBBegin({sliceBIndex, int64_t(0), int64_t(0)});
                        const Coord blockBEnd = blockABEnd;
                        
                        tools::readSubarray(data, blockABegin, blockAEnd, sliceDataFlat3DViewA);
                        tools::readSubarray(data, blockBBegin, blockBEnd, sliceDataFlat3DViewB);
                        
                        auto sliceDataA = sliceDataFlat3DViewA.squeezedView();
                        auto sliceDataB = sliceDataFlat3DViewB.squeezedView();

                        marray::Marray<DataType> sliceFiltersA(filtersShape.begin(), filtersShape.end() );
                        marray::Marray<DataType> sliceFiltersB(filtersShape.begin(), filtersShape.end() );

                        applyFilters(sliceDataA, sliceFiltersA);
                        applyFilters(sliceDataB, sliceFiltersB);

                        for(size_t c = 0; c < applyFilters.numberOfChannels(); ++c) {
                                
                            // accumulate the filter responses for the inner slice edges (of lower slice == sliceA)
                            nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                                const auto lU = sliceLabelsA(coord.asStdArray());
                                for(size_t axis=0; axis<2; ++axis){
                                    Coord2 coord2 = coord;
                                    ++coord2[axis];
                                    if(coord2[axis] < sliceShape2[axis]){
                                        const auto lV = sliceLabelsA(coord2.asStdArray());
                                        if(lU != lV){
                                            
                                            const auto edge = rag.findEdge(lU,lV);
                                            const auto dataU = sliceFiltersA(c,coord[0],coord[1]);
                                            const auto dataV = sliceFiltersA(c,coord2[0],coord2[1]);

                                            VigraCoord vigraCoordU;
                                            VigraCoord vigraCoordV;
                                            vigraCoordU[0] = sliceAIndex;
                                            vigraCoordV[0] = sliceAIndex;

                                            for(size_t d=1; d<3; ++d){
                                                vigraCoordU[d] = coord[d-1];
                                                vigraCoordV[d] = coord2[d-1];
                                            }

                                            accChainVector[edge][c].updatePassN(dataU, vigraCoordU, pass);
                                            accChainVector[edge][c].updatePassN(dataV, vigraCoordV, pass);
                                        
                                        }
                                    }
                                }
                            });
                            
                            // accumulate the filter responses for the between slice edges
                            nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                            
                                const auto lU = sliceLabelsA(coord.asStdArray());
                                const auto lV = sliceLabelsB(coord.asStdArray());
                                const auto edge = rag.findEdge(lU,lV);

                                const auto dataU = sliceFiltersA(c,coord[0],coord[1]);
                                const auto dataV = sliceFiltersB(c,coord[0],coord[1]);
                                
                                VigraCoord vigraCoordU;
                                VigraCoord vigraCoordV;

                                vigraCoordU[0] = sliceAIndex;
                                vigraCoordV[0] = sliceBIndex;
                                for(size_t d=1; d<3; ++d){
                                    vigraCoordU[d] = coord[d-1];
                                    vigraCoordV[d] = coord[d-1];
                                }

                                accChainVector[edge][c].updatePassN(dataU, vigraCoordU, pass);
                                accChainVector[edge][c].updatePassN(dataV, vigraCoordV, pass);
                            });
                        }
                        
                        // accumulate the filter response for the edges inside of the last slice
                        if(sliceBIndex == numberOfSlices - 1) {
                            
                            for(size_t c = 0; c < applyFilters.numberOfChannels(); ++c) {
                                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                                    const auto lU = sliceLabelsB(coord.asStdArray());
                                    for(size_t axis=0; axis<2; ++axis){
                                        Coord2 coord2 = coord;
                                        ++coord2[axis];
                                        if(coord2[axis] < sliceShape2[axis]){
                                            const auto lV = sliceLabelsB(coord2.asStdArray());
                                            if(lU != lV){

                                                const auto edge = rag.findEdge(lU,lV);
                                                const auto dataU = sliceFiltersB(c,coord[0],coord[1]);
                                                const auto dataV = sliceFiltersB(c,coord2[0],coord2[1]);

                                                VigraCoord vigraCoordU;
                                                VigraCoord vigraCoordV;
                                                vigraCoordU[0] = sliceBIndex;
                                                vigraCoordV[0] = sliceBIndex;

                                                for(size_t d=1; d<3; ++d){
                                                    vigraCoordU[d] = coord[d-1];
                                                    vigraCoordV[d] = coord2[d-1];
                                                }

                                                accChainVector[edge][c].updatePassN(dataU, vigraCoordU, pass);
                                                accChainVector[edge][c].updatePassN(dataV, vigraCoordV, pass);
                                            
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    }
                });
            } 
        }
        
        // call functor with finished acc chain
        f(accChainVector);
    }
    
    
    template<size_t DIM, class RAG, class DATA, class FEATURE_TYPE>
    void accumulateEdgeStatisticsFromFilters(
        const RAG & rag,
        const DATA & data,
        marray::View<FEATURE_TYPE> & out,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;

        typedef FEATURE_TYPE DataType;
        // TODO set nbins from outside ?
        typedef acc::StandardQuantiles<acc::UserRangeHistogram<64> > Quantiles;
        // TODO what is this DataArg buisness ?
        typedef acc::Select< acc::DataArg<1>, acc::Mean, acc::Sum, acc::Minimum, acc::Maximum, acc::Variance, Quantiles> SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> EdgeAccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
    
        // allocate a ach chain vector for each thread
        accumulateEdgeFeaturesFromFiltersWithAccChain<EdgeAccChainType>(rag, data, pOpts, threadpool, true,
        [&](
            const std::vector<std::vector<EdgeAccChainType>> & accChainVec
        ){
            size_t numberOfChannels = accChainVec.front().size();
            size_t numberOfStats    = 10;
            parallel::parallel_foreach(threadpool, accChainVec.size(),[&](
                const int tid, const int64_t edge
            ){
                vigra::TinyVector<DataType,7> quantiles;
                size_t offset = 0;
                for(size_t c = 0; c < numberOfChannels; ++c) {
                    offset = c*numberOfStats;
                    out(edge, offset)   = acc::get<acc::Mean>(accChainVec[edge][c]);
                    out(edge, offset+1) = acc::get<acc::Sum>(accChainVec[edge][c]);
                    out(edge, offset+2) = acc::get<acc::Minimum>(accChainVec[edge][c]);
                    out(edge, offset+3) = acc::get<acc::Maximum>(accChainVec[edge][c]);
                    out(edge, offset+4) = acc::get<acc::Variance>(accChainVec[edge][c]);
                    quantiles = acc::get<Quantiles>(accChainVec[edge][c]);
                    out(edge, offset+5) = quantiles[1];
                    out(edge, offset+6) = quantiles[2];
                    out(edge, offset+7) = quantiles[3];
                    out(edge, offset+8) = quantiles[4];
                    out(edge, offset+9) = quantiles[5];
                }
            });
        });
    }
    
    
    // TODO check out if 1 pass vs. 2 pass filts make a significant diff once the pipeline is running
    /* 
    template<size_t DIM, class RAG, class DATA, class FEATURE_TYPE>
    void accumulateEdgeStatisticsFromFiltersTwoPass(
        const RAG & rag,
        const DATA & data,
        marray::View<FEATURE_TYPE> & out,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;

        typedef FEATURE_TYPE DataType;
        // TODO set nbins from outside ?
        typedef acc::StandardQuantiles<acc::AutoRangeHistogram<64> > Quantiles;
        // TODO what is this DataArg buisness ?
        typedef acc::Select< acc::DataArg<1>, acc::Mean, acc::Sum, acc::Minimum, acc::Maximum, acc::Variance, acc::Skewness, acc::Kurtosis, Quantiles> SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> EdgeAccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
    
        // allocate a ach chain vector for each thread
        accumulateEdgeFeaturesFromFiltersWithAccChain<EdgeAccChainType>(rag, data, pOpts, threadpool, false,
        [&](
            const std::vector<std::vector<EdgeAccChainType>> & accChainVec
        ){
            size_t numberOfChannels = accChainVec.front().size();
            size_t numberOfStats    = 12;
            parallel::parallel_foreach(threadpool, accChainVec.size(),[&](
                const int tid, const int64_t edge
            ){
                vigra::TinyVector<DataType,7> quantiles;
                size_t offset = 0;
                for(size_t c = 0; c < numberOfChannels; ++c) {
                    offset = c*numberOfStats;
                    out(edge, offset)   = acc::get<acc::Mean>(accChainVec[edge][c]);
                    out(edge, offset+1) = acc::get<acc::Sum>(accChainVec[edge][c]);
                    out(edge, offset+2) = acc::get<acc::Minimum>(accChainVec[edge][c]);
                    out(edge, offset+3) = acc::get<acc::Maximum>(accChainVec[edge][c]);
                    out(edge, offset+4) = acc::get<acc::Variance>(accChainVec[edge][c]);
                    out(edge, offset+5) = acc::get<acc::Variance>(accChainVec[edge][c]);
                    out(edge, offset+6) = acc::get<acc::Variance>(accChainVec[edge][c]);
                    quantiles = acc::get<Quantiles>(accChainVec[edge][c]);
                    out(edge, offset+7) = quantiles[1];
                    out(edge, offset+8) = quantiles[2];
                    out(edge, offset+9) = quantiles[3];
                    out(edge, offset+10) = quantiles[4];
                    out(edge, offset+11) = quantiles[5];
                }
            });
        });
    }
    */





} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_HXX */
