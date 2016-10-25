#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FROM_BOUNDING_BOXES_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FROM_BOUNDING_BOXES_HXX

#include "nifty/graph/rag/grid_rag_accumulate.hxx"

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

        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LabelsProxyType::LabelType LabelType;
        typedef typename DATA::DataType DataType;

        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        const auto & shape = rag.shape();

        std::vector< EdgeAccChainVectorType * > perThreadEdgeAccChainVector(actualNumberOfThreads);


        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadEdgeAccChainVector[i] = new EdgeAccChainVectorType(rag.edgeIdUpperBound()+1);
        });

        const auto passesRequired = (*perThreadEdgeAccChainVector.front()).front().passesRequired();

        if(accOptions.setMinMax){
            parallel::parallel_foreach(threadpool, actualNumberOfThreads,
            [&](int tid, int i){

                vigra::HistogramOptions histogram_opt;
                histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal); 

                auto & edgeAccVec = *(perThreadEdgeAccChainVector[i]);
                for(auto & edgeAcc : edgeAccVec){
                    edgeAcc.setHistogramOptions(histogram_opt);
                }
            });
        }
        
        Coord blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(blockShape[d]+1, shape[d]);
        }
        
        const size_t nBlocks = startCoordinates.size();
        marray::Marray<LabelType> labelsBlockView(blockShapeWithBorder.begin(),blockShapeWithBorder.end());
        marray::Marray<DataType> dataBlockView(blockShapeWithBorder.begin(),blockShapeWithBorder.end());

        // do N passes of accumulator
        for(auto pass=1; pass <= passesRequired; ++pass){

            std::cout << "Pass: " << pass << std::endl;

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
                    tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                    tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);
                }
                catch (const std::runtime_error & e) {
                    std::cout << "Overlap violating and reduced" << std::endl;
                    overlapViolates = true;
                    for(auto d=0; d<DIM; ++d) {
                        blockEnd[d] -= 1;
                        blockShapeWithBorder[d] -= 1;
                    }
                    labelsBlockView.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    dataBlockView.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                    tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);
                }

                nifty::tools::parallelForEachCoordinate(
                    threadpool,
                    blockShapeWithBorder,
                    [&](int tid, const Coord & coordU) {
                    
                        // get the accumulator vector for this thread
                        auto & accVec = *(perThreadEdgeAccChainVector[tid]);

                        const auto lU = labelsBlockView(coordU.asStdArray());
                        for(size_t axis=0; axis<DIM; ++axis){
                            auto coordV = makeCoord2(coordU, axis);
                            if(coordV[axis] < blockShapeWithBorder[axis]){
                                const auto lV = labelsBlockView(coordV.asStdArray());
                                if(lU != lV){
                                    const auto edge = rag.findEdge(lU,lV);

                                    const auto dataU = dataBlockView(coordU.asStdArray());
                                    const auto dataV = dataBlockView(coordV.asStdArray());

                                    
                                    VigraCoord vigraCoordU;
                                    VigraCoord vigraCoordV;

                                    for(size_t d=0; d<DIM; ++d){
                                        vigraCoordU[d] = coordU[d];
                                        vigraCoordV[d] = coordV[d];
                                    }

                                    accVec[edge].updatePassN(dataU, vigraCoordU, pass);
                                    accVec[edge].updatePassN(dataV, vigraCoordV, pass);
                                }
                            }
                        }
                });
                
                // reset shape and marray if overlap was violating
                if(overlapViolates) {
                    for(auto d=0; d<DIM; ++d)
                        blockShapeWithBorder[d] += 1;
                    labelsBlockView.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    dataBlockView.resize(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                }
            }
        }

        auto & resultAccVec = *(perThreadEdgeAccChainVector.front());


        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, resultAccVec.size(), 
        [&](const int tid, const int64_t edge){

            for(auto t=1; t<actualNumberOfThreads; ++t){
                resultAccVec[edge].merge((*(perThreadEdgeAccChainVector[t]))[edge]);
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
    
    
    // accumulator with data
    template<class NODE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F, class COORD>
    void accumulateNodeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f,
        const AccOptions & accOptions = AccOptions()
    ){
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LabelsProxyType::LabelType LabelType;
        typedef typename DATA::DataType DataType;

        typedef array::StaticArray<int64_t, DIM> Coord;

        typedef NODE_ACC_CHAIN NodeAccChainType;
        typedef std::vector<NodeAccChainType> NodeAccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        const auto & shape = rag.shape();

        std::vector< NodeAccChainVectorType * > perThreadNodeAccChainVector(actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadNodeAccChainVector[i] = new NodeAccChainVectorType(rag.nodeIdUpperBound()+1);
        });

        const auto numberOfPasses = (*perThreadNodeAccChainVector.front()).front().passesRequired();

        if(accOptions.setMinMax){
            parallel::parallel_foreach(threadpool, actualNumberOfThreads,
            [&](int tid, int i){
                vigra::HistogramOptions histogram_opt;
                histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal); 

                auto & nodeAccVec = *(perThreadNodeAccChainVector[i]);
                for(auto & nodeAcc : nodeAccVec){
                    nodeAcc.setHistogramOptions(histogram_opt);
                }
            });
        }
        
        Coord blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(blockShape[d]+1, shape[d]);
        }
        
        const size_t nBlocks = startCoordinates.size();
        marray::Marray<LabelType> labelsBlockView(blockShapeWithBorder.begin(),blockShapeWithBorder.end());
        marray::Marray<DataType> dataBlockView(blockShapeWithBorder.begin(),blockShapeWithBorder.end());

        // do N passes of accumulator
        for(auto pass=1; pass <= numberOfPasses; ++pass){

            std::cout << "Pass: " << pass << std::endl;

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
                    tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                    tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);
                }
                catch (const std::runtime_error & e) {
                    std::cout << "Overlap violating and reduced" << std::endl;
                    overlapViolates = true;
                    for(auto d=0; d<DIM; ++d) {
                        blockEnd[d] -= 1;
                        blockShapeWithBorder[d] -= 1;
                    }
                    labelsBlockView = marray::Marray<LabelType>(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    dataBlockView = marray::Marray<DataType>(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                    tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);
                }

                nifty::tools::parallelForEachCoordinate(
                    threadpool,
                    blockShapeWithBorder,
                    [&](int tid, const Coord & coordU) {
                        
                        // get the accumulator vector for this thread
                        auto & nodeAccVec = *(perThreadNodeAccChainVector[tid]);

                        const auto lU = labelsBlockView(coordU.asStdArray());
                        const auto dataU = dataBlockView(coordU.asStdArray());
                        
                        VigraCoord vigraCoordU;
                        for(size_t d=0; d<DIM; ++d)
                            vigraCoordU[d] = coordU[d] + blockBegin[d];

                        nodeAccVec[lU].updatePassN(dataU, vigraCoordU, pass);
                });
                
                // reset shape and marray if overlap was violating
                if(overlapViolates) {
                    for(auto d=0; d<DIM; ++d)
                        blockShapeWithBorder[d] += 1;
                    labelsBlockView = marray::Marray<LabelType>(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                    dataBlockView = marray::Marray<DataType>(blockShapeWithBorder.begin(),blockShapeWithBorder.end());  
                }
            }
        }

        auto & nodeResultAccVec = *perThreadNodeAccChainVector.front();

        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, nodeResultAccVec.size(), 
        [&](const int tid, const int64_t node){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                auto & accChainVec = *(perThreadNodeAccChainVector[t]);
                nodeResultAccVec[node].merge(accChainVec[node]);           
            }
        });
        
        // call functor with finished acc chain
        f(nodeResultAccVec);


        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            delete perThreadNodeAccChainVector[i];
        });

    }
    
    
    // 9 features
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateEdgeStandartFeaturesOnePass(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        const double minVal,
        const double maxVal,
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
                const std::vector<AccChainType> & edgeAccChainVec
            ){
                using namespace vigra::acc;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    edgeFeaturesOut(edge, 0) = replaceIfNotFinite(mean,     0.0);
                    edgeFeaturesOut(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi)
                        edgeFeaturesOut(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                }); 
            },
            AccOptions(minVal, maxVal)
        );

    }
    
    // 11
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateEdgeStandartFeaturesTwoPass(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        const double minVal,
        const double maxVal,
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
                const std::vector<AccChainType> & edgeAccChainVec
            ){
                using namespace vigra::acc;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    edgeFeaturesOut(edge, 0) = replaceIfNotFinite(mean,     0.0);
                    edgeFeaturesOut(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    edgeFeaturesOut(edge, 2) = replaceIfNotFinite(get<acc::Skewness>(chain), 0.0);
                    edgeFeaturesOut(edge, 3) = replaceIfNotFinite(get<acc::Kurtosis>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi)
                        edgeFeaturesOut(edge, 4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                }); 
            },
            AccOptions(minVal, maxVal)
        );

    }


    // 9 features
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateNodeStandartFeaturesOnePass(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        const double minVal,
        const double maxVal,
        marray::View<FEATURE_TYPE> & nodeFeaturesOut,
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
        
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();
        

        accumulateNodeFeaturesWithAccChain<AccChainType>(
            rag, 
            data, 
            startCoordinates,
            blockShape,
            pOpts, 
            threadpool,
            [&](const std::vector<AccChainType> & nodeAccChainVec) {
                using namespace vigra::acc;
                parallel::parallel_foreach(threadpool, nodeAccChainVec.size(),[&](
                    const int tid, const int64_t node
                ){
                    const auto & chain = nodeAccChainVec[node];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    nodeFeaturesOut(node, 0) = replaceIfNotFinite(mean,     0.0);
                    nodeFeaturesOut(node, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi){
                        nodeFeaturesOut(node, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                });
            },
            AccOptions(minVal, maxVal)
        );
    }
    
    
    // 11 features
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE, class COORD>
    void accumulateNodeStandartFeaturesTwoPass(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const std::vector<COORD> & startCoordinates,
        const COORD & blockShape,
        const double minVal,
        const double maxVal,
        marray::View<FEATURE_TYPE> & nodeFeaturesOut,
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
        
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();
        

        accumulateNodeFeaturesWithAccChain<AccChainType>(
            rag, 
            data, 
            startCoordinates,
            blockShape,
            pOpts, 
            threadpool,
            [&](const std::vector<AccChainType> & nodeAccChainVec) {
                using namespace vigra::acc;
                parallel::parallel_foreach(threadpool, nodeAccChainVec.size(),[&](
                    const int tid, const int64_t node
                ){
                    const auto & chain = nodeAccChainVec[node];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    nodeFeaturesOut(node, 0) = replaceIfNotFinite(mean,     0.0);
                    nodeFeaturesOut(node, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    nodeFeaturesOut(node, 1) = replaceIfNotFinite(get<acc::Skewness>(chain), 0.0);
                    nodeFeaturesOut(node, 1) = replaceIfNotFinite(get<acc::Kurtosis>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi){
                        nodeFeaturesOut(node, 4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                });
            },
            AccOptions(minVal, maxVal)
        );
    }

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FROM_BOUNDING_BOXES_HXX */
