#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif

#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace graph{

    template<class T, unsigned int NBINS>
    class DefaultAcc{
        
    public:
        DefaultAcc()
        :   minVal_(std::numeric_limits<T>::infinity()),
            maxVal_(-1.0*std::numeric_limits<T>::infinity()),
            sumVal_(0.0),
            countVal_(0.0)
        {

        }
        void accumulate(const T & val){
            minVal_ = std::min(val, minVal_);
            maxVal_ = std::max(val, maxVal_);
            sumVal_ += val;
            ++countVal_;
        }
        void merge(const DefaultAcc & other){
            minVal_ = std::min(other.minVal_, minVal_);
            maxVal_ = std::max(other.maxVal_, maxVal_);
            sumVal_ += other.sumVal_;
            countVal_ +=other.countVal_;
        }
        void getFeatures(T * featuresOut){
            featuresOut[0] = minVal_;
            featuresOut[1] = maxVal_;
            featuresOut[2] = sumVal_;
            featuresOut[3] = sumVal_/static_cast<T>(countVal_);
            featuresOut[4] = static_cast<T>(countVal_);
        }
    private:
        T minVal_,maxVal_,sumVal_;
        uint64_t countVal_;
    };


    template<class GRAPH, class T, unsigned int NBINS, template<class> class ITEM_MAP >
    class DefaultAccMapBase 
    {
    public:
        typedef GRAPH Graph;
        DefaultAccMapBase(const Graph & graph, const T globalMinVal = std::numeric_limits<T>::infinity(), const T globalMaxVal=-1.0*std::numeric_limits<T>::infinity())
        :   graph_(graph),
            currentPass_(0),
            globalMinVal_(globalMinVal),
            globalMaxVal_(globalMaxVal),
            accs_(graph){

        }
        const Graph & graph()const{
            return graph_;
        }
        void accumulate(const uint64_t item, const T & val){
            if(currentPass_ == 0){
                accs_[item].accumulate(val);
            }
        }
        void startPass(const size_t passIndex){
            currentPass_ = passIndex;
        }
        size_t numberOfPasses()const{
            return 1.0;
        }
        void merge(const uint64_t alive,  const uint64_t dead){
            accs_[alive].merge(accs_[dead]);
        }
        uint64_t numberOfFeatures() const {
            return 5;
        }
        void getFeatures(const uint64_t item, T * featuresOut){
            accs_[item].getFeatures(featuresOut);
        }   
        void resetFrom(const DefaultAccMapBase & other){
            std::copy(other.accs_.begin(), other.accs_.end(), accs_.begin());
            globalMinVal_ = other.globalMinVal_;
            globalMinVal_ = other.globalMaxVal_;
            currentPass_ = other.currentPass_;
        }
    private:
        const GRAPH & graph_;
        size_t currentPass_;
        T globalMinVal_;
        T globalMaxVal_;
        ITEM_MAP<DefaultAcc<T, NBINS> > accs_;
    };



    template<class GRAPH, class T, unsigned int NBINS=40>
    class DefaultAccEdgeMap : 
        public  DefaultAccMapBase<GRAPH, T, NBINS, GRAPH:: template EdgeMap >
    {
    public:
        typedef GRAPH Graph;
        //typedef typename Graph:: template EdgeMap<DefaultAcc<T, NBINS> > BasesBaseType;
        typedef DefaultAccMapBase<Graph, T, NBINS, Graph:: template EdgeMap >  BaseType;
        using BaseType::BaseType;
    };

    template<class GRAPH, class T, unsigned int NBINS=40>
    class DefaultAccNodeMap : 
        public  DefaultAccMapBase<GRAPH, T, NBINS, GRAPH:: template NodeMap >
    {
    public:
        typedef GRAPH Graph;
        //typedef typename Graph:: template NodeMap<DefaultAcc<T, NBINS> > BasesBaseType;
        typedef DefaultAccMapBase<Graph, T, NBINS, Graph:: template NodeMap >  BaseType;
        using BaseType::BaseType;
    };



    

    

    template< size_t DIM, class LABELS_TYPE, class T, class EDGE_MAP, class NODE_MAP>
    void gridRagAccumulateFeatures(
        const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
        nifty::marray::View<T> data,
        EDGE_MAP & edgeMap,
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, DIM> Coord;

        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto labels = labelsProxy.labels(); 
        
        const auto numberOfPasses =  std::max(edgeMap.numberOfPasses(),nodeMap.numberOfPasses());
        for(size_t p=0; p<numberOfPasses; ++p){
            // start path p
            edgeMap.startPass(p);
            nodeMap.startPass(p);

            nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
                const auto lU = labels(coord);
                const auto dU = data(coord);
                nodeMap.accumulate(lU, dU);
                for(size_t axis=0; axis<DIM; ++axis){
                    Coord coord2 = coord;
                    coord2[axis] += 1;
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2);
                        if(lU != lV){
                            const auto e = graph.findEdge(lU, lV);
                            const auto dV = data(coord2);
                            edgeMap.accumulate(e, dU);
                            edgeMap.accumulate(e, dV);
                        }
                    }
                }
            });
        }
    }



    template<size_t DIM, class LABELS_TYPE, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
        nifty::marray::View<LABELS> data,
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, DIM> Coord;

        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto labels = labelsProxy.labels(); 

        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());
        


        nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
            const auto x = coord[0];
            const auto y = coord[1];
            const auto node = labels(x,y);            
            const auto l  = data(x,y);
            overlaps[node][l] += 1;
        });

        for(const auto node : graph.nodes()){
            const auto & ol = overlaps[node];
            // find max ol 
            uint64_t maxOl = 0 ;
            uint64_t maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[node] = maxOlLabel;
        }
    }
    
    template<class LABELS_PROXY, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const GridRagStacked2D<LABELS_PROXY> & graph,
        const LABELS & data,                         
        NODE_MAP &  nodeMap,
        const int numberOfThreads = -1
    ){
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LABELS_PROXY::LabelType LabelType;
        typedef typename LabelsProxyType::BlockStorageType LabelsBlockStorage;
        typedef typename tools::BlockStorageSelector<LABELS>::type DataBlockStorage;
        
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        
        NIFTY_CHECK_OP(data.shape(0),==,shape[0], "Shape along z does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(data.shape(2),==,shape[2], "Shape along x does not agree")
        
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});


        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());
        
        LabelsBlockStorage sliceLabelsStorage(threadpool, sliceShape3, nThreads);
        DataBlockStorage   sliceDataStorage(threadpool, sliceShape3, nThreads);

        // TODO could probably use for forEachBlock instead
        parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

            // fetch the data for the slice
            auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);
            auto sliceDataFlat3DView   = sliceDataStorage.getView(tid);
            
            const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
            const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
            
            tools::readSubarray(labelsProxy, blockBegin, blockEnd, sliceLabelsFlat3DView);
            tools::readSubarray(data, blockBegin, blockEnd, sliceDataFlat3DView);
            
            auto sliceLabels = sliceLabelsFlat3DView.squeezedView();
            auto sliceData = sliceDataFlat3DView.squeezedView();
            
            nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                const auto node = sliceLabels( coord.asStdArray() );            
                const auto l    = sliceData( coord.asStdArray() );
                overlaps[node][l] += 1;
            });

        });
        
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


} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX */
