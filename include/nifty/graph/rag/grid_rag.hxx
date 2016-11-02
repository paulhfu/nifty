#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_HXX


#include <random>
#include <functional>
#include <ctime>
#include <stack>
#include <algorithm>

//#include <parallel/algorithm>
#include <unordered_set>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{

template<class COORD>
COORD makeCoord2(const COORD & coord,const size_t axis){
    COORD coord2 = coord;
    coord2[axis] += 1;
    return coord2;
};



template<size_t DIM, class LABEL_TYPE>
class ExplicitLabels;

template<class LABELS_PROXY>
struct RefHelper{
    typedef const LABELS_PROXY & type;
};

template<size_t DIM, class LABEL_TYPE>
struct RefHelper<ExplicitLabels<DIM, LABEL_TYPE>>{
    typedef ExplicitLabels<DIM, LABEL_TYPE> type;
};



template<size_t DIM, class LABELS_PROXY>
class GridRag : public UndirectedGraph<>{
public:
    struct DontComputeRag{};
    typedef LABELS_PROXY LabelsProxy;
    struct Settings{
        Settings()
        :   numberOfThreads(-1),
            blockShape()
        {
            for(auto d=0; d<DIM; ++d)
                blockShape[d] = 100;
        }
        int numberOfThreads;
        array::StaticArray<int64_t, DIM> blockShape;
    };

    typedef UndirectedGraph<> BaseType;
    typedef GridRag<DIM, LABELS_PROXY> SelfType;
    typedef array::StaticArray<int64_t, DIM> ShapeType;
    typedef typename LabelsProxy::LabelType LabelType;
    typedef typename BaseType::EdgeInternalType EdgeType;

    friend class detail_rag::ComputeRag< SelfType >;


    GridRag(const LabelsProxy & labelsProxy, const Settings & settings = Settings())
    :   settings_(settings),
        labelsProxy_(labelsProxy)
    {
        detail_rag::ComputeRag< SelfType >::computeRag(*this, settings_);
    }
    
    // need to expose this for deserializing the rag
    GridRag(const LabelsProxy & labelsProxy, const Settings & settings, const DontComputeRag)
    :   settings_(settings),
        labelsProxy_(labelsProxy){

    }
    

    template<class ITER>
    GridRag(
        const LabelsProxy & labelsProxy, 
        ITER serializationBegin,
        const Settings & settings = Settings()
    )
    :   settings_(settings),
        labelsProxy_(labelsProxy)
    {
        this->deserialize(serializationBegin);
    }

    const LabelsProxy & labelsProxy() const {
        return labelsProxy_;
    }

    const ShapeType & shape()const{
        return labelsProxy_.shape();
    }
    
    /* On second thought, we don't really need these apart from the construction 
    // reimplement serialization due to the Settings
    uint64_t serializationSize() const{
        uint64_t size = BaseType::serializationSize();
        size += 1;
        size += DIM;
        return size;
    }

    template<class ITER>
    void serialize(ITER iter) const {
        BaseType::serialize(iter);
        *iter = settings_.numberOfThreads;  
        iter++;
        for(auto d = 0; d < DIM; ++d) {
            *iter = settings_.blockShape[d];  
            iter++;
        }
    }
    
    template<class ITER>
    void deserialize(ITER iter) {
        BaseType::deserialize(iter);
        settings_.numberOfThreads = *iter;  
        iter++;
        for(auto d = 0; d < DIM; ++d) {
            settings_.blockShape[d] = *iter;  
            iter++;
        }
    }
    */


protected:
    typedef typename RefHelper<LABELS_PROXY>::type StorageType;
    Settings settings_;
    StorageType labelsProxy_;

};


template<class LABELS_PROXY>
class GridRag3DStacked2D;


template<unsigned int DIM, class LABEL_TYPE>
using ExplicitLabelsGridRag = GridRag<DIM, ExplicitLabels<DIM, LABEL_TYPE> > ; 


} // end namespace graph
} // end namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_HXX */
