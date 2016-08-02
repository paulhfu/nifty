#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_CHUNKED_HXX

#include <random>
#include <functional>
#include <ctime>
#include <stack>
#include <algorithm>

// for strange reason travis does not find the boost flat set
#ifdef WITHIN_TRAVIS
#include <set>
#define __setimpl std::set
#else
#include <boost/container/flat_set.hpp>
#define __setimpl boost::container::flat_set
#endif

//#include <parallel/algorithm>
#include <unordered_set>

#include "nifty/graph/rag/grid_rag_labels_chunked.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag_chunked.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{

template<class LabelsProxy>
//class GridRagSliced : public GridRag<3, LabelsProxy>{
class GridRagSliced : public UndirectedGraph<>{

public:
    
    typedef GridRagSliced<LabelsProxy> SelfType;
    friend class detail_rag::ComputeRag< SelfType >;
    
    // TODO find the meaningful settings for the gridrag
    struct Settings{
        int numberOfThreads{1};
    };
    
    GridRagSliced(const std::string & labelFile, const std::string & labelKey, const Settings & settings = Settings())
    :   settings_(settings),
        labelsProxy_(labelFile, labelKey)
    {
        // make sure that we have chunks of shape (1,Y,X) TODO check the chunk shape again, but only give a warning if it doesn't fit
        //NIFTY_CHECK_OP(labelsProxy.labels().chunkShape(0),==,1,"Z chunks have to be of size 1 for sliced rag")
        detail_rag::ComputeRag< SelfType >::computeRag(*this, settings_);
    }
    
    const LabelsProxy & labelsProxy() const {
        return labelsProxy_;
    }

    const typename LabelsProxy::ViewType & labels() const {
        return labelsProxy_.labels();
    }
        
    

private:
    Settings settings_;
    LabelsProxy labelsProxy_;

};


template<class LABEL_TYPE>
using ChunkedLabelsGridRagSliced = GridRagSliced<ChunkedLabels<3, LABEL_TYPE> >; 


} // namespace graph
} // namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_CHUNKED_HXX */
