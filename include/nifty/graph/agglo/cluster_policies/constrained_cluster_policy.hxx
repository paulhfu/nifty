#pragma once

#include <functional>
#include <array>
#include <iostream>

#include "nifty/histogram/histogram.hxx"
#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"


namespace nifty{
namespace graph{
namespace agglo{







template<
    class GRAPH,bool ENABLE_UCM
>
class ConstrainedPolicy{

    typedef ConstrainedPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

private:
    typedef typename GRAPH:: template EdgeMap<uint64_t> UInt64EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;
    typedef typename GRAPH:: template NodeMap<int> Int64NodeMap;

public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgeIndicatorsType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;


    typedef Int64NodeMap                                GTlabelsType;
    typedef UInt64EdgeMap                               MergeTimesType;

    struct SettingsType : public EdgeWeightedClusterPolicySettings
    {

        float threshold{0.5};
        int nb_iterations{-1};
        int ignore_label{-1};
        bool constrained{true}; // Avoid merging across GT boundary
        int bincount{256};  // This is the parameter that can be passed to the policy
        bool verbose{false};
    };
    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types
    // const static size_t NumberOfBins = 256;
    typedef nifty::histogram::Histogram<float> HistogramType;
    //typedef std::array<float, NumberOfBins> HistogramType;     
    typedef typename GRAPH:: template EdgeMap<HistogramType> EdgeHistogramMap;


    typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

public:

//    TODO: why this template? I guess it's for PyBind11 types
    template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES, class GT_LABELS>
    ConstrainedPolicy(const GraphType &,
//                      const EdgeContractionGraphType & ,
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const NODE_SIZES & ,
                              const GT_LABELS & ,
                              const SettingsType & settings = SettingsType());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone();

    bool edgeIsConstrained(const uint64_t);

    // callback called by edge contraction graph
    
    EdgeContractionGraphType & edgeContractionGraph();


    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);


    const EdgeIndicatorsType & edgeIndicators() const {
        return edgeIndicators_;
    }
    const EdgeSizesType & edgeSizes() const {
        return edgeSizes_;
    }
    const MergeTimesType & mergeTimes() const {
        return mergeTimes_;
    }

    const NodeSizesType & nodeSizes() const {
        return nodeSizes_;
    }

    const EdgeIndicatorsType & loss_targets() const {
        return loss_targets_;
    }

    const EdgeIndicatorsType & loss_weights() const {
        return loss_weights_;
    }



    bool isReallyDone() const {
        return isReallyDone_;
    }


private:
    float histogramToMedian(const uint64_t edge) const;

    // const EdgeIndicatorsType & edgeIndicators() const {
    //     return edgeIndicators_;
    // }
    // const EdgeSizesType & edgeSizes() const {
    //     return edgeSizes_;
    // }
    // const NodeSizesType & nodeSizes() const {
    //     return nodeSizes_;
    // }
    
private:
    // INPUT
    const GraphType &   graph_;
    EdgeIndicatorsType  edgeIndicators_;
    EdgeSizesType       edgeSizes_;
    NodeSizesType       nodeSizes_;


    MergeTimesType      mergeTimes_;
    GTlabelsType        GTlabels_;
    EdgeIndicatorsType  loss_targets_;
    EdgeIndicatorsType  loss_weights_;
    uint64_t            nb_correct_steps_;
    uint64_t            nb_wrong_steps_;


    SettingsType            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

    EdgeHistogramMap histograms_;

    uint64_t time_;
    // Keeps track of edges in PQ (some are alive, but deleted from the PQ because constrained
    uint64_t nb_active_edges_;

//    uint64_t nb_steps_;
//    uint64_t ignore_label_;
    bool     isReallyDone_;


};

//    Define the methods:

//        TODO: can I copy this twice with the passed edgeContractionGraph
template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES, class GT_LABELS>
inline ConstrainedPolicy<GRAPH, ENABLE_UCM>::
ConstrainedPolicy(
    const GraphType & graph,
//    const EdgeContractionGraphType & contractionGraph,
    const EDGE_INDICATORS & edgeIndicators,
    const EDGE_SIZES      & edgeSizes,
    const NODE_SIZES      & nodeSizes,
    const GT_LABELS    & GTlabels,
    const SettingsType & settings
)
:   graph_(graph),
    edgeIndicators_(graph),
    edgeSizes_(graph),
    nodeSizes_(graph),
    mergeTimes_(graph, graph_.numberOfNodes()),
    GTlabels_(graph),
    settings_(settings),
//    TODO: can this work...?
//    edgeContractionGraph_(contractionGraph),
    edgeContractionGraph_(graph, *this),
//    TODO: change with a contracted graph:
    nb_active_edges_(graph_.numberOfEdges()),
    nb_correct_steps_(0),
    nb_wrong_steps_(0),
    loss_weights_(graph, 0.),
    loss_targets_(graph, 0.),
    pq_(graph.edgeIdUpperBound()+1),
    histograms_(graph, HistogramType(0,1,settings.bincount)),
    time_(0),
    isReallyDone_(false)
{
    graph_.forEachEdge([&](const uint64_t edge){


        const auto val = edgeIndicators[edge];


        // currently the value itself
        // is the median
        const auto size = edgeSizes[edge];
        edgeSizes_[edge] = size;

        // put in histogram
        histograms_[edge].insert(val, size);

        // put in pq
        pq_.push(edge, val);

    });

    graph_.forEachNode([&](const uint64_t node){
        // Save GT labels:
        const auto GT_lab = GTlabels[node];
        GTlabels_[node] = GT_lab;

        nodeSizes_[node] = nodeSizes[node];
    });
    //this->initializeWeights();
}


//        This should be called only when we are sure that a not constrained edge is still
//        available in PQ
template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
edgeIsConstrained(
        const uint64_t edge
){
    const auto uv = edgeContractionGraph_.uv(edge);
    const auto u = uv.first;
    const auto v = uv.second;
    const auto reprU = edgeContractionGraph_.findRepresentativeNode(u);
    const auto reprV = edgeContractionGraph_.findRepresentativeNode(v);

    if (GTlabels_[reprU] == GTlabels_[reprV]) {
         std::cout << "EdgeNC" << edge << "\n";
        return false;
    } else {
        std::cout << "EdgeC" << edge << "\n";
        return true;
    }
}


template<class GRAPH, bool ENABLE_UCM>
inline bool 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
isDone() {
    if (time_>=settings_.nb_iterations)
        return true;

    // Find the next not-constrained edge:
    while (true) {
        if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop) {
            isReallyDone_ = true;
            std::cout << "1 node, stop\n";
            return true;
        }
        if(edgeContractionGraph_.numberOfEdges() <= settings_.numberOfEdgesStop) {
            isReallyDone_ = true;
            std::cout << "0 edges, stop\n";
            return  true;
        }

        // All remaining edges could be constrained, so PQ is empty
        // although there are alive edges:
        if (nb_active_edges_<=0) {
            std::cout << "0 active edges, stop\n";
            isReallyDone_ = true;
            return  true;
        }
        if (pq_.topPriority() >= settings_.threshold) {
            std::cout << "threashold reached, stop" << pq_.topPriority() <<"\n";
            isReallyDone_ = true;
            return  true;
        }

        if (!settings_.constrained)
            break;

        const auto edgeToContractNextAndPriority = this->edgeToContractNext();
        const auto edgeToContractNext = edgeToContractNextAndPriority.first;

        if (! this->edgeIsConstrained(edgeToContractNext))
            break;

        // Delete constrained edge from PQ:
        pq_.deleteItem(edgeToContractNext);
        --nb_active_edges_;

        // Remember about correct step:
        loss_targets_[edgeToContractNext] = -1.; // We should not merge (and we would have)
        loss_weights_[edgeToContractNext] = 1.; // For the moment all equally weighted
        ++nb_wrong_steps_;

    }

    return false;
}


template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    // This is currently not used. Merge time is done in the agglomeration class:
    mergeTimes_[edgeToContract] = time_;
    std::cout << "Contract edge: " << edgeToContract << "\n";
    ++time_;

    if (settings_.constrained) {
        // Remember about correct step:
        loss_targets_[edgeToContract] = 1.; // We should merge (and we did)
        loss_weights_[edgeToContract] = 1.; // For the moment all equally weighted
        ++nb_correct_steps_;
    }


    pq_.deleteItem(edgeToContract);
    --nb_active_edges_;
}

template<class GRAPH, bool ENABLE_UCM>
inline typename ConstrainedPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType &
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    std::cout << "Merge edges: " << aliveEdge << " "<< deadEdge << "\n";
    pq_.deleteItem(deadEdge);
    --nb_active_edges_;

    // merging the histogram is just adding
    auto & ha = histograms_[aliveEdge];
    auto & hd = histograms_[deadEdge];
    ha.merge(hd);
    pq_.push(aliveEdge, histogramToMedian(aliveEdge));
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){

}

template<class GRAPH, bool ENABLE_UCM>
inline float 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
histogramToMedian(
    const uint64_t edge
) const{
    // todo optimize me
    float median;
    const float rank=0.5;
    nifty::histogram::quantiles(histograms_[edge],&rank,&rank+1,&median);
    return median;
}





} // namespace agglo
} // namespace nifty::graph
} // namespace nifty


