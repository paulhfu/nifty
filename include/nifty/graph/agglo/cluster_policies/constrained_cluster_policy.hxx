#pragma once

#include <functional>
#include <array>
#include <iostream>
#include <vector>

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
    typedef typename GRAPH:: template EdgeMap<bool> BoolEdgeMap;
    typedef typename GRAPH:: template EdgeMap<std::vector<uint64_t>> VectorEdgeMap;


public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgeIndicatorsType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;


    typedef BoolEdgeMap                                 EdgeFlagType;
    typedef Int64NodeMap                                GTlabelsType;
    typedef UInt64EdgeMap                               MergeTimesType;
    typedef VectorEdgeMap                               BacktrackEdgesType;

    struct SettingsType : public EdgeWeightedClusterPolicySettings
    {

        float threshold{0.5};
        int ignore_label{-1};
        bool constrained{true}; // Avoid merging across GT boundary
        bool computeLossData{true}; // Option to avoid edge UF backtracking
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

//    TODO: in this policy the edge_indicators are better passed to runMileStep() method
    template<class EDGE_INDICATORS, class EDGE_SIZES, class NODE_SIZES, class GT_LABELS>
    ConstrainedPolicy(const GraphType &,
//                      const EdgeContractionGraphType & ,
                              const EDGE_INDICATORS & , 
                              const EDGE_SIZES & , 
                              const NODE_SIZES & ,
                              const GT_LABELS & ,
                              const SettingsType & settings = SettingsType());

    template<class EDGE_INDICATORS>
    void updateEdgeIndicators(EDGE_INDICATORS  &);

    template<class NODE_SIZES,
            class NODE_LABELS,
            class EDGE_SIZES,
            class EDGE_INDICATORS,
            class DEND_HIGH,
            class MERGING_TIME,
            class LOSS_TARGETS,
            class LOSS_WEIGHTS>
    void collectDataMilestep(
            NODE_SIZES        & ,
            NODE_LABELS        & ,
            EDGE_SIZES        & ,
            EDGE_INDICATORS   & ,
            DEND_HIGH         & ,
            MERGING_TIME      & ,
            LOSS_TARGETS      & ,
            LOSS_WEIGHTS      &
    ) const;


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone();

    bool edgeIsConstrained(const uint64_t);
    void computeFinalTargets();
    template<class EDGE_INDICATORS>
    bool runMileStep(const int, EDGE_INDICATORS &);

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
    const GraphType & graph() const {
        return graph_;
    }


    const MergeTimesType & mergeTimes() const {
        return mergeTimes_;
    }

    const NodeSizesType & nodeSizes() const {
        return nodeSizes_;
    }

//    template<class EDGE_MAP>
//    void lossTargets(EDGE_MAP & edgeMap) const {
////        const auto & cgraph = edgeContractionGraph_;
////        const auto & graph = cgraph.graph();
//        for(const auto edge : graph_.edges()){
//            const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
//            edgeMap[edge] = loss_targets_[cEdge];
//        }
//    }
//
//    template<class EDGE_MAP>
//    void lossWeights(EDGE_MAP & edgeMap) const {
//        for(const auto edge : graph_.edges()){
//            const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
//            edgeMap[edge] = loss_weights_[cEdge];
//        }
//    }

//    const EdgeIndicatorsType & lossWeights() const {
//        return loss_weights_;
//    }

    uint64_t time() const {
        return time_;
    }


    bool isReallyDone() const {
        return isReallyDone_;
    }

    void resetDataBeforeMilestep(
            const int nb_iterations_in_milestep
    ) {
        nb_iterations_in_milestep_ = nb_iterations_in_milestep;
        mileStepTimeOffset_ = time_;
        std::fill(loss_targets_.begin(), loss_targets_.end(), 0.);
        std::fill(loss_weights_.begin(), loss_weights_.end(), 0.);
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
    EdgeFlagType        flagAliveEdges_;
    BacktrackEdgesType  backtrackEdges_;
    EdgeIndicatorsType  weightedSum_; // Used for the average

    MergeTimesType      mergeTimes_;
    EdgeIndicatorsType  dendHeigh_;
    GTlabelsType        GTlabels_;
    EdgeIndicatorsType  loss_targets_;
    EdgeIndicatorsType  loss_weights_;
    uint64_t            nb_correct_mergers_;
    uint64_t            nb_wrong_mergers_;
    uint64_t            nb_correct_splits_;
    uint64_t            nb_wrong_splits_;
    int                 nb_iterations_in_milestep_;


    SettingsType            settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

    EdgeHistogramMap histograms_;

    uint64_t time_;
    uint64_t mileStepTimeOffset_;
    // Keeps track of edges in PQ (some are alive, but deleted from the PQ because constrained
    uint64_t nb_active_edges_;

//    uint64_t nb_steps_;
//    uint64_t ignore_label_;
    bool     isReallyDone_;


};

//    Define the methods:

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
    weightedSum_(graph),
    flagAliveEdges_(graph, true),
    // TODO: bad, find better way to initialize this..
    dendHeigh_(graph, 2.),
    GTlabels_(graph),
    settings_(settings),
    edgeContractionGraph_(graph, *this),
    nb_active_edges_(graph_.numberOfEdges()),
    nb_correct_mergers_(0),
    nb_wrong_mergers_(0),
    nb_correct_splits_(0),
    nb_wrong_splits_(0),
    nb_iterations_in_milestep_(-1),
    loss_weights_(graph, 0.),
    loss_targets_(graph, 0.),
    pq_(graph.edgeIdUpperBound()+1),
    histograms_(graph, HistogramType(0,1,settings.bincount)),
    time_(0),
    backtrackEdges_(graph),
    mileStepTimeOffset_(0),
    isReallyDone_(false)
{
    graph_.forEachEdge([&](const uint64_t edge){


        const auto val = edgeIndicators[edge];
        edgeIndicators_[edge] = val;
        if (settings_.computeLossData)
            backtrackEdges_[edge].reserve(5);

        // currently the value itself
        // is the median
        const auto size = edgeSizes[edge];
        edgeSizes_[edge] = size;

        // put in histogram
        histograms_[edge].insert(val, size);

        // Data for average:
        weightedSum_[edge] = size*val;

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
//         std::cout << "EdgeNC" << edge << "\n";
        return false;
    } else {
//        std::cout << "EdgeC" << edge << "\n";
        return true;
    }
}

template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS>
inline void
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
updateEdgeIndicators(EDGE_INDICATORS & newEdgeIndicators) {
    // TODO: what does it happen to the old memory...? Clear is
//    auto newHistograms_ = EdgeHistogramMap(graph_, HistogramType(0,1,settings_.bincount));
//    histograms_ = newHistograms_;
//    std::fill(edgeIndicators_.begin(), edgeIndicators_.end(), 0.);

    auto reprEdgesBoolMap = BoolEdgeMap(graph_, false);
    // Reset this to the number of active edges in the edge_contraction_graph:
    nb_active_edges_ =  (uint64_t) 0;
    graph_.forEachEdge([&](const uint64_t edge){
        // Reset historgram and edgeIndicatorsMap. Previous statistics in the histogram are lost.
        histograms_[edge].clear();
        edgeIndicators_[edge] = 0.;
        weightedSum_[edge] = 0.;

        // Insert updated values in histogram and PQ (only for alive representative edges):
        const auto reprEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
        if (flagAliveEdges_[reprEdge] && !reprEdgesBoolMap[reprEdge]) {
            ++nb_active_edges_;
            const auto val = newEdgeIndicators[reprEdge];
            edgeIndicators_[reprEdge] = val;

            const auto size = edgeSizes_[reprEdge];
            edgeSizes_[reprEdge] = size;

            // Put in histogram and update values in PQ:
            std::cout << "(edge = " << reprEdge;
            std::cout << "; val = " << val;
            std::cout << "; size = " << size << ")\n";
            histograms_[reprEdge].insert(val, size);
            pq_.push(reprEdge, val);
            weightedSum_[reprEdge] = val*size;

            reprEdgesBoolMap[reprEdge] = true;
        }
    });

}


// TODO: optimize me: here we loop over all initial edges...
template<class GRAPH, bool ENABLE_UCM>
inline void
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
computeFinalTargets() {
    if (settings_.constrained && settings_.computeLossData) {
        // Loop over all alive edges (only parents ID are fine, we map later):
        for (const auto edge : graph_.edges()) {
            const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
            if (flagAliveEdges_[cEdge]) {
                loss_weights_[edge] = 1.; // For the moment all equally weighted
                if (this->edgeIsConstrained(edge)) {
                    loss_targets_[edge] = -1.; // We should not merge (and we did not)
//                  TODO: find better way to compute these numbers
//                    ++nb_correct_splits_;
                } else {
                    loss_targets_[edge] = 1.; // We should merge (and we did not)
//                    ++nb_wrong_splits_;
                }
            }
        }

        std::cout << "FINAL STATS HC:\n";
        std::cout << "Mergers: (+" << nb_correct_mergers_ << ", -" << nb_wrong_mergers_ << ")\n";
        std::cout << "Final splits: (+" << nb_correct_splits_ << ", -" << nb_wrong_splits_ << ")\n";
    }
}

template<class GRAPH, bool ENABLE_UCM>
inline bool 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
isDone() {
    const auto mileStepTime = time_ - mileStepTimeOffset_;
    if ( mileStepTime>=nb_iterations_in_milestep_)
        return true;

    // Find the next not-constrained edge:
    while (true) {
        if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop) {
            isReallyDone_ = true;
            this->computeFinalTargets();
//            std::cout << "1 node, stop\n";
            return true;
        }
        if(edgeContractionGraph_.numberOfEdges() <= settings_.numberOfEdgesStop) {
            isReallyDone_ = true;
            this->computeFinalTargets();
//            std::cout << "0 edges, stop\n";
            return  true;
        }

        // All remaining edges could be constrained, so PQ is empty
        // although there are alive edges:
        if (nb_active_edges_<=0) {
//            std::cout << "0 active edges, stop\n";
            isReallyDone_ = true;
            this->computeFinalTargets();
            return  true;
        }
        if (pq_.topPriority() >= settings_.threshold) {
//            std::cout << "threashold reached, stop" << pq_.topPriority() <<"\n";
            isReallyDone_ = true;
            this->computeFinalTargets();
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

        // Remember about wrong step:
        if (settings_.computeLossData) {
            loss_targets_[edgeToContractNext] = -1.; // We should not merge (and we would have)
            loss_weights_[edgeToContractNext] = 1.; // For the moment all equally weighted
            for (auto it = backtrackEdges_[edgeToContractNext].begin();
                 it != backtrackEdges_[edgeToContractNext].end(); it++) {
                const auto edge = *it;
                std::cout << "Write mistake in " << edgeToContractNext << "to: " << edge << "\n";
                loss_targets_[edge] = -1.; // We should not merge (and we would have)
                loss_weights_[edge] = 1.;
            }
            std::cout << "Moving to next edge in PQ";
        }
        ++nb_wrong_mergers_;

    }

    return false;
}

// UCM of data after a mileStep:
template<class GRAPH, bool ENABLE_UCM>
template<class NODE_SIZES,
        class NODE_LABELS,
        class EDGE_SIZES,
        class EDGE_INDICATORS,
        class DEND_HIGH,
        class MERGING_TIME,
        class LOSS_TARGETS,
        class LOSS_WEIGHTS>
inline void
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
collectDataMilestep(
    NODE_SIZES        & nodeSizes,
    NODE_LABELS        & nodeLabels,
    EDGE_SIZES        & edgeSizes,
    EDGE_INDICATORS   & edgeIndicators,
    DEND_HIGH         & dendHeigh,
    MERGING_TIME      & mergeTimes,
    LOSS_TARGETS      & lossTargets,
    LOSS_WEIGHTS      & lossWeights
) const {
    for(const auto edge : graph_.edges()){
        const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
        dendHeigh[edge] = dendHeigh_[cEdge];
        mergeTimes[edge] = mergeTimes_[cEdge];
        lossTargets[edge] = loss_targets_[cEdge];
        lossWeights[edge] = loss_weights_[cEdge];
        // Map size and indicators only to alive edges:
        if (flagAliveEdges_[cEdge]) {
            edgeSizes[edge]     = edgeSizes_[cEdge];
            edgeIndicators[edge] = edgeIndicators_[cEdge];
        } else {
            edgeSizes[edge]     = -2.0;
            edgeIndicators[edge] = -2.0;
        }
    }

    for(const auto node : graph_.nodes()) {
        const auto cNode = edgeContractionGraph_.findRepresentativeNode(node);
        nodeSizes[node] = nodeSizes_[cNode];
        nodeLabels[node] = (float) cNode;
    }
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    // Update data:
    mergeTimes_[edgeToContract] = time_;
    dendHeigh_[edgeToContract] = pq_.topPriority();
    std::cout << "Contract edge: " << edgeToContract << "\n";
    ++time_;

    if (settings_.constrained) {
        // Remember about correct step:
        loss_targets_[edgeToContract] = 1.; // We should merge (and we did)
        loss_weights_[edgeToContract] = 1.; // For the moment all equally weighted
        ++nb_correct_mergers_;
    }


    pq_.deleteItem(edgeToContract);
    flagAliveEdges_[edgeToContract] = false;
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
    nodeSizes_[aliveNode] = nodeSizes_[deadNode] + nodeSizes_[aliveNode];
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

    std::cout << "Prev: 1 =" << edgeIndicators_[aliveEdge];
    std::cout << "; 2 = " << edgeIndicators_[deadEdge];

    // Update size:
    const auto newSize = edgeSizes_[aliveEdge] + edgeSizes_[deadEdge];
    edgeSizes_[aliveEdge] = newSize;

//    // merging the histogram is just adding
//    auto & ha = histograms_[aliveEdge];
//    auto & hd = histograms_[deadEdge];
//    ha.merge(hd);
//    const auto newEdgeIndicator = histogramToMedian(aliveEdge);

    const auto newWeightedSum = weightedSum_[aliveEdge] + weightedSum_[deadEdge];
    weightedSum_[aliveEdge] = newWeightedSum;
    const auto newEdgeIndicator = newWeightedSum / newSize;

    std::cout << "; post: " << newEdgeIndicator << "\n";

    // Update edge-data:
    pq_.push(aliveEdge, newEdgeIndicator);
    edgeIndicators_[aliveEdge] = newEdgeIndicator;
    // TODO: optimize me. Check size and reserve...?
    if (settings_.computeLossData) {
//        const auto cap =  backtrackEdges_[aliveEdge].capacity();
//        const auto size =  backtrackEdges_[aliveEdge].size();
//        if (cap-size==0)
//            backtrackEdges_[aliveEdge].reserve(cap + (uint64_t) 5);
        backtrackEdges_[aliveEdge].push_back(deadEdge);
    }
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


// TODO: Bad design, this should be somehow moved to the agglomeration class..
template<class GRAPH, bool ENABLE_UCM>
template<class EDGE_INDICATORS>
inline bool
ConstrainedPolicy<GRAPH, ENABLE_UCM>::
runMileStep(const int nb_iterations_in_milestep,
            EDGE_INDICATORS & newEdgeIndicators) {
    this->updateEdgeIndicators(newEdgeIndicators);
    this->resetDataBeforeMilestep(nb_iterations_in_milestep);

    while(!this->isDone()){
        const auto edgeToContractNextAndPriority = this->edgeToContractNext();
        const auto edgeToContractNext = edgeToContractNextAndPriority.first;
        const auto priority = edgeToContractNextAndPriority.second;
//            if(verbose){
//                const auto & cgraph = clusterPolicy_.edgeContractionGraph();
//                std::cout<<"Nodes "<<cgraph.numberOfNodes()<<" p="<<priority<<"\n";
//            }
        this->edgeContractionGraph().contractEdge(edgeToContractNext);
    }

    return this->isReallyDone();
}



} // namespace agglo
} // namespace nifty::graph
} // namespace nifty


