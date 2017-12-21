#pragma once

#include <functional>
#include <array>
#include <iostream>
#include <vector>

#include "nifty/histogram/histogram.hxx"
#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"
#include "nifty/tools/runtime_check.hxx"

namespace nifty{
namespace graph{
namespace agglo{







template<
    class GRAPH,bool ENABLE_UCM
>
class ConstrainedGeneralizedMeanFixationClusterPolicy{

    typedef ConstrainedGeneralizedMeanFixationClusterPolicy<
        GRAPH, ENABLE_UCM
    > SelfType;

private:
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<uint64_t> UInt64EdgeMap;
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<double> FloatNodeMap;
    typedef typename GRAPH:: template NodeMap<uint64_t> UInt64NodeMap;
    typedef typename GRAPH:: template EdgeMap<bool> BoolEdgeMap;
    typedef typename GRAPH:: template EdgeMap<std::vector<uint64_t>> VectorEdgeMap;

    typedef boost::container::flat_set<uint64_t> SetType;

public:
    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgeIndicatorsType;
    typedef FloatEdgeMap                                EdgePrioType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;


    typedef BoolEdgeMap                                 EdgeFlagType;
    typedef UInt64NodeMap                               GTlabelsType;
    typedef UInt64EdgeMap                               MergeTimesType;
    typedef VectorEdgeMap                               BacktrackEdgesType;

    struct SettingsType : public EdgeWeightedClusterPolicySettings
    {
        enum UpdateRule{
            SMOOTH_MAX,
            GENERALIZED_MEAN
            //HISTOGRAM_RANK
        };

        UpdateRule updateRule0{GENERALIZED_MEAN};
        UpdateRule updateRule1{GENERALIZED_MEAN};

        bool zeroInit = false;
        double p0{1.0};
        double p1{1.0};


        float threshold{0.5};
        uint64_t ignore_label{(uint64_t) -1};
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
//    typedef nifty::histogram::Histogram<float> HistogramType;
    //typedef std::array<float, NumberOfBins> HistogramType;     
//    typedef typename GRAPH:: template EdgeMap<HistogramType> EdgeHistogramMap;


    typedef nifty::tools::ChangeablePriorityQueue< double , std::greater<double> > QueueType;

    double pqMergePrio(const uint64_t edge) const;

public:

    template<class IS_LOCAL_EDGE, class EDGE_SIZES, class NODE_SIZES, class GT_LABELS>
    ConstrainedGeneralizedMeanFixationClusterPolicy(const GraphType &,
                              const IS_LOCAL_EDGE & ,
                              const EDGE_SIZES & , 
                              const NODE_SIZES & ,
                              const GT_LABELS & ,
                              const SettingsType & settings = SettingsType());

    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS>
    void updateEdgeIndicators(const MERGE_PRIOS &,
                              const NOT_MERGE_PRIOS &);

    template<class NODE_SIZES,
            class NODE_LABELS,
            class NODE_GT_LABELS,
            class EDGE_SIZES,
            class MERGE_PRIOS,
            class DEND_HIGH,
            class MERGING_TIME,
            class LOSS_TARGETS,
            class LOSS_WEIGHTS>
    void collectDataMilestep(
            NODE_SIZES        & ,
            NODE_LABELS        & ,
            NODE_GT_LABELS        & ,
            EDGE_SIZES        & ,
            MERGE_PRIOS   & ,
            DEND_HIGH         & ,
            MERGING_TIME      & ,
            LOSS_TARGETS      & ,
            LOSS_WEIGHTS      &
    ) const;


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone();

    bool edgeIsConstrained(const uint64_t);
    bool edgeInvolvesIgnoredNodes(const uint64_t);

    void computeFinalTargets();
    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS>
    bool runMileStep(const int, const MERGE_PRIOS &, const NOT_MERGE_PRIOS &);

    // callback called by edge contraction graph

    EdgeContractionGraphType & edgeContractionGraph();



    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);

    bool isMergeAllowed(const uint64_t edge){
        if(isLocalEdge_[edge]){
            // todo this isPureLocal_ seems to be legacy
            // check if needed
            return isPureLocal_[edge] ? true : mergePrios_[edge] > notMergePrios_[edge];
        }
        else{
            return false;
        }
    }

    // TODO: these should no longer be necessary..
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

    // -----------
    // Added methods:
    const EdgePrioType & mergePrios() const {
        return mergePrios_;
    }
    const EdgePrioType & notMergePrios() const {
        return notMergePrios_;
    }



private:
//    float histogramToMedian(const uint64_t edge) const;

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

    const double getMergePrio(const uint64_t edge)const{
        return mergePrios_[edge];
    }

    const double notMergePrio(const uint64_t edge)const{
        return notMergePrios_[edge];
    }

    // INPUT
    const GraphType &   graph_;

    EdgePrioType mergePrios_;
    EdgePrioType notMergePrios_;

    UInt8EdgeMap isLocalEdge_;
    UInt8EdgeMap isPureLocal_;
    UInt8EdgeMap isPureLifted_;

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

//    EdgeHistogramMap histograms_;

    uint64_t time_;
    uint64_t mileStepTimeOffset_;
    // Keeps track of edges in PQ (some are alive, but deleted from the PQ because constrained
//    uint64_t nb_active_edges_;

//    uint64_t nb_steps_;
//    uint64_t ignore_label_;
    bool     isReallyDone_;

    // TODO: probably not necessary
    uint64_t edgeToContractNext_;
    double   edgeToContractNextMergePrio_;

};

//    Define the methods:

template<class GRAPH, bool ENABLE_UCM>
template<class IS_LOCAL_EDGE, class EDGE_SIZES, class NODE_SIZES, class GT_LABELS>
inline ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
ConstrainedGeneralizedMeanFixationClusterPolicy(
    const GraphType & graph,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const NODE_SIZES      & nodeSizes,
    const GT_LABELS    & GTlabels,
    const SettingsType & settings
)
:   graph_(graph),
    mergePrios_(graph, -1.),
    notMergePrios_(graph, -1.),
    isLocalEdge_(graph),
    isPureLocal_(graph),
    isPureLifted_(graph),
    edgeSizes_(graph),
    nodeSizes_(graph),
    mergeTimes_(graph, graph_.numberOfNodes()),
    weightedSum_(graph),
    flagAliveEdges_(graph, true),
    // TODO: bad, find better way to initialize this..
    dendHeigh_(graph, -1.),
    GTlabels_(graph),
    settings_(settings),
    edgeContractionGraph_(graph, *this),
//    nb_active_edges_(graph_.numberOfEdges()),
    nb_correct_mergers_(0),
    nb_wrong_mergers_(0),
    nb_correct_splits_(0),
    nb_wrong_splits_(0),
    nb_iterations_in_milestep_(-1),
    loss_weights_(graph, 0.),
    loss_targets_(graph, 0.),
    pq_(graph.edgeIdUpperBound()+1),
//    histograms_(graph, HistogramType(0,1,settings.bincount)),
    time_(0),
    backtrackEdges_(graph),
    mileStepTimeOffset_(0),
    isReallyDone_(false)
{
    graph_.forEachEdge([&](const uint64_t edge){
        isLocalEdge_[edge] = isLocalEdge[edge];
        isPureLocal_[edge] = isLocalEdge[edge];
        isPureLifted_[edge] = !isLocalEdge[edge];


        if (settings_.computeLossData)
            backtrackEdges_[edge].push_back(edge);

        const auto size = edgeSizes[edge];
        edgeSizes_[edge] = size;

        // FIXME: what about average mean????
    });

    graph_.forEachNode([&](const uint64_t node){
        // Save GT labels:
        const auto GT_lab = GTlabels[node];
        GTlabels_[node] = GT_lab;
        nodeSizes_[node] = nodeSizes[node];
    });
}

template<class GRAPH, bool ENABLE_UCM>
inline double
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
pqMergePrio(
        const uint64_t edge
) const {
    return isLocalEdge_[edge] ?  mergePrios_[edge] : -1.0;
}



//        This should be called only when we are sure that a not constrained edge is still
//        available in PQ
template<class GRAPH, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
edgeToContractNext() const {
    return std::pair<uint64_t, double>(pq_.top(),pq_.topPriority()) ;
}

template<class GRAPH, bool ENABLE_UCM>
inline bool
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
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
        inline bool
        ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
        edgeInvolvesIgnoredNodes(
                const uint64_t edge
        ){
            // TODO: ugly repetition of code here and in the isConstrained
            const auto uv = edgeContractionGraph_.uv(edge);
            const auto u = uv.first;
            const auto v = uv.second;
            const auto reprU = edgeContractionGraph_.findRepresentativeNode(u);
            const auto reprV = edgeContractionGraph_.findRepresentativeNode(v);

            if (GTlabels_[reprU] == settings_.ignore_label ||
                GTlabels_[reprV] == settings_.ignore_label ) {
                return true;
            } else {
                return false;
            }
        }


template<class GRAPH, bool ENABLE_UCM>
template<class MERGE_PRIOS, class NOT_MERGE_PRIOS>
inline void
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
updateEdgeIndicators(const MERGE_PRIOS & newMergePrios,
                     const NOT_MERGE_PRIOS & newNotMergePrios) {
    graph_.forEachEdge([&](const uint64_t edge){



        mergePrios_[edge] = -1.;
        notMergePrios_[edge] = -1.;
        weightedSum_[edge] = 0.;

        // Insert updated values in histogram and PQ (only for alive representative edges):
        if (flagAliveEdges_[edge]) {  // && !reprEdgesBoolMap[reprEdge]
//            const auto reprEdge = edgeContractionGraph_.findRepresentativeEdge(edge);

            notMergePrios_[edge] = newNotMergePrios[edge];
            mergePrios_[edge] = newMergePrios[edge];

            if(settings_.zeroInit){
                if(isLocalEdge_[edge])
                    notMergePrios_[edge] = 0.0;
                else
                    mergePrios_[edge] = 0.0;
            }

//            ++nb_active_edges_;

            pq_.push(edge, this->pqMergePrio(edge));
        }
    });

}


//        ***************************************

// TODO: optimize me: here we loop over all initial edges...
template<class GRAPH, bool ENABLE_UCM>
inline void
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
computeFinalTargets() {
    if (settings_.constrained && settings_.computeLossData) {
        // Loop over all alive edges (only parents ID are fine, we map later):
        for (const auto edge : graph_.edges()) {
            if (flagAliveEdges_[edge]) {
                const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
                NIFTY_TEST_OP(edge,==,cEdge);
                const auto edgeInvolvesIgnoredNodes = this->edgeInvolvesIgnoredNodes(cEdge);
                if (! edgeInvolvesIgnoredNodes) {
                    const auto isConstrained = this->edgeIsConstrained(cEdge);
                    const auto target_value = (isConstrained) ? (-1.) : (1.);

                    for (auto it = backtrackEdges_[cEdge].begin(); it != backtrackEdges_[cEdge].end(); it++) {
                        const auto subEdge = *it;
                        loss_targets_[subEdge] = target_value;
                        loss_weights_[subEdge] = 1.;
                    }

                    if (isConstrained)
                        ++nb_correct_splits_;
                    else
                        ++nb_wrong_splits_;
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
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
isDone() {
    const auto mileStepTime = time_ - mileStepTimeOffset_;
    if ( mileStepTime>=nb_iterations_in_milestep_)
        return true;

    // Find the next not-constrained edge:
    while (true) {
        if (edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop ||
                edgeContractionGraph_.numberOfEdges() <= settings_.numberOfEdgesStop ||
                pq_.empty() ||
                pq_.topPriority() < -0.0000001    ||
                pq_.topPriority() <= settings_.threshold ) {
            isReallyDone_ = true;
            this->computeFinalTargets();
//            std::cout << "1 node, stop\n";
            return true;
        }

        // Check if merge is allowed (merge>split)
        const auto nextActioneEdge = pq_.top();
        if(! this->isMergeAllowed(nextActioneEdge)){
            pq_.push(nextActioneEdge, -1.0);
            continue;
        }

        if (!settings_.constrained)
            return false;

        // Check if GT is constraining the merge:
        const auto edgeToContractNext = nextActioneEdge;

        if (! this->edgeIsConstrained(edgeToContractNext))
            return false;

        // Set infinite cost in PQ:
        pq_.push(edgeToContractNext, -1.0);

        if (! this->edgeInvolvesIgnoredNodes(edgeToContractNext)) {
            // Remember about wrong step:
            if (settings_.computeLossData) {
//            loss_targets_[edgeToContractNext] = -1.; // We should not merge (and we would have)
//            loss_weights_[edgeToContractNext] = 1.; // For the moment all equally weighted
                for (auto it = backtrackEdges_[edgeToContractNext].begin();
                     it != backtrackEdges_[edgeToContractNext].end(); it++) {
                    const auto edge = *it;
//                std::cout << "Write mistake in " << edgeToContractNext << "to: " << edge << "\n";
                    loss_targets_[edge] = -1.; // We should not merge (and we would have)
                    loss_weights_[edge] = 1.;
                }
//            std::cout << "Moving to next edge in PQ";
            }
            ++nb_wrong_mergers_;
        }

    }

    return false;
}



// UCM of data after a mileStep:
template<class GRAPH, bool ENABLE_UCM>
template<class NODE_SIZES,
        class NODE_LABELS,
        class NODE_GT_LABELS,
        class EDGE_SIZES,
        class MERGE_PRIOS,
        class DEND_HIGH,
        class MERGING_TIME,
        class LOSS_TARGETS,
        class LOSS_WEIGHTS>
inline void
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
collectDataMilestep(
    NODE_SIZES        & nodeSizes,
    NODE_LABELS        & nodeLabels,
    NODE_GT_LABELS      & nodeGTLabels,
    EDGE_SIZES        & edgeSizes,
    MERGE_PRIOS   & mergePrios,
    DEND_HIGH         & dendHeigh,
    MERGING_TIME      & mergeTimes,
    LOSS_TARGETS      & lossTargets,
    LOSS_WEIGHTS      & lossWeights
) const {
    for(const auto edge : graph_.edges()){
        const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
        dendHeigh[edge] = dendHeigh_[cEdge];
        lossTargets[edge] = loss_targets_[cEdge];
        lossWeights[edge] = loss_weights_[cEdge];
        // Map size and indicators only to alive edges:
        if (flagAliveEdges_[cEdge]) {
            mergeTimes[edge] = mergeTimes_[cEdge];
            edgeSizes[edge]     = edgeSizes_[cEdge];
            mergePrios[edge] = mergePrios_[cEdge];
        } else {
            mergeTimes[edge] = -1.0;
            edgeSizes[edge]     = -1.0;
            mergePrios[edge] = -1.0;
        }
    }

    for(const auto node : graph_.nodes()) {
        const auto cNode = edgeContractionGraph_.findRepresentativeNode(node);
        nodeSizes[node] = nodeSizes_[cNode];
        nodeLabels[node] = (float) cNode;
        nodeGTLabels[node] = (float) GTlabels_[cNode];
    }
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    // Update data:
    mergeTimes_[edgeToContract] = time_;
    dendHeigh_[edgeToContract] = pq_.topPriority();
//    std::cout << "Contract edge: " << edgeToContract << "\n";
    ++time_;

    if (settings_.constrained  ) {
        const auto edgeInvolvesIgnoredNodes = this->edgeInvolvesIgnoredNodes(edgeToContract);
        if (! edgeInvolvesIgnoredNodes) {
            // Remember about correct step:
            loss_targets_[edgeToContract] = 1.; // We should merge (and we did)
            loss_weights_[edgeToContract] = 1.; // For the moment all equally weighted
            ++nb_correct_mergers_;
        }
    }


    pq_.deleteItem(edgeToContract);
    flagAliveEdges_[edgeToContract] = false;
////    --nb_active_edges_;
}

template<class GRAPH, bool ENABLE_UCM>
inline typename ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::EdgeContractionGraphType &
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}



template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    const auto GTAlive = GTlabels_[aliveNode];
    const auto GTDead = GTlabels_[deadNode];

    // Propagate ignore label to the merged node:
    if (GTAlive==settings_.ignore_label || GTDead==settings_.ignore_label)
        GTlabels_[aliveNode] = settings_.ignore_label;

    nodeSizes_[aliveNode] = nodeSizes_[deadNode] + nodeSizes_[aliveNode];
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){
    NIFTY_CHECK_OP(aliveEdge,!=,deadEdge,"");
    NIFTY_CHECK(pq_.contains(aliveEdge),"");
    NIFTY_CHECK(pq_.contains(deadEdge),"");

//    std::cout << "Merge edges: " << aliveEdge << " "<< deadEdge << "\n";
    pq_.deleteItem(deadEdge);
    flagAliveEdges_[deadEdge] = false;

//    std::cout << "Prev: 1 =" << edgeIndicators_[aliveEdge];
//    std::cout << "; 2 = " << edgeIndicators_[deadEdge];

    auto generalized_mean = [](
            const long double a,
            const long double d,
            const long double wa,
            const long double wd,
            const long double p
    ){
        const long double  eps = 0.000000001;
        if(std::isinf(p)){
            // max
            if(p>0){
                return std::max(a,d);
            }
                // min
            else{
                return std::min(a,d);
            }
        }
        else if(p > 1.0-eps && p< 1.0+ eps){
            return (wa*a + wd*d)/(wa+wd);
        }
        else{
            const auto wad = wa+wd;
            const auto nwa = wa/wad;
            const auto nwd = wd/wad;
            const auto sa = nwa * std::pow(a, p);
            const auto sd = nwd * std::pow(d, p);
            return std::pow(sa+sd, 1.0/p);
        }
    };

    auto smooth_max = [](
            const long double a,
            const long double d,
            const long double wa,
            const long double wd,
            const long double p
    ){
        const long double  eps = 0.000000001;
        if(std::isinf(p)){
            // max
            if(p>0){
                return std::max(a,d);
            }
                // min
            else{
                return std::min(a,d);
            }
        }
        else if(p > 0.0-eps && p< 0.0+ eps){
            return (wa*a + wd*d)/(wa+wd);
        }
        else{

            const auto eaw = wa * std::exp(a*p);
            const auto edw = wd * std::exp(d*p);
            return (a*eaw + d*edw)/(eaw + edw);
        }
    };


    //  sizes
    const auto sa = edgeSizes_[aliveEdge];
    const auto sd = edgeSizes_[deadEdge];
    const auto zi = settings_.zeroInit ;



    // update merge prio
    if(zi && isPureLifted_[aliveEdge] && !isPureLifted_[deadEdge]){
        mergePrios_[aliveEdge] = mergePrios_[deadEdge];
    }
    else if(zi && !isPureLifted_[aliveEdge] && isPureLifted_[deadEdge]){
        mergePrios_[deadEdge] = mergePrios_[aliveEdge];
    }
    else{
        if(settings_.updateRule0 == SettingsType::GENERALIZED_MEAN){
            mergePrios_[aliveEdge]    = generalized_mean(mergePrios_[aliveEdge],     mergePrios_[deadEdge],    sa, sd, settings_.p0);
        }
        else if(settings_.updateRule0 == SettingsType::SMOOTH_MAX){
            mergePrios_[aliveEdge]    = smooth_max(mergePrios_[aliveEdge],     mergePrios_[deadEdge],    sa, sd, settings_.p0);
        }
        else{
            NIFTY_CHECK(false,"not yet implemented");
        }
    }


    // update notMergePrio
    if(zi && isPureLocal_[aliveEdge] && !isPureLocal_[deadEdge]){
        notMergePrios_[aliveEdge] = notMergePrios_[deadEdge];
    }
    else if(zi && !isPureLocal_[aliveEdge] && isPureLocal_[deadEdge]){
        notMergePrios_[aliveEdge] = notMergePrios_[deadEdge];
    }
    else{
        if(settings_.updateRule0 == SettingsType::GENERALIZED_MEAN){
            notMergePrios_[aliveEdge] = generalized_mean(notMergePrios_[aliveEdge] , notMergePrios_[deadEdge], sa, sd, settings_.p1);
        }
        else if(settings_.updateRule0 == SettingsType::SMOOTH_MAX){
            notMergePrios_[aliveEdge] = smooth_max(notMergePrios_[aliveEdge] , notMergePrios_[deadEdge], sa, sd, settings_.p1);
        }
        else{
            NIFTY_CHECK(false,"not yet implemented");
        }
    }




    edgeSizes_[aliveEdge] = sa + sd;

    const auto deadIsLocalEdge = isLocalEdge_[deadEdge];
    auto & aliveIsLocalEdge = isLocalEdge_[aliveEdge];
    aliveIsLocalEdge = deadIsLocalEdge || aliveIsLocalEdge;

    isPureLocal_[aliveEdge] = isPureLocal_[aliveEdge] && isPureLocal_[deadEdge];
    isPureLifted_[aliveEdge] = isPureLifted_[aliveEdge] && isPureLifted_[deadEdge];



    pq_.push(aliveEdge, this->pqMergePrio(aliveEdge));




    if (settings_.computeLossData) {
        backtrackEdges_[aliveEdge].insert(backtrackEdges_[aliveEdge].end(), backtrackEdges_[deadEdge].begin(), backtrackEdges_[deadEdge].end() );
        // TODO: free memory deadEdge?
    }
}

template<class GRAPH, bool ENABLE_UCM>
inline void 
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){

}


template<class GRAPH, bool ENABLE_UCM>
template<class MERGE_PRIOS, class NOT_MERGE_PRIOS>
inline bool
ConstrainedGeneralizedMeanFixationClusterPolicy<GRAPH, ENABLE_UCM>::
runMileStep(const int nb_iterations_in_milestep,
            const MERGE_PRIOS & newMergePrios,
            const NOT_MERGE_PRIOS & newNotMergePrios) {
    this->updateEdgeIndicators(newMergePrios, newNotMergePrios);
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


