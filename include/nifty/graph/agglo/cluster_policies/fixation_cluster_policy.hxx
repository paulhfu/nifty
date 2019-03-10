#pragma once

#include <functional>
#include <set>
#include <unordered_set>
#include <boost/container/flat_set.hpp>
#include <string>
#include <cmath>        // std::abs
#include <math.h>  // sqrt

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"
#include <iostream>


namespace nifty{
namespace graph{
namespace agglo{


template<
    class GRAPH, class ACC_0, bool ENABLE_UCM
>
class FixationClusterPolicy{

    typedef FixationClusterPolicy<
        GRAPH, ACC_0,  ENABLE_UCM
    > SelfType;

private:
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<float> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<float> FloatNodeMap;

    typedef boost::container::flat_set<uint64_t> SetType;
    typedef typename GRAPH:: template NodeMap<SetType > NonLinkConstraints;

    typedef ACC_0 Acc0Type;
public:
    typedef typename Acc0Type::SettingsType Acc0SettingsType;


    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgePrioType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;

    struct SettingsType{

        // TODO: make threshold-check optional
        Acc0SettingsType updateRule0;
        bool zeroInit = false; // DEPRECATED
        bool initSignedWeights = false; // DEPRECATED
        uint64_t numberOfNodesStop{1};
        double sizeRegularizer{0.};
        double sizeThreshMin{10.}; // DEPRECATED
        double sizeThreshMax{30.}; // DEPRECATED
        bool postponeThresholding{true}; // DEPRECATED
        double threshold{0.5}; // Merge all: 0.0; split all: 1.0 // DEPRECATED
        //uint64_t numberOfBins{40};
        bool costsInPQ{true}; // DEPRECATED
        bool checkForNegCosts{true}; // DEPRECATED
        bool addNonLinkConstraints{false};
        bool removeSmallSegments{false}; // DEPRECATED
        uint64_t smallSegmentsThresh{10}; // DEPRECATED
    };

    enum class EdgeStates : uint8_t {
        PURE_LOCAL = 0,
        LOCAL = 1,
        LIFTED = 2,
        PURE_LIFTED = 3
    };

    enum class EdgeSizeStates : uint8_t {
        SMALL = 0,
        FROM_SMALL_TO_BIG = 1,
        BIG = 2,
    };



    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types


    typedef nifty::tools::ChangeablePriorityQueue< float , std::greater<float> > QueueType;


public:

    template<class MERGE_PRIOS, class IS_LOCAL_EDGE, class EDGE_SIZES, class NODE_SIZES>
    FixationClusterPolicy(const GraphType &,
                              const MERGE_PRIOS & ,
                              const IS_LOCAL_EDGE &,
                              const EDGE_SIZES & ,
                              const NODE_SIZES & ,
                              const SettingsType & settings = SettingsType());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone();

    // callback called by edge contraction graph

    EdgeContractionGraphType & edgeContractionGraph();

private:
    double pqMergePrio(const uint64_t edge) const;
    double computeWeight(const uint64_t edge) const;

public:
    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);

    bool isEdgeConstrained(const uint64_t edge){
        const auto uv = edgeContractionGraph_.uv(edge);
        const auto u = uv.first;
        const auto v = uv.second;
        const auto & setU  = nonLinkConstraints_[u];
        const auto & setV  = nonLinkConstraints_[v];
        NIFTY_CHECK((setU.find(v)!=setU.end()) == (setV.find(u)!=setV.end()),"");
        if(setU.find(v)!=setU.end()){// || setV.find(u)!=setV.end()){
            return true;
        } else{
            return false;
        }
    }


    bool isMergeAllowed(const uint64_t edge) const{
        // Here we do not care about the fact that an edge is lifted or not.
        // We just look at the priority
        return acc0_[edge] > 0.;
    }

    double edgeCostInPQ(const uint64_t edge) const{
        const auto priority = acc0_[edge];
        if (settings_.addNonLinkConstraints) {
            return std::abs(priority);
        } else {
            return priority;
        }
    }

    void addNonLinkConstraint(const uint64_t edge){
        //std::cout<<"add non link constraint\n";
        const auto uv = edgeContractionGraph_.uv(edge);
        const auto u = uv.first;
        const auto v = uv.second;
        nonLinkConstraints_[uv.first].insert(uv.second);
        nonLinkConstraints_[uv.second].insert(uv.first);
    }

    auto exportAgglomerationData(){
        typename xt::xtensor<float, 2>::shape_type retshape;
        retshape[0] = graph_.nodeIdUpperBound()+1;
        retshape[1] = 4;
        xt::xtensor<float, 2> out(retshape);

        graph_.forEachNode([&](const uint64_t node) {
            out(node, 0) = maxNodeSize_per_iter_[node];
            out(node, 1) = maxCostInPQ_per_iter_[node];
            out(node, 2) = meanNodeSize_per_iter_[node];
            out(node, 3) = variance_[node];
        });
        return out;
    }



private:



    // INPUT
    const GraphType &   graph_;

    NonLinkConstraints nonLinkConstraints_;

//    int phase_;

    ACC_0 acc0_;
    NodeSizesType nodeSizes_;
    NodeSizesType maxNodeSize_per_iter_;
    NodeSizesType meanNodeSize_per_iter_;
    NodeSizesType variance_;
    NodeSizesType maxCostInPQ_per_iter_;
    uint64_t max_node_size_;
    uint64_t sum_node_size_;
    uint64_t quadratic_sum_node_size_;
    uint64_t nb_performed_contractions_;

    bool mean_rule_;



    typename GRAPH:: template EdgeMap<EdgeStates>  edgeState_;
    typename GRAPH:: template EdgeMap<EdgeSizeStates>  edgeSizeState_;
    typename GRAPH:: template NodeMap<EdgeSizeStates>  nodeSizeState_;

    SettingsType        settings_;

    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

    uint64_t edgeToContractNext_;
    double   edgeToContractNextMergePrio_;
};

//    TODO: rename MERGE_PRIO in something like ATTRACTIVE_COSTS

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
template<class MERGE_PRIOS, class IS_LOCAL_EDGE,class EDGE_SIZES,class NODE_SIZES>
inline FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
FixationClusterPolicy(
    const GraphType & graph,
    const MERGE_PRIOS & signedWeights,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const NODE_SIZES      & nodeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    nonLinkConstraints_(graph),
    acc0_(graph, signedWeights, edgeSizes, settings.updateRule0),
    edgeState_(graph),
    nodeSizes_(graph),
    edgeSizeState_(graph),
    nodeSizeState_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this),
    maxNodeSize_per_iter_(graph),
    maxCostInPQ_per_iter_(graph),
    variance_(graph),
    meanNodeSize_per_iter_(graph),
    max_node_size_(0),
    sum_node_size_(0),
    quadratic_sum_node_size_(0),
    nb_performed_contractions_(0)
{
    // FIXME: are ignored segments with both -1 handled well in general?
    if (settings_.removeSmallSegments && (!settings_.checkForNegCosts || settings_.addNonLinkConstraints || settings_.costsInPQ) ) {
        NIFTY_CHECK(false,"Small segments not supported atm without checkForNegCosts, with logCosts or with constraints!");
    }

//    phase_ = 0;
    graph_.forEachNode([&](const uint64_t node) {
        nodeSizes_[node] = nodeSizes[node];
        sum_node_size_ += nodeSizes[node];
        quadratic_sum_node_size_ += nodeSizes[node] * nodeSizes[node];
        if (nodeSizes[node] > max_node_size_)
            max_node_size_ = uint8_t(nodeSizes[node]);

        // FIXME: only true if we start from pixels!
        nodeSizeState_[node] = EdgeSizeStates::SMALL;
    });

//    std::cout << "Size reg:" << settings_.sizeRegularizer << "\n";
    graph_.forEachEdge([&](const uint64_t edge){

        const auto loc = isLocalEdge[edge];

//        mean_rule_ = acc0_.name() == std::string("ArithmeticMean");
//        if (mean_rule_) {
//            if (acc0_[edge] < 0.) {
//                NIFTY_ASSERT_OP(acc1_[edge],>=,0.);
//                acc0_.set(edge, 0.5 - acc1_[edge], acc1_.weight(edge));
//            } else {
//                acc0_.set(edge, 0.5 + acc0_[edge], acc0_.weight(edge));
//            }
//        }

        // FIXME: only true if we start from pixels!
        edgeSizeState_[edge] = EdgeSizeStates::SMALL;

        // TODO: get rid of this
//        if(settings_.zeroInit){
//            edgeState_[edge] = (loc ? EdgeStates::PURE_LOCAL : EdgeStates::PURE_LIFTED);
//
//            if(loc == 1){
//                acc1_.set(edge, 0.0, edgeSizes[edge]);
//            }
//            else{
//                acc0_.set(edge, 0.0, edgeSizes[edge]);
//            }
//        } else {
        edgeState_[edge] = (loc == 1 ? EdgeStates::LOCAL : EdgeStates::LIFTED);

        pq_.push(edge, this->computeWeight(edge));

//        pq_.push(edge, this->pqMergePrio(edge));
    });
}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline std::pair<uint64_t, double>
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
edgeToContractNext() const {
    return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline bool
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::isDone(
){
    while(true) {
        while(!pq_.empty() && !isNegativeInf(pq_.topPriority())){
            // Here we already know that the edge is not lifted
            // (Otherwise we would have inf cost in PQ)
            const auto nextActioneEdge = pq_.top();

            // Check if some early constraints were enforced:
            if (settings_.addNonLinkConstraints) {
                if(this->isEdgeConstrained(nextActioneEdge)) {
                    pq_.push(nextActioneEdge, -1.0*std::numeric_limits<double>::infinity());
                    continue;
                }
            }

            // Here we check if we are allowed to make the merge:
            if(this->isMergeAllowed(nextActioneEdge)){
                edgeToContractNext_ = nextActioneEdge;
                edgeToContractNextMergePrio_ = pq_.topPriority();
                return false;
            }
            else{
                if (! settings_.addNonLinkConstraints) {
                    // In this case we know that we reached priority zero, so we can already stop
                    return true;
                }
                this->addNonLinkConstraint(nextActioneEdge);
                pq_.push(nextActioneEdge, -1.0*std::numeric_limits<double>::infinity());
            }
        }
        return true;
    }
}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline double
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
pqMergePrio(
    const uint64_t edge
) const {
    const auto s = edgeState_[edge];
    double costInPQ;
    if(s == EdgeStates::LOCAL || s==EdgeStates::PURE_LOCAL){
        costInPQ = this->edgeCostInPQ(edge);
    }
    else{
        // In this case the edge is lifted, so we need to be careful.
        // It can be inserted in the PQ to constrain, but not to merge.
        // REMARK: The second condition  "isMergeAllowed" is actually not necessary, because it is anyway checked
        // again in isDone before to actually contract the edge...
        if (!settings_.addNonLinkConstraints || this->isMergeAllowed(edge))
            costInPQ = -1.0*std::numeric_limits<double>::infinity();
        else {
            costInPQ = this->edgeCostInPQ(edge);
        }
    }
    return costInPQ;
}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline void
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    // Remember about the highest cost in PQ:
//    std::cout << edgeToContractNextMergePrio_ << "\n";
              //    maxCostInPQ_per_iter_[nb_performed_contractions_] = edgeToContractNextMergePrio_;
    maxCostInPQ_per_iter_[nb_performed_contractions_] = edgeContractionGraph_.numberOfEdges();
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline typename FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::EdgeContractionGraphType &
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline void
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode,
    const uint64_t deadNode
){
    // Save data about max_node_size
    maxNodeSize_per_iter_[nb_performed_contractions_] = max_node_size_;
    const auto remaining_nodes = edgeContractionGraph_.numberOfNodes();
    meanNodeSize_per_iter_[nb_performed_contractions_] = float(sum_node_size_) / float(remaining_nodes);
    variance_[nb_performed_contractions_] =  float(quadratic_sum_node_size_) / float(remaining_nodes) - std::pow(meanNodeSize_per_iter_[nb_performed_contractions_], 2);

    quadratic_sum_node_size_ += std::pow(nodeSizes_[deadNode] + nodeSizes_[aliveNode], 2) - std::pow(nodeSizes_[aliveNode], 2) - std::pow(nodeSizes_[deadNode], 2);

    nodeSizes_[aliveNode] += nodeSizes_[deadNode];
    if (nodeSizes_[aliveNode] > max_node_size_)
        max_node_size_ = uint64_t(nodeSizes_[aliveNode]);


    if (settings_.addNonLinkConstraints) {
        auto  & aliveNodeNlc = nonLinkConstraints_[aliveNode];
        const auto & deadNodeNlc = nonLinkConstraints_[deadNode];
        aliveNodeNlc.insert(deadNodeNlc.begin(), deadNodeNlc.end());


        for(const auto v : deadNodeNlc){
            auto & nlc = nonLinkConstraints_[v];

            // best way to change values in set...
            nlc.erase(deadNode);
            nlc.insert(aliveNode);
        }

        aliveNodeNlc.erase(deadNode);
    }

}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline void
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge,
    const uint64_t deadEdge
){

    NIFTY_ASSERT_OP(aliveEdge,!=,deadEdge);
    NIFTY_ASSERT(pq_.contains(aliveEdge));
    NIFTY_ASSERT(pq_.contains(deadEdge));

    pq_.deleteItem(deadEdge);

    // update priority:
    acc0_.merge(aliveEdge, deadEdge);


    // update state
    auto & sa = edgeState_[aliveEdge];
    const auto  sd = edgeState_[deadEdge];
    if(sa == EdgeStates::PURE_LIFTED &&  sd == EdgeStates::PURE_LIFTED){
        sa =  EdgeStates::PURE_LIFTED;
    }
    else if(sa == EdgeStates::PURE_LOCAL &&  sd == EdgeStates::PURE_LOCAL){
        sa = EdgeStates::PURE_LOCAL;
    }
    else if(
        sa == EdgeStates::PURE_LOCAL ||  sa == EdgeStates::LOCAL ||
        sd == EdgeStates::PURE_LOCAL ||  sd == EdgeStates::LOCAL
    ){
        sa = EdgeStates::LOCAL;
    }
    else if((sa == EdgeStates::PURE_LIFTED ||  sa == EdgeStates::LIFTED)  &&
            (sd == EdgeStates::PURE_LIFTED ||  sd == EdgeStates::LIFTED) )
    {
        sa = EdgeStates::LIFTED;
    }

    const auto sr = settings_.sizeRegularizer;
    if (sr < 0.000001)
        pq_.push(aliveEdge, this->computeWeight(aliveEdge));

}


template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline void
FixationClusterPolicy<GRAPH, ACC_0, ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    // HERE WE UPDATE the PQ when a SizeReg is used:
    const auto sr = settings_.sizeRegularizer;
    if (sr > 0.000001) {
        const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
        for(auto adj : edgeContractionGraph_.adjacency(u)){
            const auto edge = adj.edge();
            pq_.push(edge, computeWeight(edge));
        }
    }

    nb_performed_contractions_++;

}

template<class GRAPH, class ACC_0, bool ENABLE_UCM>
inline double
FixationClusterPolicy<GRAPH, ACC_0,  ENABLE_UCM>::
computeWeight(
        const uint64_t edge
) const {
    const auto fromEdge = this->pqMergePrio(edge); // This -inf if the edge was lifted
    const auto sr = settings_.sizeRegularizer;

    if (sr > 0.000001 and !isNegativeInf(fromEdge))
    {
        const auto uv = edgeContractionGraph_.uv(edge);
        const auto sizeU = nodeSizes_[uv.first];
        const auto sizeV = nodeSizes_[uv.second];
        const auto sFac = 2.0 / ( 1.0/std::pow(sizeU,sr) + 1.0/std::pow(sizeV,sr) );
//        if (edge < 100) {
//            std::cout << this->pqMergePrio(edge) << "; regFact: " << sFac<<"; sizeU: " << sizeU<<"; sizeV: " << sizeV<<"; final: " << fromEdge*(1. / sFac) << "\n";
//        }
        return fromEdge * (1. / sFac);
    } else {
        return fromEdge;
    }


}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

