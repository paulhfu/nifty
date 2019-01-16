#pragma once

#include <functional>
#include <set>
#include <unordered_set>
#include <boost/container/flat_set.hpp>
#include <string>
#include <cmath>        // std::abs

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"
#include <iostream>


namespace nifty{
namespace graph{
namespace agglo{


template<
    class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM
>
class FixationClusterPolicy{

    typedef FixationClusterPolicy<
        GRAPH, ACC_0, ACC_1, ENABLE_UCM
    > SelfType;

private:    
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<float> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<float> FloatNodeMap;

    typedef boost::container::flat_set<uint64_t> SetType;
    typedef typename GRAPH:: template NodeMap<SetType > NonLinkConstraints;

    typedef ACC_0 Acc0Type;
    typedef ACC_1 Acc1Type;
public:
    typedef typename Acc0Type::SettingsType Acc0SettingsType;
    typedef typename Acc1Type::SettingsType Acc1SettingsType;


    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgePrioType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;

    struct SettingsType{

        // TODO: make threshold-check optional
        Acc0SettingsType updateRule0;
        Acc1SettingsType updateRule1;
        bool zeroInit = false; // DEPRECATED
        bool initSignedWeights = false; // DEPRECATED
        uint64_t numberOfNodesStop{1};
        double sizeRegularizer{0.};
        double sizeThreshMin{10.}; // DEPRECATED
        double sizeThreshMax{30.}; // DEPRECATED
        bool postponeThresholding{true}; // DEPRECATED
        double threshold{0.5}; // Merge all: 0.0; split all: 1.0
        //uint64_t numberOfBins{40};
        bool costsInPQ{true};
        bool checkForNegCosts{true};
        bool addNonLinkConstraints{false};
        bool removeSmallSegments{false};
        uint64_t smallSegmentsThresh{10};
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

    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE, class EDGE_SIZES, class NODE_SIZES>
    FixationClusterPolicy(const GraphType &, 
                              const MERGE_PRIOS & , 
                              const NOT_MERGE_PRIOS &,
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
        if (settings_.removeSmallSegments
            && !settings_.addNonLinkConstraints && !settings_.costsInPQ && edgeSizeState_[edge] == EdgeSizeStates::SMALL) {
            return true;
        }

        // Here we do not care about the fact that an edge is lifted or not.
        // We just look at the costs
        if (settings_.checkForNegCosts && acc0_[edge] < 0)
            return false;
        else if (settings_.checkForNegCosts && acc1_[edge] < 0)
            return true;
        else {
            return acc0_[edge] - acc1_[edge] > 2 * (settings_.threshold - 0.5);
        }
    }

    double edgeCostInPQ(const uint64_t edge) const{
        if (settings_.costsInPQ) {
            auto attrFromEdge = acc0_[edge];
            auto repFromEdge = acc1_[edge];
            if (settings_.checkForNegCosts && attrFromEdge < 0)
                attrFromEdge = 0.;
            if (settings_.checkForNegCosts && repFromEdge < 0)
                repFromEdge = 0.;

            const auto cost = attrFromEdge - repFromEdge;

            // With constraints, we take the absolute value
            if (settings_.addNonLinkConstraints)
                return std::abs(cost);
            else
                return cost;
        }
        else {
            if (settings_.addNonLinkConstraints) {
                return std::max(acc0_[edge], acc1_[edge]);
            } else {
                return acc0_[edge];
            }
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



private:



    // INPUT
    const GraphType &   graph_;

    NonLinkConstraints nonLinkConstraints_;

//    int phase_;

    ACC_0 acc0_;
    ACC_1 acc1_;
    NodeSizesType nodeSizes_;

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

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE,class EDGE_SIZES,class NODE_SIZES>
inline FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
FixationClusterPolicy(
    const GraphType & graph,
    const MERGE_PRIOS & mergePrios,
    const NOT_MERGE_PRIOS & notMergePrios,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const NODE_SIZES      & nodeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    nonLinkConstraints_(graph),
    acc0_(graph, mergePrios,    edgeSizes, settings.updateRule0),
    acc1_(graph, notMergePrios, edgeSizes, settings.updateRule1),
    edgeState_(graph),
    nodeSizes_(graph),
    edgeSizeState_(graph),
    nodeSizeState_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
//    phase_ = 0;
    graph_.forEachNode([&](const uint64_t node) {
        nodeSizes_[node] = nodeSizes[node];
        // FIXME: only true if we start from pixels!
        nodeSizeState_[node] = EdgeSizeStates::SMALL;
    });

//    std::cout << "Size reg:" << settings_.sizeRegularizer << "\n";
    graph_.forEachEdge([&](const uint64_t edge){

        const auto loc = isLocalEdge[edge];

        // FIXME: only true if we start from pixels!
        edgeSizeState_[edge] = EdgeSizeStates::SMALL;

        // TODO: better possible solution: pure_repulsive, pure_attractive
//        if(settings_.initSignedWeights) {
//            if (mergePrios[edge] > settings_.threshold) {
//                // Set repulsion to zero:
//                acc1_.set(edge, -1.0, edgeSizes[edge]);
//            } else {
////            if(loc == 1)
////                // Set repulsion to zero:
////                acc1_.set(edge, -1.0, edgeSizes[edge]);
////            else {
////                // Set attraction to zero:
//                acc0_.set(edge, -1.0, edgeSizes[edge]);
////            }
//
//            }
////        if(loc == 1){
////            // Set repulsion to zero:
////            acc1_.set(edge, -1.0, edgeSizes[edge]);
////        } else {
////            // Set attraction to zero:
////
////            acc0_.set(edge, -1.0, edgeSizes[edge]);
////        }
//        }

        // TODO: get rid of this
        if(settings_.zeroInit){
            edgeState_[edge] = (loc ? EdgeStates::PURE_LOCAL : EdgeStates::PURE_LIFTED);

            if(loc == 1){
                acc1_.set(edge, 0.0, edgeSizes[edge]);
            }
            else{
                acc0_.set(edge, 0.0, edgeSizes[edge]);
            }
        } else {
            edgeState_[edge] = (loc == 1 ? EdgeStates::LOCAL : EdgeStates::LIFTED);
        }

        pq_.push(edge, this->computeWeight(edge));

//        pq_.push(edge, this->pqMergePrio(edge));
    });
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
edgeToContractNext() const {    
    return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline bool 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::isDone(
){
    while(true) {
        while(!pq_.empty() && !isNegativeInf(pq_.topPriority())){

            // Here we already know that the edge is not lifted
            // (Otherwise we would have inf cost in PQ)
            const auto nextActioneEdge = pq_.top();

            // Here we check if some early constraints were enforced:
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
                // TODO: re-introduce an smart early stop (mean, sum, when thresh is reached, I can stop)
//                const auto mean = std::string("ArithmeticMean");
//                if (settings_.sizeThreshMin == 0 && not settings_.postponeThresholding && not settings_.zeroInit
//                    && acc0_.name() == mean
//                    && acc1_.name() == mean) {
//                    return true;
//                }
                if (settings_.addNonLinkConstraints) {
                    this->addNonLinkConstraint(nextActioneEdge);
                }
                pq_.push(nextActioneEdge, -1.0*std::numeric_limits<double>::infinity());
            }
        }
//        if (phase_ != 0 || settings_.sizeThreshMin == 0 || not settings_.postponeThresholding){
            return  true;
//        } else {
//            phase_ = 1;
//            std::cout << "Phase 1 done ";
//            // Insert again all edges in PQ without SizeRegularizer
//            int counter = 0;
//            graph_.forEachEdge([&](const uint64_t edge) {
//                const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
//                const auto uv = edgeContractionGraph_.uv(edge);
//                if (cEdge == edge && cEdge>=0 && uv.first!=uv.second) {
//                    counter++;
//                    pq_.push(edge, this->computeWeight(cEdge));
//                }
//            });
//            std::cout << "--> "<< counter<<" new edges inserted!\n";
//        }
    }
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline double 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
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
        if (!settings_.addNonLinkConstraints || this->isMergeAllowed(edge))
            costInPQ = -1.0*std::numeric_limits<double>::infinity();
        else {
            costInPQ = this->edgeCostInPQ(edge);
        }
    }
    return costInPQ;
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline typename FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::EdgeContractionGraphType & 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
    nodeSizes_[aliveNode] +=nodeSizes_[deadNode];
    if (settings_.removeSmallSegments
        && !settings_.addNonLinkConstraints && !settings_.costsInPQ) {
        if (nodeSizes_[aliveNode] >= settings_.smallSegmentsThresh &&
                (nodeSizeState_[aliveNode] == EdgeSizeStates::SMALL || nodeSizeState_[deadNode] == EdgeSizeStates::SMALL)) {
            nodeSizeState_[aliveNode] = EdgeSizeStates::FROM_SMALL_TO_BIG;
        }
    }

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

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){

    NIFTY_ASSERT_OP(aliveEdge,!=,deadEdge);
    NIFTY_ASSERT(pq_.contains(aliveEdge));
    NIFTY_ASSERT(pq_.contains(deadEdge));

    pq_.deleteItem(deadEdge);
   
    // update merge prio
    
    auto & sa = edgeState_[aliveEdge];
    const auto  sd = edgeState_[deadEdge];

    const auto mergePrioAlive = acc0_[aliveEdge];
    const auto mergePrioDead = acc0_[deadEdge];
    const auto notMergePrioAlive = acc1_[aliveEdge];
    const auto notMergePrioDead = acc1_[deadEdge];
//    std::cout << "[MP<0 "<< mergePrioAlive < 0. || mergePrioDead < 0. <<" ]; ";
//    std::cout << "[nMP<0 "<< notMergePrioAlive< 0. || notMergePrioAlive < 0. <<" ]; ";

    if (settings_.checkForNegCosts && mergePrioAlive < 0.) {
//        std::cout << "[MergePrioDead "<< mergePrioDead <<" ]; ";
        acc0_.setFrom(aliveEdge, deadEdge);
    } else if (settings_.checkForNegCosts && mergePrioDead < 0.) {
//        std::cout << "[MergePrioAlive "<< mergePrioAlive <<" ]; ";
    }
    else {
        acc0_.merge(aliveEdge, deadEdge);
    }

    if (settings_.checkForNegCosts && notMergePrioAlive < 0.) {
//        std::cout << ".";
        acc1_.setFrom(aliveEdge, deadEdge);
    } else if (settings_.checkForNegCosts && notMergePrioDead < 0.) {
//        std::cout << "/";
//        std::cout << "[notMergePrioAlive "<< notMergePrioAlive <<" ]; ";
    }
    else {
//        std::cout << "[bothRep]";
        acc1_.merge(aliveEdge, deadEdge);
    }

//    if(settings_.zeroInit  && sa == EdgeStates::PURE_LIFTED &&  sd != EdgeStates::PURE_LIFTED)
//        acc0_.setValueFrom(aliveEdge, deadEdge);
//    else if (settings_.zeroInit  && sa != EdgeStates::PURE_LIFTED &&  sd == EdgeStates::PURE_LIFTED) {}
//    else
//        acc0_.merge(aliveEdge, deadEdge);
//
//    // update notMergePrio
//    if(settings_.zeroInit  && sa == EdgeStates::PURE_LOCAL &&  sd !=  EdgeStates::PURE_LOCAL)
//        acc1_.setValueFrom(aliveEdge, deadEdge);
//        // FIXME: here weight is not updated!!!
//    else if(settings_.zeroInit  && sa != EdgeStates::PURE_LOCAL &&  sd ==  EdgeStates::PURE_LOCAL) {}
//    else
//        acc1_.merge(aliveEdge, deadEdge);
    

    // update state
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


template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
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
    // Only in the case when the segments size went over the threshold,
    // we update the edgeSizeLabels and the edge costs in PQ:
    if (settings_.removeSmallSegments
        && !settings_.addNonLinkConstraints && !settings_.costsInPQ) {
        const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
        if (nodeSizeState_[u] == EdgeSizeStates::FROM_SMALL_TO_BIG) {
            for(auto adj : edgeContractionGraph_.adjacency(u)){
                const auto edge = adj.edge();
                const auto v = adj.node();
                if (nodeSizeState_[v] == EdgeSizeStates::BIG) {
                    edgeSizeState_[edge] = EdgeSizeStates::BIG;
                    pq_.push(edge, computeWeight(edge));
                }
//              else {
//                    edgeSizeState_[edge] = EdgeSizeStates::SMALL;
//                }

            }
            nodeSizeState_[u] = EdgeSizeStates::BIG;
        }
    }
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline double
FixationClusterPolicy<GRAPH, ACC_0, ACC_1, ENABLE_UCM>::
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
        if (settings_.removeSmallSegments && (edgeSizeState_[edge] == EdgeSizeStates::SMALL)
            && !settings_.addNonLinkConstraints && !settings_.costsInPQ) {
            // Here we increase the priority of edges involving small segments:
            return fromEdge + 1.0;
        }
        return fromEdge;
    }


}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

