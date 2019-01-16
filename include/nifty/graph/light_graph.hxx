#pragma once

#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "xtensor/xtensor.hpp"

namespace nifty {
namespace graph {

    // A simple undirected graph,

    typedef uint64_t NodeType;
    typedef int64_t EdgeIndexType;

    typedef std::pair<NodeType, NodeType> EdgeType;
//    typedef boost::hash<EdgeType> EdgeHash;

    // Perfoemance for extraction of 50 x 512 x 512 cube (real labels)
    // (including some overhead (python cals, serializing the graph, etc.))
    // using normal set:    1.8720 s
    // using unordered set: 1.8826 s
    // Note that we would need an additional sort to make the unordered set result correct.
    // As we do not see an improvement, stick with the set for now.
    // But for operations on larger edge / node sets, we should benchmark the unordered set again
    typedef std::set<NodeType> NodeSet;
    typedef std::set<EdgeType> EdgeSet;


    class LightGraph {
        // private graph typedefs

        // NodeAdjacency: maps nodes that are adjacent to a given node to the corresponding edge-id
        typedef std::map<NodeType, EdgeIndexType> NodeAdjacency;
        // NodeStorage: storage of the adjacency for all nodes
        typedef std::unordered_map<NodeType, NodeAdjacency> NodeStorage;
        // EdgeStorage: dense storage of pairs of edges
        typedef std::vector<EdgeType> EdgeStorage;
    public:


        template<class EDGE_ARRAY>
        LightGraph(const size_t number_of_labels,
            const xt::xexpression<EDGE_ARRAY> & uvs_exp) : nodeMaxId_(0) {

            // casts
            const auto & uvs = uvs_exp.derived_cast();
            const size_t num_edges = uvs.shape()[0];

//            Create edge_ID vector
//            std::vector<size_t> indices(num_edges + num_mutex);
//            std::iota(indices.begin(), indices.end(), 0);
//            typedef uint64_t NodeType;
//            typedef int64_t EdgeIndexType;
//
//            typedef std::pair<NodeType, NodeType> EdgeType;
            edges_.resize(num_edges);
            for(size_t edgeId = 0; edgeId < num_edges; ++edgeId) {
                edges_[edgeId] = std::make_pair(uvs(edgeId, 0), uvs(edgeId, 1));
            }

            // init the graph
            initGraph();
        }

        // non-constructor API

        // Find edge-id corresponding to the nodes u, v
        // returns -1 if no such edge exists
        EdgeIndexType findEdge(NodeType u, NodeType v) const {
            // find the node iterator
            auto uIt = nodes_.find(u);
            // don't find the u node -> return -1
            if(uIt == nodes_.end()) {
                return -1;
            }
            // check if v is in the adjacency of u
            auto vIt = uIt->second.find(v);
            // v node is not in u's adjacency -> return -1
            if(vIt == uIt->second.end()) {
                return -1;
            }
            // otherwise we have found the edge and return the edge id
            return vIt->second;
        }

        // get the node adjacency
        const NodeAdjacency & nodeAdjacency(const NodeType node) const {
            return nodes_.at(node);
        }


        // number of nodes and edges
        size_t numberOfNodes() const {return nodes_.size();}
        size_t numberOfEdges() const {return edges_.size();}
        size_t nodeMaxId() const {return nodeMaxId_;}

        const EdgeStorage & edges() const {return edges_;}

        void nodes(std::set<NodeType> & out) const{
            for(auto nodeIt = nodes_.begin(); nodeIt != nodes_.end(); ++nodeIt) {
                out.insert(nodeIt->first);
            }
        }

        void nodes(std::vector<NodeType> & out) const{
            out.clear();
            out.resize(numberOfNodes());
            size_t nodeId = 0;
            for(auto nodeIt = nodes_.begin(); nodeIt != nodes_.end(); ++nodeIt, ++nodeId) {
                out[nodeId] = nodeIt->first;
            }
        }

    private:
        // init the graph from the edges
        void initGraph() {
            // iterate over the edges we have
            NodeType u, v;
            NodeType maxNode;
            EdgeIndexType edgeId = 0;
            for(const auto & edge : edges_) {
                u = edge.first;
                v = edge.second;

                // insert v in the u adjacency
                auto uIt = nodes_.find(u);
                if(uIt == nodes_.end()) {
                    // if u is not in the nodes vector yet, insert it
                    nodes_.insert(std::make_pair(u, NodeAdjacency{{v, edgeId}}));
                } else {
                    uIt->second[v] = edgeId;
                }

                // insert u in the v adjacency
                auto vIt = nodes_.find(v);
                if(vIt == nodes_.end()) {
                    // if v is not in the nodes vector yet, insert it
                    nodes_.insert(std::make_pair(v, NodeAdjacency{{u, edgeId}}));
                } else {
                    vIt->second[u] = edgeId;
                }

                // increase the edge id
                ++edgeId;

                // update the node max id
                maxNode = std::max(u, v);
                if(maxNode > nodeMaxId_) {
                    nodeMaxId_ = maxNode;
                }
            }
        }

        NodeType nodeMaxId_;
        NodeStorage nodes_;
        EdgeStorage edges_;
    };



}
}
