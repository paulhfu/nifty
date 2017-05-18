#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/marray/marray.hxx"

namespace nifty {
namespace skeletons {

    template<unsigned DIM, typename MARKER_TYPE>
    class SkeletonGraph {
    
    public:
        typedef MARKER_TYPE MarkerType;
        typedef UndirectedGraph<> Graph;
        typedef array::StaticArray<int64_t,DIM> Coordinate;
        typedef std::vector<Coordinate> Coordinates;

        SkeletonGraph(const marray::View<MarkerType> & skeleton, double anisotropy, unsigned anisotropicDimension = 0);

        const Graph & graph() const{
            return graph_;
        }

        const Coordinates & edgeCoordinates(const int64_t edgeId) const {
            return edgeCoordinates_[edgeId];
        }

        size_t edgeLength(const int64_t edgeId) const;

        const Coordinate & nodeCoordinate(nodeId) const{
            return (nodeId < terminalOffset_) ? terminalNodes_[nodeId] : branchNodes_[nodeId - terminalOffset_];
        }


    private:

        void findNodes(marray::View<bool> & nodeMap);
        void growToBranchNodes(marray::View<bool> & visited, const marray::View<bool> & nodeMap);
        void constructGraph(   marray::View<bool> & visited, const marray::View<bool> & nodeMap);

        // return node type of coord (0 -> no node, 1 -> terminal node, 2 -> branch node) 
        inline uint8_t checkAdjacencyBox(const Coordinate & coord);
        
        // fill nextCoordinate vector with next coordinates (only 1 if this is a node !)
        inline void checkAdjacencyBoxWithVisited(const Coordinate & coord, marray::View<bool> & visited, Coordinates & nextCoordinates);

        const marray::View<MarkerType> & skeleton_;
        double anisotropy_;
        unsigned anisotropicDimension_;
        Graph graph_;

        Coordinates terminalNodes_;
        Coordinates branchNodes_;
        size_t branchOffset_;
        std::vector<Coordinates> edgeCoordinates_;

    };

    template<typename MARKER_TYPE>
    SkeletonGraph<MARKER_TYPE>::SkeletonGraph(const marray::View<MARKER_TYPE> & skeleton, double anisotropy, unsigned anisotropicDimension) :
        skeleton_(skeleton),
        anisotropy_(anisotropy),
        anisotropicDimension_(anisotropicDimension),
        graph_(0) {

        // find all nodes
        marray::Marray<int64_t> nodeMap(skeleton_.shapeBegin(), skeleton_.shapeEnd());
        findNodes(nodeMap);

        // grow from terminals to first set of branch nodes
        marray::Marray<bool> visited(skeleton_.shapeBegin(), skeleton_.shapeEnd());
        growToBranchNodes(visited, nodeMap);

        // construct the inner graph
        constructGraph(visited, nodeMap);
    }
    
    
    template<typename MARKER_TYPE>
    void SkeletonGraph<MARKER_TYPE>::findNodes(const marray::View<int64_t> & nodeMap) {
        // iterate over the volume and for each pixel check the adjacency box
        // for 1 neighbor add a terminal node
        // for > 2 neighbors add a branch node
        // keep track of node coordinates and insert nodes into node map
        
        auto numberOfNodes = terminalNodes_.size() + branchNodes_.size()
        graph.assign(numberOfNodes);
    }


} // namespace::graph
} // namespace::nifty
