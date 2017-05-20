#pragma once

#include <deque>

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
        inline void checkAdjacencyBoxWithVisited(const Coordinate & coord, const marray::View<bool> & visited, Coordinates & nextCoordinates);

        const marray::View<MarkerType> & skeleton_;
        double anisotropy_;
        unsigned anisotropicDimension_;
        Graph graph_;

        Coordinate shape_;

        Coordinates terminalNodes_;
        Coordinates branchNodes_;
        std::vector<Coordinates> edgeCoordinates_;

    };

    template<unsigned DIM, typename MARKER_TYPE>
    SkeletonGraph<DIM,MARKER_TYPE>::SkeletonGraph<DIM,MARKER_TYPE>(const marray::View<MARKER_TYPE> & skeleton, double anisotropy, unsigned anisotropicDimension) :
        skeleton_(skeleton),
        anisotropy_(anisotropy),
        anisotropicDimension_(anisotropicDimension),
        graph_(0) {

        for(size_t d = 0; d < DIM; ++d)
            shape_[d] = skeleton_.shape(d);

        // find all nodes
        marray::Marray<int64_t> nodeMap(skeleton_.shapeBegin(), skeleton_.shapeEnd());
        findNodes(nodeMap);

        // grow from terminals to first set of branch nodes
        marray::Marray<bool> visited(skeleton_.shapeBegin(), skeleton_.shapeEnd());
        growToBranchNodes(visited, nodeMap);

        // construct the inner graph
        constructGraph(visited, nodeMap);
    }

 
    // iterate over the volume and for each pixel check the adjacency box
    // for 1 neighbor add a terminal node
    // for > 2 neighbors add a branch node
    // keep track of node coordinates and insert nodes into node map
    template<unsigned DIM, typename MARKER_TYPE>
    void SkeletonGraph<MARKER_TYPE, DIM>::findNodes(marray::View<int64_t> & nodeMap) {

        nifty::tools::forEachCoordinate(shape_, [&](const Coordinate & coord) {
            auto nodeDegree = checkAdjacencyBox(coord);  
            if(nodeDegree == 1) {
                nodeMap(coord.asStdArray()) = terminalNodes_.size(); // the node id corresponds to the current number of terminal nodes
                terminalNodes_.push_back(coord);
            }
            else if(nodeDegree == 2) {
                nodeMap(coord.asStdArray()) = terminalNodes_.size() + branchNodes_.size(); // the node id corresponds to the current number of terminal nodes + branch nodes
                branchNodes_.push_back(coord);
            }
        });
        
        auto numberOfNodes = terminalNodes_.size() + branchNodes_.size()
        graph.assign(numberOfNodes);
    }
 
    
    // iterate over the terminal nodes and grow each terminal node to its nearest branch 
    template<unsigned DIM, typename MARKER_TYPE>
    void SkeletonGraph<MARKER_TYPE, DIM>::growToBranchNodes(
            marray::View<bool> & visited,
            const marray::View<int64_t> & nodeMap) {

        for(size_t u = 0; u < terminalNodes_.size(); ++u) {
            
            Coordiate coord = terminalNodes_[u];
            Coordinates nextCoords;
            Coordinates edgeCoords;
            
            // we loop until we find a branch node and then break
            while(true) {
                
                edgeCoords.push_back(coord);
                visited(coord.asStdArray()) = true; // TODO do we need to do this before or after checking the adjacency box
                checkAdjacencyBoxWithVisited(coord, visited, nextCoords); 

                if( nextCoords.size() == 1) {
                    coord = nextCoords[0];
                }
                else if(nextCoordinates.size() > 1) {
                    auto v      = nodeMap(coord.asStdArray());
                    auto edgeId = graph_.insertEdge(u, v);
                    NIFTY_ASSERT_OP(edgeCoordinates_.size(),==,edgeId,"Edge Ids out of sync");
                    edgeCoordinates_.push_back(edgeCoords);
                    break;
                }
                else {
                    throw std::runtime_error("No next coordinate for skeleton found. This should never happen!");
                }
            }

        }
    }
        
    
    // schedule the edges at the first branch node and then build the graph
    template<unsigned DIM, typename MARKER_TYPE>
    void SkeletonGraph<MARKER_TYPE, DIM>::constructGraph(
            marray::View<bool> & visited,
            const marray::View<bool> & nodeMap) {
        
        // TODO proper type for deque ?!
        // define the queue element: pair(Coordinate, StartNodeId)
        typedef std::pair<Coordinate,int64_t> QueueElemType;
        deque<QueueElemType> queue;
        
        // initialize the queue with the edges leaving from the first branching node
        auto u = terminalNodes_.size();
        auto currentCoord = nodeCoordinate(u);

        Coordinates nextCoords;
        checkAdjacencyBoxWithVisited(currentCoord, visited, nextCoords);
        for(auto & coord : nextCoords) {
            queue.emplace_front( std::make_pair(coord, u) );
        }
        nextCoords.clear();

        // build the inner graph as long as the queue is not empty
        // TODO use as FIFO or LIFO ?
        Coordinates edgeCoords;
        while(queue.size()) {
            
            // clear edge coordinates
            edgeCoords.clear();
            
            auto elem = queue.pop_front(); // FIFO ?! 
            currentCoord = queue.first;
            u = queue.second; 
        
            // we loop until we find the next node, put its edges (if any left) on the queue and break
            while(true) {
                
                nextCoords.clear()
                edgeCoords.push_back(currentCoord);
                visited(currentCoord.asStdArray()) = true; // TODO do we need to do this before or after checking the adjacency box
                checkAdjacencyBoxWithVisited(currentCoord, visited, nextCoords); 

                if( nextCoords.size() == 1) { // if we only find a single next coord, we just keep on going
                    currentCoord = nextCoords[0];
                }
                else { // else we make an edge and push the new edges (if any) on the queue 
                    auto v      = nodeMap(currentCoord.asStdArray());
                    auto edgeId = graph_.insertEdge(u, v);
                    NIFTY_ASSERT_OP(edgeCoordinates_.size(),==,edgeId,"Edge Ids out of sync");
                    edgeCoordinates_.push_back(edgeCoords);
                    for(auto & coord : nextCoords) {
                        queue.emplace_front( std::make_pair(coord, v) );
                    }
                    break;
                }
            }
        
        }
    }


        
} // namespace::graph
} // namespace::nifty
