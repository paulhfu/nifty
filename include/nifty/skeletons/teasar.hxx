#include <math.h>
#include "vigra/multi_distance.hxx"
#include <nifty/graph/undirected_grid_graph.hxx>
#include <nifty/graph/shortest_path_dijkstra.hxx>
#include <nifty/xtensor/xtensor.hxx>



namespace nifty {
namespace skeletons {

    enum NodeLabel {
        Outside    = 0, /* voxel not in mask */
		Inside     = 1, /* ordinary inside voxels and initial values */
		Boundary   = 2, /* inside voxels on boundary */
		Explained  = 3, /* boundary voxels that are within a threshold distance to skeleton voxels */
		OnSkeleton = 4, /* skeleton voxels */
		Visited    = 5  /* skeleton voxels that have been added to Skeleton datastructure (eventually, all OnSkeleton voxels)*/
	};

    template<class GRAPH, class MASK>
    inline void find_nodes_and_boundaries(const GRAPH & graph, const MASK & mask,
                                          std::vector<NodeLabel> & node_labels, std::vector<int64_t> & boundary_nodes) {

        // 1st pass: set all nodes to inside / outside
        for(std::size_t node_id = 0; node_id < graph.numberOfNodes(); ++node_id) {
            const auto & coord = graph.nodeToCoordinate(node_id);
            if(xtensor::read(mask, coord)) {
                node_labels[node_id] = NodeLabel::Inside;
            } else {
                node_labels[node_id] = NodeLabel::Outside;
            }
        }

        // NOTE change to 26 for indirect nhood
        const int max_neighbors = 6;
        // 2nd pass: find boundary nodes
        for(std::size_t node_id = 0; node_id < graph.numberOfNodes(); ++node_id) {
            if(node_labels[node_id] == NodeLabel::Outside) {
                continue;
            }

            int num_neighbors = 0;
            for(auto adj = graph.adjacencyBegin(node_id); adj != graph.adjacencyEnd(node_id); ++node_id) {
                const int64_t adj_node = adj->node();
                if(node_labels[adj_node] != NodeLabel::Outside) {
                    ++num_neighbors;
                }
            }

            if(num_neighbors < max_neighbors) {
                node_labels[node_id] = NodeLabel::Boundary;
                boundary_nodes.push_back(node_id);
            }
        }
    }


    template<class GRAPH, class MASK>
    inline uint64_t compute_edge_penalties(const GRAPH & graph, const MASK & mask,
                                           const std::vector<NodeLabel> & node_labels, const std::vector<double> & pixel_pitch,
                                           const double boundary_weight, std::vector<float> & penalties) {
        // init the boundary distance array
        // NOTE this needs to be a vigra array, because we use the vigra distance trafo
        // would be nice to use / implement something compatible with xtensor in the future
        const auto & shape = mask.shape();
	    vigra::MultiArray<3, float> boundary_distance(vigra::Shape3(shape[0],
                                                                    shape[1],
                                                                    shape[2]));
        // set all boundary distances of valid nodes to 1.
        for(std::size_t z = 0; z < shape[0]; ++z) {
            for(std::size_t y = 0; y < shape[1]; ++y) {
                for(std::size_t x = 0; x < shape[2]; ++x) {
                    boundary_distance(z, y, x) = mask(z, y, x);
                }
            }
        }

        // compute squared distance to the outside
	    vigra::separableMultiDistSquared(boundary_distance, boundary_distance,
                                         false, pixel_pitch);

        // find the center point (= point with maximal boundary distance)
        double max_boundary_distance = 0.;
        uint64_t center_node = 0;
        for(std::size_t node_id = 0; node_id < graph.numberOfNodes(); ++node_id) {
            if(node_labels[node_id] == NodeLabel::Outside) {
                continue;
            }

            const auto & coord = graph.nodeToCoordinate(node_id);
            const float dist = boundary_distance(coord[0], coord[1], coord[2]);
            if(dist > max_boundary_distance) {
                center_node = node_id;
                max_boundary_distance = dist;
            }

        }

	    // multiply with Euclidean node distances
	    //
	    // The TEASAR paper suggests to add the Euclidean distances. However, for
	    // the penalty to be meaningful in anistotropic volumes, it should be
	    // multiplied with the Euclidean distance between the nodes (otherwise, it
	    // is more expensive to move in the high-resolution dimensions). Therefore,
	    // the final value is
	    //   penalty*euclidean + euclidean = euclidean*(penalty + 1)

        // create initial boundary penalties
        for(std::size_t edge_id = 0; edge_id < graph.numberOfEdges(); ++edge_id) {
            const auto & u = graph.nodeToCoordinate(graph.u(edge_id));
            const auto & v = graph.nodeToCoordinate(graph.v(edge_id));
            const float du = boundary_distance(u[0], u[1], u[2]);
            const float dv = boundary_distance(v[0], v[1], v[2]);
            const double penalty = boundary_weight * (1. - sqrt(0.5 * (du + dv) / max_boundary_distance));
            const double euclidean = (u[0] != v[0]) ? pixel_pitch[0] : ((u[1] != v[1]) ? pixel_pitch[1] : pixel_pitch[2]);
            penalties[edge_id] = euclidean * (penalty + 1);
        }

        return center_node;

        /* Note this is the euclidean distance computation for indirect nh
	    float nodeDistances[8];
	    nodeDistances[0] = 0;
	    nodeDistances[1] = _graphVolume.getResolutionZ();
	    nodeDistances[2] = _graphVolume.getResolutionY();
	    nodeDistances[3] = sqrt(pow(_graphVolume.getResolutionY(), 2) + pow(_graphVolume.getResolutionZ(), 2));
	    nodeDistances[4] = _graphVolume.getResolutionX();
	    nodeDistances[5] = sqrt(pow(_graphVolume.getResolutionX(), 2) + pow(_graphVolume.getResolutionZ(), 2));
	    nodeDistances[6] = sqrt(pow(_graphVolume.getResolutionX(), 2) + pow(_graphVolume.getResolutionY(), 2));
	    nodeDistances[7] = sqrt(pow(_graphVolume.getResolutionX(), 2) + pow(_graphVolume.getResolutionY(), 2) + pow(_graphVolume.getResolutionZ(), 2));
        */

        /*
	    for (GraphVolume::EdgeIt e(_graphVolume.graph()); e != lemon::INVALID; ++e) {

	    	Position u = _graphVolume.positions()[_graphVolume.graph().u(e)];
	    	Position v = _graphVolume.positions()[_graphVolume.graph().v(e)];

	    	int i = 0;
	    	if (u[0] != v[0]) i |= 4;
	    	if (u[1] != v[1]) i |= 2;
	    	if (u[2] != v[2]) i |= 1;

	    	_distanceMap[e] = nodeDistances[i]*(_distanceMap[e] + 1);
	    }
        */
    }

    template<class DIJKSTRA, class PENALTIES>
    inline int64_t find_root(DIJKSTRA & dijkstra, const PENALTIES & penalties,
                             const std::vector<int64_t> & boundary_nodes, const int64_t center_node) {

    	dijkstra.runSingleSourceMultiTarget(penalties, center_node, boundary_nodes);
        const auto & distances = dijkstra.distances();

    	// find furthest point on the boundary
        int64_t root = 0;
    	float max_dist = -1;
    	for (const int64_t node_id : boundary_nodes) {
    		if (distances[node_id] > max_dist) {
    			root = node_id;
    			max_dist = distances[node_id];
    		}
    	}
    	if (max_dist == -1) {
            throw std::runtime_error("could not find a root boundary point");
        }
        return root;
    }


    template<class DIJKSTRA, class PENALTIES, class LABELS>
    inline bool extract_longest_segment(DIJKSTRA & dijkstra, const PENALTIES & penalties,
                                        const std::vector<int64_t> & boundary_nodes, const int64_t root_node,
                                        LABELS & labels, int64_t & current_node,
                                        const bool skip_explained_nodes, const std::size_t min_segment_length) {
	    // compute distances from current root node to all boundary points
        dijkstra.runSingleSourceMultiTarget(current_node, boundary_nodes);
        const auto & distances = dijkstra.distances();

	    // find furthest point on boundary
	    float max_dist = -1;
        int64_t furthest = 0;
	    for(const auto node_id : boundary_nodes) {
            // TODO what does this parameter mean ?
	    	if (skip_explained_nodes && node_labels[node_id] == NodeLabel::Explained) {
	    		continue;
            }

	    	if (distances[node_id] > max_dist) {
	    		furthest = node_id;
	    		max_value = distances[node_id];
	    	}
	    }

	    // no more points or length smaller then min segment length
	    if (max_dist == -1 || max_dist < min_segment_length) {
	    	return false;
        }

	    // walk backwards to next skeleton point
	    int64_t n = furthest;
	    while (node_labels[n] != OnSkeleton) {

	    	node_labels[n] = OnSkeleton;

	    	if (_parameters.skipExplainedNodes)
	    		drawExplanationSphere(_graphVolume.positions()[n]);

	    	GraphVolume::Edge pred = _dijkstra.predMap()[n];
	    	GraphVolume::Node u = _graphVolume.graph().u(pred);
	    	GraphVolume::Node v = _graphVolume.graph().v(pred);

	    	n = (u == n ? v : u);
	    	_distanceMap[pred] = 0.0;
	    }

	    // first segment?
	    if (n == _root) {
	    	LOG_DEBUG(skeletonizelog) << "longest segment has length " << maxValue << std::endl;
	    	_parameters.minSegmentLength = std::max(_parameters.minSegmentLength, _parameters.minSegmentLengthRatio*maxValue);
	    	LOG_DEBUG(skeletonizelog) << "setting min segment length to " << _parameters.minSegmentLength << std::endl;
	    }
	    return true;
    }


    // TODO gather all relevant params and output type(s)
    template<class MASK>
    inline void skeletonize(const MASK & mask, const std::vector<double> & pixel_pitch,
                            const double boundary_weight, const std::size_t max_num_segments) {

        typedef nifty::array::StaticArray<int64_t, 3> Coord;
        typedef graph::UndirectedGridGraph<3, true> GridType;

        // NOTE for now, the nifty grid graph only supports
        // direc neighborhood. Original teasar is implemented
        // using inderect nhood. I don't think that will make much
        // of a difference, but would still be nice to support both
        // (actually not sure if this is true, the graph has the
        // 'SIMPLE_NH' flag, could try that at some point

        // build the 3d grid graph
        const auto & shape = mask.shape();
        const Coord graph_shape = {shape[0], shape[1], shape[2]};
        const GridType grid_graph(graph_shape);

        // find the ndoes inside of the mask and at its boundaries
        std::vector<NodeLabel> node_labels(grid_graph.numberOfNodes());
        std::vector<int64_t> boundary_nodes;
        find_nodes_and_boundaries(grid_graph, mask,
                                  node_labels, boundary_nodes);

        // make the edge penalty map
        // TODO is it correct to set penalties for invalid edges (= edges between masked nodes)
        // to zero?
        std::vector<float> penalties(grid_graph.numberOfEdges());
        const int64_t center_node = compute_edge_penalties(grid_graph, mask, node_labels,
                                                           pixel_pitch, boundary_weight, penalties);

        // initialize the shortest path computer
        graph::ShortestPathDijkstra<GridType, float> dijkstra(grid_graph);

        // find the root node
        int64_t root_node = find_root(dijkstra, penalties, boundary_nodes, center_node);
        node_labels[root_node] = NodeLabel::OnSkeleton;

        // extract skeleton segments until we reach the maximal number
        // or there are no more segments longer than the min length to extract
        int64_t current_node = root_node;
        for (int ii = 0; ii < max_num_segments; ++ii) {
            const bool continue_extraction = extract_longest_segment(dijkstra, penalties, boundary_nodes, root_node,
                                                                     node_labels, current_node);
 		    if (!continue_extraction){
                break;
            }
	    }
	    // return parseVolumeSkeleton();
    }


}
}
