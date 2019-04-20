#pragma once

#include <queue>
#include <algorithm>
#include "nifty/xtensor/xtensor.hxx"


namespace nifty {
namespace skeletons {

    typedef std::array<std::size_t, 3> Coord;

    inline float _sq(const float wa, const float wb) {
        return std::sqrt(wa * wa + wb * wb);
    }

    inline float _cu(const float wa, const float wb, const float wc) {
        return std::sqrt(wa * wa + wb * wb + wc * wc);
    }


    // TODO neighborhoods
    inline std::array<Coord, 26> precomputed_nhoods() {
        std::array<Coord, 26> nhoods;
        return nhoods;
    }


    // TODO multipliers
    inline std::array<float, 26> precomputed_multipliers(const std::array<float, 3> & voxel_size) {
        std::array<float, 26> multiplier;
        return multiplier;
        /*
        const float vz, vy, vx = voxel_size[0], voxel_size[1], voxel_size[2];
        std::array<float, nhood_size> multipliers = {
            // direct links
            vz, vz, vy, vy, vx, vx,

            // square diagonal links
			_sq(vz, vy), _sq(vz, vy), _sq(vz, vy), _sq(vz, vy),
			_sq(vy, vx), _sq(vy, vx), _sq(vy, vx), _sq(vy, vx),
			_sq(vz, vx), _sq(vz, vx), _sq(vz, vx), _sq(vz, vx),

            // cube diagonal links
			_cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx),
			_cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx)
        };
        */
    }


    template<class MASK, class DIST>
    inline void euclidean_distance(const MASK & mask, const Coord & root,
                                   const std::array<float, 3> & voxel_size,
                                   DIST & distance) {
        // size of indirect 3d nhood
        constexpr unsigned nhood_size = 26;

        // number of voxels in volume and in plane
        const auto & shape = mask.shape();
        const std::size_t n_voxels = std::accumulate(shape.begin(), shape.end(), 1,
                                                     std::multiplies<std::size_t>());
        const std::size_t n_voxels_xy = std::accumulate(shape.begin() + 1, shape.end(), 1,
                                                        std::multiplies<std::size_t>());

        // set all distances to inf; root distance to 0
        // TODO better way to fill xtensor ?
        const float inf = std::numeric_limits<float>::infinity();
        std::fill(distance.begin(), distance.end(), inf);
        xt::xindex root_coord(3);
        std::copy(root.begin(), root.end(), root_coord.begin());
        distance[root_coord] = -0;

        // pq for distance nodes
        typedef std::pair<xt::xindex, float> Node;
        auto node_comp = [](const Node & a, const Node & b){return a.second < b.second;};
        std::priority_queue<Node, std::vector<Node>, decltype(node_comp)> pq(node_comp);
        pq.emplace(root_coord, 0.0);

        // precomputed nhoods and distance multiplier
        const auto nhoods = precomputed_nhoods();
        const auto multipliers = precomputed_multipliers(voxel_size);

        // compute all distances
        while(!pq.empty()) {
            const auto node = pq.top();
            pq.pop();
            const auto & coord = node.first;

            float & dist = distance[coord];
            // negative weights : node was already visited
            if(std::signbit(dist)) {
                continue;
            }

            // iterate over the indirect neighborhood
            for(int ngb = 0; ngb < nhoods.size(); ++ngb) {
                const auto & nhood = nhoods[ngb];

                // make 3d ngb coordinate and check that it's valid
                xt::xindex ngb_coord(3);
                std::copy(coord.begin(), coord.end(), ngb_coord.begin());
                bool out_of_range = false;
                for(unsigned d = 0; d < 3; ++d) {
                    ngb_coord[d] += nhood[d];
                    if(ngb_coord[d] < 0 || ngb_coord[d] > shape[d]) {
                        out_of_range = true;
                        break;
                    }
                }
                if(out_of_range) {
                    continue;
                }

                // check that ngb is in mask
                if(!mask[ngb_coord]) {
                    continue;
                }

                float & ngb_dist = distance[ngb_coord];
                // check if ngb was visited
                if(std::signbit(ngb_dist)) {
                    continue;
                }

                ngb_dist = dist + multipliers[ngb];
                pq.emplace(ngb_coord, ngb_dist);
            }

            // mark this position as visited
            dist *= -1;
        }

    }


    template<class DIST, class PATH>
    inline void dijkstra(const DIST & dist, const Coord & src,
                         const Coord & target, PATH & path) {

    }


    template<class MASK, class DIST, class PATH>
    inline void compute_path_mask(const MASK & mask, const DIST & distance, const PATH & path,
                                  const double mask_scale, const double mask_min_radius,
                                  const Coord & voxel_size, MASK & out) {

    }


    template<class MASK>
    inline void boundary_voxel(const MASK & mask, Coord & out) {
        const auto & shape = mask.shape();
        for(int64_t z = 0; z < shape[0]; ++z) {
            for(int64_t y = 0; y < shape[1]; ++y) {
                for(int64_t x = 0; x < shape[2]; ++x) {
                    if(mask(z, y, x)) {
                        out[0] = z;
                        out[1] = y;
                        out[2] = x;
                        return;
                    }
                }
            }
        }
    }

}
}
