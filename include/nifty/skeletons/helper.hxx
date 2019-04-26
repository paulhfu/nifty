#pragma once

#include <queue>
#include <algorithm>
#include "nifty/xtensor/xtensor.hxx"
#include "xtensor/xstrides.hpp"


namespace nifty {
namespace skeletons {

    typedef std::array<int64_t, 3> Coord;

    inline float _sq(const float wa, const float wb) {
        return std::sqrt(wa * wa + wb * wb);
    }

    inline float _cu(const float wa, const float wb, const float wc) {
        return std::sqrt(wa * wa + wb * wb + wc * wc);
    }


    inline std::array<Coord, 26> precomputed_nhoods() {
        // direct links (6)
        std::array<Coord, 26> nhoods;
        nhoods[0] = {-1, 0, 0};
        nhoods[1] = {1,  0, 0};
        nhoods[2] = {0, -1, 0};
        nhoods[3] = {0,  1, 0};
        nhoods[4] = {0, 0, -1};
        nhoods[5] = {0, 0,  1};
        // square diagonal links (12)
        nhoods[6]  = {-1,-1, 0};
        nhoods[7]  = {-1, 1, 0};
        nhoods[8]  = {1, -1, 0};
        nhoods[9]  = {1,  1, 0};
        nhoods[10] = {0, -1, -1};
        nhoods[11] = {0, -1,  1};
        nhoods[12] = {0,  1, -1};
        nhoods[13] = {0,  1,  1};
        nhoods[14] = {-1, 0, -1};
        nhoods[15] = {-1, 0,  1};
        nhoods[16] = {1,  0, -1},
        nhoods[17] = {1,  0,  1};
        // cube diagonal links (8)
        nhoods[18] = {-1, -1, -1};
        nhoods[19] = { 1, -1, -1};
        nhoods[20] = { 1,  1, -1};
        nhoods[21] = { 1, -1,  1};
        nhoods[22] = {-1,  1, -1};
        nhoods[23] = {-1, -1,  1};
        nhoods[24] = {-1,  1,  1};
        nhoods[25] = { 1,  1,  1};
        return nhoods;
    }


    inline std::array<float, 26> precomputed_multipliers(const std::array<float, 3> & voxel_size) {
        const float vz = voxel_size[0];
        const float vy = voxel_size[1];
        const float vx = voxel_size[2];
        std::array<float, 26> multipliers = {
            // direct links (6)
            vz, vz, vy, vy, vx, vx,

            // square diagonal links (12)
			_sq(vz, vy), _sq(vz, vy), _sq(vz, vy), _sq(vz, vy),
			_sq(vy, vx), _sq(vy, vx), _sq(vy, vx), _sq(vy, vx),
			_sq(vz, vx), _sq(vz, vx), _sq(vz, vx), _sq(vz, vx),

            // cube diagonal links (8)
			_cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx),
			_cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx), _cu(vz, vy, vx)
        };
        return multipliers;
    }


    template<class MASK, class DIST>
    inline void euclidean_distance(const MASK & mask, const Coord & root,
                                   const std::array<float, 3> & voxel_size,
                                   DIST & distance) {

        // number of voxels in volume and in plane
        const auto & shape = mask.shape();

        // get the root coordinate as xindex
        xt::xindex root_coord(3);
        std::copy(root.begin(), root.end(), root_coord.begin());

        // pq for distance nodes
        typedef std::pair<xt::xindex, float> Node;
        auto node_comp = [](const Node & a, const Node & b){return a.second < b.second;};
        std::priority_queue<Node, std::vector<Node>, decltype(node_comp)> pq(node_comp);

        pq.emplace(root_coord, 0.0);

        // precomputed nhoods and distance multiplier
        const auto nhoods = precomputed_nhoods();
        const auto multipliers = precomputed_multipliers(voxel_size);
        const int nhood_size = nhoods.size();

        std::cout << "Start distance computation from voxel" << std::endl;
        std::cout << root[0] << " " << root[1] << " " << root[2] << std::endl;
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
            for(int ngb = 0; ngb < nhood_size; ++ngb) {
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

                // TODO Jan uses dist * (1 + multipliers)
                ngb_dist = dist + multipliers[ngb];
                // FIXME this is line sometimes segfaults
                // (segfault occurs when compiled in Release, not in Debug)
                pq.emplace(ngb_coord, ngb_dist);
            }

            // mark this position as visited
            dist *= -1;
        }

        // TODO can we use xt::abs / xt::fabs ?
        // make all distances positive again
        for(auto dit = distance.begin(); dit != distance.end(); ++dit) {
            *dit = std::fabs(*dit);
        }
        std::cout << "Done distance computation" << std::endl;
    }


    template<class COORD, class STRIDES>
    inline int64_t ravel_from_strides(const COORD & coord, const STRIDES & strides) {
        int64_t ret = 0;
        for(unsigned dim = 0; ++dim; dim < 3) {
            ret += coord[dim] * strides[dim];
        }
        return ret;
    }


    template<class PARENTS, class PATH>
    inline void backtrace_path(const PARENTS & parents, const xt::xindex & target,
                               PATH & path) {
        const auto & strides = parents.strides();
        const auto & layout = parents.layout();

        int64_t current_id = ravel_from_strides(target, strides);
        xt::xindex current_coord = target;

        // NOTE: parents[src] = 0
        while(parents[current_coord]) {
            path.emplace_back(current_coord);
            current_id = parents[current_coord];
            // THIS IS STUPID ....
            // need to copy unravel_from_strides ret type (= std::array) to xindex
            const auto tmp_coord = xt::unravel_from_strides(current_id, strides, layout);
            std::copy(tmp_coord.begin(), tmp_coord.end(), current_coord.begin());
        }
    }


    template<class FIELD, class PATH>
    inline void dijkstra(const FIELD & field, const Coord & src,
                         const Coord & target, PATH & path) {

        // shape and strides
        const auto & shape = field.shape();
        const auto & strides = field.strides();

        // initialize distances and coordiantes
        xt::xtensor<float, 3> distances = xt::zeros<float>(shape);
        xt::xindex src_coord(3);
        std::copy(src.begin(), src.end(), src_coord.begin());

        xt::xindex target_coord(3);
        std::copy(target.begin(), target.end(), target_coord.begin());

        // make parents array
        xt::xtensor<int64_t, 3> parents = xt::zeros<int64_t>(shape);

        // priority queue
        typedef std::pair<xt::xindex, float> Node;
        auto node_comp = [](const Node & a, const Node & b){return a.second < b.second;};
        std::priority_queue<Node, std::vector<Node>, decltype(node_comp)> pq(node_comp);
        pq.emplace(src_coord, 0.0);

        // precomputed indirect nhood
        const auto nhoods = precomputed_nhoods();
        const int nhood_size = nhoods.size();

        // compute all distances
        while(!pq.empty()) {
            const auto node = pq.top();
            pq.pop();
            const auto & coord = node.first;

            float & dist = distances[coord];
            // negative weights : node was already visited
            if(std::signbit(dist)) {
                continue;
            }

            const int64_t prnt_flat = ravel_from_strides(coord, strides);
            // iterate over the indirect neighborhood
            for(int ngb = 0; ngb < nhood_size; ++ngb) {
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

                // get the new distance value
                float & ngb_dist = distances[ngb_coord];

                // check if ngb was visited
                if(std::signbit(ngb_dist)) {
                    continue;
                }
                // get the diff field value at ngb coord
                const float delta = field[ngb_coord];

                // update new distance and the parent field
                parents[ngb_coord] = prnt_flat;
                ngb_dist = dist + delta;

                // break if we found the target
                // (using goto convinience ...)
                if(ngb_coord == target_coord) {
                    goto DONE;
                }

                pq.emplace(ngb_coord, ngb_dist);
            }

            // mark this position as visited
            dist *= -1;
        }

        // we goto done when we find the target
        DONE:
        // backtrace the path
        backtrace_path(parents, target_coord, path);
    }


    template<class MASK, class DIST, class PATH>
    inline void compute_path_mask(const DIST & distance, const PATH & path,
                                  const double mask_scale, const double mask_min_radius,
                                  const Coord & voxel_size, MASK & out) {
        const auto & shape = distance.shape();
        const std::size_t n_vox = distance.size();
        const std::size_t n_vox_xy = shape[1] * shape[2];

        // allocate the topology
        // TODO I don't really understand yet how this works
        std::vector<int16_t> topology(n_vox);

        // preallocate local and global min / max
        std::array<int64_t, 3> local_min, local_max;
        std::array<int64_t, 3> global_min;
        std::copy(shape.begin(), shape.end(), global_min.begin());
        std::array<int64_t, 3> global_max = {0, 0, 0};

        // iterate over the coordinates in the path and compute the topology
        for(const auto & coord: path) {
            // local radius
            const float radius = mask_scale * distance[coord] + mask_min_radius;

            // compute local min and max
            for(unsigned dim = 0; dim < 3; ++dim) {
                local_min[dim] = std::max(0L, static_cast<int64_t>(coord[dim] - radius / voxel_size[dim]));
                local_max[dim] = std::min(shape[dim] - 1, static_cast<int64_t>(0.5 + coord[dim] + radius / voxel_size[dim]));

                global_min[dim] = std::min(local_min[dim], global_min[dim]);
                global_max[dim] = std::max(local_max[dim], global_max[dim]);
            }

            // TODO does this hold up with different axis order ?
            // do stuff i don't understand yet ;(
            for(int z = local_min[0]; z <= local_max[0]; ++z) {
                for(int y = local_min[1]; y <= local_max[1]; ++y) {
                    topology[local_min[2] + shape[2] * y + n_vox_xy * z] += 1;
                    topology[local_max[2] + shape[2] * y + n_vox_xy * z] -= 1;
                }
            }
        }

        // make path mask
        for(int z = global_min[0]; z <= global_max[0]; ++z) {
            for(int y = global_min[1]; y <= global_max[1]; ++y) {
                const std::size_t yz_off = shape[0] * y + n_vox_xy * z;
                int coloring = 0;

                for(int x = global_min[2]; x <= global_min[2]; ++x) {
                    coloring += topology[x + yz_off];
                    if(coloring > 0 || topology[x + yz_off]) {
                        // set to masked
                        // note mask is flat and needs to be reshaped later 1
                        out[x + yz_off] = 1;
                    }
                }
            }
        }
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
