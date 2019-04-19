#pragma once

#include <algorithm>
#include "nifty/libdivide.h"
#include "nifty/xtensor/xtensor.hxx"


namespace nifty {
namespace skeletons {

    inline float _sq(const float wa, const float wb) {
        return std::sqrt(wa * wa + wb * wb);
    }

    inline float _cu(const float wa, const float wb, const float wc) {
        return std::sqrt(wa * wa + wb * wb + wc * wc);
    }

    typedef std::array<std::size_t, 3> Coord;

    template<class MASK, class DIST>
    inline void euclidean_distance(const MASK & mask, const Coord & root,
                                   const std::array<float, 3> voxel_size, DIST & distance) {
        // size of indirect 3d nhood
        constexpr unsigned nhood_size = 26;

        // number of voxels in volume and in plane
        const auto & shape = mask.shape();
        const std::size_t n_voxels = std::accumulate(shape.begin(), shape.end(), 1,
                                                     std::multiplies<std::size_t>());
        const std::size_t n_voxels_xy = std::accumulate(shape.begin() + 1, shape.end(), 1,
                                                        std::multiplies<std::size_t>());

        const libdivide::divider<std::size_t> div_plane(n_voxels_xy);
        const libdivide::divider<std::size_t> div_vol(n_voxels);

        // check if the plane shape is a power of 2 and calculate shifts
        const bool power_of_two = !((shape[1] & (shape[1] - 1)) || (shape[2] & (shape[2] - 1)));
        const int shfty = std::log2(shape[1]);
        const int shftx = std::log2(shape[2]);

        // set all distances to inf; root distance to 0
        // TODO better way to fill xtensor ?
        std::fill(distance.begin(), distance.end(), +INFINTY);
        xt::xindex root_coord(3);
        std::copy(root.begin(), root.end(), root_coord.begin());
        distance[root_coord] = -0;

        // make the neighborhood and distance multipliers
        std::array<int, nhood_size> neighborhood;
        const float vz, vy, vx = voxel_size[0], voxel_size[1], voxel_size[2];
        std::array<float, nhood_size> nhood_multipliers = {
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

    }

}
}
