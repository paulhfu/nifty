#pragma once

namespace nifty {
namespace skeletons {

    typedef std::arrray<std::size_t, 3> Coord;

    template<class MASK, class DIST>
    inline void euclidean_distance(const MASK & mask, const Coord & root,
                                   const Coord voxel_size, DIST & distance) {

    }


    template<class DIST, class PATH>
    inline void dijkstra(const DIST & dist, const Coord & src, const Coord & target, PATH & path) {

    }


    template<class MASK, class DIST, class PATH>
    inline void compute_path_mask(const MASK & mask, const DIST & distance, const PATH & path,
                                  const double mask_scale, const double mask_min_radius, const Coord & voxel_size,
                                  MASK & out) {

    }


    template<class MASK>
    inline void boundary_voxel(const MASK & mask, Coord & out) {

    }

}
}
