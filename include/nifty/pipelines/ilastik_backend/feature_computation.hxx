#pragma once

#include <tuple>
#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/pipelines/ilastik_backend/interactive_pixel_classification.hxx"
#include "nifty/tools/blocking.hxx"

namespace nifty {
namespace pipelines {
namespace ilastik_backend {
        
// TODO change filter names from strings to enum!
using selected_feature_type = std::pair<std::vector<std::pair<std::string, std::vector<bool> > >, std::vector<double>>;
    
template<unsigned DIM, typename IN_DATA_TYPE, typename OUT_DATA_TYPE>
void feature_computation(
    const size_t blockId,
    const marray::View<IN_DATA_TYPE> & in,
    marray::View<OUT_DATA_TYPE> & out,
    selected_feature_type selected_features,
    const double window_ratio = 2.) {
    
    std::cout << "feature_computation for block " << blockId << std::endl;
    
    using in_data_type  = IN_DATA_TYPE; 
    using out_data_type = OUT_DATA_TYPE;
    using coordinate = array::StaticArray<int64_t,DIM>;
    using apply_type = nifty::features::ApplyFilters<DIM>;
    using multichan_coordinate = array::StaticArray<int64_t,DIM+1>;

    apply_type applyFilters(selected_features.second, selected_features.first);
    
    // copy data to float
    coordinate in_shape;
    for(int d = 0; d < DIM; ++d)
        in_shape[d] = in.shape(d);
    marray::Marray<out_data_type> in_float(in_shape.begin(), in_shape.end());
    tools::forEachCoordinate(in_shape, [&](const coordinate & coord){
        in_float(coord.asStdArray()) = in(coord.asStdArray());    
    });

    // set the window ratio
    applyFilters.setWindowRatio(window_ratio);
    
    // apply the filter via the functor
    // TODO consider passing the tbb threadpool here
    //apply_(in_float, outTransposedView);
    applyFilters(in_float, out);
    
    // For now we do not permute the dimensions here, but in the rf
    //size_t permutation[out.dimension()];
    //permutation[DIM] = 0;
    //for(int d = 0; d < DIM; ++d)
    //    permutation[d] = d+1;
    //out.permute(permutation);
}

// TODO also use the type of the selected features
template<unsigned DIM>
array::StaticArray<int64_t,DIM> getHaloShape(const selected_feature_type & selected_features) {
    array::StaticArray<int64_t,DIM> ret;
    const auto & sigmas = selected_features.second;
    size_t halo_size = size_t( round(3.5 * *std::max_element(sigmas.begin(), sigmas.end())));
    std::fill(ret.begin(), ret.end(), halo_size);
    return ret;
}

template<unsigned DIM>
size_t getNumberOfChannels(const selected_feature_type & selected_features) {
    using apply_type = nifty::features::ApplyFilters<DIM>;
    return apply_type::numberOfChannels(selected_features.first, selected_features.second);
}

} // namespace ilastik_backend
} // namespace pipelines
} // namespace nifty
