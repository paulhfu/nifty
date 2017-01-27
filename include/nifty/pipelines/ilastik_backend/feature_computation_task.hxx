#ifndef _OPERATORS_FILTEROPERATOR_H_
#define _OPERATORS_FILTEROPERATOR_H_

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <tbb/concurrent_lru_cache.h>

#include <tuple>
#include <algorithm>

#include <tbb/flow_graph.h>

#include "nifty/marray/marray.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/pipelines/ilastik_backend/interactive_pixel_classification.hxx"
#include "nifty/tools/blocking.hxx"

namespace nifty {
namespace pipelines {
namespace ilastik_backend {
    
    template<unsigned DIM>
    class feature_computation_task /*: public tbb::task*/
    {
    public:
        using apply_type = nifty::features::ApplyFilters<DIM>;
        using in_data_type = uint8_t;
        using out_data_type = float;
        using raw_cache = hdf5::Hdf5Array<in_data_type>;
        using out_array_view = nifty::marray::View<out_data_type>;
        using in_array_view = nifty::marray::View<in_data_type>;
        using selected_feature_type = std::pair<std::vector<std::pair<std::string, std::vector<bool> > >, std::vector<double>>;
        using coordinate = array::StaticArray<int64_t,DIM>;
        using multichan_coordinate = array::StaticArray<int64_t,DIM+1>;
        using blocking_type = nifty::tools::Blocking<DIM, int64_t>;
        using block_with_halo = nifty::tools::BlockWithHalo<DIM>;

    public:
        feature_computation_task(
                size_t block_id,
                raw_cache & rc,
                out_array_view & out,
                const selected_feature_type & selected_features,
                const blocking_type& blocking,
                const block_with_halo & halo_block,
                const double window_ratio = 2.):
            blockId_(block_id),
            rawCache_(rc),
            outArray_(out),
            apply_(selected_features.second, selected_features.first), // TODO need to rethink if we want to apply different outer scales for the structure tensor eigenvalues
            blocking_(blocking),
            halo_block_(halo_block),
            window_ratio_(window_ratio)
        {
        }
        
        tbb::task* execute()
        {
            // ask for the raw data
            const auto & outerBlock = halo_block_.outerBlock();
            const auto & outerBlockBegin = outerBlock.begin();
            const auto & outerBlockEnd = outerBlock.end();
            const auto & outerBlockShape = outerBlock.shape();
            
            nifty::marray::Marray<in_data_type> in(outerBlockShape.begin(), outerBlockShape.end());
            {
            std::lock_guard<std::mutex> lock(s_mutex);
            rawCache_.readSubarray(outerBlockBegin.begin(), in);
            }
            compute(in);
            return NULL;
        }

        void compute(const in_array_view & in)
        {
            // TODO set the correct window ratios
            //copy input data from uint8 to float
            coordinate in_shape;
            for(int d = 0; d < DIM; ++d)
                in_shape[d] = in.shape(d);
            marray::Marray<out_data_type> in_float(in_shape.begin(), in_shape.end());
            tools::forEachCoordinate(in_shape, [&](const coordinate & coord){
                in_float(coord.asStdArray()) = in(coord.asStdArray());    
            });

            // TODO set the window ratio
            apply_.setWindowRatio(window_ratio_);
            
            // apply the filter via the functor
            // TODO consider passing the tbb threadpool here
            //apply_(in_float, outTransposedView);
            apply_(in_float, outArray_);

            size_t permutation[outArray_.dimension()];
            permutation[DIM] = 0;
            for(int d = 0; d < DIM; ++d)
                permutation[d] = d+1;
            outArray_.permute(permutation);

        }

        // TODO also use the type of the selected features
        static coordinate get_halo(const selected_feature_type & selected_features) {
            const auto & sigmas = selected_features.second;
            size_t halo_size = size_t( round(3.5 * *std::max_element(sigmas.begin(), sigmas.end())));
            coordinate ret;
            std::fill(ret.begin(), ret.end(), halo_size);
            return ret;
        }

    private:
	    static std::mutex s_mutex;
        size_t blockId_;
        raw_cache & rawCache_;
        out_array_view & outArray_;
        apply_type apply_; // the functor for applying the filters
        const blocking_type & blocking_;
        const block_with_halo & halo_block_;
        double window_ratio_;
    };
    
    template <unsigned DIM>
    std::mutex feature_computation_task<DIM>::s_mutex;

} // namespace ilastik_backend
} // namespace pipelines
} // namespace nifty


#endif // _OPERATORS_FILTEROPERATOR_H_
