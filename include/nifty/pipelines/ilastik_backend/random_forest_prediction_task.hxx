#ifndef _RANDOM_FOREST_PREDICTION_TASK_H_
#define _RANDOM_FOREST_PREDICTION_TASK_H_

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <tbb/concurrent_lru_cache.h>

#include <algorithm>

#include <tbb/tbb.h>
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include <nifty/marray/marray.hxx>

namespace nifty
{
namespace pipelines
{
namespace ilastik_backend
{
    
    template<unsigned DIM>
    class random_forest_prediction_task /*: public tbb::task */
    {
    public:
        // typedefs
        using data_type = float;
        using float_array_view = nifty::marray::View<data_type>;
        using float_array = nifty::marray::Marray<data_type>;
        
        using random_forest = nifty::pipelines::ilastik_backend::RandomForestType;
        using feature_cache = tbb::concurrent_lru_cache<size_t, float_array, std::function<float_array(size_t)>>;
        
        using out_shape_type = array::StaticArray<int64_t,DIM+1>;
        using feature_shape_type = array::StaticArray<int64_t,2>;

    public:
        // API
        random_forest_prediction_task(
            size_t blockId,
            feature_cache& fc,
            float_array_view& out,
            random_forest& random_forest
        ):
            blockId_(blockId),
            feature_cache_(fc),
            out_array_(out),
            random_forest_(random_forest)
        {
        }

        tbb::task* execute()
        {
            // ask for features. This blocks if it's not present
            feature_cache::handle ho = feature_cache_[blockId_];
            float_array_view& features = ho.value();
            compute(features);
            return NULL;
        }

        // TODO fix the reshapes to avoid 1 data copy
        void compute(const float_array_view & in)
        {
            std::cout << "rf_task::compute for block " << blockId_ << std::endl;
            size_t num_classes = random_forest_.num_classes();
            size_t num_features = random_forest_.num_features();
            assert(num_features == in.shape(0));

            //std::cout << "Filters" << std::endl;
            //std::cout << in(0) << std::endl;

            size_t pixel_count = 1;
            out_shape_type in_shape;
            for(int d = 0; d < DIM+1; ++d) {
                in_shape[d] = in.shape(d);
                if(d < DIM)
                    pixel_count *= in.shape(d);
            }
            //std::cout << in_shape << std::endl;
            //std::cout << pixel_count << std::endl;

            feature_shape_type feature_shape({pixel_count, num_features});
            feature_shape_type prediction_shape({pixel_count, num_classes});
            
            // FIXME TODO copy for now, but we should be able to do this w/o
            //std::cout << "Flattened shape:" << std::endl;
            //auto in_flatten    = in.reshapedView(feature_shape.begin(), feature_shape.end());
            marray::Marray<data_type> in_flatten(feature_shape.begin(), feature_shape.end());
            tools::forEachCoordinate(in_shape, [&in, &in_flatten](const out_shape_type& coord)
            {
                size_t pixel = (DIM == 2) ? coord[0] + coord[1]*in.shape(0) : coord[0] + coord[1] * in.shape(0) + coord[2] * (in.shape(0) * in.shape(1));
                in_flatten(pixel, coord[DIM]) = in(coord.asStdArray());
            });
            
            for(int d = 0; d < in_flatten.dimension(); ++d)
                std::cout << in_flatten.shape(d) << std::endl;
            
            marray::Marray<data_type> prediction(prediction_shape.begin(), prediction_shape.end());

            // loop over all random forests for prediction probabilities
            random_forest_.predict_probs(in_flatten, prediction);
            prediction /= random_forest_.num_trees();

            //auto miMa = std::minmax_element(prediction.begin(), prediction.end());
            //std::cout << "MinMax predition " << *miMa.first << " " << *miMa.second << std::endl;

            // transform back to marray
            out_shape_type output_shape;
            for(size_t d = 0; d < DIM; ++d) {
                output_shape[d] = out_array_.shape(d);
            }
            
            output_shape[DIM] = num_classes;
            //float_array_view & tmp_out_array = prediction.reshapedView(output_shape.begin(), output_shape.end());
            //out_array_ = prediction.reshapedView(output_shape.begin(), output_shape.end());
            
            tools::forEachCoordinate(output_shape, [&prediction, this](const out_shape_type& coord)
            {
                size_t pixel = (DIM == 2) ? coord[0] + coord[1]*this->out_array_.shape(0) 
                    : coord[0] + coord[1] * this->out_array_.shape(0) + coord[2] * (this->out_array_.shape(0) * this->out_array_.shape(1));
                this->out_array_(coord.asStdArray()) = prediction(pixel, coord[DIM]);
            });
            
            std::cout << "rf_task::compute done" << std::endl;
        }

    private:
        // members
        size_t blockId_;
        feature_cache & feature_cache_;
        float_array_view & out_array_;
        random_forest & random_forest_;
    };

} // namespace ilastik_backend
} // namespace pipelines
} // namespace nifty

#endif // _RANDOM_FOREST_PREDICTION_TASK_H_
