#ifndef _RANDOM_FOREST_PREDICTION_TASK_H_
#define _RANDOM_FOREST_PREDICTION_TASK_H_

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <tbb/concurrent_lru_cache.h>

#include <algorithm>

#include <tbb/tbb.h>
#include "nifty/pipelines/ilastik_backend/random_forest_3_loader.hxx"
#include <nifty/marray/marray.hxx>
// TODO include appropriate vigra stuff
#include <vigra/random_forest_hdf5_impex.hxx>

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
                
                using random_forest_vector = nifty::pipelines::ilastik_backend::RandomForestVectorType;
                using feature_cache = tbb::concurrent_lru_cache<size_t, float_array, std::function<float_array(size_t)>>;
                
                using out_shape_type = array::StaticArray<int64_t,DIM+1>;
                using feature_shape_type = array::StaticArray<int64_t,2>;

            public:
                // API
                random_forest_prediction_task(
                    size_t blockId,
                    feature_cache& fc,
                    float_array_view& out,
                    random_forest_vector& random_forests
                ):
                    blockId_(blockId),
                    feature_cache_(fc),
                    out_array_(out),
                    random_forest_vector_(random_forests)
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
                    // TODO: transform data to vigra array?!
                    size_t num_pixel_classification_labels = random_forest_vector_[0].num_classes();
                    size_t num_required_features = random_forest_vector_[0].num_features();
                    assert(num_required_features == in.shape(0));

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

                    feature_shape_type feature_shape({pixel_count, num_required_features});
                    feature_shape_type prediction_shape({pixel_count, num_pixel_classification_labels});
                    
                    //std::cout << "Flattened shape:" << std::endl;
                    //auto in_flatten    = in.reshapedView(feature_shape.begin(), feature_shape.end());
                    
                    // TODO copy for now, but we should be able to do this w/o
                    marray::Marray<data_type> in_flatten(feature_shape.begin(), feature_shape.end());

                    // copy data from marray to vigra. TODO: is the axes order correct??
                    tools::forEachCoordinate(in_shape, [&in, &in_flatten](const out_shape_type& coord)
                    {
                        size_t pixel = (DIM == 2) ? coord[0] + coord[1]*in.shape(0) : coord[0] + coord[1] * in.shape(0) + coord[2] * (in.shape(0) * in.shape(1));
                        in_flatten(pixel, coord[DIM]) = in(coord.asStdArray());
                    });
                    
                    for(int d = 0; d < in_flatten.dimension(); ++d)
                        std::cout << in_flatten.shape(d) << std::endl;
                    
                    marray::Marray<data_type> prediction(prediction_shape.begin(), prediction_shape.end());

                    // loop over all random forests for prediction probabilities
                    random_forest_vector_[0].predict_probs(in_flatten, prediction);
                    prediction /= random_forest_vector_[0].num_trees();

                    //auto miMa = std::minmax_element(prediction.begin(), prediction.end());
                    //std::cout << "MinMax predition " << *miMa.first << " " << *miMa.second << std::endl;

                    // transform back to marray
                    out_shape_type output_shape;
                    for(size_t d = 0; d < DIM; ++d) {
                        output_shape[d] = out_array_.shape(d);
                    }
                    
                    output_shape[DIM] = num_pixel_classification_labels;
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
                feature_cache& feature_cache_;
                float_array_view& out_array_;
                random_forest_vector& random_forest_vector_;
            };
        
        } // namespace ilastik_backend
    } // namespace pipelines
} // namespace nifty

#endif // _RANDOM_FOREST_PREDICTION_TASK_H_
