#ifndef _RANDOM_FOREST_PREDICTION_TASK_H_
#define _RANDOM_FOREST_PREDICTION_TASK_H_

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <tbb/concurrent_lru_cache.h>

#include <algorithm>

#include <tbb/tbb.h>
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
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
                    std::cout << "rf_task::execute()" << std::endl;
                    // ask for features. This blocks if it's not present
                    feature_cache::handle ho = feature_cache_[blockId_];
                    float_array_view& features = ho.value();
                    compute(features);
                    return NULL;
                }

                void compute(const float_array_view & in)
                {
                    std::cout << "rf_task::compute for block " << blockId_ << std::endl;
                    // TODO: transform data to vigra array?!
                    size_t num_pixel_classification_labels = random_forest_vector_[0].class_count();
                    size_t num_required_features = random_forest_vector_[0].feature_count();
                    
                    assert(num_required_features == in.shape(0));

                    std::cout << "Filters" << std::endl;
                    std::cout << in(0) << std::endl;

                    size_t pixel_count = 1;
                    out_shape_type in_shape;
                    for(int d = 0; d < DIM+1; ++d) {
                        in_shape[d] = in.shape(d);
                        if(d > 0)
                            pixel_count *= in.shape(d);
                    }
                    std::cout << in_shape << std::endl;
                    std::cout << pixel_count << std::endl;

                    feature_shape_type feature_shape({pixel_count, num_required_features});
                    
                    //vigra::MultiArray<2, data_type> vigra_in(vigra::Shape2(pixel_count, num_required_features));
                    vigra::MultiArrayView<2, data_type> vigra_in(vigra::Shape2(pixel_count, num_required_features), &in(0));
                    
                    std::cout << "rf_task::compute vigra copies " << std::endl;
                    // copy data from marray to vigra. TODO: is the axes order correct??
                    //tools::forEachCoordinate(in_shape, [&in, &vigra_in, &in_shape, pixel_count](const out_shape_type& coord)
                    //{
                    //    size_t pixel = (DIM == 2) ? coord[1] * (in_shape[2] + in_shape[3]) + coord[2] * (in_shape[3]) :
                    //                   coord[1] * (in_shape[2] + in_shape[3] + in_shape[4]) + coord[2] * (in_shape[3] + in_shape[4]) + coord[3] * in_shape[4];
                    //    if(pixel >= pixel_count) {
                    //        std::cout << "WAAAAHAAHHAA" << std::endl;
                    //        std::cout << pixel << " " << pixel_count << std::endl;
                    //        std::cout << coord << std::endl;
                    //    }
                    //    if(coord[0] >= 2)
                    //        std::cout << "Whoooooosa " << coord[0] << std::endl;
                    //    
                    //    //std::cout << coord << std::endl;
                    //    //std::cout << pixel << std::endl;
                    //    //std::cout << vigra_in(pixel, coord[0]) << std::endl;
                    //    //std::cout << "Accessed vigra array " << std::endl;
                    //    //std::cout << in(coord.asStdArray()) << std::endl;
                    //    //std::cout << "Accessed in array " << std::endl;
                    //    vigra_in(pixel, coord[0]) = in(coord.asStdArray());
                    //});
                    //std::cout << "rf_task::compute vigra copies done 1" << std::endl;
                    
                    vigra::MultiArray<2, data_type> prediction_map_view(vigra::Shape2(pixel_count, num_pixel_classification_labels));

                    // loop over all random forests for prediction probabilities
                    std::cout << "rf_task::compute predicting" << std::endl;
                    for(size_t rf = 0; rf < random_forest_vector_.size(); ++rf)
                    {
                        std::cout << "AAA" << std::endl;
                        vigra::MultiArray<2, data_type> prediction_temp(pixel_count, num_pixel_classification_labels);
                        std::cout << "BBB" << std::endl;
                        random_forest_vector_[rf].predictProbabilities(vigra_in, prediction_temp);
                        std::cout << "CCC" << std::endl;
                        prediction_map_view += prediction_temp;
                        std::cout << "Prediction done for rf: " << rf << std::endl;
                        auto pred_mm = std::minmax_element(prediction_map_view.begin(), prediction_map_view.end()); 
                        std::cout << "RF-min prediction: " << *(pred_mm.first) << " RF-max prediction: " << *(pred_mm.second) << std::endl;
                    }
                    std::cout << "rf_task::compute prediction done" << std::endl;

                    auto pred_mm = std::minmax_element(prediction_map_view.begin(), prediction_map_view.end()); 
                    std::cout << "RF-min prediction: " << *(pred_mm.first) << " RF-max prediction: " << *(pred_mm.second) << std::endl;
                    // divide probs by num random forests
                    prediction_map_view /= random_forest_vector_.size();
                    pred_mm = std::minmax_element(prediction_map_view.begin(), prediction_map_view.end()); 
                    std::cout << "RF-min prediction: " << *(pred_mm.first) << " RF-max prediction: " << *(pred_mm.second) << std::endl;

                    // transform back to marray
                    out_shape_type output_shape;
                    for(size_t d = 0; d < DIM; ++d) {
                        output_shape[d] = out_array_.shape(d);
                    }
                    
                    output_shape[DIM] = num_pixel_classification_labels;
                    float_array_view& tmp_out_array = out_array_;
                    
                    std::cout << "rf_task::compute black copy magic" << std::endl;
                    tools::forEachCoordinate(output_shape, [&tmp_out_array, &prediction_map_view, output_shape](const out_shape_type& coord)
                    {
                        size_t pixelRow = coord[0] * (output_shape[1] + output_shape[2]) + coord[1] * (output_shape[2]);
                        if(DIM == 3)
                        {
                            pixelRow = coord[0] * (output_shape[1] + output_shape[2] + output_shape[3]) + coord[1] * (output_shape[2] + output_shape[3]) + coord[2] * output_shape[3];
                        }
                        tmp_out_array(coord.asStdArray()) = prediction_map_view(pixelRow, coord[DIM]);
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
