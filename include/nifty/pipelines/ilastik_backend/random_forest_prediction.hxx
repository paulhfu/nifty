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
    
using random_forest2 = nifty::pipelines::ilastik_backend::RandomForest2Type;
using random_forest3 = nifty::pipelines::ilastik_backend::RandomForest3Type;

template<unsigned DIM, typename DATA_TYPE>
void random_forest3_prediction(
        const size_t blockId,
        const nifty::marray::View<DATA_TYPE> & in,
        nifty::marray::View<DATA_TYPE> & out,
        const random_forest3 & rf,
        const int n_threads = 1) {
    
    std::cout << "random_forest3_prediction for block " << blockId << std::endl;
    
    // typedefs
    using data_type = DATA_TYPE;
    using multichan_coordinate = array::StaticArray<int64_t,DIM+1>;
    using flattened_coordinate = array::StaticArray<int64_t,2>;
    
    size_t num_classes  = rf.num_classes();
    size_t num_features = rf.num_features();
    assert(num_features == in.shape(0));

    size_t pixel_count = 1;
    multichan_coordinate in_shape;
    for(int d = 0; d < DIM+1; ++d) {
        in_shape[d] = in.shape(d);
        if(d > 0)
            pixel_count *= in.shape(d);
    }

    flattened_coordinate feature_shape({pixel_count, num_features});
    flattened_coordinate prediction_shape({pixel_count, num_classes});
            
    // FIXME TODO we should be able to do this w/o a copy, but somehow this is broken...
    //if(! in.isSimple() )
    //    throw std::runtime_error("In view is not simple");
    //auto in_flatten    = in.reshapedView(feature_shape.begin(), feature_shape.end());
    
    marray::Marray<data_type> in_flatten(feature_shape.begin(), feature_shape.end());
    tools::forEachCoordinate(in_shape, [&in, &in_flatten](const multichan_coordinate& coord)
    {
        size_t pixel = (DIM == 2) ? coord[1] + coord[2] * in.shape(1) 
                                  : coord[1] + coord[2] * in.shape(1) + coord[3] * in.shape(1) * in.shape(2);
        in_flatten(pixel, coord[0]) = in(coord.asStdArray());
    });
            
    marray::Marray<data_type> prediction(prediction_shape.begin(), prediction_shape.end());
    
    // FIXME TODO we should be able to do this w/o a copy, but somehow this is broken...
    //if(! out.isSimple() )
    //    throw std::runtime_error("Out view is not simple");
    //auto prediction = out.reshapedView(prediction_shape.begin(), prediction_shape.end());
    
    rf.predict_probabilities(in_flatten, prediction, n_threads);

    // transform back to marray
    multichan_coordinate output_shape;
    for(size_t d = 0; d < DIM; ++d) {
        output_shape[d] = out.shape(d);
    }
    output_shape[DIM] = num_classes;

    tools::forEachCoordinate(output_shape, [&prediction, &out](const multichan_coordinate& coord)
    {
        size_t pixel = (DIM == 2) ? coord[0] + coord[1] * out.shape(0) 
                                  : coord[0] + coord[1] * out.shape(0) + coord[2] * out.shape(0) * out.shape(1);
        out(coord.asStdArray()) = prediction(pixel, coord[DIM]);
    });
}


template<unsigned DIM, typename DATA_TYPE>
void random_forest2_prediction(
        const size_t blockId,
        const nifty::marray::View<DATA_TYPE> & in,
        nifty::marray::View<DATA_TYPE> & out,
        const random_forest2 & rf,
        const int n_threads = 0) {
    
    std::cout << "random_forest2_prediction for block " << blockId << std::endl;
    
    // typedefs
    using data_type = DATA_TYPE;
    using multichan_coordinate = array::StaticArray<int64_t,DIM+1>;
    
    size_t num_classes  = rf.class_count();
    size_t num_features = rf.feature_count();
    assert(num_features == in.shape(0));

    size_t pixel_count = 1;
    multichan_coordinate in_shape;
    for(int d = 0; d < DIM+1; ++d) {
        in_shape[d] = in.shape(d);
        if(d > 0)
            pixel_count *= in.shape(d);
    }
    
    // FIXME TODO we should be able to do this w/o a copy, but somehow this is broken...
    //vigra::MultiArrayView<2, data_type> in_flatten( vigra::Shape2(pixel_count, num_features), &in(0) );
    
    vigra::MultiArray<2,data_type> in_flatten( vigra::Shape2(pixel_count, num_features) );
    tools::forEachCoordinate(in_shape, [&in, &in_flatten](const multichan_coordinate& coord)
    {
        size_t pixel = (DIM == 2) ? coord[1] + coord[2] * in.shape(1) 
                                  : coord[1] + coord[2] * in.shape(1) + coord[3] * in.shape(1) * in.shape(2);
        in_flatten(pixel, coord[0]) = in(coord.asStdArray());
    });
    
    // FIXME TODO we should be able to do this w/o a copy, but somehow this is broken...
    //vigra::MultiArrayView<2,data_type> prediction( vigra::Shape2(pixel_count, num_classes), &out(0) );
    
    vigra::MultiArray<2,data_type> prediction( vigra::Shape2(pixel_count, num_classes) );
    rf.predictProbabilities(in_flatten, prediction);
    
    // transform back to marray
    multichan_coordinate output_shape;
    for(size_t d = 0; d < DIM; ++d) {
        output_shape[d] = out.shape(d);
    }
    output_shape[DIM] = num_classes;

    tools::forEachCoordinate(output_shape, [&prediction, &out](const multichan_coordinate& coord)
    {
        size_t pixel = (DIM == 2) ? coord[0] + coord[1] * out.shape(0) 
                                  : coord[0] + coord[1] * out.shape(0) + coord[2] * out.shape(0) * out.shape(1);
        out(coord.asStdArray()) = prediction(pixel, coord[DIM]);
    });
}

} // namespace ilastik_backend
} // namespace pipelines
} // namespace nifty

#endif // _RANDOM_FOREST_PREDICTION_TASK_H_
