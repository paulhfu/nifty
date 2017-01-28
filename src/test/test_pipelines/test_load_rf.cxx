#define BOOST_TEST_MODULE pipelines_load_rf

#include <boost/test/unit_test.hpp>

#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

BOOST_AUTO_TEST_CASE(RandomForestLoadingTest)
{
    using namespace nifty::pipelines::ilastik_backend;

    using data_type = float;
    
    using coordinate = nifty::array::StaticArray<int64_t, 4>;
    using coordinate_flatten = nifty::array::StaticArray<int64_t,2>;

    const std::string rf_file = "./testPC.ilp";
    const std::string rf_key = "/PixelClassification/ClassifierForests/Forest0000";

    const std::string feature_key  = "exported_data";

    // load the features
    auto ff = nifty::hdf5::openFile("./denk_raw_Features.h5");
    nifty::hdf5::Hdf5Array<float> features(ff, feature_key);

    std::vector<size_t> shape;
    std::vector<size_t> start;
    
    for(int d = 0; d < features.dimension(); ++d) {
        shape.push_back(features.shape(d));
        start.push_back(0);
    }

    nifty::marray::Marray<data_type> feature_array(shape.begin(), shape.end());
    features.readSubarray(start.begin(), feature_array);

    auto in = feature_array.squeezedView();
    
    coordinate shapeSq;
    coordinate chunks;
    for(int d = 0; d < in.dimension(); ++d) {
        shapeSq[d] = in.shape(d);
        chunks[d]  = std::min(size_t(64), in.shape(d));
    }
    
    // load the rf
    auto rf = get_rf3_from_file(rf_file, rf_key);
                    
    size_t num_labels   = rf.num_classes();
    size_t num_features = rf.num_features();

    size_t pixel_count = 1;
    for(int d = 0; d < 3; ++d) {
        pixel_count *= in.shape(d);
    }
            
    coordinate_flatten feature_shape(   {pixel_count, num_labels});
    coordinate_flatten prediction_shape({pixel_count, num_features});
            
    // FIXME TODO copy for now, but we should be able to do this w/o
    //std::cout << "Flattened shape:" << std::endl;
    //auto in_flatten    = in.reshapedView(feature_shape.begin(), feature_shape.end());
    nifty::marray::Marray<data_type> in_flatten(feature_shape.begin(), feature_shape.end());
    nifty::tools::forEachCoordinate(shapeSq, [&in, &in_flatten](const coordinate & coord)
    {
        size_t pixel = coord[0] + coord[1] * in.shape(0) + coord[2] * (in.shape(0) * in.shape(1));
        in_flatten(pixel, coord[3]) = in(coord.asStdArray());
    });
    
    for(int d = 0; d < in_flatten.dimension(); ++d)
        std::cout << in_flatten.shape(d) << std::endl;
    
    nifty::marray::Marray<data_type> prediction(prediction_shape.begin(), prediction_shape.end());

    // loop over all random forests for prediction probabilities
    rf.predict_probabilities(in_flatten, prediction);

    coordinate out_shape;
    coordinate out_start;
    
    for(int d = 0; d < 4; d++){ 
        out_shape[d] = shapeSq[d];
        out_start[d] = 0;
    }
    nifty::marray::Marray<data_type> out_array(out_shape.begin(), out_shape.end());
                    
    nifty::tools::forEachCoordinate(out_shape, [&prediction, &out_array](const coordinate& coord)
    {
        size_t pixel = coord[0] + coord[1] * out_array.shape(0) + coord[2] * (out_array.shape(0) * out_array.shape(1));
        out_array(coord.asStdArray()) = prediction(pixel, coord[3]);
    });

    auto out_file = nifty::hdf5::createFile("./out_rf.h5");
    nifty::hdf5::Hdf5Array<float> out(out_file, "data", shapeSq.begin(), shapeSq.end(), chunks.begin() );

    out.writeSubarray(out_start.begin(), out_array);
}
