#define BOOST_TEST_MODULE pipelines_load_rf

#include <boost/test/unit_test.hpp>

#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include "nifty/hdf5/hdf5_array.hxx"

BOOST_AUTO_TEST_CASE(RandomForestLoadingTest)
{
    using namespace nifty::pipelines::ilastik_backend;

    const std::string rf_file = "./testPC.ilp";
    const std::string rf_key = "/PixelClassification/ClassifierForests/Forest";

    const std::string feature_file = "";
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

    nifty::marray::Marray<float> feature_array(shape.begin(), shape.end());
    features.readSubarray(start.begin(), feature_array);

    auto features_squeezed = feature_array.squeezedView();
    
    std::vector<size_t> shapeSq;
    std::vector<size_t> chunks;
    for(int d = 0; d < features_squeezed.dimension(); ++d) {
        shapeSq.push_back(features_squeezed.shape(d));
        chunks.push_back( std::min( size_t(64), features_squeezed.shape(d)) );
    }
    
    // load the rf
    RandomForestVectorType rf_vec;
    get_rfs_from_file(rf_vec, rf_file, rf_key, 4);
                    
    size_t num_labels   = rf_vec[0].class_count();
    size_t num_features = rf_vec[0].feature_count();

    // transform to vigra features and predict the rf
    
    size_t pixel_count = 1;
    for(int d = 0; d < 3; ++d) {
        pixel_count *= features_squeezed.shape(d);
    }

    vigra::MultiArrayView<2,float> vigra_in(vigra::Shape2(pixel_count, num_features), &features_squeezed(0));
    vigra::MultiArray<2, float> prediction_map_view(vigra::Shape2(pixel_count, num_labels));

    // loop over all random forests for prediction probabilities
    std::cout << "rf_task::compute predicting" << std::endl;
    for(size_t rf = 0; rf < rf_vec.size(); ++rf)
    {
        vigra::MultiArray<2, float> prediction_temp(pixel_count, num_labels);
        rf_vec[rf].predictProbabilities(vigra_in, prediction_temp);
        prediction_map_view += prediction_temp;
        std::cout << "Prediction done for rf: " << rf << std::endl;
    }
    std::cout << "rf_task::compute prediction done" << std::endl;
                    
    // TODO check the output
    //tools::forEachCoordinate(output_shape, [&tmp_out_array, &prediction_map_view, output_shape](const out_shape_type& coord)
    //{
    //    size_t pixelRow = coord[0] * (output_shape[1] + output_shape[2]) + coord[1] * (output_shape[2]);
    //    if(DIM == 3)
    //    {
    //        pixelRow = coord[0] * (output_shape[1] + output_shape[2] + output_shape[3]) + coord[1] * (output_shape[2] + output_shape[3]) + coord[2] * output_shape[3];
    //    }
    //    tmp_out_array(coord.asStdArray()) = prediction_map_view(pixelRow, coord[DIM]);
    //});

    //nifty::hdf5::createFile("./out_rf.h5")
    //nifty::hdf5::Hdf5Array<float> prediction( shapeSq.begin(), shapeSq.end(), chunks );

    
}
