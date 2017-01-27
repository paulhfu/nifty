#define BOOST_TEST_MODULE pipelines_ilastik_pixel_classification

#include <tbb/tbb.h>
#include <boost/test/unit_test.hpp>

#include "nifty/pipelines/ilastik_backend/batch_prediction_task.hxx"

BOOST_AUTO_TEST_CASE(PixelClassificationPredictionTest)
{   
    using namespace nifty::pipelines::ilastik_backend;
    
    constexpr size_t dim  = 3;

    using data_type = float;
    using in_data_type = uint8_t;
    using float_array = nifty::marray::Marray<data_type>;
    using float_array_view = nifty::marray::View<data_type>;
    using prediction_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    constexpr size_t max_num_entries = 100;
    using raw_cache = Hdf5Input<in_data_type, dim, false, in_data_type>;
    using coordinate = nifty::array::StaticArray<int64_t, dim>;

    // load random forests
    const std::string rf_filename = "./testPC.ilp";
    const std::string rf_path = "/PixelClassification/ClassifierForests/Forest0000";
    
    //const std::string rf_filename = "/home/consti/Work/data_neuro/ilastik_hackathon/hackathon_flyem_forest.h5";
    //const std::string rf_path     = "Forest0000";

    std::string raw_file = "./testraw.h5";
    //std::string raw_file = "/home/consti/Work/data_neuro/ilastik_hackathon/data_200_8bit.h5";
    
    coordinate roiBegin({0,0,0});
    coordinate roiEnd({128,128,128});
    
    //coordinate roiBegin({2500,2600,0});
    //coordinate roiEnd({2800,2900,200});
    
    coordinate block_shape({64,64,64});
    //coordinate block_shape({128,128,128});

    auto selected_features = std::make_pair(
        std::vector<std::pair<std::string, std::vector<bool>>>({std::make_pair<std::string, std::vector<bool> >("GaussianSmoothing", {true, true})}),
        std::vector<double>({2.,3.5})
    );

    batch_prediction_task<dim>& batch = *new(tbb::task::allocate_root()) batch_prediction_task<dim>(
            raw_file, "exported_data",
            rf_filename, rf_path,
            selected_features,
            block_shape, max_num_entries,
            roiBegin, roiEnd);
    std::cout << "Spawning Main Task" << std::endl;
    tbb::task::spawn_root_and_wait(batch);
    std::cout << "Main Task done" << std::endl;
}

