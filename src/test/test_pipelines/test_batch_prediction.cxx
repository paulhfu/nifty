#define BOOST_TEST_MODULE pipelines_ilastik_batch_classification

#include <tbb/tbb.h>
#include <boost/test/unit_test.hpp>

#include "nifty/pipelines/ilastik_backend/batch_prediction_task.hxx"

BOOST_AUTO_TEST_CASE(BatchPredictionTest)
{   
    using namespace nifty::pipelines::ilastik_backend;
    
    constexpr size_t dim  = 3;

    using data_type = float;
    using in_data_type = uint8_t;
    using float_array = nifty::marray::Marray<data_type>;
    using float_array_view = nifty::marray::View<data_type>;
    using prediction_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    using raw_cache = Hdf5Input<in_data_type, dim, false, in_data_type>;
    using coordinate = nifty::array::StaticArray<int64_t, dim>;

    bool use_small_data = false;

    std::string raw_file = "/home/consti/Work/data_neuro/ilastik_hackathon/data_200_8bit_squeezed.h5";
    std::string raw_key  = "data";
    
    std::string rf_filename = "/home/consti/Work/data_neuro/ilastik_hackathon/hackathon_flyem_forest.h5";
    std::string rf_path     = "Forest0000";
    
    coordinate roiBegin({2500,2600,0});
    coordinate roiEnd({2800,2900,200});
    coordinate blockShape({64,64,64});
    
    auto selected_features = std::make_pair(
        std::vector<std::pair<std::string, std::vector<bool>>>(
            {std::make_pair<std::string,std::vector<bool>>("GaussianSmoothing",         {true, true, true,  true, true, true}),
             std::make_pair<std::string,std::vector<bool>>("LaplacianOfGaussian",       {false,true, false, true, false,true}),
             std::make_pair<std::string,std::vector<bool>>("GaussianGradientMagnitude", {false,false,true, false, true, false}),
             std::make_pair<std::string,std::vector<bool>>("HessianOfGaussianEigenvalues",{false,false,true, false, true, false})}
                                                                                                                            ),
                                                                    std::vector<double>({0.3,  1.0,  1.6,   3.5,  5.0,  10.0})
    );

    if(use_small_data) {
        rf_filename = "./testPC.ilp";
        rf_path = "/PixelClassification/ClassifierForests/Forest0000";
        
        raw_file = "./testraw.h5";
        raw_key  = "exported_data";

        roiBegin = coordinate({0,0,0});
        roiEnd   = coordinate({128,128,128});
        blockShape = coordinate({64, 64, 64});
    
        selected_features = std::make_pair(
        std::vector<std::pair<std::string, std::vector<bool>>>({std::make_pair<std::string,std::vector<bool>>("GaussianSmoothing", {true, true})}),
        std::vector<double>({2.,3.5}));
    }
    
    std::string out_filename = "./out.h5";
    std::string out_key = "data";

    batch_prediction_task<dim>& batch = *new(tbb::task::allocate_root()) batch_prediction_task<dim>(
            raw_file, raw_key,
            rf_filename, rf_path,
            out_filename, out_key,
            selected_features,
            blockShape,
            roiBegin, roiEnd);
    std::cout << "Spawning Main Task" << std::endl;
    tbb::task::spawn_root_and_wait(batch);
    std::cout << "Main Task done" << std::endl;
}

