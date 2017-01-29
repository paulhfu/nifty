#pragma once

#include <nifty/marray/marray.hxx>

// includes for vigra::rf3
#include <vigra/random_forest_3.hxx>
#include <vigra/random_forest_3_hdf5_impex.hxx>

// includes for vigra::rf2
#include <vigra/random_forest_hdf5_impex.hxx>
#include <hdf5_hl.h>

namespace nifty
{
namespace pipelines
{
namespace ilastik_backend
{
    using LabelType = size_t;
    using FeatureType = float;
    using Labels = nifty::marray::View<LabelType>; 
    using Features = nifty::marray::View<FeatureType>; 
    using RandomForest2Type = vigra::RandomForest<LabelType>;
    using RandomForest3Type = vigra::rf3::DefaultRF<Features,Labels>::type;

    RandomForest2Type get_rf2_from_file(
        const std::string& fn,
        const std::string& path_in_file)
    {
        hid_t h5file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert(h5file > 0);
        RandomForest2Type rf;
        vigra::rf_import_HDF5(rf, fn, path_in_file);
        H5Fclose(h5file);
        return rf;
    }
    
    std::vector<RandomForest2Type> get_multiple_rf2_from_file(
        const std::string& fn,
        const std::vector<std::string> & paths_in_file)
    {
        hid_t h5file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert(h5file > 0);
        std::vector<RandomForest2Type> rf_vector( paths_in_file.size(), RandomForest2Type() );
        for(int i = 0; i < paths_in_file.size(); ++i) {
            const auto & key = paths_in_file[i];
            vigra::rf_import_HDF5(rf_vector[i], fn, key);
        }
        H5Fclose(h5file);
        return rf_vector;
    }
    
    RandomForest3Type get_rf3_from_file(
        const std::string& fn,
        const std::string& path_in_file)
    {
        auto h5file = vigra::HDF5File(fn);
        return vigra::rf3::random_forest_import_HDF5<Features,Labels>(h5file, path_in_file);
    }
}
}
}
