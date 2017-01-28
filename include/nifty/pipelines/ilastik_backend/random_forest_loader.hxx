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
