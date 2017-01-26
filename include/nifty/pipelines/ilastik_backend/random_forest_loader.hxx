#ifndef _RANDOM_FOREST_LOADER_HXX_
#define _RANDOM_FOREST_LOADER_HXX_

#pragma once

#include <nifty/marray/marray.hxx>

#include <vigra/random_forest_3.hxx>
#include <vigra/random_forest_3_hdf5_impex.hxx>
//#include <hdf5_hl.h>

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
    using RandomForestType = vigra::rf3::DefaultRF<Features,Labels>::type;

    std::string zero_padding(int num, int n_zeros)
    {
        std::ostringstream ss;
        ss << std::setw(n_zeros) << std::setfill('0') << num;
        return ss.str();
    }

    /** 
     * @brief Read the random forests from the hdf5 files.
     * 
     * WARNING: this shows some warnings on the command line because we try to read one more
     *          tree than is available. But that seems to be the easiest option to get all RFs in the group.
     * 
     */
    /*
    bool get_rf_from_multi_file(
        RandomForestType& rf,
        const std::string& fn,
        const std::string& path_in_file,
        int n_leading_zeros)
    {
        
        bool read_successful = true;
        int n_forests = -1;
        auto h5file = vigra::HDF5File(fn);
        //hid_t h5file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        //assert(h5file > 0);

        do
        {
            n_forests++;
            std::string rf_path = path_in_file + zero_padding(n_forests, n_leading_zeros);
            hid_t rf_group = H5Gopen1(h5file, rf_path.c_str());
            
            if(rf_group > 0) 
            {
                auto rf_tmp = vigra::rf3::random_forest_import_HDF5<Features,Labels>(fn, rf_path);
                H5Gclose(rf_group);
                rf.merge(rf_tmp);
            }
            else
            {
                read_successful = false;
            }
        } while(read_successful);

        H5Fclose(h5file);
        std::cout << "Read " << n_forests << " Random Forests" << std::endl;

        return n_forests > 0;
    }
    */
    
    RandomForestType get_rf_from_file(
        const std::string& fn,
        const std::string& path_in_file)
    {
        auto h5file = vigra::HDF5File(fn);
        return vigra::rf3::random_forest_import_HDF5<Features,Labels>(h5file, path_in_file);
    }
}
}
}

#endif // _RANDOM_FOREST_LOADER_HXX_
