#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURE_FUNCTOR_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURE_FUNCTOR_HXX

#include "nifty/marray/marray.hxx"
#include "fastfilters/fastfilters.h"

namespace nifty{
namespace graph{

namespace detail_functor {
    
    template <typename fastfilters_array_t>
    void convert_marray2ff(marray::View<float> & array, fastfilters_array_t & ff) {
        
        const unsigned int ff_ndim = ff_ndim_t<fastfilters_array_t>::ndim;
        auto shape = array.shape();
    
        if (array.dimensin() >= (int)ff_ndim) {
            // TODO need to get pointer to the data
            //ff.ptr = (float *) np_info.ptr;
            
            // TODO need to get strides
    
            ff.n_x = shape[ff_ndim - 1];
            //ff.stride_x = np_info.strides[ff_ndim - 1] / sizeof(float);
    
            ff.n_y = np_info.shape[ff_ndim - 2];
            //ff.stride_y = np_info.strides[ff_ndim - 2] / sizeof(float);
    
            if (ff_ndim == 3) {
                ff_ndim_t<fastfilters_array_t>::set_z(shape[ff_ndim - 3], ff);
                //ff_ndim_t<fastfilters_array_t>::set_stride_z(np_info.strides[ff_ndim - 3] / sizeof(float), ff);
            }
        } else {
            throw std::logic_error("Too few dimensions.");
        }
    
        if (array.dimension() == ff_ndim) {
            ff.n_channels = 1;
        } else if ((array.dimension() == ff_ndim + 1) && shape[ff_ndim] < 8 && np_info.strides[ff_ndim] == sizeof(float)) {
            ff.n_channels = shape[ff_ndim];
        } else {
            throw std::logic_error("Invalid number of dimensions or too many channels or stride between channels.");
        }
    }

    template <typename fastfilters_array_t>
    void convert_ff2marray(fastfilters_array_t & ff, marray::View<float> & array) {

    }

} //namespace detail_functor

    // wrap fastfilters in a functor
    template<class T, unsigned DIM>
    struct FilterFunctor {
        
        typedef typename std::conditional<ndim == 2, fastfilters_fir_laplacian2d, fastfilters_fir_laplacian3d>::type ff_laplacian;
        typedef typename std::conditional<ndim == 2, fastfilters_fir_convolve2d, fastfilters_fir_convolve3d>::type ff_gaussian;

        // TODO make different filters from fastfilters available either through constructor or templates
        // TODO make sigmas available
        FilterFunctor()
        {}
        
        void operator(const marray::View<T> & in, marray::View<T> & out) {
    
            typedef typename std::conditional<ndim == 2, fastfilters_array2d_t, fastfilters_array3d_t>::type ff_array_t;
            ff_array_t ff;
            ff_array_t ff_out;

            // TODO different filter + sigma
            ff_gaussian(&ff, sigma, &ff_out, opts);

            convert_ff2marray(ff_out, out);


        }

        //TODO get number of channels 
        

    };
 

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HDF5_HXX */
