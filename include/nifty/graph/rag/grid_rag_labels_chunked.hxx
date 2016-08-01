#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_LABELS_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_LABELS_CHUNKED_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/runtime_check.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"
//
#include "nifty/hdf5/hdf5_array.hxx"

namespace nifty{
namespace graph{


// TODO Chunked labels and chunked labels sliced
template<size_t DIM, class LABEL_TYPE>
class ChunkedLabels{
public:

    typedef nifty::hdf5::Hdf5Array<LABEL_TYPE> ViewType;

    // enable setting the chunk size by overloading the constuctor
    ChunkedLabels(const ViewType & labels )
    : labels_( labels ), shape_() {
        for(size_t i=0; i<DIM; ++i)
            shape_[i] = labels_.shape(i);
    }

    
    // part of the API
    // TODO only calculate this once -> move to constructor and introduce new variable
    uint64_t numberOfLabels() const {
    
        size_t sliceShape[] = {size_t(shape_[0]), size_t(shape_[1]), 1};
        marray::Marray<LABEL_TYPE> currentSlice(sliceShape, sliceShape + 3);
        
        LABEL_TYPE maxLabel = 0;
        
        // TODO we can do all kind of checks here (being sliced, being consecutive, etc.)
        // TODO iterate over the slices in parellel !
        for(size_t z = 0; z < shape_[2]; z++) {
            size_t sliceStart[] = {0, 0, z};
            labels_.readSubarray(sliceStart, currentSlice);
            LABEL_TYPE maxSlice = *std::max_element(currentSlice.begin(), currentSlice.end());
            maxLabel = std::max(maxLabel, maxSlice ); 
        }
        
        return uint64_t(maxLabel) + 1;
    }

    // not part of the general API
    const ViewType & labels() const{
        return labels_;
    }

    const std::array<int64_t, DIM> & shape() const{
        return  shape_;
    }


private:
    std::array<int64_t, DIM> shape_;
    std::string label_file_;
    std::string label_key_;
    ViewType labels_;

};



} // namespace graph
} // namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_LABELS_CHUNKED_HXX */
