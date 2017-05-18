#pragma once

#include "nifty/graph/rag/grid_rag.hxx"

namespace nifty{
namespace graph{

    template<unsigned DIM, class LABELS_PROXY>
    void edgesFromPaths( 
        const GridRag<DIM, LABELS_PROXY> & rag,
        const std::vector<array::StaticArray<int64_t, DIM>> & coordinates,
        std::vector<int64_t> & edgeIds
        ) {
        
        // TODO iterate over coordinates, find labels and append edge to edgeIds if we find a label transition
        // TODO for explicit and hdf5 labels (2 different functions if necessary)

    }

}
}
