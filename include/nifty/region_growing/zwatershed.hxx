// Edge based watershed and zwatershed:
// Algorithms defined in:
// Zlateski et al., Image Segmentation by Size-Dependent Single Linkage Clustering of a Watershed Basin Graph
// https://arxiv.org/abs/1505.00249
// Implementation inspired by:
// https://github.com/TuragaLab/zwatershed

#pragma once

#include "nifty/ufd/ufd.hxx"
#include "nifty/region_growing/edge_based_watershed.hxx"

namespace nifty {
namespace region_growing {

template< unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
void
getRegionGraph(
    const marray::View<DATA_TYPE> & affinityMap,
    const marray::View<LABEL_TYPE> & segmentation,
    const size_t maxId,
    std::vector<std::tuple<DATA_TYPE,LABEL_TYPE,LABEL_TYPE>> & out) {

    typedef DATA_TYPE DataType;
    typedef LABEL_TYPE LabelType;
    
    typedef array::StaticArray<int64_t, DIM> Coord;
    typedef array::StaticArray<int64_t, DIM+1> CoordAndChannel;
    
    Coord shape;
    for(int d = 0; d < DIM; d++)
        shape[d] = segmentation.shape(d);
    
    auto decrementCoord = [](const Coord & coord, const size_t axis){
        Coord coord2 = coord;
        coord2[axis] -= 1;
        return coord2;
    };
    
    // initialize the edges and region graph
    std::vector<std::map<LabelType,DataType>> edges(maxId+1);

    // iterate over all coordinates and get he edges
    nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
        
        CoordAndChannel affinityCoord;
        for(size_t d = 0; d < DIM; ++d)
            affinityCoord[d] = coord[d];
        
        for(size_t axis = 0; axis < DIM; ++axis) {
            Coord coord2 = decrementCoord(coord, axis);
            if( (coord[axis]>0) && segmentation(coord.asStdArray()) && segmentation(coord2.asStdArray()) ) {
                auto minmax = std::minmax( segmentation(coord.asStdArray()), segmentation(coord2.asStdArray()) );
                DataType & maxEdgeValue = edges[minmax.first][minmax.second];
                affinityCoord[DIM+1] = axis;
                maxEdgeValue = std::max( maxEdgeValue, affinityMap(affinityCoord.asStdArray()) );
            }
        }
    });

    for(LabelType id = 1; id <= maxId; ++id) {
        for(const auto & edge : edges[id])
            out.emplace_back(edge.second, id, edge.first);
    }

    std::stable_sort(out.begin(), out.end(),
        std::greater<std::tuple<DataType,LabelType,LabelType>>());
}

template<class DATA_TYPE, class LABEL_TYPE>
void
mergeSegments(
        marray::View<LABEL_TYPE> & segmentation,
        const std::vector<std::tuple<DATA_TYPE,LABEL_TYPE,LABEL_TYPE>> & regionGraph,
        std::vector<size_t> & counts,
        const DATA_TYPE mergeThreshold,
        const DATA_TYPE lowThreshold,
        const size_t sizeThreshold) {
    
    typedef DATA_TYPE DataType;
    typedef LABEL_TYPE LabelType;
    
    // TODO I don't really get what this score is...
    auto computeMergeScore = [&]( DataType val ){
        return (val < lowThreshold ) ? 0 : val * val * mergeThreshold;
    };

    ufd::Ufd<LabelType> sets(counts.size());

    for(auto & edge : regionGraph) {
        // we compare to size_t here, because we compare to the counts later
        size_t mergeScore = computeMergeScore( std::get<0>(edge) );
        if(mergeScore==0)
            break;

        LabelType s1 = sets.find(std::get<1>(edge));
        LabelType s2 = sets.find(std::get<2>(edge));

        if( s1 != s2 && s1 && s2) {
            if( (counts[s1] < mergeScore) || (counts[s2] < mergeScore) ) {
                counts[s1] += counts[s2];
                counts[s2] = 0;
                sets.merge(s1,s2);
                LabelType s = sets.find(s1); 
                std::swap(counts[s],counts[s1]);
            }
        }
    }

    std::vector<LabelType> mapping(counts.size());
    LabelType next = 1;

    for( LabelType id = 0; id < counts.size(); ++id) {
        LabelType s = sets.find(id);
        if( s && (mapping[s]==0) && (counts[s] >= sizeThreshold) ) {
            mapping[s] == next;
            counts[next] = counts[s];
            ++next;
        }
    }
    counts.resize(next);

    for( auto segIt = segmentation.begin(); segIt != segmentation.end(); ++segIt)
        *segIt = mapping[sets.find(*segIt)];
    
    // in the original code the regiongraph is reconstructed, but I don't see why we need this
}

template< unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
void
zWatershed(
    const marray::View<DATA_TYPE>  & affinityMap,
    const DATA_TYPE mergeThreshold,
    const size_t    sizeThreshold,
    const DATA_TYPE lowThreshold,
    const DATA_TYPE highThreshold,
    marray::View<LABEL_TYPE> & out ) {
    
    // run the initial watershed
    auto counts = edgeBasedWatershed<DIM>(affinityMap, lowThreshold, highThreshold, out);
    // build the region graph
    std::vector<std::tuple<DATA_TYPE,LABEL_TYPE,LABEL_TYPE>> regionGraph;
    LABEL_TYPE maxId = counts.size() - 1;
    getRegionGraph<DIM>(affinityMap, out, maxId, regionGraph);
    // merge the segments
    mergeSegments(out, regionGraph, counts, mergeThreshold, lowThreshold, sizeThreshold);
}



} // namsepace region_growing
} // namespae nifty
