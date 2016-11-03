// Edge based watershed and zwatershed:
// Algorithms defined in:
// Coutsy et al., Watershed cuts: Minimum spanning forsets and the drop of water principle.
// Zlateski et al., Image Segmentation by Size-Dependent Single Linkage Clustering of a Watershed Basin Graph
// https://arxiv.org/abs/1505.00249
// Implementation inspired by:
// https://github.com/TuragaLab/zwatershed

#pragma once

#include <iostream>

#include "nifty/marray/marray.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty {
namespace region_growing {

template< unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
std::vector<size_t>
edgeBasedWatershed(
        const marray::View<DATA_TYPE> & affinityMap,
        const DATA_TYPE low,
        const DATA_TYPE high,
        marray::View<LABEL_TYPE> & out)
{
    std::cout << "Starting" << std::endl;
    
    typedef DATA_TYPE DataType;
    typedef LABEL_TYPE LabelType;

    typedef array::StaticArray<int64_t, DIM> Coord;

    Coord shape;
    for(int d = 0; d < DIM; d++)
        shape[d] = affinityMap.shape(d);
    
    std::vector<size_t> counts(1,0);
    
    //
    // initialize bitvalues encoding the directions
    //

    // TODO is this consistent with the strides ?!
    // need to check for non-symetric input
    //const std::ptrdiff_t direction[2*DIM] = { -1, -shape[1], 1, shape[1] };
    //const LabelType dirmask[2*DIM]  = { 0x01, 0x02, 0x04, 0x08 };
    //const LabelType idirmask[2*DIM] = { 0x04, 0x08, 0x01, 0x02 };
    std::ptrdiff_t direction[2*DIM];
    LabelType dirmask[2*DIM];
    LabelType idirmask[2*DIM];
    LabelType currentBit = 0x01;
    std::ptrdiff_t currentDirection = -1;
    for( size_t dir = 0; dir < 2*DIM; dir++) {
        // we need to change the sign of the direction after DIM directions
        if(dir == DIM)
            currentDirection = 1;
        
        direction[dir] = currentDirection;
        dirmask[dir] = currentBit;
        idirmask[dir] = (dir < DIM) ? currentBit << DIM : currentBit >> DIM;
        
        currentBit = currentBit << 1;
        // TODO is this consistent with the strides ?
        currentDirection *= shape[DIM-((dir%DIM)+1)];
    }
    LabelType maxBit = currentBit;

    //
    // initialize the connecitivity according due to the affinity map
    //
    
    auto makeCoordDir = [](const Coord & coord,const size_t dir){
        Coord coordDir = coord;
        coordDir[dir%DIM] += (dir<DIM) ? -1 : 1;
        return coordDir;
    };
    
    // dimension independent!
    nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
        
        LabelType & segId = out(coord.asStdArray());

        std::vector<DataType> affinityValues;
        for( size_t dir = 0; dir < 2*DIM; ++dir ) {
            Coord coordDir = makeCoordDir(coord, dir);
            DataType val = (dir<DIM) ? ( (coordDir[dir%DIM]>0) ? affinityMap(coordDir.asStdArray()) : low )
                : ( (coordDir[dir%DIM]<shape[dir%DIM]-1) ? affinityMap(coordDir.asStdArray()) : low );
            affinityValues.push_back(val);
        }
        DataType maxAffinity = *(std::max_element(affinityValues.begin(), affinityValues.end()));

        if( maxAffinity > low ) {
            for( size_t dir = 0; dir < 2*DIM; ++dir ) {
                if( affinityValues[dir] == maxAffinity || affinityValues[dir] >= high )
                    segId |= dirmask[dir];
            }
        }
    });


    //
    // get plateau corners
    //

    std::vector<std::ptrdiff_t> bfs;
    size_t size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<int64_t>() );

    for ( auto segIt = out.begin(); segIt != out.end(); ++segIt ) {
        for ( std::ptrdiff_t dir = 0; dir < 2*DIM; ++dir ) {
            if ( *segIt & dirmask[dir] ) {
                if ( !( *(segIt+direction[dir]) & idirmask[dir]) ) {
                    *segIt |= maxBit;
                    bfs.push_back(std::distance(out.begin(),segIt));
                    break;
                }
            }
        }
    }

    //
    // divide the plateaus
    //

    std::size_t bfsIndex = 0;

    while ( bfsIndex < bfs.size() ) {
        
        std::ptrdiff_t idx = bfs[bfsIndex];
        auto segIt = out.begin() + idx;
        LabelType toSet = 0;

        for ( std::ptrdiff_t dir = 0; dir < 2*DIM; ++dir ) {
            if ( *segIt & dirmask[dir] ) {
                if ( *(segIt+direction[dir]) & idirmask[dir] ) {
                    if ( !( *(segIt+direction[dir]) & maxBit ) ) {
                        bfs.push_back(idx+direction[dir]);
                        *(segIt+direction[dir]) |= maxBit;
                        }
                }
                else {
                    toSet = dirmask[dir];
                }
            }
        }
        *segIt = toSet;
        ++bfsIndex;
    }

    bfs.clear();

    //
    // main watershed
    // 

    LabelType next = 1;
    
    // FIXME this is only correct for uint32
    LabelType highBit = 0x80000000;

    for ( std::ptrdiff_t idx = 0; idx < size; ++idx ) {
        auto segIt = out.begin() + idx;
        
        // point is must not link (all affinities < low)
        if ( *segIt == 0 ) {
            // why do we activate the highest bit for 0s ?? -> probably this is like a 'visited' flag
            *segIt |= highBit;
            ++counts[0];
        }

        // point not visited and is not 0
        if ( !( *segIt & highBit ) && *segIt ) {
            
            bfs.push_back(idx);
            bfsIndex = 0;
            *segIt |= maxBit;

            // loop over all points in the stream
            while ( bfsIndex < bfs.size() ) {
                std::ptrdiff_t me = bfs[bfsIndex];
                auto segMe = out.begin() + me;

                for ( std::ptrdiff_t dir = 0; dir < 2*DIM; ++dir ) {
                    
                    // we found an upstream point
                    if ( *segMe & dirmask[dir] ) {
                        auto segHim = segMe + direction[dir];
                        if ( *segHim & highBit ) {
                            counts[ *segHim & ~highBit ] += bfs.size();

                            for ( const auto & it: bfs )
                                *(out.begin() + it) = *segHim;

                            bfs.clear();
                            break;
                        }
                        else if ( !( *segHim & maxBit ) ) {
                            *segHim |= maxBit;
                            bfs.push_back( std::distance(out.begin(),segHim) );
                        }
                    }
                }
                ++bfsIndex;
            }

            // current point is the upstream
            if ( bfs.size() )
            {
                counts.push_back( bfs.size() );
                for ( const auto & it: bfs )
                    *(out.begin()+it) = highBit | next;
                ++next;
                bfs.clear();
            }
        }
    }

    std::cout << "found: " << (next-1) << " components\n";

    // TODO need to change this for different LabelTypes
    const uint32_t mask = 0x7FFFFFFF;
    // splice away the high bit
    for( auto segIt = out.begin(); segIt != out.end(); segIt++ )
        *segIt &= mask;
    
    
    return counts;
}

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

} // namsepace region_growing
} // namespae nifty
