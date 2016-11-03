// Edge based watershed and zwatershed:
// Algorithms defined in:
// Coutsy et al., Watershed cuts: Minimum spanning forsets and the drop of water principle.
// Implementation inspired by:
// https://github.com/TuragaLab/zwatershed

#pragma once

#include <iostream>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

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

} // namsepace region_growing
} // namespae nifty
