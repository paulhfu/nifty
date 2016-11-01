#pragma once

#include <iostream>

#include "nifty/marray/marray.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty {
namespace region_growing {

// TODO implement DIM > 2
template< unsigned DIM, class DATA_TYPE, class LABEL_TYPE>
std::vector<size_t>
edgeBasedWatershed(
        const marray::View<DATA_TYPE> & affinityMap,
        const DATA_TYPE low,
        const DATA_TYPE high,
        marray::View<LABEL_TYPE> & out)
{
    typedef DATA_TYPE DataType;
    typedef LABEL_TYPE LabelType;

    typedef array::StaticArray<int64_t, DIM> Coord;

    Coord shape;
    for(int d = 0; d < DIM; d++)
        shape[d] = affinityMap.shape(d);
    
    std::vector<size_t> counts(1,0);

    //
    // initialize the connecitivity according due to the affinity map
    //
    
    // TODO dimension independent!
    //nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
    //});

    for(size_t x = 0; x < shape[0]; x++) {
        for(size_t y = 0; y < shape[1]; y++) {

            LabelType & segId = out(x,y);
                
            DataType negx = (x>0) ? affinityMap(x,y,0) : low;
            DataType negy = (y>0) ? affinityMap(x,y,1) : low;
            DataType posx = (x<(shape[0]-1)) ? affinityMap(x+1,y,0) : low;
            DataType posy = (y<(shape[1]-1)) ? affinityMap(x,y+1,1) : low;

            DataType m = std::max({negx,negy,posx,posy});

            if( m > low ) {
                if ( negx == m || negx >= high ) { segId |= 0x01; } // bit magic: 01 -> connected to left x
                if ( negy == m || negy >= high ) { segId |= 0x02; } // 02 ->
                if ( posx == m || posx >= high ) { segId |= 0x04; } // 04
                if ( posy == m || posy >= high ) { segId |= 0x08; } // 08
            }
        }
    }

    // TODO is this consistent with the strides ?!
    // need to check for non-symetric input
    const std::ptrdiff_t direction[2*DIM] = { -1, -shape[1], 1, shape[1] };
    const LabelType dirmask[2*DIM]  = { 0x01, 0x02, 0x04, 0x08 };
    const LabelType idirmask[2*DIM] = { 0x04, 0x08, 0x01, 0x02 };

    //
    // get plateau corners
    //

    //std::vector<Coord> bfs;
    std::vector<std::ptrdiff_t> bfs;
    size_t size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<int64_t>() );

    for ( auto segIt = out.begin(); segIt != out.end(); segIt++ ) {
        for ( std::ptrdiff_t dir = 0; dir < 2*DIM; ++dir ) {
            if ( *segIt & dirmask[dir] ) {
                if ( !( *(segIt+direction[dir]) & idirmask[dir]) ) {
                    *segIt |= 0x10;
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
                    if ( !( *(segIt+direction[dir]) & 0x10 ) ) {
                        bfs.push_back(idx+direction[dir]);
                        *(segIt+direction[dir]) |= 0x10;
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
        
        // ponit was not visited yet
        if ( *segIt == 0 ) {
            // why do we activate the highest bit for 0s ?? -> probably this is like a 'visited' flag
            *segIt |= highBit;
            ++counts[0];
        }

        // point visited and is not 0
        if ( !( *segIt & highBit ) && *segIt ) {
            
            bfs.push_back(idx);
            bfsIndex = 0;
            *segIt |= 0x10;

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
                        else if ( !( *segHim & 0x10 ) ) {
                            *segHim |= 0x10;
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

    // I don't get why we do this...
    //for ( std::ptrdiff_t idx = 0; idx < size; ++idx )
    //    seg_raw[idx] &= traits::mask;
    //
    
    return counts;
}

} // namsepace region_growing
} // namespae nifty
