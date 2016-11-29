// Edge based watershed and zwatershed:
// Algorithms defined in:
// Coutsy et al., Watershed cuts: Minimum spanning forsets and the drop of water principle.

#pragma once
#ifndef NIFTY_REGION_GROWING_EDGE_BASED_WATERSHED_HXX
#define NIFTY_REGION_GROWING_EDGE_BASED_WATERSHED_HXX

#include <iostream>
#include <deque>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty {
namespace region_growing {

template<unsigned DIM, class VALUE_TYPE> void
nodeWeightsFromEdgeWeights(
        marray::View<VALUE_TYPE> const & edgeWeights,
        VALUE_TYPE const lowerThreshold,
        marray::View<VALUE_TYPE> & out,
        bool const ignoreBorder = false) {

    typedef array::StaticArray<int64_t,DIM> Coord;
    typedef array::StaticArray<int64_t,DIM+1> WeightCoord;
    typedef VALUE_TYPE ValueType; 

    //ValueType infinity = *std::max_element(edgeWeights.begin(), edgeWeights.end()) + .1;
    ValueType infinity = 1.;
    
    Coord shape;
    for(int d = 0; d < DIM; ++d)
        shape[d] = edgeWeights.shape(d);

    nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){

        std::vector<ValueType> thisWeights;
        
        // TODO make sure that this conforms with the CNN conventions
        for(int d = 0; d < DIM; ++d) {
            
            if( coord[d] > 0) {
                
                WeightCoord newCoord;
                for(int dd = 0; dd < DIM; ++dd)
                    newCoord[dd] = coord[dd];
                newCoord[d] -= 1;
                newCoord[DIM] = d;
                
                thisWeights.push_back( edgeWeights(newCoord.asStdArray()) );
            }

            if( coord[d] < shape[d] - 1) {
                
                WeightCoord newCoord;
                for(int dd = 0; dd < DIM; ++dd)
                    newCoord[dd] = coord[dd];
                newCoord[DIM] = d;
                
                thisWeights.push_back( edgeWeights(newCoord.asStdArray()) );

            }


        }

        ValueType maxWeight = *std::max_element(thisWeights.begin(), thisWeights.end() ); 
        out( coord.asStdArray() ) = (maxWeight > lowerThreshold ) ? infinity : maxWeight;

        bool atBorder = false;
        for(int d = 0; d < DIM; ++d)
            atBorder = (coord[d] == 0 || coord[d] == shape[d] - 1) ? true : false;
        if( ignoreBorder && atBorder )
            out( coord.asStdArray() ) = infinity;
    });
}

template<unsigned DIM, class VALUE_TYPE> void
thresholdEdgeWeights(marray::View<VALUE_TYPE> & edgeWeights, VALUE_TYPE const upperThreshold ) {
    
    typedef array::StaticArray<int64_t,DIM> Coord;
    typedef array::StaticArray<int64_t,DIM+1> WeightCoord;
    typedef VALUE_TYPE ValueType; 
    
    WeightCoord shape;
    for(int d = 0; d < DIM+1; ++d)
        shape[d] = edgeWeights.shape(d);
    
    //ValueType infinity = *std::max_element(edgeWeights.begin(), edgeWeights.end()) + .1;
    ValueType infinity = 1.;

    nifty::tools::forEachCoordinate(shape,[&](const WeightCoord & coord){
        for(int d = 0; d < DIM+1; ++d) {
            if( edgeWeights(coord.asStdArray()) > upperThreshold )
                edgeWeights(coord.asStdArray()) = infinity;
        }
    });

}

template<unsigned DIM, class VALUE_TYPE, class LABEL_TYPE, class COORD> inline 
LABEL_TYPE stream(COORD const & coord, 
        marray::View<VALUE_TYPE> const & edgeWeights,
        marray::View<VALUE_TYPE> const & nodeWeights,
        marray::View<LABEL_TYPE> const & labels, 
        std::vector<COORD> & streamCoordinates
        ) {

    typedef VALUE_TYPE ValueType;
    typedef LABEL_TYPE LabelType;
    typedef COORD Coord;
    typedef array::StaticArray<int64_t, DIM+1> WeightCoord;
    
    // add pixel to the stream coordinates
    streamCoordinates.push_back(coord);

    // initialize pixel queue
    std::deque<Coord> queue;
    queue.push_back(coord);
    
    while( ! queue.empty() ) {
       
        Coord nextCoord = queue.front();
        queue.pop_front(); 
        
        // get neighbors and weights for this pixel
        std::vector<Coord> coordinates;
        std::vector<ValueType> weights;

        for(int d = 0; d < DIM; ++d) {
            

            if(nextCoord[d] < edgeWeights.shape(d) - 1 ) {
                
                WeightCoord weightCoord;
                for(int dd = 0; dd < DIM; ++dd)
                    weightCoord[dd] = nextCoord[dd];
                weightCoord[DIM] = d;

                Coord newCoord = nextCoord;
                newCoord[d] += 1;

                coordinates.push_back( nextCoord );
                weights.push_back( edgeWeights(weightCoord.asStdArray()) );
            }
            
            if( nextCoord[d] > 0) {
                
                WeightCoord weightCoord;
                for(int dd = 0; dd < DIM; ++dd)
                    weightCoord[dd] = nextCoord[dd];
                weightCoord[d] -= 1;
                weightCoord[DIM] = d;

                Coord newCoord = nextCoord;
                newCoord[d] -=1;
            
                coordinates.push_back( newCoord );
                weights.push_back( edgeWeights(weightCoord.asStdArray()) );
            }

        }
        
        ValueType maxWeight = nodeWeights(nextCoord.asStdArray()); 

        auto itWeights = weights.begin();
        auto itCoordinates = coordinates.begin();

        for(;itWeights != weights.end(); ++itWeights, ++itCoordinates) {
            
            // only consider a pixel, if it is not in the stream yet and if its weight is equal to the nodes max-weight
            if( std::find(streamCoordinates.begin(), streamCoordinates.end(), *itCoordinates) == streamCoordinates.end() && fabs(*itWeights - maxWeight) <= std::numeric_limits<ValueType>::epsilon() ) {

                // if we hit a labeled pixel, return the stream 
                if( labels( itCoordinates->asStdArray() ) != 0 )
                    return labels( itCoordinates->asStdArray() );

                // if the node weight of the considered pixel is smaller, we start depth first search from it 
                else if( nodeWeights( itCoordinates->asStdArray() ) < maxWeight ) {
                    
                    streamCoordinates.push_back(*itCoordinates);
                    queue.clear();
                    queue.push_back(*itCoordinates);
                    // break looping over the current neighbors and go to new pix
                    break;
                }
                else {
                    streamCoordinates.push_back(*itCoordinates);
                    queue.push_back(*itCoordinates);
                }
            }
        }
    }
    
    // return 0, if we have not found a labeled pixel in the stream
    return 0;
}



template<unsigned DIM, class VALUE_TYPE, class LABEL_TYPE>  void
runEdgeBasedWatershed(
        marray::View<VALUE_TYPE> const & edgeWeights,
        marray::Marray<VALUE_TYPE> const & nodeWeights,
        marray::View<LABEL_TYPE> & out, 
        bool const ignoreBorder = false ) {
    
    typedef VALUE_TYPE ValueType;
    typedef LABEL_TYPE LabelType;
    typedef array::StaticArray<int64_t, DIM> Coord;

    LabelType next = 1;

    Coord shape;
    for(int d = 0; d < DIM; ++d)
        shape[d] = edgeWeights.shape(d);

    // iterate over all pixel
    nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){

            bool atBorder = false;
            for(int d = 0; d < DIM; ++d)
                atBorder = (coord[d] == 0 || coord[d] == shape[d] - 1) ? true : false;
            
            if( ignoreBorder && atBorder )
                return;
                        
            // if the pixel is already labeled, continue
            if( out(coord.asStdArray()) != 0 )
                return;

            // call stream -> finds the stream belonging to the current label and pixel coordinates belonging to the stream
            std::vector< Coord > streamCoordinates;
            LabelType label = stream<DIM>( coord, edgeWeights, nodeWeights, out, streamCoordinates);

            // if stream returns 0, we found a new stream
            if( label == 0 )
                label = next++;
            
            // update labels
            for( auto streamCoord : streamCoordinates)
                out(streamCoord.asStdArray()) = label;

    });
}


template<unsigned DIM, class VALUE_TYPE, class LABEL_TYPE>  void
edgeBasedWatershed(
        marray::View<VALUE_TYPE> & edgeWeights,
        VALUE_TYPE const lowerThreshold,
        VALUE_TYPE const upperThreshold,
        marray::View<LABEL_TYPE> & out, 
        bool const ignoreBorder = false ) {
    
    size_t shape[DIM];
    for(int d = 0; d < DIM; ++d)
        shape[d] = edgeWeights.shape(d);

    marray::Marray<VALUE_TYPE> nodeWeights(shape, shape + DIM);
    nodeWeightsFromEdgeWeights<DIM>(edgeWeights, lowerThreshold, nodeWeights, ignoreBorder);
    thresholdEdgeWeights<DIM>(edgeWeights, upperThreshold);
    runEdgeBasedWatershed<DIM>(edgeWeights, nodeWeights, out, ignoreBorder);

}



template<unsigned DIM, class VALUE_TYPE, class LABEL_TYPE> void
regionWeights(
        marray::View<VALUE_TYPE> const & edgeWeights,
        marray::View<LABEL_TYPE> const & labels,
        std::map<std::pair<LABEL_TYPE,LABEL_TYPE>,VALUE_TYPE> & out) {
    
    typedef VALUE_TYPE ValueType;
    typedef LABEL_TYPE LabelType;
    typedef array::StaticArray<int64_t, DIM> Coord;
    typedef array::StaticArray<int64_t,DIM+1> WeightCoord;
    typedef std::pair<LabelType,LabelType> EdgeType;
    
    Coord shape;
    for(int d = 0; d < DIM; ++d)
        shape[d] = edgeWeights.shape(d);

    // iterate over all pixel
    nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){

        LabelType lU = labels( coord.asStdArray() );

        WeightCoord weightCoord;
        for(int d = 0; d < DIM; ++d)
            weightCoord[d] = coord[d];

        for(int d = 0; d < DIM; ++d) {
            
            if(coord[d] != shape[d] -1) {
                Coord coordV = coord;
                coordV[d] += 1;
                
                LabelType lV = labels(coordV.asStdArray());
                
                if( lU != lV ) {
                    weightCoord[DIM] = d;
                    ValueType weight = edgeWeights(weightCoord.asStdArray());    

                    EdgeType edge = (lU > lV) ? std::make_pair(lV,lU) : std::make_pair(lU,lV);
                    
                    auto edgeIt = out.find(edge);
                    if( edgeIt == out.end() ) // this was the other way round in the old code, but it should be correct like this
                        out.insert( std::make_pair(edge, weight) );
                    else
                        edgeIt->second = std::max(weight, edgeIt->second);

                }

            }

        }

    });
}


template<unsigned DIM, class VALUE_TYPE, class LABEL_TYPE> void 
sizeFilter(
        std::map<std::pair<LABEL_TYPE,LABEL_TYPE>, VALUE_TYPE> const & regionWeightMap,
        size_t const sizeThreshold,
        VALUE_TYPE const weightThreshold,
        marray::View<LABEL_TYPE> & labels ) {
    
    typedef VALUE_TYPE ValueType;
    typedef LABEL_TYPE LabelType;
    
    typedef array::StaticArray<int64_t, DIM> Coord;
    
    typedef std::pair<LabelType,LabelType> EdgeType;
    typedef std::pair<EdgeType,ValueType> regionEdgeType;
    
    Coord shape;
    for(int d = 0; d < DIM; ++d)
        shape[d] = labels.shape(d);
    
    // dump map into a vector and sort it by value
    auto regionWeightsVec = std::vector< std::pair<EdgeType,ValueType> >(regionWeightMap.begin(), regionWeightMap.end());

    auto sortByValDecreasing = [] (regionEdgeType edgeA, regionEdgeType edgeB) {
        return (edgeA.second >= edgeB.second) ? false : true;
    };

    std::sort( regionWeightsVec.begin(), regionWeightsVec.end(), sortByValDecreasing );
    
    // + 1 because 0 label is reserved
    size_t nRegions = *std::max_element(labels.begin(), labels.end()) + 1;

    // find sizes of regions
    std::vector<size_t> sizes(nRegions);
    
    nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
        ++sizes[labels(coord.asStdArray())];
    });
    
    ufd::Ufd<LabelType> ufd(nRegions);

    // merge regions
    for( auto & edgeAndWeight : regionWeightsVec ) {
        
        // if we have reached the value threshold, we stop filtering
        if( edgeAndWeight.second < weightThreshold )
            break;
        
        auto edge = edgeAndWeight.first;
        
        auto s1 = ufd.find(edge.first);
        auto s2 = ufd.find(edge.second);

        // merge two regions, if at least one of them is below the size threshold
        if( s1 != s2 && (sizes[s1] < sizeThreshold || sizes[s2] < sizeThreshold) ) {
            
            size_t size = sizes[s1] + sizes[s2];
            sizes[s1] = 0;
            sizes[s2] = 0;
            ufd.merge(s1,s2);
            sizes[ufd.find(s1)] = size;
        }
    }
    
    std::map<LabelType,LabelType> newLabels;
    ufd.representativeLabeling(newLabels);
    
    // filter out small regions
    // we might not filter here for a dense segmentatation 
    LabelType next = 1;
    for( auto it = newLabels.begin(); it != newLabels.end(); ++it) {
        if( sizes[it->first] < sizeThreshold )
            it->second = 0;
        else
            it->second = next++;
    }
    
    nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
        labels(coord.asStdArray()) = newLabels[ ufd.find(labels(coord.asStdArray())) ];
    });
}



template<unsigned DIM, class VALUE_TYPE, class LABEL_TYPE>  void
zWatershed(
        marray::View<VALUE_TYPE> & edgeWeights,
        VALUE_TYPE const lowerThreshold,
        VALUE_TYPE const upperThreshold,
        size_t     const sizeThreshold,
        VALUE_TYPE const weightThreshold,
        marray::View<LABEL_TYPE> & out, 
        bool const ignoreBorder = false ) {

    edgeBasedWatershed<DIM>(edgeWeights, lowerThreshold, upperThreshold, out, ignoreBorder);
    std::map<std::pair<LABEL_TYPE,LABEL_TYPE>,VALUE_TYPE> regionWeightMap; 
    regionWeights<DIM>(edgeWeights,out,regionWeightMap);
    sizeFilter<DIM>(regionWeightMap, sizeThreshold, weightThreshold, out);
}



} // namespace region_growing
} // namespace nifty

#endif // #ifndef NIFTY_REGION_GROWING_EDGE_BASSED_WATERSHED_HXX
