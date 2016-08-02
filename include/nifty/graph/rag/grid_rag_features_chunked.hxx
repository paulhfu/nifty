#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_CHUNKED_HXX

#include "nifty/graph/rag/grid_rag_chunked.hxx"
#include "nifty/graph/rag/grid_rag_features.hxx"

namespace nifty{
namespace graph{
    
    template< class LABELS_TYPE, class T, class EDGE_MAP, class NODE_MAP>
    void gridRagAccumulateFeatures(
        const ChunkedLabelsGridRagSliced<LABELS_TYPE> & graph,
        const  marray::View<T> & data,
        EDGE_MAP & edgeMap,
        NODE_MAP &  nodeMap,
        const size_t z0
    ){
        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & labels = labelsProxy.labels();

        // check that the data covers a whole slice in xy
        // need to take care of different axis ordering...
        NIFTY_CHECK_OP(data.shape(0),==,shape[0], "Shape along x does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(z0+data.shape(2),<=,shape[2], "Z offset is too large")

        size_t sliceShape[] = {size_t(shape[0]), size_t(shape[1]), 1};
        marray::Marray<LABELS_TYPE> currentSlice(sliceShape, sliceShape+3);
        marray::Marray<LABELS_TYPE> nextSlice(sliceShape, sliceShape+3);
        
        const auto numberOfPasses =  std::max(edgeMap.numberOfPasses(),nodeMap.numberOfPasses());
        for(size_t p=0; p<numberOfPasses; ++p){
            // start pass p
            edgeMap.startPass(p);
            nodeMap.startPass(p);

            for( size_t z = 0; z < data.shape(2); z++ )
            {
                size_t sliceStart[] = {0,0,z+z0};
                labels.readSubarray(sliceStart, currentSlice);

                if( z < data.shape(2) - 1) {
                    size_t nextStart[] = {0,0,z+z0+1};
                    labels.readSubarray(nextStart, nextSlice);
                }
                
                // TODO parallelize
                for(size_t x = 0; x < shape[0]; x++) {
                    for(size_t y = 0; y < shape[1]; y++) {
                        
                        const auto lU = currentSlice(x,y,0);
                        const auto dU = data(x,y,z);
                        nodeMap.accumulate(lU, dU);
                        
                        if( x + 1 < shape[0] ) {
                            const auto lV = currentSlice(x+1,y,0);
                            const auto dV = data(x+1,y,z);
                            if( lU != lV) {
                                const auto e = graph.findEdge(lU, lV);
                                edgeMap.accumulate(e, dU);
                                edgeMap.accumulate(e, dV);
                            }
                        }
                        
                        if( y + 1 < shape[1] ) {
                            const auto lV = currentSlice(x,y+1,0);
                            const auto dV = data(x,y+1,z);
                            if( lU != lV) {
                                const auto e = graph.findEdge(lU, lV);
                                edgeMap.accumulate(e, dU);
                                edgeMap.accumulate(e, dV);
                            }
                        }
                        
                        if( z + 1 < data.shape(2)) {
                            const auto lV = nextSlice(0,y,x);
                            const auto dV = data(x,y,z+1);
                            // we don't need to check if the labels are different, due to the sliced labels
                            const auto e = graph.findEdge(lU, lV);
                            edgeMap.accumulate(e, dU);
                            edgeMap.accumulate(e, dV);
                        }
                    }
                }
            }
        }
    }


    
    template<class LABELS_TYPE, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const ChunkedLabelsGridRagSliced<LABELS_TYPE> & graph,
        const nifty::hdf5::Hdf5Array<LABELS> & data, // template for the chunked data (expected to be a chunked vigra type)
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, 2> Coord;

        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & labels = labelsProxy.labels(); 

        // check that the data covers a whole slice in xy
        // need to take care of different axis ordering...
        NIFTY_CHECK_OP(data.shape(0),==,shape[0], "Shape along x does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(data.shape(2),==,shape[2], "Shape along z does not agree")

        size_t sliceShape[] = { size_t(shape[0]), size_t(shape[1]), 1};

        marray::Marray<LABELS_TYPE> currentLabels(sliceShape, sliceShape+3);
        marray::Marray<LABELS> currentData(sliceShape, sliceShape+3);
        
        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());

        for(size_t z = 0; z < shape[2]; z++) {

            // checkout this slice
            size_t sliceStart[] = {0, 0, z};
            labels.readSubarray(sliceStart, currentLabels);
            data.readSubarray(sliceStart, currentData);
            
            // TODO parallel versions of the code
            
            nifty::tools::forEachCoordinate(std::array<int64_t,2>({(int64_t)shape[0],(int64_t)shape[1]}),[&](const Coord & coord){
                const auto x = coord[0];
                const auto y = coord[1];
                const auto node = currentLabels(x,y,0);            
                const auto l  = currentData(x,y,0);
                overlaps[node][l] += 1;
            });
        }
        
        for(const auto node : graph.nodes()){
            const auto & ol = overlaps[node];
            // find max ol 
            uint64_t maxOl = 0 ;
            uint64_t maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[node] = maxOlLabel;
        }

    }


} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_CHUNKED_HXX */
