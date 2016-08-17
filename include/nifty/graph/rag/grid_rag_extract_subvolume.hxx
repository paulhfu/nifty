#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_EXTRACT_SUBVOLUME_HXX
#define NIFTY_GRAPH_RAG_GRID_EXTRACT_SUBVOLUME_HXX

#include <map>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/graph/rag/grid_rag.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif

namespace nifty{
namespace graph{

    /* For now, we use the node list variant instead
    template<class RAG, class COORD>
    void extractNodesAndEdgesFromSubVolume(const RAG & rag, const COORD & blockBegin, const COORD & blockEnd, 
        marray::View<typename RAG::EdgeType> & innerEdgesOut, marray::View<typename RAG::EdgeType> & outerEdgesOut, marray::View<typename RAG::LabelType> & uvIdsOut) {

        // TODO s
        
        // For now I have chosen to implement this s.t. the whole label block is read out of memory and then processed
        // could also loop over subblocks here, if the label blocks get too large
        
        // Also, this is only implemented for 3d, but should be pretty easy to implement dimension independent 

        typedef typename RAG::LabelType LabelType;
        // check out labels in subvolume
        COORD subShape( {blockEnd[0] - blockBegin[0], blockEnd[1] - blockBegin[1],blockEnd[2] - blockBegin[2]} );
        marray::Marray<LabelType> subLabels( subShape.begin(), subShape.end() );
        nifty::tools::readSubarray(blockBegin, blockEnd, subLabels);
        
        // loop over the coordinates in parallel
    } */
    
    template<class RAG>
    void extractNodesAndEdgesFromNodeList(const RAG & rag,
            const marray::View<typename RAG::LabelType> & nodeList, 
            std::vector<typename RAG::EdgeType> & innerEdgesOut,
            std::vector<typename RAG::EdgeType> & outerEdgesOut,
            std::vector<std::pair<typename RAG::LabelType,typename RAG::LabelType>> & uvIdsOut) {
        
        typedef typename RAG::LabelType LabelType;
        typedef typename RAG::EdgeType EdgeType;

        std::map<LabelType,LabelType> globalToLocalNodes;
        for(size_t i = 0; i < nodeList.size(); i++)
            globalToLocalNodes.insert( std::make_pair(nodeList(i), LabelType(i)) );

        for(auto nodeIt = nodeList.begin(); nodeIt != nodeList.end(); ++nodeIt) {
            auto u = *nodeIt;
            for(auto adjacencyIt = rag.adjacencyBegin(u); adjacencyIt != rag.adjacencyEnd(u); ++adjacencyIt) {
                auto v = adjacencyIt->node();
                auto e = rag.findEdge(u, v);
                if( std::find(nodeList.begin(), nodeList.end(), v) != nodeList.end() )
                    innerEdgesOut.push_back(e);
                else
                    outerEdgesOut.push_back(e);
            }
        }
        
        // sort the edges and make them unique
        std::vector<EdgeType> temp(innerEdgesOut.size());
        std::sort(innerEdgesOut.begin(),innerEdgesOut.end());
        auto it = std::unique_copy(innerEdgesOut.begin(), innerEdgesOut.end(), temp.begin());
        temp.resize(std::distance(temp.begin(),it));
        innerEdgesOut = temp;

        temp.clear();
        temp.resize(outerEdgesOut.size());
        std::sort(outerEdgesOut.begin(),outerEdgesOut.end());
        it = std::unique_copy(outerEdgesOut.begin(), outerEdgesOut.end(), temp.begin());
        temp.resize(std::distance(temp.begin(),it));
        outerEdgesOut = temp;

        uvIdsOut.resize(innerEdgesOut.size());

        // get the local uv - ids
        for(size_t i = 0; i < innerEdgesOut.size(); ++i) {
            auto uv = rag.uv(innerEdgesOut[i]);
            uvIdsOut[i].first = globalToLocalNodes[uv.first];
            uvIdsOut[i].second = globalToLocalNodes[uv.second];
        }
    }

} // end namespace graph
} // end namespace nifty

#endif
