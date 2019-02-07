#pragma once

#include <unordered_set>
#include <set>
#include <boost/functional/hash.hpp>

#include "xtensor/xtensor.hpp"
#include "xtensor/xadapt.hpp"

#include "z5/multiarray/xtensor_access.hxx"
#include "z5/dataset_factory.hxx"
#include "z5/groups.hxx"
#include "z5/attributes.hxx"

#include "nifty/array/static_array.hxx"
#include "nifty/xtensor/xtensor.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace fs = boost::filesystem;

namespace nifty {
namespace distributed {


    ///
    // graph typedefs at nifty.distributed level
    ///

    typedef uint64_t NodeType;
    typedef int64_t EdgeIndexType;

    typedef std::pair<NodeType, NodeType> EdgeType;
    typedef boost::hash<EdgeType> EdgeHash;

    // Perfoemance for extraction of 50 x 512 x 512 cube (real labels)
    // (including some overhead (python cals, serializing the graph, etc.))
    // using normal set:    1.8720 s
    // using unordered set: 1.8826 s
    // Note that we would need an additional sort to make the unordered set result correct.
    // As we do not see an improvement, stick with the set for now.
    // But for operations on larger edge / node sets, we should benchmark the unordered set again
    typedef std::set<NodeType> NodeSet;
    typedef std::set<EdgeType> EdgeSet;
    //typedef std::unordered_set<EdgeType, EdgeHash> EdgeSet;

    // xtensor typedefs
    typedef xt::xtensor<NodeType, 1> Tensor1;
    typedef xt::xtensor<NodeType, 2> Tensor2;
    typedef typename Tensor1::shape_type Shape1Type;
    typedef typename Tensor2::shape_type Shape2Type;
    typedef xt::xtensor<NodeType, 3> Tensor3;
    typedef typename Tensor3::shape_type Shape3Type;

    // nifty typedef
    typedef nifty::array::StaticArray<int64_t, 3> CoordType;


    ///
    // helper functions (we might turn this into detail namespace ?!)
    ///


    template<class NODES>
    inline void loadNodes(const std::string & graphPath,
                          NODES & nodes) {
        const std::vector<size_t> zero1Coord({0});
        // get handle and dataset
        z5::handle::Group graph(graphPath);
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the node set
        Shape1Type nodeShape({nodeDs->shape(0)});
        Tensor1 tmpNodes(nodeShape);
        z5::multiarray::readSubarray<NodeType>(nodeDs, tmpNodes, zero1Coord.begin());
        nodes.insert(tmpNodes.begin(), tmpNodes.end());
    }


    inline void loadNodes(const std::string & graphPath,
                          std::vector<NodeType> & nodes,
                          const size_t offset,
                          const int nThreads=1) {
        const std::vector<size_t> zero1Coord({0});
        // get handle and dataset
        z5::handle::Group graph(graphPath);
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the node set
        Shape1Type nodeShape({nodeDs->shape(0)});
        Tensor1 tmpNodes(nodeShape);
        z5::multiarray::readSubarray<NodeType>(nodeDs, tmpNodes, zero1Coord.begin(), nThreads);
        nodes.resize(nodes.size() + nodeShape[0]);
        std::copy(tmpNodes.begin(), tmpNodes.end(), nodes.begin() + offset);
    }


    template<class NODES>
    inline void loadNodesToArray(const std::string & graphPath,
                                 xt::xexpression<NODES> & nodesExp,
                                 const int nThreads=1) {
        auto & nodes = nodesExp.derived_cast();
        const std::vector<size_t> zero1Coord({0});
        // get handle and dataset
        z5::handle::Group graph(graphPath);
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the array
        z5::multiarray::readSubarray<NodeType>(nodeDs, nodes, zero1Coord.begin(), nThreads);
    }


    inline bool loadEdgeIndices(const std::string & graphPath,
                                std::vector<EdgeIndexType> & edgeIndices,
                                const std::size_t offset,
                                const int nThreads=1) {
        const std::vector<std::size_t> zero1Coord({0});
        const std::vector<std::string> keys = {"numberOfEdges"};

        // get handle and check if we have edges
        z5::handle::Group graph(graphPath);
        nlohmann::json j;
        z5::readAttributes(graph, keys, j);
        size_t numberOfEdges = j[keys[0]];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get id dataset
        auto idDs = z5::openDataset(graph, "edgeIds");
        // read the nodes and inset them into the node set
        Shape1Type idShape({idDs->shape(0)});
        xt::xtensor<EdgeIndexType, 1> tmpIds(idShape);
        z5::multiarray::readSubarray<EdgeIndexType>(idDs, tmpIds, zero1Coord.begin(), nThreads);
        edgeIndices.resize(idShape[0] + edgeIndices.size());
        std::copy(tmpIds.begin(), tmpIds.end(), edgeIndices.begin() + offset);
        return true;
    }


    template<class EDGES>
    inline bool loadEdges(const std::string & graphPath,
                          EDGES & edges) {
        const std::vector<size_t> zero2Coord({0, 0});
        const std::vector<std::string> keys = {"numberOfEdges"};

        // get handle and check if we have edges
        z5::handle::Group graph(graphPath);
        nlohmann::json j;
        z5::readAttributes(graph, keys, j);
        size_t numberOfEdges = j[keys[0]];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get edge dataset
        auto edgeDs = z5::openDataset(graph, "edges");
        // read the edges and inset them into the edge set
        Shape2Type edgeShape({edgeDs->shape(0), 2});
        Tensor2 tmpEdges(edgeShape);
        z5::multiarray::readSubarray<NodeType>(edgeDs, tmpEdges, zero2Coord.begin());
        for(size_t edgeId = 0; edgeId < edgeShape[0]; ++edgeId) {
            edges.insert(std::make_pair(tmpEdges(edgeId, 0), tmpEdges(edgeId, 1)));
        }
        return true;
    }


    inline bool loadEdges(const std::string & graphPath,
                          std::vector<EdgeType> & edges,
                          const size_t offset,
                          const int nThreads=1) {
        const std::vector<size_t> zero2Coord({0, 0});
        const std::vector<std::string> keys = {"numberOfEdges"};

        // get handle and check if we have edges
        z5::handle::Group graph(graphPath);
        nlohmann::json j;
        z5::readAttributes(graph, keys, j);
        size_t numberOfEdges = j[keys[0]];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get edge dataset
        auto edgeDs = z5::openDataset(graph, "edges");

        // read the edges and inset them into the edge set
        Shape2Type edgeShape({edgeDs->shape(0), 2});
        Tensor2 tmpEdges(edgeShape);
        z5::multiarray::readSubarray<NodeType>(edgeDs, tmpEdges, zero2Coord.begin(), nThreads);
        edges.resize(edges.size() + edgeShape[0]);
        for(size_t edgeId = 0; edgeId < edgeShape[0]; ++edgeId) {
            edges[edgeId + offset] = std::make_pair(tmpEdges(edgeId, 0), tmpEdges(edgeId, 1));
        }
        return true;
    }


    // Using templates for node and edge storages here,
    // because we might use different datastructures at different graph levels
    // (set or unordered_set)
    template<class NODES, class EDGES, class COORD>
    void serializeGraph(const std::string & pathToGraph,
                        const std::string & saveKey,
                        const NODES & nodes,
                        const EDGES & edges,
                        const COORD & roiBegin,
                        const COORD & roiEnd,
                        const bool ignoreLabel=false,
                        const int numberOfThreads=1,
                        const std::string & compression="raw") {

        const size_t nNodes = nodes.size();
        const size_t nEdges = edges.size();

        // create the graph group
        auto graphPath = fs::path(pathToGraph);
        graphPath /= saveKey;
        z5::handle::Group group(graphPath.string());
        z5::createGroup(group, false);

        // threadpool for parallel writing
        parallel::ThreadPool tp(numberOfThreads);

        // serialize the graph (nodes)
        std::vector<size_t> nodeShape = {nNodes};
        std::vector<size_t> nodeChunks = {std::min(nNodes, 2*262144UL)};
        // FIXME For some reason only raw compression works,
        // because the precompiler flags for activating compression schemes
        // are not properly set (although we can read datasets with compression,
        // so this doesn't make much sense)
        // std::cout << "Writing " << nNodes << " nodes to " << pathToGraph << std::endl;
        auto dsNodes = z5::createDataset(group, "nodes",
                                         "uint64", nodeShape,
                                         nodeChunks, false,
                                         compression);

        const size_t numberNodeChunks = dsNodes->numberOfChunks();
        // std::cout << "Serialize nodes" << std::endl;
        parallel::parallel_foreach(tp, numberNodeChunks, [&](const int tId,
                                                             const size_t chunkId){
            const size_t nodeStart = chunkId * nodeChunks[0];
            const size_t nodeStop = std::min((chunkId + 1) * nodeChunks[0],
                                             nodeShape[0]);

            const size_t nNodesChunk = nodeStop - nodeStart;
            Shape1Type nodeSerShape({nNodesChunk});
            Tensor1 nodeSer(nodeSerShape);

            auto nodeIt = nodes.begin();
            std::advance(nodeIt, nodeStart);
            for(size_t i = 0; i < nNodesChunk; i++, nodeIt++) {
                nodeSer(i) = *nodeIt;
            }

            const std::vector<size_t> nodeOffset({nodeStart});
            z5::multiarray::writeSubarray<NodeType>(dsNodes, nodeSer,
                                                    nodeOffset.begin());
        });
        // std::cout << "done" << std::endl;

        // serialize the graph (edges)
        // std::cout << "Serialize edges" << std::endl;
        if(nEdges > 0) {
            // std::cout << "Writing " << nEdges << " edges to " << pathToGraph << std::endl;
            std::vector<size_t> edgeShape = {nEdges, 2};
            std::vector<size_t> edgeChunks = {std::min(nEdges, 262144UL), 2};
            // FIXME For some reason only raw compression works,
            // because the precompiler flags for activating compression schemes
            // are not properly set (although we can read datasets with compression,
            // so this doesn't make much sense)
            auto dsEdges = z5::createDataset(group, "edges", "uint64",
                                             edgeShape, edgeChunks, false,
                                             compression);
            const size_t numberEdgeChunks = dsEdges->numberOfChunks();

            parallel::parallel_foreach(tp, numberEdgeChunks, [&](const int tId,
                                                                 const size_t chunkId){
                const size_t edgeStart = chunkId * edgeChunks[0];
                const size_t edgeStop = std::min((chunkId + 1) * edgeChunks[0],
                                                 edgeShape[0]);

                const size_t nEdgesChunk = edgeStop - edgeStart;
                Shape2Type edgeSerShape({nEdgesChunk, 2});
                Tensor2 edgeSer(edgeSerShape);

                auto edgeIt = edges.begin();
                std::advance(edgeIt, edgeStart);
                for(size_t i = 0; i < nEdgesChunk; i++, edgeIt++) {
                    edgeSer(i, 0) = edgeIt->first;
                    edgeSer(i, 1) = edgeIt->second;
                }

                const std::vector<size_t> edgeOffset({edgeStart, 0});
                z5::multiarray::writeSubarray<NodeType>(dsEdges, edgeSer,
                                                        edgeOffset.begin());
            });
        }
        // std::cout << "done" << std::endl;

        // serialize metadata (number of edges and nodes and position of the block)
        nlohmann::json attrs;
        attrs["numberOfNodes"] = nNodes;
        attrs["numberOfEdges"] = nEdges;
        attrs["roiBegin"] = std::vector<size_t>(roiBegin.begin(), roiBegin.end());
        attrs["roiEnd"] = std::vector<size_t>(roiEnd.begin(), roiEnd.end());
        attrs["ignoreLabel"] = ignoreLabel;

        z5::writeAttributes(group, attrs);
    }


    inline void makeCoord2(const CoordType & coord,
                           CoordType & coord2,
                           const size_t axis) {
        coord2 = coord;
        coord2[axis] += 1;
    };


    ///
    // Workflow functions
    ///


    template<class COORD>
    void extractGraphFromRoi(const std::string & pathToLabels,
                             const std::string & keyToLabels,
                             const COORD & roiBegin,
                             const COORD & roiEnd,
                             NodeSet & nodes,
                             EdgeSet & edges,
                             const bool ignoreLabel=false,
                             const bool increaseRoi=true) {

        // open the n5 label dataset
        auto path = fs::path(pathToLabels);
        path /= keyToLabels;
        auto ds = z5::openDataset(path.string());

        // if specified, we decrease roiBegin by 1.
        // this is necessary to capture edges that lie in between of block boundaries
        // However, we don't want to add the nodes to nodes in the sub-graph !
        COORD actualRoiBegin = roiBegin;
        std::array<bool, 3> roiIncreasedAxis = {false, false, false};
        if(increaseRoi) {
            for(int axis = 0; axis < 3; ++axis) {
                if(actualRoiBegin[axis] > 0) {
                    --actualRoiBegin[axis];
                    roiIncreasedAxis[axis] = true;
                }
            }
        }

        // load the roi
        Shape3Type shape;
        CoordType blockShape, coord2;

        for(int axis = 0; axis < 3; ++axis) {
            shape[axis] = roiEnd[axis] - actualRoiBegin[axis];
            blockShape[axis] = shape[axis];
        }
        Tensor3 labels(shape);
        z5::multiarray::readSubarray<NodeType>(ds, labels, actualRoiBegin.begin());

        // iterate over the the roi and extract all graph nodes and edges
        // we want ordered iteration over nodes and edges in the end,
        // so we use a normal set instead of an unordered one

        NodeType lU, lV;
        nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {

            lU = xtensor::read(labels, coord.asStdArray());
            // we don't add the nodes in the increased roi
            if(increaseRoi) {
                bool insertNode = true;
                for(int axis = 0; axis < 3; ++axis) {
                    if(coord[axis] == 0 && roiIncreasedAxis[axis]) {
                        insertNode = false;
                        break;
                    }
                }
                if(insertNode) {
                    nodes.insert(lU);
                }
            }
            else {
                nodes.insert(lU);
            }

            // skip edges to zero if we have an ignoreLabel
            if(ignoreLabel && (lU == 0)) {
                return;
            }

            for(size_t axis = 0; axis < 3; ++axis){
                makeCoord2(coord, coord2, axis);
                if(coord2[axis] < blockShape[axis]){
                    lV = xtensor::read(labels, coord2.asStdArray());
                    // skip zero if we have an ignoreLabel
                    if(ignoreLabel && (lV == 0)) {
                        return;
                    }
                    if(lU != lV){
                        edges.insert(std::make_pair(std::min(lU, lV),
                                     std::max(lU, lV)));
                    }
                }
            }
        });
    }


    template<class COORD>
    inline void computeMergeableRegionGraph(const std::string & pathToLabels,
                                            const std::string & keyToLabels,
                                            const COORD & chunkId,
                                            const std::string & nodeDsPath,
                                            const std::string & edgeDsPath,
                                            const bool ignoreLabel=false,
                                            const bool increaseRoi=false) {
        // get the roi corresponding to this chunk id
        const auto dsNodes = z5::openDataset(nodeDsPath);
        const auto & chunking = dsNodes->chunking();
        std::vector<std::size_t> roiBegin, roiEnd;
        chunking.getBlockBeginAndEnd(chunkId, roiBegin, roiEnd);

        // extract graph nodes and edges from roi
        NodeSet nodes;
        EdgeSet edges;
        extractGraphFromRoi(pathToLabels, keyToLabels,
                            roiBegin, roiEnd,
                            nodes, edges,
                            ignoreLabel, increaseRoi);

        // TODO handle empty blocks !!!
        // serialize the sub-graph nodes and edges
        std::vector<NodeType> nodeVec(nodes.begin(), nodes.end());
        dsNodes->writeChunk(chunkId, &nodeVec[0], true, nodeVec.size());

        const auto dsEdges = z5::openDataset(edgeDsPath);
        // flatten the edges for storage
        std::vector<NodeType> edgeVec(2 * edges.size());
        std::size_t edgeId = 0;
        for(const auto & edge: edges) {
            edgeVec[2 * edgeId] = edge.first;
            edgeVec[2 * edgeId + 1] = edge.second;
            ++edgeId;
        }
        dsEdges->writeChunk(chunkId, &edgeVec[0], true, edgeVec.size());
    }


    inline void mergeSubgraphsSingleThreaded(const fs::path & graphPath,
                                             const std::string & dsNodePath,
                                             const std::string & dsEdgePath,
                                             const std::vector<std::size_t> & chunkIds,
                                             NodeSet & nodes,
                                             EdgeSet & edges,
                                             std::vector<std::size_t> & roiBegin,
                                             std::vector<std::size_t> & roiEnd) {
        const auto dsNodes = z5::openDataset(dsNodePath);
        const auto dsEdges = z5::openDataset(dsEdgePath);
        const auto & chunking = dsNodes->chunking();

        for(std::size_t chunkId : chunkIds) {

            std::vector<std::size_t> chunkIndex;
            chunking.blockIdToBlockCoordinate(chunkId, chunkIndex);

            // merge the rois
            std::vector<std::size_t> blockBegin, blockEnd;
            chunking.getBlockBeginAndEnd(chunkIndex, blockBegin, blockEnd);

            for(int axis = 0; axis < 3; ++axis) {
                roiBegin[axis] = std::min(roiBegin[axis],
                                          static_cast<size_t>(blockBegin[axis]));
                roiEnd[axis] = std::max(roiEnd[axis],
                                        static_cast<size_t>(blockEnd[axis]));
            }

            // load nodes
            bool isVarlen;
            const std::size_t nNodes = dsNodes->getDiscChunkSize(chunkIndex, isVarlen);
            if(nNodes == 0) {
                continue;
            }
            std::vector<NodeType> nodeVec(nNodes);
            dsNodes->readChunk(chunkIndex, &nodeVec[0]);
            nodes.insert(nodeVec.begin(), nodeVec.end());

            // load edges
            const std::size_t nEdges = dsEdges->getDiscChunkSize(chunkIndex, isVarlen);
            if(nEdges == 0) {
                continue;
            }
            std::vector<NodeType> edgeVec(nEdges);
            dsEdges->readChunk(chunkIndex, &edgeVec[0]);
            for(std::size_t edgeId = 0; edgeId < nEdges / 2; ++edgeId) {
                edges.insert(std::make_pair(edgeVec[2 * edgeId], edgeVec[2 * edgeId + 1]));
            }
        }

    }


    inline void mergeSubgraphsMultiThreaded(const fs::path & graphPath,
                                            const std::string & dsNodePath,
                                            const std::string & dsEdgePath,
                                            const std::vector<size_t> & chunkIds,
                                            NodeSet & nodes,
                                            EdgeSet & edges,
                                            std::vector<size_t> & roiBegin,
                                            std::vector<size_t> & roiEnd,
                                            const int numberOfThreads) {
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        auto nThreads = threadpool.nThreads();

        // initialize thread data
        struct PerThreadData {
            std::vector<std::size_t> roiBegin;
            std::vector<std::size_t> roiEnd;
            NodeSet nodes;
            EdgeSet edges;
        };
        std::vector<PerThreadData> threadData(nThreads);
        const std::size_t maxSizeT = std::numeric_limits<std::size_t>::max();
        for(int t = 0; t < nThreads; ++t) {
            threadData[t].roiBegin = std::vector<size_t>({maxSizeT, maxSizeT, maxSizeT});
            threadData[t].roiEnd = std::vector<size_t>({0, 0, 0});
        }

        const auto dsNodes = z5::openDataset(dsNodePath);
        const auto dsEdges = z5::openDataset(dsEdgePath);
        const auto & chunking = dsNodes->chunking();

        // merge nodes and edges multi threaded
        size_t nChunks = chunkIds.size();
        nifty::parallel::parallel_foreach(threadpool, nChunks, [&](const int tid,
                                                                   const int chunkIndex){

            // get the thread data
            const auto chunkId = chunkIds[chunkIndex];
            // for thread 0, we use the input sets instead of our thread data
            // to avoid one sequential merge in the end
            auto & threadNodes = (tid == 0) ? nodes : threadData[tid].nodes;
            auto & threadEdges = (tid == 0) ? edges : threadData[tid].edges;
            auto & threadBegin = threadData[tid].roiBegin;
            auto & threadEnd = threadData[tid].roiEnd;

            std::vector<std::size_t> chunkCoord;
            chunking.blockIdToBlockCoordinate(chunkId, chunkCoord);

            // merge the rois
            std::vector<std::size_t> blockBegin, blockEnd;
            chunking.getBlockBeginAndEnd(chunkCoord, blockBegin, blockEnd);

            for(int axis = 0; axis < 3; ++axis) {
                threadBegin[axis] = std::min(threadBegin[axis],
                                             static_cast<size_t>(blockBegin[axis]));
                threadEnd[axis] = std::max(threadEnd[axis],
                                           static_cast<size_t>(blockEnd[axis]));
            }

            // load nodes
            bool isVarlen;
            const std::size_t nNodes = dsNodes->getDiscChunkSize(chunkCoord, isVarlen);
            if(nNodes == 0) {
                return;
            }
            std::vector<NodeType> nodeVec(nNodes);
            dsNodes->readChunk(chunkCoord, &nodeVec[0]);
            threadNodes.insert(nodeVec.begin(), nodeVec.end());

            // load edges
            const std::size_t nEdges = dsEdges->getDiscChunkSize(chunkCoord, isVarlen);
            if(nEdges == 0) {
                return;
            }
            std::vector<NodeType> edgeVec(nEdges);
            dsEdges->readChunk(chunkCoord, &edgeVec[0]);
            for(std::size_t edgeId = 0; edgeId < nEdges / 2; ++edgeId) {
                threadEdges.insert(std::make_pair(edgeVec[2 * edgeId], edgeVec[2 * edgeId + 1]));
            }
        });

        // merge into final nodes and edges
        // (note that thread 0 was already used for the input nodes and edges)
        for(int tid = 1; tid < nThreads; ++tid) {
            nodes.insert(threadData[tid].nodes.begin(), threadData[tid].nodes.end());
            edges.insert(threadData[tid].edges.begin(), threadData[tid].edges.end());
        }

        // merge the rois
        for(int tid = 0; tid < nThreads; ++tid) {
            const auto & threadBegin = threadData[tid].roiBegin;
            const auto & threadEnd = threadData[tid].roiEnd;
            for(int axis = 0; axis < 3; ++axis) {
                roiBegin[axis] = std::min(roiBegin[axis],
                                          static_cast<size_t>(threadBegin[axis]));
                roiEnd[axis] = std::max(roiEnd[axis],
                                        static_cast<size_t>(threadEnd[axis]));
            }
        }
    }


    inline void mergeSubgraphs(const std::string & pathToGraph,
                               const std::string & dsNodePath,
                               const std::string & dsEdgePath,
                               const std::vector<size_t> & chunkIds,
                               const std::string & outKey,
                               const bool ignoreLabel,
                               const int numberOfThreads=1) {
        NodeSet nodes;
        EdgeSet edges;

        const std::size_t maxSizeT = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> roiBegin({maxSizeT, maxSizeT, maxSizeT});
        std::vector<std::size_t> roiEnd({0, 0, 0});

        if(numberOfThreads == 1) {
            mergeSubgraphsSingleThreaded(pathToGraph,
                                         dsNodePath, dsEdgePath,
                                         chunkIds,
                                         nodes, edges,
                                         roiBegin, roiEnd);
        } else {
            mergeSubgraphsMultiThreaded(pathToGraph, dsNodePath, dsEdgePath,
                                        chunkIds, nodes, edges,
                                        roiBegin, roiEnd,
                                        numberOfThreads);
        }

        // we can only use compression for
        // big enough blocks (too small chunks will result in zlib error)
        // as a proxy we use the number of threads to determine if we use compression
        std::string compression = (numberOfThreads > 1) ? "gzip" : "raw";
        // serialize the merged graph
        serializeGraph(pathToGraph, outKey,
                       nodes, edges,
                       roiBegin, roiEnd,
                       ignoreLabel,
                       numberOfThreads,
                       compression);
    }


    inline void mapEdgeIds(const std::string & pathToGraph,
                           const std::string & graphKey,
                           const std::string & edgeDsPath,
                           const std::string & outputDsPath,
                           const std::vector<size_t> & chunkIds,
                           const int numberOfThreads=1) {

        const std::vector<size_t> zero1Coord({0});
        // we load the edges into a vector, because
        // it will be sorted by construction and we can take
        // advantage of O(logN) search with std::lower_bound
        std::vector<EdgeType> edges;
        fs::path graphPath(pathToGraph);
        graphPath /= graphKey;
        loadEdges(graphPath.string(), edges, 0);

        // iterate over the chunks and insert the nodes and edges
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        auto nThreads = threadpool.nThreads();
        const std::size_t nChunks = chunkIds.size();

        const auto dsEdges = z5::openDataset(edgeDsPath);
        const auto dsOut = z5::openDataset(outputDsPath);
        const auto & chunking = dsEdges->chunking();

        // handle all the chunks in parallel
        nifty::parallel::parallel_foreach(threadpool, nChunks, [&](const int tid,
                                                                   const int chunkIndex){

            const auto chunkId = chunkIds[chunkIndex];
            std::vector<std::size_t> chunkCoord;
            chunking.blockIdToBlockCoordinate(chunkId, chunkCoord);

            // load the chunk edges
            bool isVarlen;
            const std::size_t nEdges = dsEdges->getDiscChunkSize(chunkCoord, isVarlen);
            if(nEdges == 0) {
                return;
            }
            std::vector<NodeType> edgeVec(nEdges);
            dsEdges->readChunk(chunkCoord, &edgeVec[0]);

            // copy to vector of pairs
            std::vector<EdgeType> chunkEdges(nEdges / 2);
            for(std::size_t edgeId = 0; edgeId < nEdges / 2; ++edgeId) {
                chunkEdges[edgeId] = std::make_pair(edgeVec[2 * edgeId], edgeVec[2 * edgeId + 1]);
            }

            // label the local edges acccording to the global edge ids
            std::vector<EdgeIndexType> edgeIds(chunkEdges.size());

            // find the first local edge in the global edges
            auto edgeIt = std::lower_bound(edges.begin(), edges.end(), chunkEdges[0]);

            // it is guaranteed that all local edges are 'above' the lowest we just found,
            // hence we start searching from this edge, and always try to increase the
            // edge iterator by one before searching again, because edges are likely to be close spatially
            for(EdgeIndexType localEdgeId = 0; localEdgeId < chunkEdges.size(); ++localEdgeId) {
                const EdgeType & edge = *edgeIt;
                if(chunkEdges[localEdgeId] == edge) {
                    edgeIds[localEdgeId] = std::distance(edges.begin(), edgeIt);
                    ++edgeIt;
                } else {
                    edgeIt = std::lower_bound(edgeIt, edges.end(), chunkEdges[localEdgeId]);
                    edgeIds[localEdgeId] = std::distance(edges.begin(), edgeIt);
                }
            }

            // serialize the edge ids
            dsOut->writeChunk(chunkCoord, &edgeIds[0], true, edgeIds.size());
        });
    }


    inline void mapEdgeIds(const std::string & pathToGraph,
                    const std::string & graphKey,
                    const std::string & edgeDsPath,
                    const std::string & outputDsPath,
                    const size_t numberOfBlocks,
                    const int numberOfThreads=1) {
        std::vector<size_t> blockIds(numberOfBlocks);
        std::iota(blockIds.begin(), blockIds.end(), 0);
        mapEdgeIds(pathToGraph, graphKey, edgeDsPath, outputDsPath, blockIds, numberOfThreads);
    }

}
}
