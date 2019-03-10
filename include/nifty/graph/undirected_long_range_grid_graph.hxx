#pragma once

#include <random>
#include <time.h>
#include <thread>
#include <stdlib.h>

#include "xtensor/xexpression.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

#include <cstdlib>
#include <ctime>
#include <iostream>

namespace nifty{
namespace graph{


    template<std::size_t DIM>
    class UndirectedLongRangeGridGraph;

    ///\cond
    namespace detail_graph{

        template<std::size_t DIM>
        class UndirectedLongRangeGridGraphAssign{
        public:
            template<class G,
                    class F>
            static void assign(
                G & graph,
                const xt::xexpression<F> & randomProbsExp,
                const int numberOfThreads

            ){
                const auto & randomProbs = randomProbsExp.derived_cast();
                const auto & shape = graph.shape();
                const auto & offsets = graph.offsets();
                const auto & offsets_probs = graph.offsetsProbs();
                const auto & isLocalOffset = graph.isLocalOffset();
                typename xt::xtensor<bool, 1>::shape_type retshape;
                retshape[0] = graph.numberOfNodes();
                xt::xtensor<int32_t, 1> ret(retshape);
//                const auto randValues = xt::random::rand<double>({shape[0], shape[1], shape[2]});
//                thread_local std::random_device rd;
//                thread_local std::mt19937 gen(rd());
//                thread_local std::uniform_real_distribution<> rng;
//                rng = std::uniform_real_distribution<>(0., 1.);


                nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                nifty::parallel::ThreadPool threadpool(pOpts);
                const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                nifty::tools::parallelForEachCoordinate(threadpool,
                    shape,
                    [&](const auto threadId, const auto & coordP){
                        const auto u = graph.coordianteToNode(coordP);
                        for(int io=0; io<offsets.size(); ++io){
                            const auto offset = offsets[io];
                            const auto coordQ = offset + coordP;
                            // Check if both coordinates are in the volume:
                            if(coordQ.allInsideShape(shape)){
                                const auto v = graph.coordianteToNode(coordQ);

                                // Check if the edge is valid
                                bool isValidEdge = true;
                                if (not isLocalOffset[io]) {
                                    // Check the longRangeStrides:
                                    for(int d=0; d<DIM; ++d){
                                        if (coordP[d] % graph.longRangeStrides()[d] != 0) {
                                            isValidEdge = false;
                                        }
                                    }
                                    // Check if the edge should be added according to the probability:
                                    array::StaticArray<int64_t, DIM + 1> noise_index;
                                    std::copy(std::begin(coordP), std::end(coordP), std::begin(noise_index));
                                    noise_index[DIM] = io;
                                    if (randomProbs[noise_index] >= offsets_probs[io])
                                        isValidEdge = false;
                                }

                                // Insert new edge in graph:
                                if (isValidEdge) {
                                    const auto e = graph.insertEdge(u, v);
                                }
                            }
                        }
                    }
                );
            }
        };
    }
    ///\endcond

    template<std::size_t DIM>
    class UndirectedLongRangeGridGraph
    :   public UndirectedGraph<>
    {
    private:
        typedef detail_graph::UndirectedLongRangeGridGraphAssign<DIM> HelperType;
    public:

        typedef array::StaticArray<int64_t, DIM>    ShapeType;
        typedef array::StaticArray<int64_t, DIM>    StridesType;
        typedef array::StaticArray<int64_t, DIM>    CoordinateType;
        typedef array::StaticArray<int64_t, DIM>    OffsetType;
//        typedef LABELS                              LabelsType;

        typedef std::vector<bool>    BoolVectorType;
        typedef std::vector<float>    OffsetProbsType;
        typedef std::vector<OffsetType>     OffsetVector;

        template<class F>
        UndirectedLongRangeGridGraph(
            const ShapeType &    shape,
            const OffsetVector & offsets,
            const StridesType & longRangeStrides,
            const OffsetProbsType & offsetsProbs,
            const BoolVectorType & isLocalOffset, // Array of 0 an 1 indicating which offsets are local
            const xt::xexpression<F> & randomProbsExp,
            const uint64_t numberOfThreads
        )
        :   UndirectedGraph<>(),
            shape_(shape),
            offsets_(offsets),
            longRangeStrides_(longRangeStrides),
            offsetsProbs_(offsetsProbs),
            isLocalOffset_(isLocalOffset),
            startFromLabelSegm_(false)
        {
            // TODO: here most probably I need to initialize labels_()... The best would be to subclass
            NIFTY_CHECK(DIM==2 || DIM==3,"wrong dimension");

//            typedef typename D::value_type value_type;

            uint64_t nNodes = shape_[0];
            for(int d=1; d<DIM; ++d){
                nNodes *= shape_[d];
            }

            this->assign(nNodes);

            strides_.back() = 1;
            for(int d=int(DIM)-2; d>=0; --d){
                strides_[d] = shape_[d+1] * strides_[d+1];
            }
            HelperType::assign(*this, randomProbsExp, numberOfThreads);
        }

        auto edgeOffsetIndex(
        )const{
            NIFTY_CHECK(not this->startFromLabelSegm_, "Update!!");
            typename xt::xtensor<int32_t, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<int32_t, 1> ret(retshape); 
            uint64_t u = 0;
            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
            auto offsetIndex = 0;
            for(const auto & offset : offsets_){
                    const auto coordQ = offset + coordP;
                    if(coordQ.allInsideShape(shape_)){
                        const auto v = this->coordianteToNode(coordQ);
                        const auto e = this->findEdge(u,v);
                        if (e>=0)
                            ret[e] = offsetIndex;
                    }
                    ++offsetIndex;
                }
                ++u;
            });
            
            return ret;
        }

        template<class D>
        auto findLocalEdges(
            const xt::xexpression<D> & nodeLabelsExp
        )const{
            NIFTY_CHECK(this->startFromLabelSegm_, "Update!!");
            NIFTY_CHECK(DIM==3,"Update");

            typedef typename D::value_type value_type;
            const auto & nodeLabels = nodeLabelsExp.derived_cast();

            const auto & shape = this->shape();
            const auto & offsets = this->offsets();
            const auto & offsets_probs = this->offsetsProbs();
            const auto & isLocalOffset = this->isLocalOffset();


            typename xt::xtensor<bool, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<bool, 1> ret(retshape);

            std::fill(ret.begin(), ret.end(), false);


            for(int p0=0; p0<shape[0]; ++p0)
                for(int p1=0; p1<shape[1]; ++p1)
                    for(int p2=0; p2<shape[2]; ++p2) {
                        const auto labelU = nodeLabels(p0, p1, p2);
                        for (int io = 0; io < offsets.size(); ++io) {
                            const int q0 = p0 + offsets[io][0];
                            const int q1 = p1 + offsets[io][1];
                            const int q2 = p2 + offsets[io][2];
                            const auto labelV = nodeLabels(q0, q1, q2);
                            if (q0 >= 0 && q0 < shape[0] && q1 >= 0 && q1 < shape[1] && q2 >= 0 && q2 < shape[2]) {
                                if (labelU != labelV) {
                                    auto e = this->findEdge(labelU,labelV);
                                    if (e>=0) {
                                        if (isLocalOffset[io]) {
//                                            std::cout << "Offset index: " << io << "\n";
//                                            std::cout << "Labels: " << labelU << " " << labelV << "\n";
//                                            std::cout << p1 << " " << p2<< "\n";
//                                            std::cout << q1 << " " << q2<< "\n";
                                            ret[e] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

            return ret;
        }

        template<class D>
        auto nodeFeatureDiffereces(
            const xt::xexpression<D> & nodeFeaturesExpression
        )const{
            NIFTY_CHECK(not this->startFromLabelSegm_, "Update!!");
            typedef typename D::value_type value_type;
            const auto & nodeFeatures = nodeFeaturesExpression.derived_cast();
            for(auto d=0; d<DIM; ++d){
                NIFTY_CHECK_OP(shape_[d],==,nodeFeatures.shape()[d], "input has wrong shape");
            }
            const auto nFeatures = nodeFeatures.shape()[DIM];

            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<value_type, 1> ret(retshape);

            if(DIM == 2){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    const auto valP = xt::view(nodeFeatures, coordP[0],coordP[1], xt::all());
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){

                            const auto valQ = xt::view(nodeFeatures, coordQ[0],coordQ[1], xt::all());
                            const auto v = this->coordianteToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                    }
                    ++u;
                });
            }
            if(DIM == 3){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    const auto valP = xt::view(nodeFeatures, coordP[0], coordP[1], coordP[2], xt::all());
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){
                            const auto valQ = xt::view(nodeFeatures, coordQ[0], coordQ[1], coordQ[2], xt::all());
                            const auto v = this->coordianteToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                    }
                    ++u;
                });
            }
            return ret;
        }

        template<class D>
        auto nodeFeatureDiffereces2(
            const xt::xexpression<D> & nodeFeaturesExpression
        )const{
            NIFTY_CHECK(not this->startFromLabelSegm_, "Update!!");
            typedef typename D::value_type value_type;
            const auto & nodeFeatures = nodeFeaturesExpression.derived_cast();
            for(auto d=0; d<DIM; ++d){
                NIFTY_CHECK_OP(shape_[d],==,nodeFeatures.shape()[d], "input has wrong shape");
            }
            const auto nFeatures = nodeFeatures.shape()[DIM];

            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<value_type, 1> ret(retshape);

            if(DIM == 2){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    
                    auto offsetIndex=0;
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){

                            const auto valP = xt::view(nodeFeatures, coordP[0],coordP[1],offsetIndex, xt::all());
                            const auto valQ = xt::view(nodeFeatures, coordQ[0],coordQ[1],offsetIndex, xt::all());
                            const auto v = this->coordianteToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                        ++offsetIndex;
                    }
                    ++u;
                });
            }
            if(DIM == 3){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    

                    auto offsetIndex=0;
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){
                            const auto valP = xt::view(nodeFeatures, coordP[0], coordP[1], coordP[2],offsetIndex, xt::all());
                            const auto valQ = xt::view(nodeFeatures, coordQ[0], coordQ[1], coordQ[2],offsetIndex, xt::all());
                            const auto v = this->coordianteToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                        offsetIndex+=1;
                    }
                    ++u;
                });
            }
            return ret;
        }


        template<class D>
        auto nodeValues(
            const xt::xexpression<D> & valuesExpression
        )const {

            NIFTY_CHECK(not this->startFromLabelSegm_, "Update!!");
            typedef typename D::value_type value_type;
            const auto &values = valuesExpression.derived_cast();

            for (auto d = 0; d < DIM; ++d) {
                NIFTY_CHECK_OP(shape_[d], == , values.shape()[d], "input has wrong shape");
            }


            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfNodes();
            xt::xtensor<value_type, 1> ret(retshape);


            nifty::tools::forEachCoordinate(shape_, [&](const auto &coordP) {
                const auto u = this->coordianteToNode(coordP);
                if (DIM == 2) {
                    const auto val = values(coordP[0], coordP[1]);
                    ret[u] = val;
                } else {
                    const auto val = values(coordP[0], coordP[1], coordP[2]);
                    ret[u] = val;
                }
            });

            return ret;
        }

        template<class D>
        auto edgeValues(
                const xt::xexpression<D> & valuesExpression
        )const{
            NIFTY_CHECK(not this->startFromLabelSegm_, "Update!!");
            typedef typename D::value_type value_type;
            const auto & values = valuesExpression.derived_cast();

            for(auto d=0; d<DIM; ++d){
                NIFTY_CHECK_OP(shape_[d],==,values.shape()[d], "input has wrong shape");
            }
            NIFTY_CHECK_OP(offsets_.size(),==,values.shape()[DIM], "input has wrong shape");


            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<value_type, 1> ret(retshape);


            uint64_t u = 0;
            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                auto offsetIndex = 0;
                for(const auto & offset : offsets_){
                    const auto coordQ = offset + coordP;
                    if(coordQ.allInsideShape(shape_)){

                        const auto v = this->coordianteToNode(coordQ);
                        const auto e = this->findEdge(u,v);
                        if (e>=0) {
                            if (DIM == 2) {
                                const auto val = values(coordP[0], coordP[1], offsetIndex);
                                ret[e] = val;
                            } else {
                                const auto val = values(coordP[0], coordP[1], coordP[2], offsetIndex);
                                ret[e] = val;
                            }
                        }
                    }
                    ++offsetIndex;
                }
                ++u;
            });

        return ret;

        }

        auto mapEdgesIDToImage(
        )const{
            NIFTY_CHECK(not this->startFromLabelSegm_, "Update!!");
            typename xt::xtensor<int64_t, DIM+1>::shape_type retshape;
            for(auto d=0; d<DIM; ++d){
                retshape[d] = shape_[d];
            }
            retshape[DIM] = offsets_.size();
            xt::xtensor<int64_t, DIM+1> ret(retshape);


            uint64_t u = 0;
            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                auto offsetIndex = 0;
                for(const auto & offset : offsets_){
                    const auto coordQ = offset + coordP;
                    int64_t e = -1;
                    if(coordQ.allInsideShape(shape_)){
                        const auto v = this->coordianteToNode(coordQ);
                        e = this->findEdge(u,v);
                    }

                    if(DIM == 2){
                        ret(coordP[0],coordP[1], offsetIndex) = e;
                    }
                    else{
                        ret(coordP[0],coordP[1],coordP[2],offsetIndex) = e;
                    }
                    ++offsetIndex;
                }
                ++u;
            });

            return ret;

        }

        auto mapNodesIDToImage(
        )const{
            NIFTY_CHECK(not this->startFromLabelSegm_, "Update!!");
            typename xt::xtensor<uint64_t, DIM>::shape_type retshape;
            for(auto d=0; d<DIM; ++d){
                retshape[d] = shape_[d];
            }
            xt::xtensor<uint64_t, DIM> ret(retshape);


            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                const auto u = this->coordianteToNode(coordP);
                if(DIM == 2){
                    ret(coordP[0],coordP[1]) = u;
                }
                else{
                    ret(coordP[0],coordP[1],coordP[2]) = u;
                }
            });

            return ret;

        }

        // template<class NODE_COORDINATE>
        // void nodeToCoordinate(
        //     const uint64_t node,
        //     NODE_COORDINATE & coordinate
        // )const{
            
        // }

        template<class NODE_COORDINATE>
        uint64_t coordianteToNode(
            const NODE_COORDINATE & coordinate
        )const{
            uint64_t n = 0;
            for(auto d=0; d<DIM; ++d){
                n +=strides_[d]*coordinate[d];
            }
            return n;
        }

        const auto & shape()const{
            return shape_;
        }
        const auto & offsets()const{
            return offsets_;
        }
        const auto & longRangeStrides()const{
            return longRangeStrides_;
        }
        const auto & offsetsProbs()const{
            return offsetsProbs_;
        }
        const auto & isLocalOffset()const{
            return isLocalOffset_;
        }
        const auto & startFromLabelSegm()const{
            return startFromLabelSegm_;
        }
    private:
        ShapeType shape_;
        StridesType strides_;
        StridesType longRangeStrides_;
        OffsetVector offsets_;
        OffsetProbsType offsetsProbs_;
        BoolVectorType isLocalOffset_;
        bool startFromLabelSegm_;

//        typedef std::unique_ptr<LabelsType> StorageType;
//        StorageType labels_;

    };


////    typedef array::StaticArray<int64_t, DIM>    ShapeType;
//    typedef array::StaticArray<int64_t, DIM>    StridesType;
//    typedef array::StaticArray<int64_t, DIM>    CoordinateType;
//    typedef array::StaticArray<int64_t, DIM>    OffsetType;
////        typedef LABELS                              LabelsType;
//
////    typedef std::vector<bool>    BoolVectorType;
//    typedef std::vector<float>    OffsetProbsType;
//    typedef std::vector<OffsetType>     OffsetVector;


    template<std::size_t DIM, class LAB_TYPE, class RAG, class F>
    auto computeLongRangeEdgesFromRagAndOffsets(
            const RAG & rag,
            const std::vector<array::StaticArray<int64_t, DIM>> & offsets,
//            const array::StaticArray<int64_t, DIM> & longRangeStrides,
            const xt::xexpression<LAB_TYPE> & nodeLabelsExp,
            const std::vector<float> & offsetsProbs,
            const xt::xexpression<F> & randomProbsExp,
            const int numberOfThreads
    ) {
        // TODO: node_labels are not necessary...
        const auto & labels = rag.labels();
        const auto & shape = rag.shape();

        NIFTY_CHECK(DIM==2 || DIM==3,"wrong dimension");

//        typedef typename LAB_TYPE::value_type value_type;
        const auto & nodeLabels = nodeLabelsExp.derived_cast();

        for(auto d=0; d<DIM; ++d){
            NIFTY_CHECK_OP(shape[d],==,nodeLabels.shape()[d], "input has wrong shape");
        }

//            uint64_t nNodes = shape_[0];
//            const auto maxValue = xt::amax(nodeLabels);
//
//            nNodes = maxValue(0) + 1;

        const auto & randomProbs = randomProbsExp.derived_cast();

        // Look for longRange edges:
        std::vector<std::vector<std::pair<uint64_t,uint64_t>>> longRangePairs(numberOfThreads);


        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        nifty::tools::parallelForEachCoordinate(threadpool,
                                                shape,
            [&](const auto threadId, const auto & coordP){
                const auto u = labels[coordP];
                for(int io=0; io<offsets.size(); ++io){
                    const auto offset = offsets[io];
                    const auto coordQ = offset + coordP;
                    // Check if both coordinates are in the volume:
                    if(coordQ.allInsideShape(shape)){
                        const auto v = labels[coordQ];

                        // Insert new edge in graph:
                        if (u != v) {
                            const auto edge = rag.findEdge(u, v);
                            if (edge < 0) {
                                // Check if the edge should be added according to the probability:
                                array::StaticArray<int64_t, DIM + 1> noise_index;
                                std::copy(std::begin(coordP), std::end(coordP), std::begin(noise_index));
                                noise_index[DIM] = io;
                                if (randomProbs[noise_index] < offsetsProbs[io]) {
                                    const std::pair <uint64_t, uint64_t> newPair(u, v);
                                    longRangePairs[threadId].push_back(newPair);
                                }
                            }
                        }
                    }
                }
            }
        );



        // Export results:
        int total_nb_edges = 0;
        std::vector<int> index_offset(numberOfThreads);
        for (int i=0; i<numberOfThreads; i++) {
            index_offset[i] = total_nb_edges;
            total_nb_edges += longRangePairs[i].size();
//            std::cout << total_nb_edges <<  "\n";
        }

//        std::cout << "DONE with edges! \n";
        // TODO: test time until here!

        typename xt::xtensor<uint64_t, 2>::shape_type retshape;
        retshape[0] = total_nb_edges;
        retshape[1] = 2;
        xt::xtensor<uint64_t, 2> out(retshape);
        for (int i=0; i<numberOfThreads; i++) {
            for (int j=0; j<longRangePairs[i].size(); j++) {
                out(index_offset[i]+j,0) = longRangePairs[i][j].first;
                out(index_offset[i]+j,1) = longRangePairs[i][j].second;
            }
        }

        return out;

    }


    }
}