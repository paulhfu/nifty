#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <algorithm>
#include <iostream>

//#include "xtensor/xexpression.hpp"
//#include "xtensor/xview.hpp"
//#include "xtensor/xmath.hpp"
//#include "xtensor-python/pyarray.hpp"
//#include "xtensor-python/pytensor.hpp"
//#include "xtensor-python/pyvectorize.hpp"

#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/parallel/threadpool.hxx"

#include "nifty/python/converter.hxx"


#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"
#include "nifty/xtensor/xtensor.hxx"


#ifdef WITH_HDF5

#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#endif


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/ufd/ufd.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

#include "nifty/external/opensimplex_noise.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;

    template<std::size_t DIM>
    void exportEvaluateSimplexNoiseOnArrayT(
            py::module & ragModule
    ) {
        ragModule.def("evaluateSimplexNoiseOnArray_impl",
                           [](
                                   std::array<int, DIM>    shape,
                                   int64_t           seed,
                                   const xt::pytensor<double, 1> & featureSize,
                                   int numberOfThreads
                           ) {
//                               FIXME: make the version in external to work and move!
                               typedef typename array::StaticArray<int64_t, DIM> ShapeType;
                               ShapeType s;
                               std::copy(shape.begin(), shape.end(), s.begin());

                              typename xt::pytensor<double, DIM>::shape_type shape_tensor;
                               for(int d = 0; d < DIM; ++d) {
                                   shape_tensor[d] = shape[d];
                               }

                              xt::pytensor<double, DIM> outArray(shape_tensor);

                               {
                                   py::gil_scoped_release allowThreads;
                                   nifty::external::evaluateSimplexNoiseOnArray(seed, s, featureSize, outArray, numberOfThreads);
                               }
                               return outArray;

                           },
                           py::arg("shape"),
                           py::arg("seed"),
                           py::arg("featureSize"),
                           py::arg("numberOfThreads") = -1
        );

    }


    /*
    template<std::size_t DIM, class RAG, class CONTR_GRAP, class DATA_T>
    void exportAccumulateAffinitiesMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateAffinitiesMeanAndLength",
        [](
            const RAG & rag,
//            xt::xexpression<DATA_T> & affinitiesExpression,
//            xt::xexpression<DATA_T> & offsetsExpression

//            xt::pytensor<DATA_T, DIM+1> affinities,
//            xt::pytensor<int, 2> offsets
            nifty::marray::PyView<DATA_T, DIM+1> affinities,
            nifty::marray::PyView<int, 2>      offsets,
            nifty::marray::PyView<DATA_T, DIM+1> affinities_weigths,
            const int numberOfThreads
        ){
            NIFTY_CHECK(false,"Max not implemented. And at some point I said there was a bug (not present in the version with undirected graph)");
            // Inputs:
//            typedef typename DATA_T::value_type value_type;
//            auto & affinities = affinitiesExpression.derived_cast();
//            auto & offsets = offsetsExpression.derived_cast();

            const auto & labels = rag.labels();
            const auto & shape = rag.shape();

            // Check inputs:
            for(auto d=0; d<DIM; ++d){
                NIFTY_CHECK_OP(shape[d],==,affinities.shape(d), "affinities have wrong shape");
            }
            NIFTY_CHECK_OP(offsets.shape(0),==,affinities.shape(DIM), "Affinities and offsets do not match");
            NIFTY_CHECK_OP(offsets.shape(0),==,affinities_weigths.shape(0), "Affinities weights and offsets do not match");

            // Create outputs:
//            typename xt::xtensor<DATA_T, 1>::shape_type retshape;
//            retshape[0] = uint64_t(rag.edgeIdUpperBound()+1);
//            typedef xt::xtensor<DATA_T, 1> XTensorType;
//            typedef std::pair<XTensorType, XTensorType>  OutType;
//            XTensorType accAff(retshape);
//            XTensorType counter(retshape);
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;

            std::array<int,2> shapeRetArray;
            shapeRetArray[0] = numberOfThreads;
            shapeRetArray[1] = uint64_t(rag.edgeIdUpperBound()+1);

            NumpyArrayType accAff(shapeRetArray.begin(), shapeRetArray.end());
            NumpyArrayType counter(shapeRetArray.begin(), shapeRetArray.end());

            std::fill(accAff.begin(), accAff.end(), 0.);
            std::fill(counter.begin(), counter.end(), 0.);

            {
                py::gil_scoped_release allowThreads;

                // Create thread pool:
                nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                nifty::parallel::ThreadPool threadpool(pOpts);
                const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                if(DIM == 3){
                    nifty::tools::parallelForEachCoordinate(threadpool,
                            shape,
                            [&](const auto threadId, const auto & coordP){
                        const auto u = labels(coordP[0],coordP[1],coordP[2]);
                        for(auto i=0; i<offsets.shape(0); ++i){
                            auto coordQ = coordP;
                            coordQ[0] += offsets(i, 0);
                            coordQ[1] += offsets(i, 1);
                            coordQ[2] += offsets(i, 2);
                            if(coordQ.allInsideShape(shape)){
                                const auto v = labels(coordQ[0],coordQ[1],coordQ[2]);
                                if(u != v){
                                    const auto edge = rag.findEdge(u,v);
                                    if(edge >=0 ){
                                        counter(threadId,edge) += affinities_weigths(i);
                                        // accAff(edge) = 0.;
                                        accAff(threadId,edge) += affinities(coordP[0],coordP[1],coordP[2],i)*affinities_weigths(i);
                                    }
                                }
                            }
                        }
                    });
                }

            }

            NumpyArrayType accAff_out({uint64_t(rag.edgeIdUpperBound()+1)});
            NumpyArrayType counter_out({uint64_t(rag.edgeIdUpperBound()+1)});

            // Normalize:
            for(auto i=0; i<uint64_t(rag.edgeIdUpperBound()+1); ++i){
                for(auto thr=1; thr<numberOfThreads; ++thr){
                    counter(0,i) += counter(thr,i);
                    accAff(0,i) += accAff(thr,i);
                }
                if(counter(0,i)>0.5){
                    accAff_out(i) = accAff(0,i) / counter(0,i);
                    counter_out(i) = counter(0,i);
                } else {
                    accAff_out(i) = 0.;
                    counter_out(i) = 0.;
                }
            }
            return OutType(accAff_out, counter_out);;

        },
        py::arg("rag"),
        py::arg("affinities"),
        py::arg("offsets"),
                      py::arg("affinitiesWeights"),
        py::arg("numberOfThreads")  = 8
        );



        ragModule.def("accumulateAffinitiesMeanAndLength",
        [](
            const RAG & rag,
            const CONTR_GRAP & contrGraph,
            nifty::marray::PyView<DATA_T, DIM+1> affinities,
            nifty::marray::PyView<int, 2>      offsets
        ){


            const auto & labels = rag.labels();
            const auto & shape = rag.shape();

            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;

            NumpyArrayType accAff({uint64_t(rag.edgeIdUpperBound()+1)});

            // std::vector<size_t> counter(uint64_t(rag.edgeIdUpperBound()+1), 0);
            NumpyArrayType counter({uint64_t(rag.edgeIdUpperBound()+1)});

            std::fill(accAff.begin(), accAff.end(), 0);
            std::fill(counter.begin(), counter.end(), 0);

            for(auto x=0; x<shape[0]; ++x){
                for(auto y=0; y<shape[1]; ++y){
                    if (DIM==3){
                        for(auto z=0; z<shape[2]; ++z){

                            const auto u = labels(x,y,z);

                            for(auto i=0; i<offsets.shape(0); ++i){
                                const auto ox = offsets(i, 0);
                                const auto oy = offsets(i, 1);
                                const auto oz = offsets(i, 2);
                                const auto xx = ox +x ;
                                const auto yy = oy +y ;
                                const auto zz = oz +z ;


                                if(xx>=0 && xx<shape[0] && yy >=0 && yy<shape[1] && zz >=0 && zz<shape[2]){
                                    const auto v = labels(xx,yy,zz);
                                    if(u != v){
                                        const auto edge = rag.findEdge(u,v);
                                        if(edge >=0 ){
                                            const auto cEdge = contrGraph.findRepresentativeEdge(edge);
                                            counter(cEdge) += 1.;
                                            accAff(cEdge) += affinities(x,y,z,i);
                                        }
                                    }
                                }
                            }
                        }
                    } else if(DIM==2) {
                        const auto u = labels(x,y);

                        for(auto i=0; i<offsets.shape(0); ++i){
                            const auto ox = offsets(i, 0);
                            const auto oy = offsets(i, 1);

                            const auto xx = ox +x ;
                            const auto yy = oy +y ;

                            if(xx>=0 && xx<shape[0] && yy >=0 && yy<shape[1]){
                                const auto v = labels(xx,yy);
                                if(u != v){
                                    const auto edge = rag.findEdge(u,v);
                                    if(edge >=0 ){
                                        const auto cEdge = contrGraph.findRepresentativeEdge(edge);
                                        counter(cEdge) +=1.;
                                        accAff(cEdge) += affinities(x,y,i);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Normalize:
            for(auto i=0; i<uint64_t(rag.edgeIdUpperBound()+1); ++i){
                if(counter(i)>0.5){
                    accAff(i) /= counter(i);
                }
            }
            return OutType(accAff, counter);;

        },
        py::arg("rag"),
        py::arg("contrGraph"),
        py::arg("affinities"),
        py::arg("offsets")
        );
    }
    */


    /*
    template<std::size_t DIM, class GRAPH, class DATA_T>
    void exportAccumulateAffinitiesMeanAndLength(
            py::module & ragModule
    ) {
        ragModule.def("accumulateAffinitiesMeanAndLength",
                      [](
                              const GRAPH &graph,
                              nifty::marray::PyView<int, DIM> labels, // LAbels less then zero are ignored
                              nifty::marray::PyView<DATA_T, DIM + 1> affinities,
                              nifty::marray::PyView<int, 2> offsets,
                              nifty::marray::PyView<DATA_T, 1> affinities_weigths,
                              const int numberOfThreads
                      ) {
                          array::StaticArray<int64_t, DIM> shape;

//                          std::array<int,DIM> shape;
                          // Check inputs:
                          for(auto d=0; d<DIM; ++d){
                              shape[d] = labels.shape(d);
                              NIFTY_CHECK_OP(shape[d],==,affinities.shape(d), "affinities have wrong shape");
                          }
                          NIFTY_CHECK_OP(offsets.shape(0),==,affinities.shape(DIM), "Affinities and offsets do not match");
                          NIFTY_CHECK_OP(offsets.shape(0),==,affinities_weigths.shape(0), "Affinities weights and offsets do not match");

                          // Create outputs:
                          typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
                          typedef std::tuple<NumpyArrayType, NumpyArrayType, NumpyArrayType>  OutType;

                          std::array<int,2> shapeRetArray;
                          shapeRetArray[0] = numberOfThreads;
                          shapeRetArray[1] = uint64_t(graph.edgeIdUpperBound()+1);

                          NumpyArrayType accAff(shapeRetArray.begin(), shapeRetArray.end());
                          NumpyArrayType maxAff(shapeRetArray.begin(), shapeRetArray.end());
                          NumpyArrayType counter(shapeRetArray.begin(), shapeRetArray.end());

                          std::fill(accAff.begin(), accAff.end(), 0.);
                          std::fill(maxAff.begin(), maxAff.end(), 0.);
                          std::fill(counter.begin(), counter.end(), 0.);

                          {
                              py::gil_scoped_release allowThreads;

                              // Create thread pool:
                              nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                              nifty::parallel::ThreadPool threadpool(pOpts);
                              const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                              if(DIM == 3){
                                  nifty::tools::parallelForEachCoordinate(threadpool,
                                                                          shape,
                                                                          [&](const auto threadId, const auto & coordP){
                                                                              const auto u = labels(coordP[0],coordP[1],coordP[2]);
                                                                              for(auto i=0; i<offsets.shape(0); ++i){
                                                                                  auto coordQ = coordP;
                                                                                  coordQ[0] += offsets(i, 0);
                                                                                  coordQ[1] += offsets(i, 1);
                                                                                  coordQ[2] += offsets(i, 2);
                                                                                  if(coordQ.allInsideShape(shape)){
                                                                                      const auto v = labels(coordQ[0],coordQ[1],coordQ[2]);
                                                                                      if(u != v && u > 0 && v > 0){
                                                                                          const auto edge = graph.findEdge(u,v);
                                                                                          if (edge >=0 ){
                                                                                              const auto aff_value = affinities(coordP[0],coordP[1],coordP[2],i);
                                                                                              if (aff_value > maxAff(threadId, edge))
                                                                                                  maxAff(threadId, edge) = aff_value;
                                                                                              counter(threadId,edge) += affinities_weigths(i);
                                                                                              // accAff(edge) = 0.;
                                                                                              accAff(threadId,edge) += aff_value*affinities_weigths(i);
                                                                                          }
                                                                                      }
                                                                                  }
                                                                              }
                                                                          });
                              }

                          }

                          NumpyArrayType accAff_out({uint64_t(graph.edgeIdUpperBound()+1)});
                          NumpyArrayType counter_out({uint64_t(graph.edgeIdUpperBound()+1)});
                          NumpyArrayType maxAff_out({uint64_t(graph.edgeIdUpperBound()+1)});

                          // Normalize:
                          for(auto i=0; i<uint64_t(graph.edgeIdUpperBound()+1); ++i){
                              maxAff_out(i) = maxAff(0,i);
                              for(auto thr=1; thr<numberOfThreads; ++thr){
                                  counter(0,i) += counter(thr,i);
                                  accAff(0,i) += accAff(thr,i);
                                  if (maxAff(thr,i) > maxAff_out(i))
                                      maxAff_out(i) = maxAff(thr,i);
                              }
                              if(counter(0,i)>0.5){
                                  accAff_out(i) = accAff(0,i) / counter(0,i);
                                  counter_out(i) = counter(0,i);
                              } else {
                                  accAff_out(i) = 0.;
                                  counter_out(i) = 0.;
                              }
                          }
                          return OutType(accAff_out, counter_out, maxAff_out);;


                      },
                      py::arg("graph"),
                      py::arg("labels"),
                      py::arg("affinities"),
                      py::arg("offsets"),
                      py::arg("affinitiesWeights"),
                      py::arg("numberOfThreads")  = 8

        );
    }

    template<std::size_t DIM, class GRAPH, class DATA_T>
    void exportAccumulateAffinitiesMeanAndLengthOnNodes(
            py::module & ragModule
    ) {
        ragModule.def("accumulateAffinitiesMeanAndLengthOnNodes",
                      [](
                              const GRAPH &graph,
                              nifty::marray::PyView<int, DIM> labels,
                              nifty::marray::PyView<DATA_T, DIM + 1> affinities,
                              nifty::marray::PyView<int, 2> offsets,
                              nifty::marray::PyView<DATA_T, 1> affinities_weigths,
                              const int numberOfThreads
                      ) {
                          array::StaticArray<int64_t, DIM> shape;

//                          std::array<int,DIM> shape;
                          // Check inputs:
                          for(auto d=0; d<DIM; ++d){
                              shape[d] = labels.shape(d);
                              NIFTY_CHECK_OP(shape[d],==,affinities.shape(d), "affinities have wrong shape");
                          }
                          NIFTY_CHECK_OP(offsets.shape(0),==,affinities.shape(DIM), "Affinities and offsets do not match");
                          NIFTY_CHECK_OP(offsets.shape(0),==,affinities_weigths.shape(0), "Affinities weights and offsets do not match");

                          // Create outputs:
                          typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
                          typedef std::tuple<NumpyArrayType, NumpyArrayType, NumpyArrayType>  OutType;

                          // TODO: try to make it more memory and computational efficient
                          // filling is slow (and the more threads, the more memory)
                          // Same for collecting the results afterwards

                          std::array<int,2> shapeRetArray;
                          shapeRetArray[0] = numberOfThreads;
                          shapeRetArray[1] = uint64_t(graph.nodeIdUpperBound()+1);

                          NumpyArrayType accAff(shapeRetArray.begin(), shapeRetArray.end());
                          NumpyArrayType maxAff(shapeRetArray.begin(), shapeRetArray.end());
                          NumpyArrayType counter(shapeRetArray.begin(), shapeRetArray.end());

                          std::fill(accAff.begin(), accAff.end(), 0.);
                          std::fill(maxAff.begin(), maxAff.end(), 0.);
                          std::fill(counter.begin(), counter.end(), 0.);

                          {
                              py::gil_scoped_release allowThreads;

                              // Create thread pool:
                              nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                              nifty::parallel::ThreadPool threadpool(pOpts);
                              const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                              if(DIM == 3){
                                  nifty::tools::parallelForEachCoordinate(threadpool,
                                                                          shape,
                                                                          [&](const auto threadId, const auto & coordP){
                                                                              const auto u = labels(coordP[0],coordP[1],coordP[2]);
                                                                              for(auto i=0; i<offsets.shape(0); ++i){
                                                                                  auto coordQ = coordP;
                                                                                  coordQ[0] += offsets(i, 0);
                                                                                  coordQ[1] += offsets(i, 1);
                                                                                  coordQ[2] += offsets(i, 2);
                                                                                  if(coordQ.allInsideShape(shape)){
                                                                                      const auto v = labels(coordQ[0],coordQ[1],coordQ[2]);
                                                                                      if(u == v){
                                                                                          const auto aff_value = affinities(coordP[0],coordP[1],coordP[2],i);
                                                                                          if (aff_value > maxAff(threadId, u))
                                                                                              maxAff(threadId, u) = aff_value;
                                                                                          counter(threadId,u) += affinities_weigths(i);
                                                                                          // accAff(edge) = 0.;
                                                                                          accAff(threadId,u) += aff_value*affinities_weigths(i);
                                                                                      }
                                                                                  }
                                                                              }
                                                                          });
                              }

                          }

                          NumpyArrayType accAff_out({uint64_t(graph.nodeIdUpperBound()+1)});
                          NumpyArrayType counter_out({uint64_t(graph.nodeIdUpperBound()+1)});
                          NumpyArrayType maxAff_out({uint64_t(graph.nodeIdUpperBound()+1)});

                          // Normalize:
                          for(auto i=0; i<uint64_t(graph.nodeIdUpperBound()+1); ++i){
                              maxAff_out(i) = maxAff(0,i);
                              for(auto thr=1; thr<numberOfThreads; ++thr){
                                  counter(0,i) += counter(thr,i);
                                  accAff(0,i) += accAff(thr,i);
                                  if (maxAff(thr,i) > maxAff_out(i))
                                      maxAff_out(i) = maxAff(thr,i);
                              }
                              if(counter(0,i)>0.5){
                                  accAff_out(i) = accAff(0,i) / counter(0,i);
                                  counter_out(i) = counter(0,i);
                              } else {
                                  accAff_out(i) = 0.;
                                  counter_out(i) = 0.;
                              }
                          }
                          return OutType(accAff_out, counter_out, maxAff_out);;


                      },
                      py::arg("graph"),
                      py::arg("labels"),
                      py::arg("affinities"),
                      py::arg("offsets"),
                      py::arg("affinitiesWeights"),
                      py::arg("numberOfThreads")  = 8

        );
    }
    */




//    template<std::size_t DIM, class DATA_T>
//    void exportGetUvIdsLongRangeRAG(
//            py::module & ragModule
//    ) {
//        ragModule.def("getUvIdsLongRangeRAG",
//                      [](
//                              xt::pytensor<uint64_t, DIM> labels,
//                              xt::pytensor<float, DIM + 1> affinities,
//                              xt::pytensor<int64_t, 2> offsets,
//                              xt::pytensor<uint64_t, 1> downscaling_factors,
//                              nifty::marray::PyView<DATA_T, 1> affinities_weigths,
//                              const int numberOfThreads
//                      ) {
//                          array::StaticArray<int64_t, DIM> shape;
//
////                          std::array<int,DIM> shape;
//                          // Check inputs:
//                          for(auto d=0; d<DIM; ++d){
//                              shape[d] = labels.shape(d);
//                              NIFTY_CHECK_OP(shape[d],==,affinities.shape(d), "affinities have wrong shape");
//                          }
//                          NIFTY_CHECK_OP(offsets.shape(0),==,affinities.shape(DIM), "Affinities and offsets do not match");
//                          NIFTY_CHECK_OP(offsets.shape(0),==,affinities_weigths.shape(0), "Affinities weights and offsets do not match");
//
//                          // Create outputs:
//                          typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
//                          typedef std::tuple<NumpyArrayType, NumpyArrayType, NumpyArrayType>  OutType;
//
//                          std::array<int,2> shapeRetArray;
//                          shapeRetArray[0] = numberOfThreads;
//                          shapeRetArray[1] = uint64_t(graph.edgeIdUpperBound()+1);
//
//                          NumpyArrayType accAff(shapeRetArray.begin(), shapeRetArray.end());
//                          NumpyArrayType maxAff(shapeRetArray.begin(), shapeRetArray.end());
//                          NumpyArrayType counter(shapeRetArray.begin(), shapeRetArray.end());
//
//                          std::fill(accAff.begin(), accAff.end(), 0.);
//                          std::fill(maxAff.begin(), maxAff.end(), 0.);
//                          std::fill(counter.begin(), counter.end(), 0.);
//
//                          {
//                              py::gil_scoped_release allowThreads;
//
//                              // Create thread pool:
//                              nifty::parallel::ParallelOptions pOpts(numberOfThreads);
//                              nifty::parallel::ThreadPool threadpool(pOpts);
//                              const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();
//
//                              if(DIM == 3){
//                                  nifty::tools::parallelForEachCoordinate(threadpool,
//                                                                          shape,
//                                                                          [&](const auto threadId, const auto & coordP){
//                                                                              const auto u = labels(coordP[0],coordP[1],coordP[2]);
//                                                                              for(auto i=0; i<offsets.shape(0); ++i){
//                                                                                  auto coordQ = coordP;
//                                                                                  coordQ[0] += offsets(i, 0);
//                                                                                  coordQ[1] += offsets(i, 1);
//                                                                                  coordQ[2] += offsets(i, 2);
//                                                                                  if(coordQ.allInsideShape(shape)){
//                                                                                      const auto v = labels(coordQ[0],coordQ[1],coordQ[2]);
//                                                                                      if(u != v){
//                                                                                          const auto edge = graph.findEdge(u,v);
//                                                                                          if (edge >=0 ){
//                                                                                              const auto aff_value = affinities(coordP[0],coordP[1],coordP[2],i);
//                                                                                              if (aff_value > maxAff(threadId, edge))
//                                                                                                  maxAff(threadId, edge) = aff_value;
//                                                                                              counter(threadId,edge) += affinities_weigths(i);
//                                                                                              // accAff(edge) = 0.;
//                                                                                              accAff(threadId,edge) += aff_value*affinities_weigths(i);
//                                                                                          }
//                                                                                      }
//                                                                                  }
//                                                                              }
//                                                                          });
//                              }
//
//                          }
//
//                          NumpyArrayType accAff_out({uint64_t(graph.edgeIdUpperBound()+1)});
//                          NumpyArrayType counter_out({uint64_t(graph.edgeIdUpperBound()+1)});
//                          NumpyArrayType maxAff_out({uint64_t(graph.edgeIdUpperBound()+1)});
//
//                          // Normalize:
//                          for(auto i=0; i<uint64_t(graph.edgeIdUpperBound()+1); ++i){
//                              maxAff_out(i) = maxAff(0,i);
//                              for(auto thr=1; thr<numberOfThreads; ++thr){
//                                  counter(0,i) += counter(thr,i);
//                                  accAff(0,i) += accAff(thr,i);
//                                  if (maxAff(thr,i) > maxAff_out(i))
//                                      maxAff_out(i) = maxAff(thr,i);
//                              }
//                              if(counter(0,i)>0.5){
//                                  accAff_out(i) = accAff(0,i) / counter(0,i);
//                                  counter_out(i) = counter(0,i);
//                              } else {
//                                  accAff_out(i) = 0.;
//                                  counter_out(i) = 0.;
//                              }
//                          }
//                          return OutType(accAff_out, counter_out, maxAff_out);;
//
//
//                      },
//                      py::arg("graph"),
//                      py::arg("labels"),
//                      py::arg("affinities"),
//                      py::arg("offsets"),
//                      py::arg("affinitiesWeights"),
//                      py::arg("numberOfThreads")  = 8
//
//        );
//    }



    /*
    template<std::size_t DIM, class DATA_T>
    void exportConnectedComponentsFromEdgeLabels(
            py::module & ragModule
    ) {
        ragModule.def("connectedComponentsFromEdgeLabels",
                      [](
                              nifty::marray::PyView<int, 1> shape_,
                              nifty::marray::PyView<int, 2> offsets,
                              nifty::marray::PyView<DATA_T, DIM + 1> edgeLabels,
                              const int numberOfThreads
                      ) {
                          NIFTY_CHECK_OP(DIM,==,3, "Connected compmponents only implemented in 3D");
                          typedef array::StaticArray<int64_t, DIM>    StridesType;
                          StridesType strides_;
                          strides_.back() = 1;
                          for(int d=int(DIM)-2; d>=0; --d){
                              strides_[d] = shape_[d+1] * strides_[d+1];
                          }

                          // Check inputs:
                          array::StaticArray<int64_t, DIM> shape;
                          uint64_t nb_nodes = 1;
                          for(auto d=0; d<DIM; ++d){
                              shape[d] = shape_(d);
                              NIFTY_CHECK_OP(shape[d],==,edgeLabels.shape(d), "EdgeLabels have wrong shape");
                              nb_nodes *= shape[d];
                          }
                          NIFTY_CHECK_OP(offsets.shape(0),==,edgeLabels.shape(DIM), "Affinities and offsets do not match");

                          // Create output nodeLabels:
                          typedef nifty::marray::PyView<DATA_T> NumpyArrayType;

                          std::array<int,DIM> shapeNodeLabels;
                          std::copy(shape.begin(), shape.end(), shapeNodeLabels.begin());

                          NumpyArrayType nodeLabels(shapeNodeLabels.begin(), shapeNodeLabels.end());

                          std::fill(nodeLabels.begin(), nodeLabels.end(), 0);
                          std::cout << "Tick 0";
                          // Create UnionFind:
                          typedef nifty::ufd::Ufd< > NodeUfdType;
                          NodeUfdType nodeUfd_(nb_nodes+1);

                          std::cout << nodeUfd_.find(0) << "Node 1: " << nb_nodes+1 << "\n";
                          std::cout << "Tick 1";
                          if(DIM == 3) {
                              nifty::tools::forEachCoordinate(shape, [&](const auto &coordP) {
                                  // Find u-node label:
                                  uint64_t u = 0;
                                  for(auto d=0; d<DIM; ++d){
                                      u +=strides_[d]*coordP[d];
                                  }

                                  for (auto i = 0; i < offsets.shape(0); ++i) {
                                      if (edgeLabels(coordP[0], coordP[1], coordP[2], i) == 0) {
                                          auto coordQ = coordP;
                                          coordQ[0] += offsets(i, 0);
                                          coordQ[1] += offsets(i, 1);
                                          coordQ[2] += offsets(i, 2);
                                          if (coordQ.allInsideShape(shape)) {
                                              // Find v-node label:
                                              uint64_t v = 0;
                                              for(auto d=0; d<DIM; ++d){
                                                  v +=strides_[d]*coordQ[d];
                                              }
                                              if (nodeUfd_.find(u) != nodeUfd_.find(v)) {
//                                                  std::cout << ".";
                                                  nodeUfd_.merge(u, v);
                                              }
                                          }
                                      }
                                  }
                              });
                          }
                           std::cout << "Tick 2";
                          // Map back the node labels to image:
                          {
                              py::gil_scoped_release allowThreads;

                              // Create thread pool:
                              nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                              nifty::parallel::ThreadPool threadpool(pOpts);
                              const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                              if(DIM == 3){
                                  nifty::tools::parallelForEachCoordinate(threadpool,
                                                                          shape,
                                                                          [&](const auto threadId, const auto & coordP){
                                                                              // Find u-node label:
                                                                              uint64_t u = 0;
                                                                              for(auto d=0; d<DIM; ++d){
                                                                                  u +=strides_[d]*coordP[d];
                                                                              }
                                                                              nodeLabels(coordP[0],coordP[1],coordP[2]) = nodeUfd_.find(u);
                                                                          });
                              }

                          }

                          return nodeLabels;


                      },
                      py::arg("shape"),
                      py::arg("offsets"),
                      py::arg("edgeLabels"),
                      py::arg("numberOfThreads")  = 8

        );
    }



    template<std::size_t DIM, class RAG, class CONTR_GRAP, class DATA_T>
    void exportMapFeaturesToBoundaries(
        py::module & ragModule
    ){
        ragModule.def("mapFeaturesToBoundaries",
        [](
            const RAG & rag,
            const CONTR_GRAP & contrGraph,
            nifty::marray::PyView<DATA_T, 2> edgeFeatures,
            nifty::marray::PyView<int, 2> offsets,
            double fillValue
        ){


            const auto & labels = rag.labels();
            const auto & shape = rag.shape();

            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;

            std::array<int,DIM+2> shapeFeatureImage;
            std::copy(shape.begin(), shape.end(), shapeFeatureImage.begin());
            shapeFeatureImage[DIM] = offsets.shape(0);
            shapeFeatureImage[DIM+1] = edgeFeatures.shape(1);

            NumpyArrayType featureImage(shapeFeatureImage.begin(), shapeFeatureImage.end());

            std::fill(featureImage.begin(), featureImage.end(), fillValue);


            for(auto x=0; x<shape[0]; ++x){
                for(auto y=0; y<shape[1]; ++y){
                    if (DIM==3){
                        for(auto z=0; z<shape[2]; ++z){

                            const auto u = labels(x,y,z);

                            for(auto i=0; i<offsets.shape(0); ++i){
                                const auto ox = offsets(i, 0);
                                const auto oy = offsets(i, 1);
                                const auto oz = offsets(i, 2);
                                const auto xx = ox +x ;
                                const auto yy = oy +y ;
                                const auto zz = oz +z ;


                                if(xx>=0 && xx<shape[0] && yy >=0 && yy<shape[1] && zz >=0 && zz<shape[2]){
                                    const auto v = labels(xx,yy,zz);
                                    if(u != v){
                                        auto edge = rag.findEdge(u,v);
                                        if(edge >=0 ){
                                            edge = contrGraph.findRepresentativeEdge(edge);
                                            for(auto f=0; f<edgeFeatures.shape(1); ++f){
                                                featureImage(x,y,z,i,f) = edgeFeatures(edge,f);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if(DIM==2) {
                        const auto u = labels(x,y);

                        for(auto i=0; i<offsets.shape(0); ++i){
                            const auto ox = offsets(i, 0);
                            const auto oy = offsets(i, 1);

                            const auto xx = ox +x ;
                            const auto yy = oy +y ;

                            if(xx>=0 && xx<shape[0] && yy >=0 && yy<shape[1]){
                                const auto v = labels(xx,yy);
                                if(u != v){
                                    auto edge = rag.findEdge(u,v);
                                    if(edge >=0 ){
                                        edge = contrGraph.findRepresentativeEdge(edge);
                                        for(auto f=0; f<edgeFeatures.shape(1); ++f){
                                            featureImage(x,y,i,f) = edgeFeatures(edge,f);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return featureImage;

        },
        py::arg("rag"),
        py::arg("contrGraph"),
        py::arg("edgeFeatures"),
        py::arg("offsets"),
        py::arg("fillValue") = 0.
        );


        ragModule.def("mapFeaturesToBoundaries",
        [](
            const RAG & rag,
            nifty::marray::PyView<DATA_T, 2> edgeFeatures,
            nifty::marray::PyView<int, 2> offsets,
            float fillValue,
            const int numberOfThreads
        ){

            // TODO: add nifty check DIM == 2 or DIM == 3

            const auto & labels = rag.labels();
            const auto & shape = rag.shape();

            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;

            std::array<int,DIM+2> shapeFeatureImage;
            std::copy(shape.begin(), shape.end(), shapeFeatureImage.begin());
            shapeFeatureImage[DIM] = offsets.shape(0);
            shapeFeatureImage[DIM+1] = edgeFeatures.shape(1);

            NumpyArrayType featureImage(shapeFeatureImage.begin(), shapeFeatureImage.end());

            std::fill(featureImage.begin(), featureImage.end(), fillValue);


            {
                py::gil_scoped_release allowThreads;

                // Create thread pool:
                nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                nifty::parallel::ThreadPool threadpool(pOpts);
                const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                if(DIM == 3){
                    nifty::tools::parallelForEachCoordinate(threadpool,
                                                            shape,
                                                            [&](const auto threadId, const auto & coordP){
                                                                const auto u = labels(coordP[0],coordP[1],coordP[2]);
                                                                for(auto i=0; i<offsets.shape(0); ++i){
                                                                    auto coordQ = coordP;
                                                                    coordQ[0] += offsets(i, 0);
                                                                    coordQ[1] += offsets(i, 1);
                                                                    coordQ[2] += offsets(i, 2);
                                                                    if(coordQ.allInsideShape(shape)){
                                                                        const auto v = labels(coordQ[0],coordQ[1],coordQ[2]);
                                                                        if(u != v){
                                                                            const auto edge = rag.findEdge(u,v);
                                                                            if(edge >=0 ){
                                                                                for(auto f=0; f<edgeFeatures.shape(1); ++f){
                                                                                    featureImage(coordP[0],coordP[1],coordP[2],i,f) = edgeFeatures(edge,f);
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            });
                } else if(DIM == 2){
                    nifty::tools::parallelForEachCoordinate(threadpool,
                                                            shape,
                                                            [&](const auto threadId, const auto & coordP){
                                                                const auto u = labels(coordP[0],coordP[1]);
                                                                for(auto i=0; i<offsets.shape(0); ++i){
                                                                    auto coordQ = coordP;
                                                                    coordQ[0] += offsets(i, 0);
                                                                    coordQ[1] += offsets(i, 1);
                                                                    if(coordQ.allInsideShape(shape)){
                                                                        const auto v = labels(coordQ[0],coordQ[1]);
                                                                        if(u != v){
                                                                            const auto edge = rag.findEdge(u,v);
                                                                            if(edge >=0 ){
                                                                                for(auto f=0; f<edgeFeatures.shape(1); ++f){
                                                                                    featureImage(coordP[0],coordP[1],i,f) = edgeFeatures(edge,f);
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            });
                }

            }

            return featureImage;

        },
        py::arg("rag"),
        py::arg("edgeFeatures"),
        py::arg("offsets"),
        py::arg("fillValue") = 0.,
                      py::arg("numberOfThreads") = 8
        );
    }



    template<std::size_t DIM, class RAG, class CONTR_GRAP, class DATA_T>
    void exportBoundaryMaskLongRange(
        py::module & ragModule
    ){
        ragModule.def("boundaryMaskLongRange",
        [](
            const RAG & rag,
            const CONTR_GRAP & contrGraph,
            nifty::marray::PyView<int, 2>      offsets
        ){


            const auto & labels = rag.labels();
            const auto & shape = rag.shape();

            typedef nifty::marray::PyView<int, DIM+1> NumpyArrayInt;
            typedef std::pair<NumpyArrayInt, NumpyArrayInt>  OutType;

            // std::cout << "Tick 1";

            std::array<int,DIM+1> new_shape;
            std::copy(shape.begin(), shape.end(), new_shape.begin());
            new_shape.back() = offsets.shape(0);

            NumpyArrayInt boundMask(new_shape.begin(), new_shape.end());
            NumpyArrayInt boundMaskIDs(new_shape.begin(), new_shape.end());

            std::fill(boundMask.begin(), boundMask.end(), 0);
            std::fill(boundMaskIDs.begin(), boundMaskIDs.end(), -1);

            // std::cout << "Tick 2";

            for(auto x=0; x<shape[0]; ++x){
                for(auto y=0; y<shape[1]; ++y){
                    if (DIM==3){
                        for(auto z=0; z<shape[2]; ++z){

                            const auto u = labels(x,y,z);
                            // std::cout << "u" << u;

                            for(auto i=0; i<offsets.shape(0); ++i){
                                const auto ox = offsets(i, 0);
                                const auto oy = offsets(i, 1);
                                const auto oz = offsets(i, 2);
                                const auto xx = ox +x ;
                                const auto yy = oy +y ;
                                const auto zz = oz +z ;



                                if(xx>=0 && xx<shape[0] && yy >=0 && yy<shape[1] && zz >=0 && zz<shape[2]){
                                    const auto v = labels(xx,yy,zz);
                                    // std::cout << "v" << v;
                                    if(u != v){
                                        auto edge = rag.findEdge(u,v);
                                        // std::cout << ".";
                                        if(edge >=0 ){
                                            auto cEdge = contrGraph.findRepresentativeEdge(edge);
                                            // std::cout << ".";
                                            boundMask(x,y,z,i) = 1;
                                            // std::cout << ".";
                                            boundMaskIDs(x,y,z,i) = cEdge;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return OutType(boundMask, boundMaskIDs);;

        },
        py::arg("rag"),
        py::arg("contrGraph"),
        py::arg("offsets")
        );

        ragModule.def("boundaryMaskLongRange",
        [](
            const RAG & rag,
            nifty::marray::PyView<int, 2>      offsets,
            const int numberOfThreads
        ){


            const auto & labels = rag.labels();
            const auto & shape = rag.shape();

            typedef nifty::marray::PyView<int, DIM+1> NumpyArrayInt;
            typedef std::pair<NumpyArrayInt, NumpyArrayInt>  OutType;


            std::array<int,DIM+1> new_shape;
            std::copy(shape.begin(), shape.end(), new_shape.begin());
            new_shape.back() = offsets.shape(0);

            NumpyArrayInt boundMask(new_shape.begin(), new_shape.end());
            NumpyArrayInt boundMaskIDs(new_shape.begin(), new_shape.end());

            std::fill(boundMask.begin(), boundMask.end(), 0);
            std::fill(boundMaskIDs.begin(), boundMaskIDs.end(), -1);


            {
                py::gil_scoped_release allowThreads;

                // Create thread pool:
                nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                nifty::parallel::ThreadPool threadpool(pOpts);
                const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                if(DIM == 3){
                    nifty::tools::parallelForEachCoordinate(threadpool,
                                                            shape,
                                                            [&](const auto threadId, const auto & coordP){
                                                                const auto u = labels(coordP[0],coordP[1],coordP[2]);
                                                                for(auto i=0; i<offsets.shape(0); ++i){
                                                                    auto coordQ = coordP;
                                                                    coordQ[0] += offsets(i, 0);
                                                                    coordQ[1] += offsets(i, 1);
                                                                    coordQ[2] += offsets(i, 2);
                                                                    if(coordQ.allInsideShape(shape)){
                                                                        const auto v = labels(coordQ[0],coordQ[1],coordQ[2]);
                                                                        if(u != v){
                                                                            const auto edge = rag.findEdge(u,v);
                                                                            if(edge >=0 ){
                                                                                boundMask(coordP[0],coordP[1],coordP[2],i) = 1;
                                                                                boundMaskIDs(coordP[0],coordP[1],coordP[2],i) = edge;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            });
                }

            }

            return OutType(boundMask, boundMaskIDs);;

        },
        py::arg("rag"),
        py::arg("offsets"),
        py::arg("numberOfThreads") = 8
        );
    }


    template<std::size_t DIM, class GRAPH, class DATA_T>
    void exportBoundaryMaskLongRange(
            py::module & ragModule
    ){
        ragModule.def("boundaryMaskLongRange",
                      [](
                              const GRAPH & graph,
                              nifty::marray::PyView<int, DIM> labels,
                              nifty::marray::PyView<int, 2>      offsets,
                              const int numberOfThreads
                      ){
                          array::StaticArray<int64_t, DIM> shape;

//                          std::array<int,DIM> shape;
                          // Check inputs:
                          for(auto d=0; d<DIM; ++d){
                              shape[d] = labels.shape(d);
                          }

                          typedef nifty::marray::PyView<int, DIM+1> NumpyArrayInt;
                          typedef std::pair<NumpyArrayInt, NumpyArrayInt>  OutType;


                          std::array<int,DIM+1> new_shape;
                          std::copy(shape.begin(), shape.end(), new_shape.begin());
                          new_shape.back() = offsets.shape(0);

                          NumpyArrayInt boundMask(new_shape.begin(), new_shape.end());
                          NumpyArrayInt boundMaskIDs(new_shape.begin(), new_shape.end());

                          std::fill(boundMask.begin(), boundMask.end(), 0);
                          std::fill(boundMaskIDs.begin(), boundMaskIDs.end(), -1);



                          {
                              py::gil_scoped_release allowThreads;

                              // Create thread pool:
                              nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                              nifty::parallel::ThreadPool threadpool(pOpts);
                              const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                              if(DIM == 3){
                                  nifty::tools::parallelForEachCoordinate(threadpool,
                                                                          shape,
                                                                          [&](const auto threadId, const auto & coordP){
                                                                              const auto u = labels(coordP[0],coordP[1],coordP[2]);
                                                                              for(auto i=0; i<offsets.shape(0); ++i){
                                                                                  auto coordQ = coordP;
                                                                                  coordQ[0] += offsets(i, 0);
                                                                                  coordQ[1] += offsets(i, 1);
                                                                                  coordQ[2] += offsets(i, 2);
                                                                                  if(coordQ.allInsideShape(shape)){
                                                                                      const auto v = labels(coordQ[0],coordQ[1],coordQ[2]);
                                                                                      if(u != v){
                                                                                          const auto edge = graph.findEdge(u,v);
                                                                                          if(edge >=0 ){
                                                                                              boundMask(coordP[0],coordP[1],coordP[2],i) = 1;
                                                                                              boundMaskIDs(coordP[0],coordP[1],coordP[2],i) = edge;
                                                                                          }
                                                                                      }
                                                                                  }
                                                                              }
                                                                          });
                              }

                          }
                          return OutType(boundMask, boundMaskIDs);;

                      },
                      py::arg("graph"),
                      py::arg("labels"),
                      py::arg("offsets"),
                      py::arg("numberOfThreads") = 8
        );
    }


    template<std::size_t DIM, class DATA_T>
    void exportMapFeaturesToLabelArray(
        py::module & ragModule
    ){
        ragModule.def("mapFeaturesToLabelArray",
        [](
            nifty::marray::PyView<int, DIM> labelArray,
            nifty::marray::PyView<DATA_T, 2> featureArray,
            int ignoreLabel,
            float fillValue,
            const int numberOfThreads
        ){
            array::StaticArray<int64_t, DIM> shape;
            for(auto d=0; d<DIM; ++d){
                shape[d] = labelArray.shape(d);
            }

            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;

            // std::cout << "Tick 0";
            NIFTY_CHECK((DIM == 3) || (DIM == 4), "Implemented dimensions: 3 and 4");

            std::array<int,DIM+1> shapeFeatureImage;
            std::copy(labelArray.shapeBegin(), labelArray.shapeEnd(), shapeFeatureImage.begin());
            shapeFeatureImage.back() = featureArray.shape(1);

            NumpyArrayType featureImage(shapeFeatureImage.begin(), shapeFeatureImage.end());

            std::fill(featureImage.begin(), featureImage.end(), fillValue);

            // std::cout << "Tick 1";
            {
                py::gil_scoped_release allowThreads;

                // Create thread pool:
                nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                nifty::parallel::ThreadPool threadpool(pOpts);
                const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                if(DIM == 3){
                    nifty::tools::parallelForEachCoordinate(threadpool,
                                                            shape,
                                                            [&](const auto threadId, const auto & coordP){
                                                                const auto label = labelArray(coordP[0],coordP[1],coordP[2]);
                                                                if (label!=ignoreLabel && label<featureArray.shape(0)) {
                                                                    for(auto f=0; f<featureArray.shape(1); ++f){
                                                                        featureImage(coordP[0],coordP[1],coordP[2],f) = featureArray(label,f);
                                                                    }
                                                                }
                                                            });
                } else if (DIM == 4) {
                    nifty::tools::parallelForEachCoordinate(threadpool,
                                                            shape,
                                                            [&](const auto threadId, const auto & coordP){
                                                                const auto label = labelArray(coordP[0],coordP[1],coordP[2],coordP[3]);
                                                                if (label!=ignoreLabel && label<featureArray.shape(0)) {
                                                                    for(auto f=0; f<featureArray.shape(1); ++f){
                                                                        featureImage(coordP[0],coordP[1],coordP[2],coordP[3],f) = featureArray(label,f);
                                                                    }
                                                                }
                                                            });
                }

            }

            return featureImage;

        },
        py::arg("labelArray"),
        py::arg("featureArray"),
        py::arg("ignoreLabel") = -1,
        py::arg("fillValue") = 0.,
        py::arg("numberOfThreads") = 8
        );
    }
    */




    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateEdgeMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeMeanAndLength",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){

            typename xt::pytensor<DATA_T, 2>::shape_type shape = {int64_t(rag.edgeIdUpperBound()+1), int64_t(2)};
            xt::pytensor<DATA_T, 2> out(shape);
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeMeanAndLength(rag, data, blockShape, out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag").noconvert(),
        py::arg("data").noconvert(),
        py::arg("blockShape")=array::StaticArray<int64_t, DIM>(100),
        py::arg("numberOfThreads")=-1
        );
    }


    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateGeometricEdgeFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateGeometricEdgeFeatures",
        [](
            const RAG & rag,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){

            xt::pytensor<DATA_T, 2> out({int64_t(rag.edgeIdUpperBound()+1), int64_t(17)});
            {
                py::gil_scoped_release allowThreads;
                accumulateGeometricEdgeFeatures(rag, blockShape, out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }


    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateMeanAndLength",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads,
            const bool saveMemory
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(2)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                accumulateMeanAndLength(rag, data, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1,
        py::arg_t<bool>("saveMemory",false)
        );
    }

    #ifdef WITH_HDF5
    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateMeanAndLengthHdf5(
        py::module & ragModule
    ){
        ragModule.def("accumulateMeanAndLength",
        [](
            const RAG & rag,
            const nifty::hdf5::Hdf5Array<DATA_T> & data,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads,
            const bool saveMemory
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(2)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                accumulateMeanAndLength(rag, data, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);;
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1,
        py::arg_t<bool>("saveMemory",false)
        );
    }
    #endif




    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateStandartFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateStandartFeatures",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(9)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateStandartFeatures(rag, data, minVal, maxVal, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }

    #ifdef WITH_HDF5
    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateStandartFeaturesHdf5(
        py::module & ragModule
    ){
        ragModule.def("accumulateStandartFeatures",
        [](
            const RAG & rag,
            const nifty::hdf5::Hdf5Array<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(9)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateStandartFeatures(rag, data, minVal, maxVal, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }

    #endif




    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateNodeStandartFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateNodeStandartFeatures",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2>nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateNodeStandartFeatures(rag, data, minVal, maxVal, blockShape, nodeOut, numberOfThreads);
            }
            return nodeOut;
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }

    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateEdgeStandartFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStandartFeatures",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2>edgeOut({int64_t(rag.edgeIdUpperBound()+1), 9L});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeatures(rag, data, minVal, maxVal, blockShape, edgeOut, numberOfThreads);
            }
            return edgeOut;
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }



    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateGeometricNodeFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateGeometricNodeFeatures",
        [](
            const RAG & rag,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(3*DIM+1)});
            {
                py::gil_scoped_release allowThreads;
                accumulateGeometricNodeFeatures(rag, blockShape, nodeOut, numberOfThreads);
            }
            return nodeOut;
        },
        py::arg("rag"),
        py::arg("blockShape")=array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")=-1
        );
    }



    void exportAccumulate(py::module & ragModule) {

        //explicit
        {
            exportEvaluateSimplexNoiseOnArrayT<2>(ragModule);
            exportEvaluateSimplexNoiseOnArrayT<3>(ragModule);
            exportEvaluateSimplexNoiseOnArrayT<4>(ragModule);


            typedef xt::pytensor<uint32_t, 2> ExplicitLabels2D;
            typedef GridRag<2, ExplicitLabels2D> Rag2d;
            typedef xt::pytensor<uint32_t, 3> ExplicitLabels3D;
            typedef GridRag<3, ExplicitLabels3D> Rag3d;

//            typedef xt::pytensor<uint32_t, 2> ExplicitPyLabels2D;
//            typedef GridRag<2, ExplicitPyLabels2D> Rag2d;

//            typedef ExplicitLabelsGridRag<2, uint64_t> Rag2d;
//            typedef ExplicitLabelsGridRag<3, uint64_t> Rag3d;


            typedef PyUndirectedGraph GraphType;
            typedef PyContractionGraph<PyUndirectedGraph> ContractionGraphType;
//
//
////            exportAccumulateAffinitiesMeanAndLength<2, Rag2d, ContractionGraphType, float>(ragModule);
            /*
            exportAccumulateAffinitiesMeanAndLength<3, Rag3d, ContractionGraphType, float>(ragModule);
            exportAccumulateAffinitiesMeanAndLength<3, GraphType, float>(ragModule);
            exportAccumulateAffinitiesMeanAndLengthOnNodes<3, GraphType, float>(ragModule);
//
//
//            exportMapFeaturesToBoundaries<2, Rag2d, ContractionGraphType, float>(ragModule);
            exportMapFeaturesToBoundaries<3, Rag3d, ContractionGraphType, float>(ragModule);

            // exportBoundaryMaskLongRange<2, Rag2d, ContractionGraphType, float>(ragModule);
            exportBoundaryMaskLongRange<3, Rag3d, ContractionGraphType, float>(ragModule);
            exportBoundaryMaskLongRange<3, GraphType, float>(ragModule);


            // TODO: move to another place!
            exportMapFeaturesToLabelArray<2, float>(ragModule);
            exportMapFeaturesToLabelArray<3, float>(ragModule);
            exportMapFeaturesToLabelArray<4, float>(ragModule);

            // TODO: move to another place!
            exportConnectedComponentsFromEdgeLabels<3, uint64_t>(ragModule);
            */

            // Previous exports:
            exportAccumulateEdgeMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateEdgeMeanAndLength<3, Rag3d, float>(ragModule);

            exportAccumulateMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateMeanAndLength<3, Rag3d, float>(ragModule);

            exportAccumulateStandartFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateStandartFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateNodeStandartFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateNodeStandartFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateEdgeStandartFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateEdgeStandartFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateGeometricNodeFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateGeometricNodeFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateGeometricEdgeFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateGeometricEdgeFeatures<3, Rag3d, float>(ragModule);

            #ifdef WITH_HDF5
            typedef nifty::hdf5::Hdf5Array<uint32_t> H5Labels;
            typedef GridRag<3, H5Labels> RagH53d;
            exportAccumulateMeanAndLengthHdf5<3,RagH53d, float>(ragModule);
            exportAccumulateStandartFeaturesHdf5<3, RagH53d, uint8_t>(ragModule);
            #endif

        }
    }

} // end namespace graph
} // end namespace nifty
