#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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




#ifdef WITH_HDF5

#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#endif


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"


#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"




namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

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
            // Inputs:
//            typedef typename DATA_T::value_type value_type;
//            auto & affinities = affinitiesExpression.derived_cast();
//            auto & offsets = offsetsExpression.derived_cast();

            const auto & labels = rag.labelsProxy().labels();
            const auto & shape = rag.labelsProxy().shape();

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


            const auto & labels = rag.labelsProxy().labels();
            const auto & shape = rag.labelsProxy().shape();

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


    template<std::size_t DIM, class GRAPH, class DATA_T>
    void exportAccumulateAffinitiesMeanAndLength(
            py::module & ragModule
    ) {
        ragModule.def("accumulateAffinitiesMeanAndLength",
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
                          typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;

                          std::array<int,2> shapeRetArray;
                          shapeRetArray[0] = numberOfThreads;
                          shapeRetArray[1] = uint64_t(graph.edgeIdUpperBound()+1);

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
                                                                                          const auto edge = graph.findEdge(u,v);
                                                                                          if (edge >=0 ){
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

                          NumpyArrayType accAff_out({uint64_t(graph.edgeIdUpperBound()+1)});
                          NumpyArrayType counter_out({uint64_t(graph.edgeIdUpperBound()+1)});

                          // Normalize:
                          for(auto i=0; i<uint64_t(graph.edgeIdUpperBound()+1); ++i){
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
                      py::arg("graph"),
                      py::arg("labels"),
                      py::arg("affinities"),
                      py::arg("offsets"),
                      py::arg("affinitiesWeights"),
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


            const auto & labels = rag.labelsProxy().labels();
            const auto & shape = rag.labelsProxy().shape();

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
            float fillValue
        ){


            const auto & labels = rag.labelsProxy().labels();
            const auto & shape = rag.labelsProxy().shape();

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
                                        const auto edge = rag.findEdge(u,v);
                                        if(edge >=0 ){
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
                                    const auto edge = rag.findEdge(u,v);
                                    if(edge >=0 ){
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
        py::arg("edgeFeatures"),
        py::arg("offsets"),
        py::arg("fillValue") = 0.
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


            const auto & labels = rag.labelsProxy().labels();
            const auto & shape = rag.labelsProxy().shape();

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


            const auto & labels = rag.labelsProxy().labels();
            const auto & shape = rag.labelsProxy().shape();

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





    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateEdgeMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeMeanAndLength",
        [](
            const RAG & rag,
            nifty::marray::PyView<DATA_T, DIM> data,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){

            nifty::marray::PyView<DATA_T> out({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                array::StaticArray<int64_t, DIM> blocKShape_;
                accumulateEdgeMeanAndLength(rag, data, blocKShape, out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("data"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }


    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateGeometricEdgeFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateGeometricEdgeFeatures",
        [](
            const RAG & rag,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){

            nifty::marray::PyView<DATA_T> out({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(17)});
            {
                py::gil_scoped_release allowThreads;
                array::StaticArray<int64_t, DIM> blocKShape_;
                accumulateGeometricEdgeFeatures(rag, blocKShape, out, numberOfThreads);
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
            nifty::marray::PyView<DATA_T, DIM> data,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads,
            const bool saveMemory
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(2)});
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                array::StaticArray<int64_t, DIM> blocKShape_;
                accumulateMeanAndLength(rag, data, blocKShape, edgeOut, nodeOut, numberOfThreads);
            }
            return OutType(edgeOut, nodeOut);;
        },
        py::arg("rag"),
        py::arg("data"),
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
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads,
            const bool saveMemory
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(2)});
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                array::StaticArray<int64_t, DIM> blocKShape_;
                accumulateMeanAndLength(rag, data, blocKShape, edgeOut, nodeOut, numberOfThreads);
            }
            return OutType(edgeOut, nodeOut);;
        },
        py::arg("rag"),
        py::arg("data"),
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
            nifty::marray::PyView<DATA_T, DIM> data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(11)});
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(11)});
            {
                py::gil_scoped_release allowThreads;
                array::StaticArray<int64_t, DIM> blocKShape_;
                accumulateStandartFeatures(rag, data, minVal, maxVal, blocKShape, edgeOut, nodeOut, numberOfThreads);
            }
            return OutType(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data"),
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
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            typedef std::pair<NumpyArrayType, NumpyArrayType>  OutType;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(11)});
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(11)});
            {
                py::gil_scoped_release allowThreads;
                array::StaticArray<int64_t, DIM> blocKShape_;
                accumulateStandartFeatures(rag, data, minVal, maxVal, blocKShape, edgeOut, nodeOut, numberOfThreads);
            }
            return OutType(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data"),
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
            nifty::marray::PyView<DATA_T, DIM> data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(11)});
            {
                py::gil_scoped_release allowThreads;
                accumulateNodeStandartFeatures(rag, data, minVal, maxVal, blocKShape, nodeOut, numberOfThreads);
            }
            return nodeOut;
        },
        py::arg("rag"),
        py::arg("data"),
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
            nifty::marray::PyView<DATA_T, DIM> data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            NumpyArrayType edgeOut({uint64_t(rag.edgeIdUpperBound()+1),uint64_t(11)});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeatures(rag, data, minVal, maxVal, blocKShape, edgeOut, numberOfThreads);
            }
            return edgeOut;
        },
        py::arg("rag"),
        py::arg("data"),
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
            array::StaticArray<int64_t, DIM> blocKShape,
            const int numberOfThreads
        ){
            typedef nifty::marray::PyView<DATA_T> NumpyArrayType;
            NumpyArrayType nodeOut({uint64_t(rag.nodeIdUpperBound()+1),uint64_t(3*DIM+1)});
            {
                py::gil_scoped_release allowThreads;
                accumulateGeometricNodeFeatures(rag, blocKShape, nodeOut, numberOfThreads);
            }
            return nodeOut;
        },
        py::arg("rag"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }



    void exportAccumulate(py::module & ragModule) {

        //explicit
        {
            typedef ExplicitLabelsGridRag<2, uint32_t> Rag2d;
            typedef ExplicitLabelsGridRag<3, uint32_t> Rag3d;

            typedef PyUndirectedGraph GraphType;
            typedef PyContractionGraph<PyUndirectedGraph> ContractionGraphType;


//            exportAccumulateAffinitiesMeanAndLength<2, Rag2d, ContractionGraphType, float>(ragModule);
            exportAccumulateAffinitiesMeanAndLength<3, Rag3d, ContractionGraphType, float>(ragModule);
            exportAccumulateAffinitiesMeanAndLength<3, GraphType, float>(ragModule);


            exportMapFeaturesToBoundaries<2, Rag2d, ContractionGraphType, float>(ragModule);
            exportMapFeaturesToBoundaries<3, Rag3d, ContractionGraphType, float>(ragModule);

            // exportBoundaryMaskLongRange<2, Rag2d, ContractionGraphType, float>(ragModule);
            exportBoundaryMaskLongRange<3, Rag3d, ContractionGraphType, float>(ragModule);
            exportBoundaryMaskLongRange<3, GraphType, float>(ragModule);


            exportMapFeaturesToLabelArray<2, float>(ragModule);
            exportMapFeaturesToLabelArray<3, float>(ragModule);
            exportMapFeaturesToLabelArray<4, float>(ragModule);

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
            typedef GridRag<3, Hdf5Labels<3, uint32_t>  >  RagH53d;
            //exportAccumulateMeanAndLengthHdf5<3,RagH53d, float>(ragModule);
            exportAccumulateStandartFeaturesHdf5<3, RagH53d, uint8_t >(ragModule);
            #endif

        }
    }

} // end namespace graph
} // end namespace nifty
