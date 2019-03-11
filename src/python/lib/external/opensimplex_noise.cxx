#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/external/opensimplex_noise.hxx"



namespace py = pybind11;

namespace nifty {
    namespace external {

        using namespace py;


        template<std::size_t DIM>
        void exportEvaluateSimplexNoiseOnArrayT(
                py::module & externalModule
        ) {
            externalModule.def("evaluateSimplexNoiseOnArray_impl",
                               [](
                                       std::array<int, DIM>    shape,
                                       int64_t           seed,
                                       const xt::pytensor<double, 1> & featureSize,
                                       int numberOfThreads
                               ) {
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

        void exportEvaluateSimplexNoiseOnArray(py::module & externalModule) {
            exportEvaluateSimplexNoiseOnArrayT<2>(externalModule);
            exportEvaluateSimplexNoiseOnArrayT<3>(externalModule);
            exportEvaluateSimplexNoiseOnArrayT<4>(externalModule);
        }

    }
}