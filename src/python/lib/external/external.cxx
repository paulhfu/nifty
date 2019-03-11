#include <pybind11/pybind11.h>
#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include <iostream>


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
    namespace external{

        void exportEvaluateSimplexNoiseOnArray(py::module &);

    }
}


PYBIND11_MODULE(_external, module) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();

    module.doc() = "external submodule of nifty";

    using namespace nifty::external;
    exportEvaluateSimplexNoiseOnArray(module);
}