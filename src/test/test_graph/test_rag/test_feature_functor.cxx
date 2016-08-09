#define BOOST_TEST_MODULE NiftyTestFeatureFunctors

#include <boost/test/unit_test.hpp>

#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/rag/feature_functors.hxx"


BOOST_AUTO_TEST_CASE(FeatureFunctorTest)
{
    std::vector<size_t> shape({50,50});
    nifty::marray::Marray<float> a(shape.begin(), shape.end()), b(shape.begin(), shape.end());
    std::fill(a.begin(), a.end(), 1.);

    nifty::graph::FilterFunctor<2> functor(1.);
    
    functor(a, b);
    
    // TODO test something meaningful
    NIFTY_TEST_OP(b.shape(0),==,50)
    NIFTY_TEST_OP(b.shape(1),==,50)

}
