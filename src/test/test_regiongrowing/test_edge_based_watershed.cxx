#define BOOST_TEST_MODULE NiftyBreadthFirstSearchTest

#include <boost/test/unit_test.hpp>

#include <iostream> 
#include <random>

#include "nifty/region_growing/edge_based_watershed.hxx"
#include "nifty/tools/runtime_check.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/hdf5/hdf5.hxx"
#endif

BOOST_AUTO_TEST_CASE(EdgebasedWatershedSimpleTest)
{
    size_t shape[] = {10,10};
    size_t shapeAff[] = {10,10,2};
    nifty::marray::Marray<float> testAffinities(shapeAff,shapeAff+3,.9);

    for(int i = 0; i < shape[1]; ++i)
        testAffinities(5,i) = 0.;

    nifty::marray::Marray<uint32_t> out(shape,shape+2);
    
    nifty::region_growing::edgeBasedWatershed<2>(testAffinities,float(.2),float(.8),out);

    // make sure that we have exactly two segments
    // TODO test for uniqueness properly
    for(auto it = out.begin(); it != out.end(); ++it) {
        NIFTY_TEST_OP(*it,<,3);
        NIFTY_TEST_OP(*it,>,0);
    }

}

BOOST_AUTO_TEST_CASE(EdgebasedWatershedReproducibilityTest)
{
    size_t shape[] = {1000,1000};
    size_t shapeAff[] = {1000,1000,2};
    nifty::marray::Marray<float> testAffinities(shapeAff,shapeAff+3,.9);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.,1.);
    auto draw = std::bind(distribution, generator);

    for(auto it = testAffinities.begin(); it != testAffinities.end(); it++)
        *it = draw();

    float low = .1;
    float high = .9;
    // first iteration
    nifty::marray::Marray<uint32_t> out1(shape,shape+2);
    nifty::region_growing::edgeBasedWatershed<2>(testAffinities,low,high,out1);
    
    // second iteration
    nifty::marray::Marray<uint32_t> out2(shape,shape+2);
    nifty::region_growing::edgeBasedWatershed<2>(testAffinities,low,high,out2);

    for(int x = 0; x < shape[0]; x++) {
        for(int y = 0; y < shape[1]; y++) {
            NIFTY_TEST_OP(out1(x,y),==,out2(x,y));
        }
    }
}

#ifdef WITH_HDF5
BOOST_AUTO_TEST_CASE(EdgebasedWatershedRealDataTest)
{
    nifty::hdf5::CacheSettings cacheSettings;
    auto testFile = nifty::hdf5::openFile("./test_data.h5", cacheSettings);
    nifty::hdf5::Hdf5Array<float> testData(testFile, "data"); 
           
    size_t shape[] = {testData.shape(0),testData.shape(1)};
    
    size_t shapeAffinities[] = {testData.shape(0),testData.shape(1),testData.shape(2)};
    size_t startAffinities[] = {0,0,0};

    nifty::marray::Marray<float> testAffinities(shapeAffinities,shapeAffinities+3);
    testData.readSubarray(startAffinities,testAffinities);
    
    float low = .1;
    float high = .9;

    // first iteration
    nifty::marray::Marray<uint32_t> out1(shape,shape+2);
    nifty::region_growing::edgeBasedWatershed<2>(testAffinities,low,high,out1);
    
    // second iteration
    nifty::marray::Marray<uint32_t> out2(shape,shape+2);
    nifty::region_growing::edgeBasedWatershed<2>(testAffinities,low,high,out2);

    for(int x = 0; x < shape[0]; x++) {
        for(int y = 0; y < shape[1]; y++) {
            //std::cout << "(" << x << " , " << y << ")" << std::endl;
            NIFTY_TEST_OP(out1(x,y),==,out2(x,y));
        }
    }
}
#endif
