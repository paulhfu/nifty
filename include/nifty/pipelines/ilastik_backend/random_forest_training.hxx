#pragma once

#include <algorithm>

#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include <nifty/marray/marray.hxx>

namespace nifty
{
namespace pipelines
{
namespace ilastik_backend
{

using random_forest2 = nifty::pipelines::ilastik_backend::RandomForest2Type;
using random_forest3 = nifty::pipelines::ilastik_backend::RandomForest3Type;

template<unsigned DIM, typename DATA_TYPE, typename LABEL_TYPE>
void random_forest2_training(
        const size_t blockId,
        const nifty::marray::View<DATA_TYPE> & features,
        const nifty::marray::View<LABEL_TYPE> & labels,
        random_forest2 & rf
        ) {
    //TODO
}
    
} // namespace ilastik_backend
} // namespace pipelines
} // namespace nifty
