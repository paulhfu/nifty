#pragma once
#ifndef NIFTY_HDF5_TOOLS
#define NIFTY_HDF5_TOOLS

#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/tools/block_access.hxx"

namespace nifty{
namespace hdf5{

    template<class T>
    void uniqueValuesInSubvolume(const Hdf5Array<T> & array,
            const array::StaticArray<int64_t,3> & begin,
            const array::StaticArray<int64_t,3> & end,
            const array::StaticArray<int64_t,3> & blockShape,
            std::vector<T> & out,
            const int numberOfThreads = -1) {
        
        typedef array::StaticArray<int64_t,3> Coord;
        typedef nifty::tools::BlockStorage<T> BlockStorageType;

        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        Coord shape;
        for(int d = 0; d < 3; d++)
            shape[d] = end[d] - begin[d];
        
        BlockStorageType blockStorage(threadpool, blockShape, nThreads);

        std::vector<std::vector<T>> perThreadUniques(nThreads);

        // TODO paralelForEachBlockWithOffset
        nifty::tools::parallelForEachBlock(threadpool, shape, blockShape, 
        [&](
            const int tid,
            const Coord & blockBegin, const Coord & blockEnd
        ){
            auto & uniques = perThreadUniques[tid];

            Coord actualBlockBegin;
            for(int d = 0; d < 3; d++)
                actualBlockBegin[d] = blockBegin[d] + begin[d];

            // we might have a singleton here, but I think that doesn't hurt
            auto view = blockStorage.getView(tid);
            array.readSubarray(actualBlockBegin.begin(), view);

            // find the unique data in this block
            auto uniquesLen = uniques.size();
            uniques.resize(uniques.size() + view.size());
            
            auto uniquesLast = uniques.begin() + uniquesLen;
            std::copy(view.begin(),view.end(),uniquesLast);
            
            std::sort(uniquesLast,uniques.end());
            auto last = std::unique(uniquesLast, uniques.end());
            uniques.erase( last, uniques.end() );
        });

        for(size_t tid = 0; tid < nThreads; tid++) {
            auto & uniques = perThreadUniques[tid];
            out.reserve(out.size() + uniques.size());
            out.insert(out.end(), uniques.begin(), uniques.end());
        }

        std::sort(out.begin(), out.end());
        auto last = std::unique(out.begin(), out.end());
        out.erase( last, out.end() );
    }
    
    /*
    template<class T>
    void uniqueValuesInSubvolumeSliced(const Hdf5Array<T> & array,
            const array::StaticArray<int64_t,3> & begin,
            const array::StaticArray<int64_t,3> & end,
            std::vector<T> & out,
            const int numberOfThreads = -1) {
        
        typedef array::StaticArray<int64_t,3> Coord;
        typedef nifty::tools::BlockStorage<T> BlockStorageType;

        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        Coord shape;
        for(int d = 0; d < 3; d++)
            shape[d] = end[d] - begin[d];

        Coord blockShape({int64_t(1),shape[1],shape[2]});

        size_t numberOfSlices = shape[0];
        
        BlockStorageType blockStorage(threadpool, blockShape, nThreads);

        std::vector<std::vector<T>> perThreadUniques(nThreads);

        // TODO paralelForEachBlockWithOffset
        parallel::parallel_foreach(threadpool, numberOfSlices , [&](
            const int tid, const int64_t slice 
        ){
            auto & uniques = perThreadUniques[tid];

            Coord actualBlockBegin({slice, shape[1], shape[2]});

            // we have a singleton dimension here, but I think that doesn't hurt
            auto view = blockStorage.getView(tid);
            array.readSubarray(actualBlockBegin.begin(), view);

            // find the unique data in this block
            auto uniques_last = uniques.end();
            uniques.reserve(uniques.size() + view.size());
            std::copy(view.begin(),view.end(),uniques_last);
            std::sort(uniques_last,uniques.end());
            auto last = std::unique(uniques_last, uniques.end());
            uniques.erase( last, uniques.end() );

        });

        for(size_t tid = 0; tid < nThreads; tid++) {
            auto & uniques = perThreadUniques[tid];
            out.reserve(out.size() + uniques.size());
            out.insert(out.end(), uniques.begin(), uniques.end());
        }

        // don't need to make unique another time, because we assume sliced structure
        std::sort(out.begin(), out.end());
    }
    */


} // namespace hdf5
} // namespace nifty

#endif  // NIFTY_HDF5_HDF5_TOOLS
