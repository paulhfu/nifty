#ifdef WITH_HDF5
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/hdf5/hdf5_tools.hxx"

namespace py = pybind11;



namespace nifty{
namespace hdf5{

    template<class T>
    void exportHdf5ToolsT(py::module & hdf5Module) {

        /*
        hdf5Module.def("uniqueValuesInSubvolumeSliced",[](
            const Hdf5Array<T> & array,
            const std::vector<size_t> & begin,
            const std::vector<size_t> & end,
            const int numberOfThreads = -1
        ){
        
            if(array.dimension() != 3) {
                throw std::runtime_error("nifty::hdf5::uniqueValuesInSubvolumeSliced only implemented for dim = 3");
            }
            
            typedef array::StaticArray<int64_t,3> Coord;

            Coord beginCoord;
            Coord endCoord;
            for(int d = 0; d < 3; ++d) {
                beginCoord[d] = begin[d];
                endCoord[d] = end[d];
            }
            
            std::vector<T> out;
            {
                py::gil_scoped_release allowThreads;
                uniqueValuesInSubvolumeSliced(array, beginCoord, endCoord, out, numberOfThreads );
            }

        },
        py::arg("array"),
        py::arg("begin"),
        py::arg("end"),
        py::arg("numberOfThreads")=-1
        );
        */
        
        hdf5Module.def("uniqueValuesInSubvolume",[](
            const Hdf5Array<T> & array,
            const std::vector<size_t> & begin,
            const std::vector<size_t> & end,
            const std::vector<size_t> & blockShape,
            const int numberOfThreads = -1
        ){

            if(array.dimension() != 3) {
                throw std::runtime_error("nifty::hdf5::uniqueValuesInSubvolumeSliced only implemented for dim = 3");
            }
            
            typedef array::StaticArray<int64_t,3> Coord;

            Coord beginCoord;
            Coord endCoord;
            Coord blockShapeCoord;
            for(int d = 0; d < 3; ++d) {
                beginCoord[d] = begin[d];
                endCoord[d] = end[d];
                blockShapeCoord[d] = blockShape[d];
            }
            
            std::vector<T> out;
            {
                py::gil_scoped_release allowThreads;
                uniqueValuesInSubvolume(array, beginCoord, endCoord, blockShapeCoord, out, numberOfThreads );
            }
            return out;

        },
        py::arg("array"),
        py::arg("begin"),
        py::arg("end"),
        py::arg("blockShape"),
        py::arg("numberOfThreads")=-1
        );

    }
    
    void correct_gt(const Hdf5Array<uint32_t> & gt,Hdf5Array<uint32_t> & new_gt, Hdf5Array<uint32_t> & touch) {
    
        size_t shape_slice[] = {gt.shape(0), gt.shape(1),1};
    
        for(size_t z = 0; z < gt.shape(2); z++) {
    
            std::cout << z << std::endl;
    
            nifty::marray::Marray<uint32_t> gt_z(shape_slice, shape_slice + 3);
            nifty::marray::Marray<uint32_t> new_z(shape_slice, shape_slice + 3);
            nifty::marray::Marray<uint32_t> touch_z(shape_slice, shape_slice + 3);
    
            size_t slice_start[] = {0,0,z};
    
            gt.readSubarray(slice_start, gt_z);
            std::copy(gt_z.begin(), gt_z.end(), new_z.begin());
    
            for(size_t x = 0; x < gt.shape(0); x++) {
                for(size_t y = 0; y < gt.shape(1); y++) {
                    auto lU = gt_z(x,y,0);
    
                    if( lU == 0 )
                        continue;
                    
                    if(x != gt.shape(0) - 1) {
                        auto lV_x = gt_z(x+1,y,0);
                        if( lU != lV_x && lV_x != 0 ) {
                            touch_z(x,y,0) = 1;
                            touch_z(x+1,y,0) = 1;
                            new_z(x,y,0) = 0;
                            new_z(x+1,y,0) = 0;
                        }
                    }
                    if(y != gt.shape(1) - 1) {
                        auto lV_y = gt_z(x,y+1,0);
                        if( lU != lV_y && lV_y != 0 ) {
                            touch_z(x,y,0) = 1;
                            touch_z(x,y+1,0) = 1;
                            new_z(x,y,0) = 0;
                            new_z(x,y+1,0) = 0;
                        }
                    }
                
                }
            }
            new_gt.writeSubarray(slice_start, gt_z);
            touch.writeSubarray(slice_start, touch_z);
        }
    }

    void export_quick_and_dirty(py::module & hdf5Module) {

        hdf5Module.def("correct_snemi_gt",&correct_gt);

    }
    
    void exportHdf5Tools(py::module & hdf5Module) {

        export_quick_and_dirty(hdf5Module);

        //exportHdf5ToolsT<uint8_t >(hdf5Module);
        //exportHdf5ToolsT<uint16_t>(hdf5Module);
        //exportHdf5ToolsT<uint32_t>(hdf5Module);
        //exportHdf5ToolsT<uint64_t>(hdf5Module);

        //exportHdf5ToolsT<int8_t >(hdf5Module);
        //exportHdf5ToolsT<int16_t>(hdf5Module);
        //exportHdf5ToolsT<int32_t>(hdf5Module);
        //exportHdf5ToolsT<int64_t>(hdf5Module);

        //exportHdf5ToolsT<float >(hdf5Module);
        //exportHdf5ToolsT<double>(hdf5Module);
    }

}
}
#endif
