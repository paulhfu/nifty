#pragma once

#include "nifty/pipelines/ilastik_backend/feature_computation.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_prediction.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"

namespace nifty{
namespace pipelines{
namespace ilastik_backend{
            
    template<unsigned DIM>
    class batch_prediction_task : public tbb::task
    {
    private:
        using data_type = float;
        using in_data_type = uint8_t;
        using coordinate = nifty::array::StaticArray<int64_t, DIM>;
        using multichan_coordinate = nifty::array::StaticArray<int64_t, DIM+1>;
        
        using float_array = nifty::marray::Marray<data_type>;
        using float_array_view = nifty::marray::View<data_type>;
        using uint_array = nifty::marray::Marray<in_data_type>;
        using uint_array_view = nifty::marray::View<in_data_type>;
        
        using raw_cache = hdf5::Hdf5Array<in_data_type>;
        using random_forest = RandomForest2Type;
        using blocking = tools::Blocking<DIM>;
        using selection_type = selected_feature_type;
        
    public:
        
        // construct batch prediction for single input
        batch_prediction_task(const std::string & in_file,
                const std::string & in_key,
                const std::string & rf_file,
                const std::string & rf_key,
                const selection_type & selected_features,
                const coordinate & block_shape,
                const coordinate & roiBegin,
                const coordinate & roiEnd) :
            blocking_(),
            rfFile_(rf_file),
            rfKey_(rf_key),
            in_file_(in_file),
            in_key_(in_key),
            selectedFeatures_(selected_features),
            blockShape_(block_shape),
            rf_(),
            roiBegin_(roiBegin),
            roiEnd_(roiEnd)
        {
            //init();
        }

        
        void init() {
            rawCache_ = std::make_unique<raw_cache>( hdf5::openFile(in_file_), in_key_ );
            
            // TODO handle the roi shapes in python
            // init the blocking
            coordinate roiShape = roiEnd_ - roiBegin_;
            
            // TODO make rf single threaded
            rf_ = get_rf2_from_file(rfFile_, rfKey_);
            blocking_ = std::make_unique<blocking>(roiBegin_, roiEnd_, blockShape_);

            //nClasses_ = rf_.num_classes();
            nClasses_ = rf_.class_count();
            
            multichan_coordinate outShape, chunkShape; 
            outShape[DIM] = nClasses_;
            chunkShape[DIM] = 1;
            for(int d = 0; d < DIM; ++d) {
                outShape[d] = roiShape[d];
                chunkShape[d] = blockShape_[d];
            }
            
            out_ = std::make_unique<hdf5::Hdf5Array<data_type>>( hdf5::createFile("./out.h5"), "data", outShape.begin(), outShape.end(), chunkShape.begin() );
        }

        
        void process_single_block(const size_t blockId, float_array_view & out_view) {
                
            std::cout << "Processing block " << blockId << " / " << blocking_->numberOfBlocks() << std::endl;

            // compute features    
            const auto halo = getHaloShape<DIM>(this->selectedFeatures_);

            // compute coordinates from blockId
            const auto & blockWithHalo = blocking_->getBlockWithHalo(blockId, halo);
            const auto & outerBlock = blockWithHalo.outerBlock();
            const auto & outerBlockShape = outerBlock.shape();

            // load the raw data
            uint_array raw(outerBlockShape.begin(), outerBlockShape.end());
            {
		    std::lock_guard<std::mutex> lock(s_mutex);
            rawCache_->readSubarray(outerBlock.begin().begin(), raw);
            }

            // allocate the feature array
            size_t nChannels = getNumberOfChannels<DIM>(this->selectedFeatures_);
            multichan_coordinate filterShape;
            filterShape[0] = nChannels;
            for(int d = 0; d < DIM; ++d)
                filterShape[d+1] = outerBlockShape[d];
            float_array feature_array(filterShape.begin(), filterShape.end());

		    feature_computation<DIM>(
                    blockId,
                    raw,
                    feature_array,
                    this->selectedFeatures_);

            // resize the out array to cut the halo
            const auto & localCore  = blockWithHalo.innerBlockLocal();
            const auto & localBegin = localCore.begin();
            const auto & localShape = localCore.shape();

            multichan_coordinate coreBegin;
            multichan_coordinate coreShape;
            for(int d = 1; d < DIM+1; d++){
                coreBegin[d] = localBegin[d-1];
                coreShape[d]  = localShape[d-1];
            }
            coreBegin[0] = 0;
            coreShape[0] = feature_array.shape(0);

            //float_array_view feature_view = feature_array.view(coreBegin.begin(), coreShape.begin());
            feature_array = feature_array.view(coreBegin.begin(), coreShape.begin());
            
            // predict the random forest
            const auto & block = blockWithHalo.innerBlock();
            const auto & outBlockShape = block.shape();
            
            multichan_coordinate predictionShape;
            predictionShape[DIM] = nClasses_;
            for(int d = 0; d < DIM; ++d)
                predictionShape[d] = outBlockShape[d];

            float_array prediction_array(predictionShape.begin(), predictionShape.end());
            random_forest2_prediction<DIM>(blockId, feature_array, out_view, this->rf_, 1);
            
            std::cout << "Block " << blockId << " / " << blocking_->numberOfBlocks() << " done" << std::endl;
        }


        tbb::task* execute() {
            
            std::cout << "batch_prediction_task::execute called" << std::endl;
            
            init();
#if 0
            // feature output for debugging only
            //auto feat_tmp = nifty::hdf5::createFile("./feat_tmp.h5");
            //size_t feat_shape[] = {128,128,128,2};
            //size_t chunk_shape[] = {64,64,64,1};
            //nifty::hdf5::Hdf5Array<float> feats(feat_tmp, "data", feat_shape, feat_shape + 4, chunk_shape );
            
            // TODO spawn the tasks to batch process the complete volume
            for(size_t blockId = 0; blockId < blocking_->numberOfBlocks(); ++blockId) {

                auto block = blocking_->getBlock(blockId);
                coordinate blockBegin = block.begin() - roiBegin_;
                const coordinate & blockShape = block.shape();
                
                // allocate the output data
                multichan_coordinate outBegin, outShape;
                for(int d = 0; d < DIM; ++d) {
                    outBegin[d] = blockBegin[d];
                    outShape[d] = blockShape[d];
                }
                
                outBegin[DIM] = 0;
                outShape[DIM] = nClasses_;
                float_array out(outShape.begin(), outShape.end());
                
                process_single_block(blockId, out);
                out_->writeSubarray(outBegin.begin(), out);
    
                //size_t permutation[feat_array.dimension()];
                //permutation[DIM] = 0;
                //for(int d = 0; d < DIM; ++d)
                //    permutation[d] = d+1;
                //feat_array.permute(permutation);
                //feats.writeSubarray(outBegin.begin(), feat_array);
            }
#else
            // TODO FIXME why two loops ?!
	        tbb::parallel_for(tbb::blocked_range<size_t>(0,blocking_->numberOfBlocks()), [this](const tbb::blocked_range<size_t> &range) {
		    for( size_t blockId=range.begin(); blockId!=range.end(); ++blockId ) {

                auto block = blocking_->getBlock(blockId);
                coordinate blockBegin = block.begin() - roiBegin_;
                const coordinate & blockShape = block.shape();
                
                // allocate the output data
                multichan_coordinate outBegin, outShape;
                for(int d = 0; d < DIM; ++d) {
                    outBegin[d] = blockBegin[d];
                    outShape[d] = blockShape[d];
                }
                outBegin[DIM] = 0;
                outShape[DIM] = nClasses_;
                float_array out(outShape.begin(), outShape.end());
                
                process_single_block(blockId, out);

                {
		        std::lock_guard<std::mutex> lock(this->s_mutex);
                out_->writeSubarray(outBegin.begin(), out);
                }
		    }		
	        });
#endif

            // TODO close the rawFile and outFile -> we need the filehandles
            return NULL;
        
        }



    private:
	    static std::mutex s_mutex;
        // global blocking
        std::unique_ptr<blocking> blocking_;
        std::unique_ptr<raw_cache> rawCache_;
        std::string in_file_;
        std::string in_key_;
        std::string rfFile_;
        std::string rfKey_;
        selection_type selectedFeatures_;
        coordinate blockShape_;
        random_forest rf_;
        std::unique_ptr<hdf5::Hdf5Array<data_type>> out_;
        coordinate roiBegin_;
        coordinate roiEnd_;
        size_t nClasses_;
    };
    
    template <unsigned DIM>
    std::mutex batch_prediction_task<DIM>::s_mutex;
    
} // namespace ilastik_backend
} // namepsace pipelines
} // namespace nifty
