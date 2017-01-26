#pragma once

#include "nifty/pipelines/ilastik_backend/feature_computation_task.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_prediction_task.hxx"
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
        
        using raw_cache = hdf5::Hdf5Array<in_data_type>;
        // TODO does not really make sense to have to identical typedefs here
        using feature_cache    = tbb::concurrent_lru_cache<size_t, float_array, std::function<float_array(size_t)>>;
        using prediction_cache = tbb::concurrent_lru_cache<size_t, float_array, std::function<float_array(size_t)>>;
        using random_forest = RandomForestType;

        using blocking = tools::Blocking<DIM>;
        
        // empty constructor doing nothig
        //batch_prediction_task() :
        //{}
        

    public:

        // TODO we want to change the strings here to some simpler flags at some point
        using selection_type = std::pair<std::vector<std::string>,std::vector<double>>;
        
        // construct batch prediction for single input
        batch_prediction_task(const std::string & in_file,
                const std::string & in_key,
                const std::string & rf_file,
                const std::string & rf_key,
                const selection_type & selected_features,
                const coordinate  & block_shape,
                const size_t max_num_cache_entries) :
            blocking_(),
            rfFile_(rf_file),
            rfKey_(rf_key),
            in_file_(in_file),
            in_key_(in_key),
            featureCache_(),
            predictionCache_(),
            selectedFeatures_(selected_features),
            blockShape_(block_shape),
            maxNumCacheEntries_(max_num_cache_entries),
            rf_()
        {
            //init();
        }

        
        void init() {

            std::cout << "batch_prediction_task::init called" << std::endl;
            rawCache_ = std::make_unique<raw_cache>( hdf5::openFile(in_file_), in_key_ );

            // init the blocking
            coordinate volBegin = coordinate({0,0,0});
            coordinate volShape;
            for(size_t i = 0; i < DIM; i++)
                volShape[i] = rawCache_->shape(i);
            blocking_ = std::make_unique<blocking>(volBegin, volShape, blockShape_);

            // init the feature cache
            std::function<float_array(size_t)> retrieve_features_for_caching = [this](size_t blockId) -> float_array {

                // TODO FIXME this should not be hard-coded, but we need some static method that calculate this given the selected features
                size_t nChannels = 2;
                multichan_coordinate filterShape;
                filterShape[0] = nChannels;
                // TODO once we have halos, we need them here -> this can also be wrapped into a static method
                for(int d = 0; d < DIM; ++d)
                    filterShape[d+1] = blockShape_[d];

                float_array out_array(filterShape.begin(), filterShape.end());

		        feature_computation_task<DIM> task(
                        blockId,
                        *(this->rawCache_),
                        out_array,
                        this->selectedFeatures_,
                        *(this->blocking_));

		        task.execute();

                /*feature_computation_task<DIM> & feat_task = *new(tbb::task::allocate_child()) feature_computation_task<DIM>(
                        blockId,
                        *(this->rawCache_),
                        out_array,
                        this->selectedFeatures_,
                        *(this->blocking_));*/
                
                // TODO why ref-count 2
                //this->set_ref_count(2);
                //this->spawn_and_wait_for_all(feat_task);
                //spawn(feat_task);
                // TODO spawn or spawn_and_wait
                return out_array;
            };
            
            featureCache_ = std::make_unique<feature_cache>(retrieve_features_for_caching, maxNumCacheEntries_);
            
            rf_ = get_rf_from_file(rfFile_, rfKey_);
            
            size_t n_classes = 2; // TODO FIXME don't hardcode, get from rf
            
            // init the prediction cache
            std::function<float_array(size_t)> retrieve_prediction_for_caching = [this](size_t blockId) -> float_array {
                
                size_t n_classes = 2; // TODO FIXME don't hardcode, get from rf
                
                multichan_coordinate predictionShape;
                predictionShape[DIM] = n_classes;
                for(int d = 0; d < DIM; ++d)
                    predictionShape[d] = this->blockShape_[d];
                float_array out_array(predictionShape.begin(), predictionShape.end());
                
                random_forest_prediction_task<DIM> task(blockId, *(this->featureCache_), out_array, this->rf_);
                task.execute();

                //random_forest_prediction_task<DIM> & rf_task = *new(tbb::task::allocate_child()) random_forest_prediction_task<DIM>(blockId, *(this->featureCache_), out_array, this->rf_);
                // TODO why ref count 2
                //this->set_ref_count(2);
                // TODO spawn or spawn_and_wait
                //this->spawn_and_wait_for_all(rf_task);
                //this->spawn(rf_task);
                return out_array;
            };

            predictionCache_ = std::make_unique<prediction_cache>(retrieve_prediction_for_caching, maxNumCacheEntries_);

            multichan_coordinate outShape, chunkShape; 
            outShape[DIM] = n_classes;
            chunkShape[DIM] = 1;
            for(int d = 0; d < DIM; ++d) {
                outShape[d] = volShape[d];
                chunkShape[d] = blockShape_[d];
            }
            out_ = std::make_unique<hdf5::Hdf5Array<data_type>>( hdf5::createFile("./out.h5"), "data", outShape.begin(), outShape.end(), chunkShape.begin() );
        }
 
        tbb::task* execute() {
            
            std::cout << "batch_prediction_task::execute called" << std::endl;
            
            init();


#if 0


            // feature output for debugging only
            auto feat_tmp = nifty::hdf5::createFile("./feat_tmp.h5");
            size_t feat_shape[] = {128,128,128,2};
            size_t chunk_shape[] = {64,64,64,1};
            nifty::hdf5::Hdf5Array<float> feats(feat_tmp, "data", feat_shape, feat_shape + 4, chunk_shape );
            
            // TODO spawn the tasks to batch process the complete volume
            for(size_t blockId = 0; blockId < blocking_->numberOfBlocks(); ++blockId) {

                std::cout << "Processing block " << blockId << " / " << blocking_->numberOfBlocks() << std::endl;
                
                auto block = blocking_->getBlock(blockId);
                coordinate blockBegin = block.begin();
                
                // need to attach the channel coordinate
                multichan_coordinate outBegin;
                for(int d = 0; d < DIM; ++d)
                    outBegin[d] = blockBegin[d];
                outBegin[DIM] = 0;

                // write the features, debugging only
                auto featHandlde = (*featureCache_)[blockId];
                auto featView = featHandlde.value();
                feats.writeSubarray(outBegin.begin(), featView);
                
                auto handle = (*predictionCache_)[blockId];
                auto outView = handle.value();

                std::cout << "Write start" << std::endl;
                std::cout << outBegin << std::endl;
                
                //std::cout << "Prediction shape" << std::endl;
                //for(int dd = 0; dd < outView.dimension(); ++dd)
                //    std::cout << outView.shape(dd) << std::endl;
                
                out_->writeSubarray(outBegin.begin(), outView);
                std::cout << "Processing block " << blockId << " done!" << std::endl;
            }
#endif
	    std::mutex m;

        // TODO FIXME why two loops ?!
	    tbb::parallel_for(tbb::blocked_range<size_t>(0,blocking_->numberOfBlocks()), [this, &m](const tbb::blocked_range<size_t> &range) {
		for( size_t blockId=range.begin(); blockId!=range.end(); ++blockId ) {

                std::cout << "Processing block " << blockId << " / " << blocking_->numberOfBlocks() << std::endl;

                auto handle = (*predictionCache_)[blockId];
                auto outView = handle.value();
                //std::cout << "handle with caution" << std::endl;

                auto block = blocking_->getBlock(blockId);
                coordinate blockBegin = block.begin();

                // need to attach the channel coordinate
                multichan_coordinate outBegin;
                for(int d = 0; d < DIM; ++d)
                    outBegin[d] = blockBegin[d];
                outBegin[DIM] = 0;

                //std::cout << "Write start" << std::endl;
                //std::cout << outBegin << std::endl;

		        std::lock_guard<std::mutex> lock(m);
                out_->writeSubarray(outBegin.begin(), outView);
                //std::cout << "Processing block " << blockId << " done!" << std::endl;
		}		
	    });

            
            // TODO close the rawFile and outFile -> we need the filehandles
            return NULL;
        }



    private:
        // global blocking
        std::unique_ptr<blocking> blocking_;
        std::unique_ptr<raw_cache> rawCache_;
        std::string in_file_;
        std::string in_key_;
        std::string rfFile_;
        std::string rfKey_;
        std::unique_ptr<feature_cache> featureCache_;
        std::unique_ptr<prediction_cache> predictionCache_;
        selection_type selectedFeatures_;
        coordinate blockShape_;
        size_t maxNumCacheEntries_;
        random_forest rf_;
        std::unique_ptr<hdf5::Hdf5Array<data_type>> out_;
    };
    
} // namespace ilastik_backend
} // namepsace pipelines
} // namespace nifty
