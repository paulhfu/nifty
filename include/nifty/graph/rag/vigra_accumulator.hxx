#pragma once
#ifndef NIFTY_VIGRA_ACCUMULATOR_HXX
#define NIFTY_VIGRA_ACCUMULATOR_HXX

#include "vigra/accumulator.hxx"

namespace nifty{
namespace graph{
    
    using namespace vigra::acc;

    template<class T, unsigned int NBINS>
    class VigraAcc{
        
    public:
        
        // AutoRangeHistogram -> needs 2 passes, but we only want 1 for the chunked rag
        //typedef StandardQuantiles<AutoRangeHistogram<NBINS> > Quantiles;
        
        // UserRangeHistogra, -> need 1 pass, but have to set the min and max
        typedef StandardQuantiles<UserRangeHistogram<NBINS> > Quantiles;

        // features are hardcoded for now, could expose them as template
        // Skewness and Kurtosis need 2 passes (at least in the current vigra impl), so we don't use them for now
        //typedef Select<Mean, Sum, Minimum, Maximum, Variance, Skewness, Kurtosis, Quantiles> Features;
        typedef Select<Mean, Sum, Minimum, Maximum, Variance, Quantiles> Features;
        typedef AccumulatorChain<T, Features> Accumulator;

        VigraAcc()
        :  accumulator_(){
        }
            
        void setMinMax(T min, T max) {
            accumulator_.setHistogramOptions(vigra::HistogramOptions().setMinMax(min, max));
        }

        void accumulate(const T & val, const size_t & pass  ){
            accumulator_.updatePassN(val, pass+1); // vigra starts countig passes at 1
        }
        
        // merge the underlying accumulator chains
        // TODO make sure that this works for the statistcs used
        void merge(const VigraAcc & other){
            accumulator_.merge(other);
        }
        
        void reset() {
            accumulator_.reset();
        }

        void getFeatures(T * featuresOut){
            featuresOut[0] = get<Mean>(accumulator_); 
            featuresOut[1] = get<Sum>(accumulator_);
            featuresOut[2] = get<Minimum>(accumulator_);
            featuresOut[3] = get<Maximum>(accumulator_);
            featuresOut[4] = get<Variance>(accumulator_);
            //featuresOut[5] = get<Skewness>(accumulator_);
            //featuresOut[6] = get<Kurtosis>(accumulator_);
            vigra::TinyVector<T, 7> quants = get<Quantiles>(accumulator_);
            featuresOut[5] = quants[1];
            featuresOut[6] = quants[2];
            featuresOut[7] = quants[3];
            featuresOut[8] = quants[4];
            featuresOut[9] = quants[5];
        }

        size_t numberOfPasses() const {
            return accumulator_.passesRequired();
        }
 
    private:
        Accumulator accumulator_;
    };
    
    template<class GRAPH, class T, unsigned int NBINS, template<class> class ITEM_MAP >
    class VigraAccMapBase 
    {
    public:
        typedef GRAPH Graph;
    
        VigraAccMapBase(const Graph & graph)
        :   graph_(graph),
            currentPass_(0),
            accs_(graph){

        }

        const Graph & graph()const{
            return graph_;
        }

        void accumulate(const uint64_t item, const T & val){
            accs_[item].accumulate(val, currentPass_);
        }

        void startPass(const size_t passIndex){
            currentPass_ = passIndex;
        }

        size_t numberOfPasses()const{
            return accs_[0].numberOfPasses();
        }
        
        void setMinMax(T min, T max) {
            for(auto &acc : accs_)
                 acc.setMinMax(min, max);
        }

        void merge(const uint64_t alive,  const uint64_t dead){
            accs_[alive].merge(accs_[dead]);
        }

        // TODO get from the accumulator chain and don't hardcode
        uint64_t numberOfFeatures() const {
            return 10;
        }

        void getFeatures(const uint64_t item, T * featuresOut){
            accs_[item].getFeatures(featuresOut);
        }   

        void resetFrom(const VigraAccMapBase & other){
            std::copy(other.accs_.begin(), other.accs_.end(), accs_.begin());
            currentPass_ = other.currentPass_;
        }

        void reset() {
            for(auto &acc : accs_)
                acc.reset();
        }

    private:
        const GRAPH & graph_;
        size_t currentPass_;
        ITEM_MAP<VigraAcc<T, NBINS> > accs_;
    };



    template<class GRAPH, class T, unsigned int NBINS=40>
    class VigraAccEdgeMap : 
        public  VigraAccMapBase<GRAPH, T, NBINS, GRAPH:: template EdgeMap >
    {
    public:
        typedef GRAPH Graph;
        //typedef typename Graph:: template EdgeMap<DefaultAcc<T, NBINS> > BasesBaseType;
        typedef VigraAccMapBase<Graph, T, NBINS, Graph:: template EdgeMap >  BaseType;
        using BaseType::BaseType;
        // need to have these for the pybindings
        void reset() {
            BaseType::reset();
        }
        void setMinMax(T min, T max) {
            BaseType::setMinMax(min, max);
        }
    };

    template<class GRAPH, class T, unsigned int NBINS=40>
    class VigraAccNodeMap : 
        public  VigraAccMapBase<GRAPH, T, NBINS, GRAPH:: template NodeMap >
    {
    public:
        typedef GRAPH Graph;
        //typedef typename Graph:: template NodeMap<DefaultAcc<T, NBINS> > BasesBaseType;
        typedef VigraAccMapBase<Graph, T, NBINS, Graph:: template NodeMap >  BaseType;
        using BaseType::BaseType;
        // need to have these for the pybindings
        void reset() {
            BaseType::reset();
        }
        void setMinMax(T min, T max) {
            BaseType::setMinMax(min, max);
        }
    };


} // namespace graph
} // namespace nifty

#endif /* NIFTY_VIGRA_ACCUMULATOR_HXX */
