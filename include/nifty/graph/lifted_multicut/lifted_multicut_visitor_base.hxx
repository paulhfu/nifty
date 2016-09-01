#pragma once
#ifndef NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_VISITOR_BASE_HXX
#define NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_VISITOR_BASE_HXX

#include <string>
#include <initializer_list>
#include <sstream>

namespace nifty {
namespace graph {
namespace lifted_multicut{

    template<class OBJECTIVE>
    class LiftedMulticutBase;


    template<class OBJECTIVE> 
    class LiftedMulticutVisitorBase{
    public:
        typedef OBJECTIVE Objective;
        typedef LiftedMulticutBase<Objective> McBase;

        // maybe the solver ptr will become a shared ptr
        virtual void begin(McBase * solver) = 0;
        virtual bool visit(McBase * solver) = 0;
        virtual void end(McBase * solver) = 0;

        virtual void addLogNames(std::initializer_list<std::string> logNames){

        }
        virtual void setLogValue(const size_t logIndex, double logValue){

        }
    };


    template<class OBJECTIVE> 
    class LiftedMulticutVerboseVisitor : public LiftedMulticutVisitorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE Objective;
        typedef LiftedMulticutBase<Objective> McBase;

        LiftedMulticutVerboseVisitor(const int printNth = 1)
        :   printNth_(printNth),
            runOpt_(true),
            iter_(1){
        }

        virtual void begin(McBase * ) {
            std::cout<<"begin inference\n";
        }
        virtual bool visit(McBase * solver) {
            if(iter_%printNth_ == 0){
                std::stringstream ss;
                ss<<solver->currentBestEnergy()<<" ";
                for(size_t i=0; i<logNames_.size(); ++i){
                    ss<<logNames_[i]<<" "<<logValues_[i]<<" ";
                }
                ss<<"\n";
                std::cout<<ss.str();
            }
            ++iter_;
            return runOpt_;
        }
        virtual void end(McBase * )   {
            std::cout<<"end inference\n";
        }
        virtual void addLogNames(std::initializer_list<std::string> logNames){
            logNames_.assign(logNames.begin(), logNames.end());
            logValues_.resize(logNames.size());
        }
        virtual void setLogValue(const size_t logIndex, double logValue){
            logValues_[logIndex] = logValue;
        }
        void stopOptimize(){
            runOpt_ = false;
        }
    private:
        bool runOpt_;
        int printNth_;
        int iter_;
        std::vector<std::string> logNames_;
        std::vector<double> logValues_;
    };

    template<class OBJECTIVE> 
    class LiftedMulticutEmptyVisitor : public LiftedMulticutVisitorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE Objective;
        typedef LiftedMulticutBase<Objective> McBase;
        virtual void begin(McBase * solver) {}
        virtual bool visit(McBase * solver) {return true;}
        virtual void end(McBase * solver)   {}
    private:
    };




    template<class OBJECTIVE>
    class LiftedMulticutVisitorProxy{
    public:
        typedef OBJECTIVE Objective;
        typedef LiftedMulticutVisitorBase<Objective> LiftedMulticutVisitorBaseType;
        typedef LiftedMulticutBase<Objective> McBase;
        LiftedMulticutVisitorProxy(LiftedMulticutVisitorBaseType * visitor)
        :   visitor_(visitor){

        }

        void addLogNames(std::initializer_list<std::string> logNames){
            if(visitor_  != nullptr){
                visitor_->addLogNames(logNames);
            }
        }
        void begin(McBase * solver) {
            if(visitor_ != nullptr){
                visitor_->begin(solver);
            }
        }
        bool visit(McBase * solver) {
            if(visitor_ != nullptr){
                return visitor_->visit(solver);
            }
            return false;
        }
        void end(McBase * solver)   {
            if(visitor_ != nullptr){
                visitor_->begin(solver);
            }
        }

        void setLogValue(const size_t logIndex, const double logValue)   {
            if(visitor_ != nullptr){
                visitor_->setLogValue(logIndex, logValue);
            }
        }

    private:
        LiftedMulticutVisitorBaseType * visitor_;
    };

}
} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_VISITOR_BASE_HXX