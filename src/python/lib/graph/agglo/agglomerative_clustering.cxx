#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/agglo/export_agglomerative_clustering.hxx"

#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/agglo/agglomerative_clustering.hxx"
#include "nifty/graph/agglo/cluster_policies/mala_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/constrained_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/constrained_generalized_mean_fixation_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/edge_weighted_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/node_and_edge_weighted_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/minimum_node_size_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/lifted_graph_edge_weighted_cluster_policy.hxx"



namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


    template<class GRAPH, class CONTR_GRAPH, bool WITH_UCM>
    void exportConstrainedPolicy(py::module & aggloModule) {

        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1> PyViewFloat1;
        typedef xt::pytensor<int, 1> PyViewInt1;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {
            // name and type of cluster operator
            typedef ConstrainedPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("constrainedHierarchicalClustering") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            auto clusterClass = py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str());


            clusterClass
                    .def("runNextMilestep", [](
                                 ClusterPolicyType * self,
                                 const int nb_iterations_in_milestep,
                                 const PyViewFloat1 & newEdgeIndicators
                         ){
                             bool out;
                             {
                                 py::gil_scoped_release allowThreads;
                                 out = self->runMileStep(nb_iterations_in_milestep,newEdgeIndicators);
                             }
                             return out;
                         },
                         py::arg("nb_iterations_in_milestep") = -1,
                         py::arg("new_edge_indicators")
                    )
                    .def("collectDataMilestep", [](
                                 ClusterPolicyType * self
                         ){
                             const auto graph = self->graph();
                             typedef xt::pytensor<float, 1> arrayFloat;
                             arrayFloat nodeSizes({size_t(graph.nodeIdUpperBound()+1)});
                             arrayFloat nodeLabels({size_t(graph.nodeIdUpperBound()+1)});
                             arrayFloat nodeGTLabels({size_t(graph.nodeIdUpperBound()+1)});
                             arrayFloat edgeSizes({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat edgeIndicators({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat dendHeigh({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat mergeTimes({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat lossTargets({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat lossWeights({size_t(graph.edgeIdUpperBound()+1)});
                             {
                                 py::gil_scoped_release allowThreds;
                                 self->collectDataMilestep(nodeSizes,nodeLabels,nodeGTLabels,edgeSizes,edgeIndicators,
                                    dendHeigh,mergeTimes,lossTargets,lossWeights);
                             }
                             return std::tuple<arrayFloat,arrayFloat,arrayFloat,arrayFloat,arrayFloat,arrayFloat,
                                     arrayFloat,arrayFloat,arrayFloat>(nodeSizes,nodeLabels,nodeGTLabels,edgeSizes,edgeIndicators,
                                                                          dendHeigh,mergeTimes,lossTargets,lossWeights);
                         }
                    )
                    .def("time", [] (
                                 ClusterPolicyType * self
                         ){
                            uint64_t time;
                            {
                                 py::gil_scoped_release allowThreds;
                                 time = self->time();
                             }
                             return time;
                         }

                    )
//                    .def("edgeContractionGraph", [](
//                                 ClusterPolicyType * self
//                         ){
//
//                            const auto graph = self->graph();
//                             nifty::marray::PyView<float> out({size_t(graph.edgeIdUpperBound()+1)});
//                             {
//                                 py::gil_scoped_release allowThreds;
//                                 self->lossTargets(out);
//                             }
//                             return out;
//                         }
//                    )
//                    .def("lossTargets", [](
//                                 ClusterPolicyType * self
//                         ){
//                             const auto graph = self->graph();
//                             nifty::marray::PyView<float> out({size_t(graph.edgeIdUpperBound()+1)});
//                             {
//                                 py::gil_scoped_release allowThreds;
//                                 self->lossTargets(out);
//                             }
//                        return out;
//                         }
//                    )
//                    .def("lossWeights", [](
//                                 ClusterPolicyType * self
//                         ){
//                             const auto graph = self->graph();
//                             nifty::marray::PyView<float> out({size_t(graph.edgeIdUpperBound()+1)});
//                             {
//                                 py::gil_scoped_release allowThreds;
//                                 self->lossWeights(out);
//                             }
//                             return out;
//                         }
//                    )
                    ;



            // factory
//            TODO: how do I add another function to get back the targets, new SP sizes, new GT labels
            aggloModule.def(clusterPolicyFacName.c_str(),
                            [](
                                    const GraphType & graph,
                                    const PyViewFloat1 & edgeIndicators,
                                    const PyViewFloat1 & edgeSizes,
                                    const PyViewFloat1 & nodeSizes,
                                    const PyViewInt1  & GTlabels,
                                    const float threshold,
                                    const uint64_t numberOfNodesStop,
                                    const int bincount,
                                    const int ignore_label,
                                    const bool constrained,
                                    const bool computeLossData,
                                    const bool verbose
                            ){
                                typename ClusterPolicyType::SettingsType s;
                                s.numberOfNodesStop = numberOfNodesStop;
                                s.bincount = bincount;
                                s.threshold = threshold;
                                s.ignore_label = (uint64_t) ignore_label;
                                s.constrained = constrained;
                                s.computeLossData = computeLossData;
                                s.verbose = verbose;
                                auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes,
                                                                 nodeSizes, GTlabels, s);
                                return ptr;
                            },
                            py::return_value_policy::take_ownership,
                            py::keep_alive<0,1>(), // graph
                            py::arg("graph"),
//                          TODO: add contractedGraph, offset for time
                            py::arg("edgeIndicators"),
                            py::arg("edgeSizes"),
                            py::arg("nodeSizes"),
                            py::arg("GTlabels"),
                            py::arg("threshold") = 0.5,
                            py::arg("numberOfNodesStop") = 1,
                            py::arg("bincount") = 40,
                            py::arg("ignore_label") = -1,
                            py::arg("constrained") = true,
                            py::arg("computeLossData") = true,
                            py::arg("verbose") = false
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }

    template<class GRAPH, class CONTR_GRAPH, bool WITH_UCM>
    void exportConstrainedGeneralizedMeanPolicy(py::module & aggloModule) {

        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1> PyViewFloat1;
        typedef xt::pytensor<int, 1> PyViewInt1;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {
            // name and type of cluster operator
            typedef ConstrainedGeneralizedMeanFixationClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("constrainedGeneralizedFixationClustering") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            auto clusterClass = py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str());


            clusterClass
                    .def("runNextMilestep", [](
                                 ClusterPolicyType * self,
                                 const int nb_iterations_in_milestep,
                                 const PyViewFloat1 & newMergePrios,
                                 const PyViewFloat1 & newNotMergePrios
                         ){
                             bool out;
                             {
                                 py::gil_scoped_release allowThreads;
                                 out = self->runMileStep(nb_iterations_in_milestep,newMergePrios,newNotMergePrios);
                             }
                             return out;
                         },
                         py::arg("nb_iterations_in_milestep") = -1,
                         py::arg("new_merge_prios"),
                         py::arg("new_not_merge_prios")
                    )
                    .def("collectDataMilestep", [](
                                 ClusterPolicyType * self
                         ){
                             const auto graph = self->graph();
                             typedef xt::pytensor<float, 1> arrayFloat;
                             // TODO this will create arrays of size 1. use xt::zeros instead
                             arrayFloat nodeSizes({size_t(graph.nodeIdUpperBound()+1)});
                             arrayFloat nodeLabels({size_t(graph.nodeIdUpperBound()+1)});
                             arrayFloat nodeGTLabels({size_t(graph.nodeIdUpperBound()+1)});
                             arrayFloat edgeSizes({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat mergePrios({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat dendHeigh({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat mergeTimes({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat lossTargets({size_t(graph.edgeIdUpperBound()+1)});
                             arrayFloat lossWeights({size_t(graph.edgeIdUpperBound()+1)});
                             {
                                 py::gil_scoped_release allowThreds;
                                 self->collectDataMilestep(nodeSizes,nodeLabels,nodeGTLabels,edgeSizes,mergePrios,
                                                           dendHeigh,mergeTimes,lossTargets,lossWeights);
                             }
                             return std::tuple<arrayFloat,arrayFloat,arrayFloat,arrayFloat,arrayFloat,arrayFloat,
                                     arrayFloat,arrayFloat,arrayFloat>(nodeSizes,nodeLabels,nodeGTLabels,edgeSizes,mergePrios,
                                                                          dendHeigh,mergeTimes,lossTargets,lossWeights);
                         }
                    )
                    .def("time", [] (
                                 ClusterPolicyType * self
                         ){
                             uint64_t time;
                             {
                                 py::gil_scoped_release allowThreds;
                                 time = self->time();
                             }
                             return time;
                         }

                    )
                    ;



            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                            [](
                                    const GraphType & graph,
                                    const PyViewFloat1 & isLocalEdge,
                                    const double p0,
                                    const double p1,
                                    const bool zeroInit,
                                    const double weight_mistakes,
                                    const double weight_successes,
                                    const PyViewFloat1 & edgeSizes,
                                    const PyViewFloat1 & nodeSizes,
                                    const PyViewInt1  & GTlabels,
                                    const float threshold,
                                    const uint64_t numberOfNodesStop,
                                    const int bincount,
                                    const int ignore_label,
                                    const bool constrained,
                                    const bool computeLossData,
                                    const bool verbose
                            ){
                                typename ClusterPolicyType::SettingsType s;
                                s.numberOfNodesStop = numberOfNodesStop;
                                s.p0 = p0;
                                s.p1 = p1;
                                s.weight_successes = weight_successes;
                                s.weight_mistakes = weight_mistakes;
                                s.zeroInit = zeroInit;
                                s.bincount = bincount;
                                s.threshold = threshold;
                                s.ignore_label = (uint64_t) ignore_label;
                                s.constrained = constrained;
                                s.computeLossData = computeLossData;
                                s.verbose = verbose;
                                auto ptr = new ClusterPolicyType(graph, isLocalEdge, edgeSizes,
                                                                 nodeSizes, GTlabels, s);
                                return ptr;
                            },
                            py::return_value_policy::take_ownership,
                            py::keep_alive<0,1>(), // graph
                            py::arg("graph"),
                            py::arg("isLocalEdge"),
                            py::arg("p0") = 1.0,
                            py::arg("p1") = 1.0,
                            py::arg("zeroInit") = false,
                            py::arg("weight_mistakes") = 1.0,
                            py::arg("weight_successes") = 1.0,
                            py::arg("edgeSizes"),
                            py::arg("nodeSizes"),
                            py::arg("GTlabels"),
                            py::arg("threshold") = 0.5,
                            py::arg("numberOfNodesStop") = 1,
                            py::arg("bincount") = 40,
                            py::arg("ignore_label") = -1,
                            py::arg("constrained") = true,
                            py::arg("computeLossData") = true,
                            py::arg("verbose") = false
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    template<class GRAPH, bool WITH_UCM>
    void exportLiftedGraphEdgeWeightedPolicy(py::module & aggloModule) {
        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1> PyViewFloat1;
        typedef xt::pytensor<bool, 1> PyViewBool1;
        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string();

        {
            // name and type of cluster operator
            typedef LiftedGraphEdgeWeightedClusterPolicy<GraphType, PyViewFloat1, PyViewFloat1, PyViewFloat1, PyViewBool1, WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("LiftedGraphEdgeWeightedClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                //.def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat1 & nodeSizes,
                    const PyViewBool1 & edgeIsLifted,
                    const uint64_t numberOfNodesStop,
                    const float sizeRegularizer
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.sizeRegularizer = sizeRegularizer;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, edgeIsLifted, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0, 1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("edgeIsLifted"),
                py::arg("numberOfNodesStop")=1,
                py::arg("sizeRegularizer")=0.5f
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    template<class GRAPH, bool WITH_UCM>
    void exportMalaClusterPolicy(py::module & aggloModule) {

        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1> PyViewFloat1;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {
            // name and type of cluster operator
            typedef MalaClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("MalaClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                //.def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat1 & nodeSizes,
                    const float threshold,
                    const uint64_t numberOfNodesStop,
                    const int bincount,
                    const bool verbose
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.bincount = bincount;
                    s.threshold = threshold;
                    s.verbose = verbose;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("threshold") = 0.5,
                py::arg("numberOfNodesStop") = 1,
                py::arg("bincount") = 40,
                py::arg("verbose") = false
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }




    template<class GRAPH, bool WITH_UCM>
    void exportEdgeWeightedClusterPolicy(py::module & aggloModule) {

        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1> PyViewFloat1;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {
            // name and type of cluster operator
            typedef EdgeWeightedClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("EdgeWeightedClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                .def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                .def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat1 & nodeSizes,
                    const uint64_t numberOfNodesStop,
                    const float sizeRegularizer
                ){
                    EdgeWeightedClusterPolicySettings s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.sizeRegularizer = sizeRegularizer;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("numberOfNodesStop") = 1,
                py::arg("sizeRegularizer") = 0.5f
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule,
                                                                           clusterPolicyBaseName);
        }
    }

    template<class GRAPH, bool WITH_UCM>
    void exportNodeAndEdgeWeightedClusterPolicy(py::module & aggloModule) {

        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1> PyViewFloat1;
        typedef xt::pytensor<float, 2> PyViewFloat2;

        const std::string withUcmStr =  WITH_UCM ? std::string("WithUcm") :  std::string() ;

        {
            // name and type of cluster operator
            typedef NodeAndEdgeWeightedClusterPolicy<GraphType,WITH_UCM> ClusterPolicyType;
            const auto clusterPolicyBaseName = std::string("NodeAndEdgeWeightedClusterPolicy") +  withUcmStr;
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                .def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                .def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat2 & nodeFeatures,
                    const PyViewFloat1 & nodeSizes,
                    const float beta,
                    const uint64_t numberOfNodesStop,
                    const float sizeRegularizer
                ){
                    typename ClusterPolicyType::SettingsType s;
                    s.numberOfNodesStop = numberOfNodesStop;
                    s.sizeRegularizer = sizeRegularizer;
                    s.beta = beta;

                    // create a MultibandArrayViewNodeMap
                    nifty::graph::graph_maps::MultibandArrayViewNodeMap<PyViewFloat2> nodeFeaturesView(nodeFeatures);

                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeFeaturesView, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeFeatures"),
                py::arg("nodeSizes"),
                py::arg("beta") = 0.5f,
                py::arg("numberOfNodesStop") = 1,
                py::arg("sizeRegularizer") = 0.5f
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }


    template<class GRAPH>
    void exportMinimumNodeSizeClusterPolicy(py::module & aggloModule) {

        typedef GRAPH GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 1> PyViewFloat1;

        {
            // name and type of cluster operator
            typedef MinimumNodeSizeClusterPolicy<GraphType> ClusterPolicyType;
            typedef typename ClusterPolicyType::SettingsType Setting;
            const auto clusterPolicyBaseName = std::string("MinimumNodeSizeClusterPolicy");
            const auto clusterPolicyClsName = clusterPolicyBaseName + graphName;
            const auto clusterPolicyFacName = lowerFirst(clusterPolicyBaseName);

            // the cluster operator cls
            py::class_<ClusterPolicyType>(aggloModule, clusterPolicyClsName.c_str())
                //.def_property_readonly("edgeIndicators", &ClusterPolicyType::edgeIndicators)
                //.def_property_readonly("edgeSizes", &ClusterPolicyType::edgeSizes)
            ;

            // factory
            aggloModule.def(clusterPolicyFacName.c_str(),
                [](
                    const GraphType & graph,
                    const PyViewFloat1 & edgeIndicators,
                    const PyViewFloat1 & edgeSizes,
                    const PyViewFloat1 & nodeSizes,
                    const double minimumNodeSize,
                    const double sizeRegularizer,
                    const double gamma
                ){
                    Setting s;
                    s.minimumNodeSize = minimumNodeSize;
                    s.sizeRegularizer = sizeRegularizer;
                    s.gamma = gamma;
                    auto ptr = new ClusterPolicyType(graph, edgeIndicators, edgeSizes, nodeSizes, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0,1>(), // graph
                py::arg("graph"),
                py::arg("edgeIndicators"),
                py::arg("edgeSizes"),
                py::arg("nodeSizes"),
                py::arg("minimumNodeSize") = 1,
                py::arg("sizeRegularizer") = 0.001,
                py::arg("gamma") = 0.999
            );

            // export the agglomerative clustering functionality for this cluster operator
            exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(aggloModule, clusterPolicyBaseName);
        }
    }



    void exportAgglomerativeClustering(py::module & aggloModule) {
        typedef PyUndirectedGraph UndirectedGraphType;
        typedef PyContractionGraph<PyUndirectedGraph> ContractionGraphType;
        {
            typedef PyUndirectedGraph GraphType;

            exportMalaClusterPolicy<GraphType, false>(aggloModule);
            exportMalaClusterPolicy<GraphType, true>(aggloModule);

            exportConstrainedPolicy<GraphType, ContractionGraphType, false>(aggloModule);
            exportConstrainedPolicy<GraphType, ContractionGraphType, true>(aggloModule);

            exportConstrainedGeneralizedMeanPolicy<GraphType, ContractionGraphType, false>(aggloModule);
            exportConstrainedGeneralizedMeanPolicy<GraphType, ContractionGraphType, true>(aggloModule);

            exportEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

//            exportNodeAndEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
//            exportNodeAndEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportMinimumNodeSizeClusterPolicy<GraphType>(aggloModule);

            exportLiftedGraphEdgeWeightedPolicy<GraphType, false>(aggloModule);
            exportLiftedGraphEdgeWeightedPolicy<GraphType, true>(aggloModule);
        }

        {
            typedef UndirectedGridGraph<2,true> GraphType;

            exportMalaClusterPolicy<GraphType, false>(aggloModule);
            exportMalaClusterPolicy<GraphType, true>(aggloModule);

            exportConstrainedPolicy<GraphType, ContractionGraphType, false>(aggloModule);
            exportConstrainedPolicy<GraphType, ContractionGraphType, true>(aggloModule);

            exportConstrainedGeneralizedMeanPolicy<GraphType, ContractionGraphType, false>(aggloModule);
            exportConstrainedGeneralizedMeanPolicy<GraphType, ContractionGraphType, true>(aggloModule);

            exportEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

//            exportNodeAndEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
//            exportNodeAndEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportMinimumNodeSizeClusterPolicy<GraphType>(aggloModule);

            exportLiftedGraphEdgeWeightedPolicy<GraphType, false>(aggloModule);
            exportLiftedGraphEdgeWeightedPolicy<GraphType, true>(aggloModule);
        }

        {
            typedef UndirectedGridGraph<3,true> GraphType;

            exportMalaClusterPolicy<GraphType, false>(aggloModule);
            exportMalaClusterPolicy<GraphType, true>(aggloModule);

            exportConstrainedPolicy<GraphType, ContractionGraphType, false>(aggloModule);
            exportConstrainedPolicy<GraphType, ContractionGraphType, true>(aggloModule);

            exportConstrainedGeneralizedMeanPolicy<GraphType, ContractionGraphType, false>(aggloModule);
            exportConstrainedGeneralizedMeanPolicy<GraphType, ContractionGraphType, true>(aggloModule);

            exportEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
            exportEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

//            exportNodeAndEdgeWeightedClusterPolicy<GraphType, false>(aggloModule);
//            exportNodeAndEdgeWeightedClusterPolicy<GraphType, true>(aggloModule);

            exportMinimumNodeSizeClusterPolicy<GraphType>(aggloModule);

            exportLiftedGraphEdgeWeightedPolicy<GraphType, false>(aggloModule);
            exportLiftedGraphEdgeWeightedPolicy<GraphType, true>(aggloModule);
        }
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
