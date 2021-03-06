#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

void undirectedGraphTest()
{
    auto e = 0;
    nifty::graph::UndirectedGraph<> graph(4);
    NIFTY_TEST_OP(graph.numberOfNodes(),==,4);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,0);

    e = graph.insertEdge(0,1);
    NIFTY_TEST_OP(e,==,0);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,1);
    NIFTY_TEST_OP(graph.u(e),==,0);
    NIFTY_TEST_OP(graph.v(e),==,1);


    e = graph.insertEdge(0,2);
    NIFTY_TEST_OP(e,==,1);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,2);
    NIFTY_TEST_OP(graph.u(e),==,0);
    NIFTY_TEST_OP(graph.v(e),==,2);

    e = graph.insertEdge(0,3);
    NIFTY_TEST_OP(e,==,2);
    NIFTY_TEST_OP(graph.numberOfEdges(),==,3);
    NIFTY_TEST_OP(graph.u(e),==,0);
    NIFTY_TEST_OP(graph.v(e),==,3);


    e = graph.insertEdge(2,3);
    NIFTY_TEST_OP(e,==,3);  
    NIFTY_TEST_OP(graph.numberOfEdges(),==,4);
    NIFTY_TEST_OP(graph.u(e),==,2);
    NIFTY_TEST_OP(graph.v(e),==,3);


    auto c=0;
    for(auto iter = graph.nodesBegin(); iter!=graph.nodesEnd(); ++iter){
        NIFTY_TEST_OP(*iter,==,c);
        ++c;
    }
    NIFTY_TEST_OP(graph.numberOfNodes(),==,c);

    c = 0;
    for(auto node : graph.nodes()){
        NIFTY_TEST_OP(node,==,c);
        ++c;
        for(auto adj : graph.adjacency(node)){
        }
    }
    NIFTY_TEST_OP(graph.numberOfNodes(),==,c);

    c = 0;
    //std::cout<<"for graph edges\n";
    for(auto & edge : graph.edges()){
        NIFTY_TEST_OP(edge,==,c);
        ++c;
    }
    NIFTY_TEST_OP(graph.numberOfEdges(),==,c);

    c = 0;
    for(auto & edge : graph.items<nifty::graph::EdgeTag>()){
        NIFTY_TEST_OP(edge,==,c);
        ++c;
    }
    NIFTY_TEST_OP(graph.numberOfEdges(),==,c);

}

int main(){
    undirectedGraphTest();
}
