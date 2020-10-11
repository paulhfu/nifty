import numpy as np
import nifty
import nifty.graph.agglo as nagglo

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def check_fully_connected():
    g = nifty.graph.undirectedGraph(5)
    edges = np.array([[0, 1],
                      [0, 2],
                      [0, 3],
                      [0, 4],
                      [1, 2],
                      [1, 3],
                      [1, 4],
                      [2, 3],
                      [2, 4],
                      [3, 4]])
    g.insertEdges(edges)

    weights = np.random.rand(len(edges))
    edge_sizes = np.ones(len(edges))
    node_sizes = np.ones(5)

    n_stop = 1

    with_ucm = True
    policy_class = nagglo.edgeWeightedClusterPolicyWithUcm if with_ucm else nagglo.edgeWeightedClusterPolicy
    policy = policy_class(graph=g,
                          edgeIndicators=weights,
                          nodeSizes=node_sizes.astype('float'),
                          edgeSizes=edge_sizes.astype('float'),
                          numberOfNodesStop=n_stop,
                          sizeRegularizer=1.)

    clustering = nagglo.agglomerativeClustering(policy)
    us, vs, dist, sizes = clustering.runAndGetLinkageMatrix(False)
    lm = [us, vs, dist, sizes]
    lm = list(map(np.array, lm))
    lm = np.concatenate([xx[:, None] for xx in lm], axis=1)
    print(lm)

    print()
    lm_scipy = linkage(weights, method='ward')
    print(lm_scipy)
    dn = dendrogram(lm)
    plt.show()


def check_toy():
    g = nifty.graph.undirectedGraph(5)
    edges = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 4]])
    g.insertEdges(edges)

    weights = np.array([0.4, 0.3, 0.9, 0.1])

    edge_sizes = np.array([1., 1., 1., 1.])
    node_sizes = np.ones(5)

    n_stop = 1

    with_ucm = True
    policy_class = nagglo.edgeWeightedClusterPolicyWithUcm if with_ucm else nagglo.edgeWeightedClusterPolicy
    policy = policy_class(graph=g,
                          edgeIndicators=weights,
                          nodeSizes=node_sizes.astype('float'),
                          edgeSizes=edge_sizes.astype('float'),
                          numberOfNodesStop=n_stop,
                          sizeRegularizer=1.)

    clustering = nagglo.agglomerativeClustering(policy)
    us, vs, dist, sizes = clustering.runAndGetLinkageMatrix(False)
    lm = [us, vs, dist, sizes]
    lm = list(map(np.array, lm))
    lm = np.concatenate([xx[:, None] for xx in lm], axis=1)

    print(lm)
    dn = dendrogram(lm)
    plt.show()


# check_fully_connected()
check_toy()
