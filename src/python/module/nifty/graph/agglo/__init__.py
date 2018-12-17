from __future__ import absolute_import
from . import _agglo as __agglo
from ._agglo import *

import numpy

__all__ = []
for key in __agglo.__dict__.keys():
    __all__.append(key)
    try:
        __agglo.__dict__[key].__module__='nifty.graph.agglo'
    except:
        pass

from ...tools import makeDense as __makeDense


def updatRule(name, **kwargs):
    if name == 'max':
        return MaxSettings()
    elif name == 'min':
        return MinSettings()
    elif name == 'sum':
        return SumSettings()
    elif name == 'mean':
        return ArithmeticMeanSettings()
    elif name in ['gmean', 'generalized_mean']:
        p = kwargs.get('p',1.0)
        return GeneralizedMeanSettings(p=float(p))
    elif name in ['smax', 'smooth_max']:
        p = kwargs.get('p',0.0)
        return SmoothMaxSettings(p=float(p))
    elif name in ['rank','quantile', 'rank_order']:
        q = kwargs.get('q',0.5)
        numberOfBins = kwargs.get('numberOfBins',40)
        return RankOrderSettings(q=float(q), numberOfBins=int(numberOfBins))
    else:
        return NotImplementedError("not yet implemented")


# def fixationClusterPolicy(graph, 
#     mergePrios=None,
#     notMergePrios=None,
#     edgeSizes=None,
#     isLocalEdge=None,
#     updateRule0="smooth_max",
#     updateRule1="smooth_max",
#     p0=float('inf'),
#     p1=float('inf'),
#     zeroInit=False):
    
#     if isLocalEdge is None:
#         raise RuntimeError("`isLocalEdge` must not be none")

#     if mergePrios is None and if notMergePrios is  None:
#         raise RuntimeError("`mergePrios` and `notMergePrios` cannot be both None")

#     if mergePrio is None:
#         nmp = notMergePrios.copy()
#         nmp -= nmp.min()
#         nmp /= nmp.max()
#         mp = 1.0 = nmp
#     elif notMergePrios is None:
#         mp = notMergePrios.copy()
#         mp -= mp.min()
#         mp /= mp.max()
#         nmp = 1.0 = mp
#     else:
#         mp = mergePrios
#         nmp = notMergePrios

#     if edgeSizes is None:
#         edgeSizes = numpy.ones(graph.edgeIdUpperBound+1)




#     if(updateRule0 == "histogram_rank" and updateRule1 == "histogram_rank"):
#         return nifty.graph.agglo.rankFixationClusterPolicy(graph=graph,
#             mergePrios=mp, notMergePrios=nmp,
#                         edgeSizes=edgeSizes, isMergeEdge=isLocalEdge,
#                         q0=p0, q1=p1, zeroInit=zeroInit)
#     elif(updateRule0 in ["smooth_max","generalized_mean"] and updateRule1 in ["smooth_max","generalized_mean"]):
        

#         return  nifty.graph.agglo.generalizedMeanFixationClusterPolicy(graph=g,
#                         mergePrios=mp, notMergePrios=nmp,
#                         edgeSizes=edgeSizes, isMergeEdge=isLocalEdge,
#                         p0=p0, p1=p1, zeroInit=zeroInit)

       













def sizeLimitClustering(graph, nodeSizes, minimumNodeSize, 
                        edgeIndicators=None,edgeSizes=None, 
                        sizeRegularizer=0.001, gamma=0.999,
                        makeDenseLabels=False):

    s = graph.edgeIdUpperBound + 1

    def rq(data):
        return numpy.require(data, 'float32')

    nodeSizes  = rq(nodeSizes)

    if edgeIndicators is None:
        edgeIndicators = numpy.ones(s,dtype='float32')
    else:
        edgeIndicators = rq(edgeIndicators)

    if edgeSizes is None:
        edgeSizes = numpy.ones(s,dtype='float32')
    else:
        edgeSizes = rq(edgeSizes)



    cp =  minimumNodeSizeClusterPolicy(graph, edgeIndicators=edgeIndicators, 
                                              edgeSizes=edgeSizes,
                                              nodeSizes=nodeSizes,
                                              minimumNodeSize=float(minimumNodeSize),
                                              sizeRegularizer=float(sizeRegularizer),
                                              gamma=float(gamma))

    agglo = agglomerativeClustering(cp)

    agglo.run()
    labels = agglo.result()

    if makeDenseLabels:
        labels = __makeDense(labels)

    return labels;




def ucmFeatures(graph, edgeIndicators, edgeSizes, nodeSizes, 
                sizeRegularizers = numpy.arange(0.1,1,0.1) ):
    
    def rq(data):
        return numpy.require(data, 'float32')
 
    edgeIndicators = rq(edgeIndicators)

    if edgeSizes is None:
        edgeSizes = numpy.ones(s,dtype='float32')
    else:
        edgeSizes = rq(edgeSizes)


    if nodeSizes is None:
        nodeSizes = numpy.ones(s,dtype='float32')
    else:
        nodeSizes = rq(nodeSizes)

    fOut = []
    # policy
    for sr in sizeRegularizers:

        sr = float(sr)
        cp = edgeWeightedClusterPolicyWithUcm(graph=graph, edgeIndicators=edgeIndicators,
                edgeSizes=edgeSizes, nodeSizes=nodeSizes, sizeRegularizer=sr)


        agglo = agglomerativeClustering(cp)



        hA = agglo.runAndGetDendrogramHeight()[:,None]
        hB = agglo.ucmTransform(cp.edgeIndicators)[:,None]

        fOut.extend([hA,hB])

    return numpy.concatenate(fOut, axis=1)


def greedyGraphEdgeContraction(graph,
                          signed_edge_weights,
                          update_rule = 'mean',
                          threshold = 0.5,
                          # unsigned_edge_weights = None,
                          add_cannot_link_constraints= False,
                          edge_sizes = None,
                          node_sizes = None,
                          is_merge_edge = None,
                          size_regularizer = 0.0,
                          ):
    """

    :param graph:
    :return:
    """
    def parse_update_rule(rule):
        accepted_rules_1 = ['max', 'min', 'mean', 'ArithmeticMean', 'sum']
        accepted_rules_2 = ['generalized_mean', 'rank', 'smooth_max']
        if not isinstance(rule, str):
            rule = rule.copy()
            assert isinstance(rule, dict)
            rule_name = rule.pop('name')
            p = rule.get('p')
            q = rule.get('q')
            assert rule_name in accepted_rules_1 + accepted_rules_2, "Passed update rule is not implemented"
            assert not (p is None and q is None), "Passed update rule is not implemented"
            parsed_rule = updatRule(rule_name, **rule)
        else:
            assert rule in accepted_rules_1, "Passed update rule is not implemented"
            parsed_rule = updatRule(rule)

        return parsed_rule

    # if unsigned_edge_weights is not None:
    #     assert signed_edge_weights is None, "Both signed and unsigned weights were given!"
    #     assert threshold is not None, "For unsigned weights it is necessary to define a threshold parameter!"
    #     signed_edge_weights = unsigned_edge_weights - threshold

    merge_prio = numpy.where(signed_edge_weights > 0, signed_edge_weights, -1.)
    not_merge_prio = numpy.where(signed_edge_weights < 0, -signed_edge_weights, -1.)

    parsed_rule = parse_update_rule(update_rule)

    costs_in_PQ = True if update_rule == 'sum' else False

    edge_sizes = numpy.ones_like(signed_edge_weights) if edge_sizes is None else edge_sizes
    is_merge_edge = numpy.ones_like(signed_edge_weights) if is_merge_edge is None else is_merge_edge
    node_sizes = numpy.ones(graph.numberOfNodes ,dtype='float32') if node_sizes is None else node_sizes


    return fixationClusterPolicy(graph=graph,
                          mergePrios=merge_prio,
                          notMergePrios=not_merge_prio,
                          isMergeEdge=is_merge_edge,
                          edgeSizes=edge_sizes,
                          nodeSizes=node_sizes,
                          updateRule0=parsed_rule,
                          updateRule1=parsed_rule,
                          zeroInit=False,
                          initSignedWeights=False,
                          sizeRegularizer=size_regularizer,
                          sizeThreshMin=0.,
                          sizeThreshMax=0.,
                          postponeThresholding=False,
                          costsInPQ=costs_in_PQ,
                          checkForNegCosts=True,
                          addNonLinkConstraints=add_cannot_link_constraints,
                          threshold=threshold)


greedyGraphEdgeContraction.__doc__ = """
Greedy edge contraction of a graph..

Accepted update rules:
 - 'mean'
 - 'max'
 - 'min'
 - 'sum'
 - {name: 'rank', q=0.5, numberOfBins=40}
 - {name: 'generalized_mean', p=2.0}   # 1.0 is mean
 - {name: 'smooth_max', p=2.0}   # 0.0 is mean
 """