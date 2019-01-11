# sphinx_gallery_thumbnail_number = 4
from __future__ import absolute_import
from . import _graph as __graph
from ._graph import *

from .. import Configuration
from . import opt

from . opt import multicut
from . opt import lifted_multicut
from . opt import mincut
from . opt import minstcut

import numpy
from functools import partial
import types
import sys

__all__ = []

for key in __graph.__dict__.keys():
    try:
        __graph.__dict__[key].__module__='nifty.graph'
    except:
        pass
    __all__.append(key)


UndirectedGraph.__module__ = "nifty.graph"


ilpSettings = multicut.ilpSettings





# multicut objective
UndirectedGraph.MulticutObjective                     = multicut.MulticutObjectiveUndirectedGraph
UndirectedGraph.EdgeContractionGraph                  = EdgeContractionGraphUndirectedGraph
EdgeContractionGraphUndirectedGraph.MulticutObjective = multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph


UndirectedGraph.MincutObjective                     = mincut.MincutObjectiveUndirectedGraph
UndirectedGraph.EdgeContractionGraph                = EdgeContractionGraphUndirectedGraph
EdgeContractionGraphUndirectedGraph.MincutObjective = mincut.MincutObjectiveEdgeContractionGraphUndirectedGraph

# #minstcut objective
# UndirectedGraph.MinstcutObjective                     = minstcut.MinstcutObjectiveUndirectedGraph
# UndirectedGraph.EdgeContractionGraph                = EdgeContractionGraphUndirectedGraph
# EdgeContractionGraphUndirectedGraph.MinstcutObjective = minstcut.MinstcutObjectiveEdgeContractionGraphUndirectedGraph

# lifted multicut objective
UndirectedGraph.LiftedMulticutObjective = lifted_multicut.LiftedMulticutObjectiveUndirectedGraph




def randomGraph(numberOfNodes, numberOfEdges):
    g = UndirectedGraph(numberOfNodes)

    uv = numpy.random.randint(low=0, high=numberOfNodes-1, size=numberOfEdges*2)
    uv = uv.reshape([-1,2])

    where = numpy.where(uv[:,0]!=uv[:,1])[0]
    uv = uv[where,:]


    g.insertEdges(uv)
    while( g.numberOfEdges < numberOfEdges):
        u,v = numpy.random.randint(low=0, high=numberOfNodes-1, size=2)
        if u != v:
            g.insertEdge(int(u),int(v))
    return g




class EdgeContractionGraphCallback(EdgeContractionGraphCallbackImpl):
    def __init__(self):
        super(EdgeContractionGraphCallback, self).__init__()

        try:
            self.contractEdgeCallback = self.contractEdge
        except AttributeError:
            pass

        try:
            self.mergeEdgesCallback = self.mergeEdges
        except AttributeError:
            pass

        try:
            self.mergeNodesCallback = self.mergeNodes
        except AttributeError:
            pass

        try:
            self.contractEdgeDoneCallback = self.contractEdgeDone
        except AttributeError:
            pass

def edgeContractionGraph(g, callback):
    Ecg = g.__class__.EdgeContractionGraph
    ecg = Ecg(g, callback)
    return ecg





def undirectedGraph(numberOfNodes):
    return UndirectedGraph(numberOfNodes)

def undirectedGridGraph(shape, simpleNh=True):
    if not simpleNh:
        raise RuntimeError("currently only simpleNh is implemented")
    s = [int(s) for s in shape]
    if(len(s) == 2):
        return UndirectedGridGraph2DSimpleNh(s)
    elif(len(s) == 3):
        return UndirectedGridGraph3DSimpleNh(s)
    else:
        raise RuntimeError("currently only 2D and 3D grid graph is exposed to python")

gridGraph = undirectedGridGraph

def undirectedLongRangeGridGraph(shape, offsets, is_local_offset, offsets_probabilities=None, labels=None, strides=None):
    """
    :param offsets_probabilities: Probability that a repulsive long-range edge is intriduced. If None a dense graph is used
    :param labels: Indices of a passed segmentation (uint64)
    :param is_local_offset: boolean array of shape=nb_offsets
    """
    # TODO: bad design. Local edges could be skipped.
    offsets = numpy.require(offsets, dtype='int64')
    assert offsets.shape[0] == is_local_offset.shape[0]
    is_local_offset = is_local_offset.astype(numpy.bool)


    shape = list(shape)
    if len(shape) == 2:
        G = UndirectedLongRangeGridGraph2D
    elif len(shape) == 3:
        G = UndirectedLongRangeGridGraph3D
    else:
        raise RuntimeError("wrong dimension: undirectedLongRangeGridGraph is only implemented for 2D and 3D")

    if labels is None:
        labels = numpy.zeros(shape, dtype=numpy.uint32)
        start_from_labels = False
    else:
        labels = labels.astype(numpy.uint32)
        start_from_labels = True


    randomProbsShape = tuple(shape) + (offsets.shape[0],)
    if offsets_probabilities is None:
        offsets_probabilities = numpy.ones((offsets.shape[0],), dtype='float')
        randomProbs = numpy.ones(randomProbsShape, dtype='float') * 0.5
    else:
        assert offsets_probabilities.shape[0] == offsets.shape[0]
        offsets_probabilities = numpy.require(offsets_probabilities, dtype='float')
        # Randomly select which edges to add or not
        # FIXME: (multi-thread bug in C++, need to do it on the python side)
        randomProbs = numpy.random.rand(randomProbsShape)

    if strides is None:
        strides = numpy.ones(len(shape), dtype='int')
    else:
        if isinstance(strides, list):
            strides = numpy.array(strides)
        assert strides.shape[0] == len(shape)


    return G(shape, offsets, strides, labels, offsets_probabilities, randomProbs, is_local_offset, start_from_labels)

longRangeGridGraph = undirectedLongRangeGridGraph


def drawGraph(graph, method='spring'):

    import networkx

    G=networkx.Graph()
    for node in graph.nodes():
        G.add_node(node)

    #uvIds = graph.uvIds()
    #for i in range(uvIds.shape[0]):
    #    u,v = uvIds[i,:]
    #    G.add_edge(u,v)
    for edge in graph.edges():
        u,v = graph.uv(edge)
        G.add_edge(u,v)

    nodeLabels = dict()

    for node in graph.nodes():
        nodeLabels[node] = str(node)
    if method == 'spring':
        networkx.draw_spring(G,labels=nodeLabels)
    else:
        networkx.draw(G, lables=nodeLabels)
