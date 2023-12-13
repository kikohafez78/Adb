# ==================================================
# Authors: 1-Omar Badr                             |
#         2-Karim Hafez                           |
#         3-Abdulhameed                           |
#         4-Seif albaghdady                       |
# Subject: ADB                                     |
# Project: HNSW +                                  |
# ==================================================
from product_quantization import quantizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import threading
import math

# import numba as nm
import random
from heapq import *

dimension = 70
segments = 14
nbits = 8

# random number generator
# I = float[-ln(uniform(0,1))*mL] mL is better selected as a non-zero element
# The authors of the paper propose choosing the optimal value of mL which is equal to 1 / ln(M). This value corresponds to the parameter p = 1 / M of the skip list being an average single element overlap between the layers.


class Node(object):
    def __init__(self, vector: np.ndarray, M: int, all_layers: int, M_MAX: int):
        self.vec = vector
        self.M = M
        self.layers = all_layers
        self.M_MAX = M_MAX
        self.friends_list = []  # list of tuples (cosine_similarity, node)

    # @nm.set(fast_math = True)
    def get_distance_n_similarity(self, vector: np.ndarray):
        return cosine_similarity(self.vec, vector)


# functions for creating a heap that are sorted by cosine similarity between elements and query vector
def sorted_list_by_cosine_similarity(heap: list, query_vector: np.ndarray):
    heap = [(cosine_similarity(node.vec, query_vector), node) for node in heap]
    heap.sort(reverse=True)  # sort descending
    return heap


# good values for M lie between 5 and 48, High M is better for high dimensionality and recall -> dim = 70 therefore M must be High
# Higher values of efConstruction imply a more profound search as more candidates are explored. However, it requires more computations. Authors recommend choosing such an efConstruction value that results at recall being close to 0.95–1 during training.
# M_MAX must be close to 2*M
# heuristic of selecting M out of efConstruction is both closest Nodes and connectivity of th graph by considering the connectivity distances between the candidates
class vector_db(object):
    def __init__(
        self, M: int, efSearch: int, efConstruction
    ):  # M is max number of links per node, M_MAX is max number of links connected to the node, efSearch is number of threads or number of nearest neighbors to use to find closest query result
        # ml 0-1 is the overlap between layers (recommended 1/ln(M)
        self.M = M
        self.M_MAX = 2 * M  # just a heuristic (based on the paper)
        self.efSearch = efSearch
        self.efConstruction = efConstruction
        self.Layers = -1
        self.max_layers = 0
        self.ml = 1.0 / math.log(M)
        self.graph = np.empty((0, 0), dtype=Node)

    """
    SEARCH-LAYER(q, ep, ef, lc)
        Input: query element q, enter points ep, number of nearest to q elements to return ef, layer number lc
        Output: ef closest neighbors to q
        1 v ← ep // set of visited elements   
        2 C ← ep // set of candidates
        3 W ← ep // dynamic list of found nearest neighbors, W is the set of ef closest neighbors to q at layer lc
        4 while │C│ > 0
        5 c ← extract nearest element from C to q
        6 f ← get furthest element from W to q
        7 if distance(c, q) > distance(f, q)
        8 break // all elements in W are evaluated
        9 for each e ∈ neighbourhood(c) at layer lc // update C and W
        10 if e ∉ v
        11 v ← v ⋃ e
        12 f ← get furthest element from W to q
        13 if distance(e, q) < distance(f, q) or │W│ < ef
        14 C ← C ⋃ e
        15 W ← W ⋃ e
        16 if │W│ > ef
        17 remove furthest element from W to q
        18 return W 
    """

    def search_layer(
        query_element: np.ndarray,
        entry_points: list[(int, Node)],
        ef_search: int,
        layer: int,
    ):
        entry_points = sorted_list_by_cosine_similarity(
            entry_points, query_element
        )  # sort entry points by cosine similarity
        v = entry_points  # visited elements
        C = entry_points  # candidates
        W = entry_points  # dynamic list of found nearest neighbors
        # while I have candidates
        while len(C) > 0:
            # extract nearest element from C to q
            c = entry_points.pop(0)
            # get furthest element from W to q
            f = W[-1]
            # if distance between c and q is greater than distance between f and q then you are done
            if c[0] > f[0]:
                break
            # iterate over friend list of c, where you will update candidates and results accordingly.
            for e in c[1].friends_list:
                if e not in v:
                    v.append(e)
                    v.sort(reversed=True)
                    f = W[-1]
                    if e[0] < f[0] or len(W) < ef_search:
                        C.append(e)
                        W.append(e)
                        C.sort(reversed=True)
                        W.sort(reversed=True)
                        if len(W) > ef_search:
                            W.pop()
        return W

    def select_neighbors_simple(
        self, query_element: np.ndarray, C: list[(int, Node)], M: int
    ):
        return sorted_list_by_cosine_similarity(C, query_element)[0:M]

    """
    SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections)
        Input: base element q, candidate elements C, number of neighbors to
        return M, layer number lc, flag indicating whether or not to extend
        candidate list extendCandidates, flag indicating whether or not to add
        discarded elements keepPrunedConnections

    """

    def select_neighbors_heuristic(
        self,
        query_element: np.ndarray,
        C : list[(int, Node)],
        M: int,
        layer: int,
        extendCandidates=False,
        keepPrunedConnections=False,
    ):
        pass

    def select_neighbors_knn(self, query_element, ef_search, layer):
        pass

    def insertion(self, q: Node, normalization_factor):
        """
        q: query vector
        normalization_factor: normalization factor of the query vector
        """

    def graph_creation(self):
        pass
