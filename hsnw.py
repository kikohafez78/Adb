# ==================================================
# Authors: 1-Omar Badr                             |
#          2-Karim Hafez                            |
#          3-Abdulhameed                            |
#          4-Seif albaghdady                        |
# Subject: ADB                                     |
# Project: HNSW +                                  |
# ==================================================
# from product_quantization import quantizer
# import threading
import math

# import numba as nm
import random
from heapq import *

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

import IVF as ivf

dimension = 70
segments = 14
nbits = 8
"""
random number generator
I = float[-ln(uniform(0,1))*mL] mL is better selected as a non-zero element
The authors of the paper propose choosing the optimal value of mL which is equal to 1 / ln(M). 
This value corresponds to the parameter p = 1 / M of the skip list being an average single element overlap between the layers.
"""


class Node(object):
    def __init__(self, vector: np.ndarray, M: int, layer: int, M_MAX: int, mL: float):
        self.vec = vector
        self.M = M
        self.layer = layer
        self.M_MAX = M_MAX
        self.friends_list: list[Node] = []  # list of Node
        self.layers = np.round(float[-math.log(random.uniform(0, 1)) * mL])

    # @nm.set(fast_math = True)
    def get_distance_n_similarity(self, vector: np.ndarray):
        return cosine_similarity(self.vec, vector)

    def get_neighbors_list(self):
        return self.friends_list


def calculate_distances(heap: list, q: np.ndarray):
    heap = [(cosine_similarity(node.vec, q), node) for node in heap]
    return heap


# functions for creating a heap that are sorted by cosine similarity between elements and query vector
def sorted_list_by_cosine_similarity(heap: list, query_vector: np.ndarray) -> list[(int, Node)]:
    heap = [(cosine_similarity(node.vec, query_vector), node) for node in heap]
    heap.sort(reverse=True)  # sort descending
    return heap


"""
- good values for M lie between 5 and 48, High M is better for high dimensionality and recall -> dim = 70 therefore M must be High
- Higher values of efConstruction imply a more profound search as more candidates are explored. However, it requires more
  computations.
  Authors recommend choosing such an efConstruction value that results at recall being close to 0.95–1 during training.
- M_MAX must be close to 2*M
- heuristic of selecting M out of efConstruction is both closest Nodes and connectivity of th graph by considering the 
  connectivity distances between the candidates
"""


class vector_db(object):
    def __init__(self, M: int, efSearch: int, efConstruction):
        """
        M is max number of links per node, M_MAX is max number of links connected to the node, efSearch is number of threads or
        number of nearest neighbors to use to find closest query result
        ml 0-1 is the overlap between layers (recommended 1/ln(M)
        """
        self.M = M
        self.M_MAX = 2 * M  # just a heuristic (based on the paper)
        self.efSearch = efSearch
        self.efConstruction = efConstruction
        self.entry_points = None
        self.max_layers = 0
        self.ml = 1.0 / math.log2(M)  # the closer to 1/ln(M) the better
        self.graph: list[list[Node]] = np.array([[]], dtype=Node)

    def search_layer(
        self,
        query_element: np.ndarray,
        entry_points: list[(int, Node)],
        ef_search: int,
        layer: int,
    ):
        entry_points = sorted_list_by_cosine_similarity(entry_points, query_element)  # sort entry points by cosine similarity
        v = entry_points.copy()  # visited elements
        C = entry_points.copy()  # candidates
        W = entry_points.copy()  # dynamic list of found nearest neighbors
        # while I have candidates
        while len(C) > 0:
            # extract nearest element from C to q
            c = C.pop(0)
            # get furthest element from W to q
            f = W[-1]
            # if distance between c and q is greater than distance between f and q then you are done
            if c[0] > f[0]:
                break
            # iterate over friend list of c, where you will update candidates and results accordingly.
            friends = calculate_distances(c[1].friends_list, query_element)
            for e in friends:
                if e not in v and e.layers >= layer:
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
        return W  # return the list of nearest neighbors

    def select_neighbors_simple(self, query_element: np.ndarray, C: list[(int, Node)], M: int) -> list[(int, Node)]:
        size = len(C) if len(C) <= M else M
        return sorted_list_by_cosine_similarity(C, query_element)[0:size]

    """
    SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates,
      keepPrunedConnections)
        Input: base element q, candidate elements C, number of neighbors to
        return M, layer number lc, flag indicating whether or not to extend
        candidate list extendCandidates, flag indicating whether or not to add
        discarded elements keepPrunedConnections
    """

    def select_neighbors_heuristic(
        self,
        query_element: np.ndarray,
        C: list[(int, Node)],
        M: int,
        layer: int,
        extendCandidates=False,
        keepPrunedConnections=False,
    ):
        R = np.ndarray([])
        W = np.copy(C[:][1])
        if extendCandidates:
            for e in C[:][1]:
                neighbors = e.get_neighbors_list()
                for neighbor in neighbors:
                    if neighbor not in W:
                        np.insert(W, Node, neighbor)
        W_d = list()
        W = sorted_list_by_cosine_similarity(W, query_element, False)
        while len(W) > 0 and len(R) < M:
            e = W.pop(0)
            if e[1].layers >= layer:
                if len(R) == 0:
                    np.insert(R, list, e)
                else:
                    if e[0] < np.min(R, axis=0):
                        np.insert(R, list, e)
                    else:
                        W_d.append(e)
        if keepPrunedConnections:
            while len(W_d) > 0 and len(R) < M:
                np.insert(R, list, W_d.pop(0))
        return R

    def select_neighbors_knn(self, query_element, ef_search, layer):
        pass

    """
    INSERT(hnsw, q, M, Mmax, efConstruction, mL)
        Input: multilayer graph hnsw, new element q, number of established
        connections M, maximum number of connections for each element
        per layer Mmax, size of the dynamic candidate list efConstruction,
        normalization factor for level generation mL
    """

    def insertion(self, q: np.ndarray, M: int, Mmax: int, efConstruction: int, mL: int):
        W = []
        entry_points = self.entry_points
        l_max = self.max_layers
        l = math.floor(-math.log(random.uniform(0, 1)) * mL)
        # if l > l_max:
        #     for layer in range(l_max, l, -1):
        #         self.graph[layer].append(Node(q, M, layer, Mmax))
        #     self.max_layers = l
        #     l_max = l
        #     entry_points = []
        for layer in range(l_max, l + 1):
            W = self.search_layer(q, entry_points, 1, layer)
            print(W, " W now ", entry_points, " entry points\n")
            entry_points = [W[0]]

        # for each layer from l_max to l
        for layer in range(l_max, l, -1):
            # if layer is not empty
            if len(self.graph[layer]) > 0:
                # get nearest neighbors from layer
                W = self.search_layer(q, entry_points, efConstruction, layer)
                entry_points = W[0:M] if len(W) > M else W
            else:
                # if layer is empty then add q to entry points
                entry_points.append((0, q))

        # for each layer from min(l_max,l) to 0
        for layer in range(min(l_max, l), 0, -1):
            # get nearest neighbors from layer
            W = self.search_layer(q, entry_points, efConstruction, layer)
            # select M nearest neighbors from W
            neighbors = self.select_neighbors_simple(q, W, M)
            # add bidirectional connections from neighbors to q at layer
            for neighbor in neighbors:
                neighbor[1].friends_list.append(q)
                q.friends_list.append(neighbor[1])
            # After adding bidirectional connections, check if the number of connections exceeds Mmax
            for neighbor in neighbors:
                if len(neighbor[1].friends_list) > Mmax:
                    # select Mmax nearest neighbors from neighbor
                    candidates = self.select_neighbors_simple(neighbor[1].vec, neighbor[1].friends_list, Mmax)
                    neighbor[1].friends_list = candidates
            entry_points = neighbors
            if l > l_max:
                self.max_layers = l
                entry_points = [Node(q, M, layer, Mmax)]

    def graph_creation(self):
        vectors = np.random.normal(size=(100, dimension))
        x = 0
        for vector in vectors:
            print(f"inserted{x}")
            self.insertion(vector, self.M, self.M_MAX, self.efConstruction, self.ml)
            x += 1
        print(len(self.graph[self.max_layers]))


# hnsw = vector_db(10, 10, 10)
# hnsw.graph_creation()

# read data from file saved_db.csv


#############################################################################


def read_data():
    data = np.genfromtxt("saved_db.csv", delimiter=",")
    return data


data = read_data()


Iv = ivf.IVFile(4096, data)

centroids = Iv.clustering()

# give centroids to hnsw then use get_closest_k_neighbors to get the closest k neighbors
hnsw = vector_db(10, 10, 10)

hnsw.graph_creation()

# loop over centroids

for centroid in centroids:
    hnsw.get_closest_k_neighbors(centroid, 10)
