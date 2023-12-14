import math
import random

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Node(object):
    def __init__(self, vector: np.ndarray, M: int):
        self.data = vector
        self.M = M
        self.friends_list: list[Node] = []  # list of Node
        # self.layers = np.round(float[-math.log(random.uniform(0, 1)) * mL])

    def get_neighbors_list(self):
        return self.friends_list

    def get_data(self):
        return self.data

    def get_consin_similarity(self, vector: np.ndarray):
        return cosine_similarity(self.data, vector)


def calculate_distances(heap: list, q: np.ndarray):
    cosine_similarities = [cosine_similarity(list[i], q) for i in heap]
    return cosine_similarities


# functions for creating a heap that are sorted by cosine similarity between elements and query vector
def sorted_list_by_cosine_similarity(heap: list, query_vector: np.ndarray) -> list[(int, Node)]:
    heap = [(cosine_similarity(node.vec, query_vector), node) for node in heap]
    heap.sort(reverse=True)  # sort descending
    return heap


class HNSW(object):
    def __init__(self, M: int, efSearch: int, efConstruction: int, heuristic=False, M0: int = None) -> None:
        self.M = M
        self.M_MAX = 2 * M  # just a heuristic (based on the paper)
        self.M0 = M0 if M0 is not None else 2 * M
        self.efSearch = efSearch
        self.efConstruction = efConstruction
        self.entry_points = None
        self.max_layers = 0
        self.ml = 1.0 / math.log2(M)  # the closer to 1/ln(M) the better
        self.graph: list[list[dict]] = []  # graph[layer][node_id] = {neighbor: distance}

    """
      1- heapify the entry points so that you have the largest cosine similarity at the top (heap[0])
      2- track the nodes you have visited
      3- loop on the condidates 
      4- pop the entry of highest cosine similarity from the heap
      5- check its neighbors, if they are not visited, add them to the heap
      6- calculate the cosine similarity between the query and newly added nodes 
      7- if there is still space in the heap (where len(ep) < ef), add the new nodes to the heap
      8- if there is no space in the heap, replace the node with the lowest cosine similarity with the new node
      9- repeat steps 4-8 until you have visited all the nodes
    """

    def search_layer(
        self,
        query_element: np.ndarray,
        entry_points: (int, int),
        ef_search: int,
        layer: int,
    ):
        candidates = entry_points
        visited = set([])

    """
      This is similar to search_layer but with ef=1, The idea is to find the closest neighbor in the graph layer
      so I could use this entry point to search in the next layer.
      1- first assign the inital entry point to the best (think of it as like finding the min. of a normal array)
      2- track the visited node (just like search_layer).
      3- loop on the candidates.
      4- pop the entry of highest cosine similarity from the heap.
      5- check its neighbors, if they are not visited, add them to the heap.
      6- calculate the cosine similarity between the query and newly added nodes.
      7- if dist > best_dist, we stop the search --> this means that I have found the closest neighbor,
        any other node will be further away.
      8- if dist < best_dist, we update best and best_dist.
      9- repeat steps 4-8 until you have visited all the nodes.
      10- return the best and best_dist --> this is a single node unlike search_layer where it returns a heap of nodes.
    """

    def search_layer_ef1(self, q: np.ndarray, dist: int, entry_point: int, layer: int) -> (int, Node):
        """
        - equivalent to search_layer(q, entry_points, ef, layer) with ef = 1
        - Mainly we used to traverse the layers of the graph that are bigger than highest layer of query element
        """
        pass

    def select_neighbors_simple(self, query_element: np.ndarray, C: list[(int, Node)], M: int) -> list[(int, Node)]:
        pass

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
      1- insert the vector in the data array and assign it an index 
      2- if the HNSW is empty, or if the L is higher the highest level in the HNSW, then we create (L - len(graphs)) levels with 
        index of the vector as the entry point.
      3- starting from entry point, and we loop on the levels that are higher than the highest level of the vector
      4- we search for the closest neighbor (single neighbor using ef=1) in the graph layer
      5- we add the closest neighbor to the entry point heap
      Now we reached the level of the vector, we have to insert it in the graph
      6- we search for ef neighbors in the bottom level (using ef=ef) and add them to the entry point heap
      7- we insert the vector in the graph using select function (naive or heuristic)
      8- we insert backlinks to the new node
      9- we repeat steps 6-8 until we reach the top level
    """

    def insert(self, element: np.ndarray):
        pass
