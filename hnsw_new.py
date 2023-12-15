import math
import random
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest

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
        self.M0 = np.ceil(np.log2(M0)) if M0 is not None else 2 * M
        self.efSearch = efSearch
        self.efConstruction = efConstruction
        self.entry_points = None
        self.max_layers = -1
        self.heuristic = heuristic
        self.ml = 1.0 / math.log2(M)  # the closer to 1/ln(M) the better
        self.graph: list[dict[int : [dict[int, int]]]] = []  # graph[layer]{indx: {neighbor: distance}}
        self.data = np.ndarray([])  # data[node_id] = vector

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
        entry_points: list[(int, int)],  # list of tuples (cosine_similarity, node_id)
        ef: int,  # in construction phase(insert), ef = ef_construction, in search phase, ef_search = ef_search
        layer: int,
    ):
        level = self.graph[layer]
        data = self.data
        visited = set([point for _, point in entry_points])
        candidates = [(similarity, point) for similarity, point in entry_points]
        # create min heap, so that the lowest cosine similarity is at the top
        heapify(candidates)

        while candidates:
            currrent_similarity, point = heappop(candidates)  # most similar node
            ref = entry_points[0][0]
            if currrent_similarity < ref:  # if new similarities are lower than the least similar node in the heap, we stop
                break
            neighbors = [neighbor for neighbor in level[point] if neighbor not in visited]
            similarities = calculate_distances([data[neighbor] for neighbor in neighbors], query_element)
            visited.update(neighbors)

            for neighbor, similarity in zip(neighbors, similarities):
                if len(entry_points) < ef:
                    heappush(entry_points, (similarity, neighbor))
                    heappush(candidates, (similarity, neighbor))
                    ref = entry_points[0][0]
                else:
                    if similarity > ref:
                        heappushpop(entry_points, (similarity, neighbor))  # replace the lowest similarity with the new one
                        heappush(candidates, (similarity, neighbor))
                        ref = entry_points[0][0]
        return entry_points

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

    def search_layer_ef1(self, q: np.ndarray, dist: int, entry_point: (int, int), layer: int) -> (int, Node):
        """
        - equivalent to search_layer(q, entry_points, ef, layer) with ef = 1
        - Mainly we used to traverse the layers of the graph that are bigger than highest layer of query element
        """
        level = self.graph[layer]
        data = self.data
        M = self.M
        visited = set([entry_point[1]])
        candidates = [entry_point]
        best = entry_point
        best_sim = dist
        while candidates:
            currrent_similarity, point = heappop(candidates)
            if currrent_similarity < best_sim:  # if the similarity is lower than the most similar node in the heap
                break
            neighbors = [neighbor for neighbor in level[point] if neighbor not in visited]
            similarities = calculate_distances([data[neighbor] for neighbor in neighbors], q)
            visited.update(neighbors)
            for neighbor, similarity in zip(neighbors, similarities):
                if similarity > best_sim:
                    best = neighbor
                    best_sim = similarity
                heappush(candidates, (similarity, neighbor))
        return best, best_sim

    """
      if it's not a heap: we will operate in simple mode (naive)
      1- first check if the node have place in friend list
      2- if it has place in friend list, add it to the friend list
      3- else, check if the node is closer to the query than the farthest friend
      4- if it is closer, add it to the friend list and remove the farthest friend
      else: we will opereate in complex mode (heap)
      1- M select the largest cosine similarity from the heap
      2- check if the node have place in friend list
      3- if it has place in friend list, then add friends to the friend list until it's full
      4- the remaining friends from M largest are not all added to the friend list.
      5  check the furthest friends in the friend list and check if you can replace them with remaining M largest
      6- Now we will go through all friends and add us to their friend list (bidirectional)
    """

    def select_neighbors_simple(self, to_be_inserted: int, C: list[(int, int)], M: int, layer: int):
        C = nlargest(M, C)
        C = [(sim, idx) for sim, idx in C if len(self.graph[layer][idx]) < self.M]  # check if the node have place in friend list
        index = to_be_inserted
        d = self.graph[layer][index]
        remaining = M - len(d)
        # d.update(to_be_inserted,C[:remaining])  # add the remaining friends to the friend list
        for sim, idx in C[:remaining]:
            d[idx] = sim
        if remaining > 0:
            # check the furthest friends in the friend list and check if you can replace them with remaining M largest
            for i in range(remaining, len(C)):
                if C[i][0] > min(d, key=d.get):
                    d.pop(min(d, key=d.get))
                    # d.update(to_be_inserted,C[i])
                    d[C[i][1]] = C[i][0]
        # Now we will go through all friends and add us to their friend list and the distance between us (bidirectional)
        for i in d:
            self.graph[layer][i][index] = d[i]  # d[i] is the distance between us

    def search(self, query_element: np.ndarray, ef: int = None, k: int = None):
        graph = self.graph
        entry_point = self.entry_points
        if entry_point is None:
            raise Exception("The graph is empty, please insert some elements first")
        if ef is None:
            ef = self.efSearch
        if k is None:
            k = self.M
        sim = cosine_similarity(query_element, self.entry_points)
        for layer in reversed(graph[1:]):  # loop on the layers till you reach layer 1
            entry_point, sim = self.search_layer_ef1(query_element, sim, entry_point, layer)
        candidates = self.search_layer(query_element, [(sim, entry_point)], ef, 0)
        candidates = nlargest(k, candidates)
        return [(sim, idxs) for sim, idxs in candidates]

    def select_neighbors_heuristic(
        self,
        query_element: np.ndarray,
        C: list[(int, int)],
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

    def get_node_layers(self):
        return np.round(-float[math.log(random.uniform(0, 1))] * self.ml)

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
        id = len(self.data)
        np.insert(self.data, np.ndarray, element)
        L = self.get_node_layers()
        if self.entry_points is not None:
            if L > self.max_layers:
                for i in range(L - self.max_layers):
                    self.graph.append({id: {}})
                self.entry_points = id
            distance = cosine_similarity(element, self.entry_points)
            P = [(distance, self.entry_points)]
            for layer in reversed(self.graph[L:]):
                W = self.search_layer_ef1(element, self.entry_points, distance, layer)
                P.append(W)
            for layer, nodes in enumerate(reversed(self.graph[:L])):
                W = self.search_layer(element, [(W[0], W[1])], self.efConstruction, layer)
                M = self.M_MAX if layer == 0 else self.M0
                self.graph[layer][id] = {}
                if not self.heuristic:
                    self.select_neighbors_simple(id, W, M, layer)
                else:
                    self.select_neighbors_heuristic(id, W, M, layer)
                self.entry_points = W
        else:
            for i in range(self.max_layers, L):
                self.graph.append({id: {}})
            self.entry_points = id
            self.max_layers = L
