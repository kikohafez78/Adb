# ==================================================
# Authors: 1-Omar Badr                             |
#          2-Karim Hafez                           |
#          3-Abdulhameed                           |
#          4-Seif albaghdady                       |
# Subject: ADB                                     |
# Project: HNSW + IVF                              |
# ==================================================
import math
import random
from heapq import nsmallest,heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from IVF import IVFile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from operator  import itemgetter
import logging as logs
import pandas as pd
import xlrd as xl
from product_quantization import quantizer


class HNSW(object):
    def cosine_distance(self, a, b):
        return -np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def __init__(self,m=5, ef=200, m0=None, heuristic=True,efSearch: int = 50, efConstruction: int = 50, vectors:np.ndarray = None,useIVF:bool = False,useQuantizer: bool = False,partitions:int = 3,segments: int = 3):
        self.data:list[np.ndarray] = []
        self.distance_func = self.cosine_distance
        self.vectorized_distance = self.vectorized_distance_
        self._m = m
        self._ef = ef
        self._m0 = np.ceil(np.log2(m0)) if m0 is not None else 2*self._m
        self._level_mult = 1 / math.log2(m)
        self._graphs: list[dict[int : [dict[int, int]]]] = []
        self._enter_point = None
        self._select = self.heuristic_selection if heuristic else self.normal_selection
        self.vectors = vectors
        self.efSearch = efSearch
        self.efConstruction = efConstruction
        if useIVF and useQuantizer:
            self.IVF = IVFile(partitions,vectors)
            self.quantizer = quantizer(len(vectors[0]),segments)
        elif useIVF and not useQuantizer:
            self.IVF = self.IVF = IVFile(partitions,vectors)
            self.quantizer = None
        elif not useIVF and useQuantizer:
            self.quantizer = quantizer(len(vectors[0]),segments)
            self.IVF = None
        else:
            self.IVF = self.quantizer = None
    def get_data(self):
        return self.data
    def similarity(self,query: np.ndarray,neighbor: np.ndarray):
        return np.dot(query,neighbor)/(np.linalg.norm(query) *np.linalg.norm(neighbor))
    def array_similarity(self,X:np.ndarray,Y:np.ndarray):
        return self.similarity(X,Y)
    def calculate_distances(self,heap: list, q: np.ndarray):
        cosine_similarities = [cosine_similarity(list[i], q) for i in heap]
        return cosine_similarities
    # functions for creating a heap that are sorted by cosine similarity between elements and query vector
    def sorted_list_by_cosine_similarity(self,heap: list, query_vector: np.ndarray) -> list[(int, int)]:
        heap = [(cosine_similarity(node.vec, query_vector), node) for node in heap]
        heap.sort(reverse=True)  # sort descending
        return heap
    def get_layer_location(self):
        return np.round(-float(math.log(random.uniform(0, 1))) * self._level_mult)
    def normal_selection(self, d: dict, to_insert:list[(float,int)], m: int, layer: dict, forward=False):
        if not forward:
            idx, dist = to_insert
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        to_insert = nlargest(m, to_insert)
        remaining = m - len(d)
        to_insert, replacements = to_insert[:remaining], to_insert[remaining:]
        possible_replacements = len(replacements)
        if possible_replacements > 0:
            best_replacements = nlargest(possible_replacements, d.items(), key=itemgetter(1))
        else:
            best_replacements = []
        for md, idx in to_insert:
            d[idx] = -md
        for (md_new, idx_new), (idx_old, d_old) in zip(replacements, best_replacements):
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new


    def heuristic_selection(self, d: dict, to_insert:list[(float,int)], m: int, g, forward: bool=False):
        nb_dicts = [g[idx] for idx in d]
        def prioritize(idx, dist):
            return any(nd.get(idx, float("inf")) < dist for nd in nb_dicts), dist, idx
        if not forward:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist) for mdist, idx in to_insert))
        unchecked = m - len(d)
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist) for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zip(checked_ins, checked_del):
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            
    def search(self, query_element: np.ndarray, ef: int = None, k: int = None):
        graph = self._graphs
        entry_point = self._enter_point
        if entry_point is None:
            raise Exception("The graph is empty, please insert some elements first")
        if ef is None:
            ef = self.efSearch
        if k is None:
            k = self._m
        sim = self.distance_func(query_element, self._enter_point)
        for layer in reversed(graph[1:]):
            entry_point, sim = self.search_layer_ef1(query_element, sim, entry_point, layer)
        candidates = self.search_layer(query_element, [(sim, entry_point)], ef, 0)
        candidates = nlargest(k, candidates)
        return [(sim, idxs) for sim, idxs in candidates]    

    def fast_insertion(self, elem: np.ndarray):
        distance = self.distance_func
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0
        idx = len(data)
        data.append(elem)
        if point is not None:
            dist = distance(elem, data[point])
            pd = [(point, dist)]
            for layer in reversed(graphs[1:]):
                point, dist = self.search_layer_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                level_m = m0 if level == 0 else m
                candidates = self.search_layer(elem, [(-dist, point)], layer, self.efConstruction)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, candidates, level_m, layer, forward=True)
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                if len(layer_idx) < level_m:
                    return
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx):
                        return
                point, dist = pd.pop()
        graphs.append({idx: {}})
        self._enter_point = idx

    def search_layer_ef1(self, q: np.ndarray, entry: int, dist: float, layer: dict):
        vectorized_distance = self.vectorized_distance
        data = self.data
        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])
        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
        return best, best_dist

    def search_layer(self, q: np.ndarray, ep:list[(float,int)], layer: dict, ef: int):
        vectorized_distance = self.vectorized_distance
        data = self.data
        candidates = [(-sim, point) for sim, point in ep]
        heapify(candidates)
        visited = set(point for _, point in ep)
        while candidates:
            dist, point = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            neighbors = [neighbor for neighbor in layer[point] if neighbor not in visited]
            visited.update(neighbors)
            dists = vectorized_distance(q, [data[neighbor] for neighbor in neighbors])
            for neighbor, dist in zip(neighbors, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, neighbor))
                    heappush(ep, (mdist, neighbor))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, neighbor))
                    heapreplace(ep, (mdist, neighbor))
                    mref = ep[0][0]

        return ep
    def index_file_creation(self):
        layer_assignment = {}
        graph = self._graphs.copy()
        for layer,plane in enumerate(reversed(self._graphs)):
            l = np.asarray([plane])
            np.savetxt(f"layer_{layer}.npy",l)

    def get_document_data(self, name):
        data = pd.read_csv("./"+name)
        pass
        
    
    def search(self, query_element: np.ndarray, ef: int = None, k: int = None):
        graph = self._graphs
        entry_point = self._enter_point
        if entry_point is None:
            raise Exception("The graph is empty, please insert some elements first")
        if ef is None:
            ef = self.efSearch
        if k is None:
            k = self._m
        sim = self.distance_func(query_element, self.data[self._enter_point])
        for layer in reversed(graph[1:]):  # loop on the layers till you reach layer 1
            entry_point, sim = self.search_layer_ef1(query_element, entry_point, sim, layer)
        candidates = self.search_layer(query_element, [(sim, entry_point)], graph[0], ef)
        candidates = nlargest(k, candidates)
        return [(sim, idxs) for sim, idxs in candidates]    
    
    def find_closest(self,element,K= 3):#find everything using IVF primitive
        if len(self._graphs) == 0 or element is None or K <= 0:
                return None
        if self.IVF is None:
            return self.search(element,K)
        else:
            graph = self._graphs
            entry_point = self._enter_point
            if entry_point is None:
                raise Exception("The graph is empty, please insert some elements first")
            if ef is None:
                ef = self.efSearch
            if k is None:
                k = self._m
            sim = self.distance_func(element, self.data[self._enter_point])
            for layer in reversed(graph[1:]):  # loop on the layers till you reach layer 1
                entry_point, sim = self.search_layer_ef1(element, entry_point, sim, layer)
            candidates = self.search_layer(element, [(sim, entry_point)], graph[0], ef)
            candidates = nlargest(k, candidates)
            return self.IVF.get_K_closest_neighbors_given_centroids(self.data[candidates[:][1]], element, K)
 
    def search_using_IVF(self,element: np.ndarray,ef:int,K:int): #finds using advanced IVF
        pass 
    def create_HNSWIVF(self):
        pass
            

    def graph_creation(self):
        if self.IVF is None:
            for vector in self.vectors:
                self.fast_insertion(vector)
            self.vectors = None
        else:
            self.clusters = self.IVF.clustering().keys()
            
            for vector in self.vectors:
                self.fast_insertion(vector)
            self.vectors = None

# dataset = np.random.normal(size=(100,3))
# hnsw = HNSW(2,0,None,True,50,50,dataset,False,False,0,0)
# hnsw.graph_creation()
# vector = np.random.normal(size=(1,3))
# print(hnsw._graphs)
# found = hnsw.search(vector,k=4,ef=30)
# for f in found:
#     print(hnsw.data[f[1]])
#     print(cosine_similarity([hnsw.data[f[1]]],vector))
# print(vector)