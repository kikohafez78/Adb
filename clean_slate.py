# ==================================================
# Authors: 1-Omar Badr                             |
#          2-Karim Hafez                           |
#          3-Abdulhameed                           |
#          4-Seif albaghdady                       |
# Subject: ADB                                     |
# Project: HNSW +                                  |
# ==================================================
import math
import random
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from IVF import IVFile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from operator  import itemgetter
import logging as logs
import pandas as pd
import xlrd as xl
from product_quantization import quantizer

def get_consin_similarity(self, vector: np.ndarray):
    return cosine_similarity(self.data, vector)




class HNSW(object):
    def __init__(self, M: int, efSearch: int, efConstruction: int, vectors: np.ndarray = None, heuristic=False, M0: int = None,useIVF:bool = False,useQuantizer: bool = False,partitions:int = 3,segments: int = 3) -> None:
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
        self.data:list[np.ndarray] = []# data[node_id] = vector
        self.vectors = vectors
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
    def create_IVF(self):
        self.assignments = self.IVF.clustering()
        self.centroids = [np.fromstring(cluster) for cluster in self.assignments.keys()]
    def get_neighbors_list(self,id):
        return self.graph[:][id]

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
        return np.round(-float(math.log(random.uniform(0, 1))) * self.ml)
    def search_layer(self, q: np.ndarray, entry_points: list[(float,int)], ef: int, layer: int):
        data = self.data
        plane = self.graph[layer]
        candidates = [(-sim, points) for sim, points in entry_points]
        visited = set(point for _, point in entry_points)
        heapify(candidates)
        while candidates:
            sim, point = heappop(candidates)
            starting_sim = entry_points[0][0]
            if sim > -starting_sim:
                break
            try:
                neighbors = [neighbor for neighbor in plane[point] if neighbor not in visited]
            except:
                neighbors = []
                logs.critical(f"search layer is skipping the neighbors @ node {point}\n")
            visited.update(neighbors)
            similarities = self.calculate_distances([data[neighbor] for neighbor in neighbors],q)
            for point, sim in zip(neighbors, similarities):
                max_sim = -sim
                if len(entry_points) < ef:
                    heappush(candidates, (sim, point))
                    heappush(entry_points, (max_sim, point))
                    starting_sim = entry_points[0][0]
                elif max_sim > starting_sim:
                    heappush(candidates, (sim, point))
                    heapreplace(entry_points, (max_sim, point))
                    starting_sim = entry_points[0][0]
        if entry_points == None or len(entry_points) == 0: logs.warning(f"empty entry point @ node {q}\n")
        return entry_points
    def search_layer_ef1(self, query:np.ndarray , entry_point: int, dist: float, layer: int):
        data = self.data
        plane = self.graph[layer]
        closest_vec = entry_point
        closest_simlarity = dist
        candidates = [(dist, entry_point)]
        visited = set([entry_point])

        while candidates:
            dist, point = heappop(candidates)
            if dist > closest_simlarity:
                break
            try:
                neighbors = [neighbor for neighbor in plane[point] if neighbor not in visited]
            except:
                neighbors = []
                logs.critical("search ef1 is skipping the neighbors @ node {point}\n")
            visited.update(neighbors)
            dists = self.calculate_distances([data[e] for e in neighbors],query)
            for neighbor, dist in zip(neighbors, dists):
                if dist < closest_simlarity:
                    closest_vec = neighbor
                    closest_simlarity = dist
                    heappush(candidates, (dist, neighbor))
        return closest_simlarity, closest_vec
    
    def get_closest_simple(self,current: (float,int),M: int,layer: int):
        pass
    
    def Insert(self,query: np.ndarray):
        data = self.data
        id = len(data)
        data.append(query)
        graph = self.graph
        entry = self.entry_points
        M = self.M
        L = int(self.get_layer_location())
        if entry is not None:
            distance_to_entry = self.array_similarity(self.data[entry],query)
            for layer,planes in enumerate(graph[L:]):
                distance_to_entry, entry = self.search_layer_ef1(query,entry,distance_to_entry,layer)
            ep = [(distance_to_entry,  entry)]
            for layer, planes in enumerate(reversed(graph[:L])):
                M = self.M_MAX if layer != 0 else self.M0
                ep = self.search_layer(query,ep,self.efConstruction,layer) 
                planes[id] = {}
                if self.heuristic:
                    pass
                else:
                    pass
                for ids,cosine_similarity in planes[id].items():
                    if self.heuristic:
                        pass
                    else:
                        pass
        for plane in range(len(graph),L):
            graph.append({id:{}})
            self.entry_points = id
        logs.log(0,f"succefully inserted node{id} @ level {L} and the number of current layers is {len(graph)}\n")
    
    def normal_selection(self,neigborhood: dict, current_node:list[(float,int)], M: int, layer: int):
        data = self.data
        sim,id = current_node
        if id not in neigborhood and len(neigborhood) < M:
            neigborhood[id] = sim
            current_node.pop(0)
        if not any(ids for _, ids in current_node):
            current_node = nlargest(M, current_node, key = itemgetter(0))
            remaining = M - len(neigborhood)
            current_node, possible_inserts = current_node[:remaining],current_node[remaining:]
            no_po_in = len(possible_inserts)
            if no_po_in > 0:
                replacements = nlargest(no_po_in,neigborhood.items(),key = itemgetter(1))
            else:
                replacements = []
                for dist,id in current_node:
                    dist[id] = -dist
                for (new_dist, new_id),(old_id,old_dist) in zip(possible_inserts,replacements):
                    if old_dist <= new_dist:
                        break
                    del neigborhood[old_id]
                    neigborhood[new_id] = -new_dist
    
    def get_document_data(self, name):
        data = pd.read_csv("./"+name)
        pass
        
        
    
    
    
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
    
    def find_closest(self,element,K= 3):
        if len(self.graph) == 0 or element is None or K <= 0:
                return None
        if self.IVF is None:
            ep = self.entry_points
            distance_ep = self.array_similarity(self.data[ep],element)
            Graph = self.graph
            for layer,nodes in enumerate(reversed(Graph)):
                if layer > 0:
                    new_ep,distance = self.search_layer_ef1(element,distance_ep,ep,layer)
                else:
                    ep = self.search_layer(element,[(distance,new_ep)],self.efSearch,layer)
                    if K > 1:
                        ep = nlargest(K,ep)
                    else:
                        sorted(ep,reverse=True)
                        return ep[0]
            return ep[:K]
        else:
            ep = self.entry_points
            distance_ep = self.array_similarity(self.data[ep],element)
            Graph = self.graph
            for layer,nodes in enumerate(reversed(Graph)):
                if layer > 0:
                    new_ep,distance = self.search_layer_ef1(element,distance_ep,ep,layer)
                else:
                    ep = self.search_layer(element,[(distance,new_ep)],self.efSearch,layer)
                    if K > 1:
                        ep = nlargest(K,ep)
                    else:
                        sorted(ep,reverse=True)
                        return self.IVF.get_K_closest_neighbors_given_centroids(self.data[ep[0][1]], element, K)
            return self.IVF.get_K_closest_neighbors_given_centroids(self.data[ep[:K][1]], element, K)
        
    def search_using_IVF(self,element: np.ndarray,ef:int,K:int):
        graph = self.graph.copy()
        closest_centroid = self.IVF.get_closest_centroids(element,K)
        cluster_data = self.IVF.get_cluster_data(closest_centroid,element,K)
        #===========================================
        for plane in graph:
            for vec in plane.keys():
                if vec not in cluster_data:
                    for link in plane[vec].keys():
                        del plane[link][vec]
                    del plane[vec]
        #===========================================
        #experimental do not try, here i terminate all nodes in search space unrelated to the cluster i am closest to then i terminate all of their backlinks  
        #decreasing search space as requested     
        entry_point = self.entry_points
        if entry_point is None:
            raise Exception("The graph is empty, please insert some elements first")
        if ef is None:
            ef = self.efSearch
        if k is None:
            k = self.M
        sim = cosine_similarity(element, self.entry_points)
        for layer in reversed(graph[1:]):  # loop on the layers till you reach layer 1
            entry_point, sim = self.search_layer_ef1(element, sim, entry_point, layer)
        candidates = self.search_layer(element, [(sim, entry_point)], ef, 0)
        candidates = nlargest(k, candidates)
        return [(sim, idxs) for sim, idxs in candidates] 
        
    def graph_creation(self):
        if self.IVF is None:
            for vector in self.vectors:
                self.Insert(vector)
            self.vectors = None
        else:
            self.clusters = self.IVF.clustering().keys()
            for vector in self.vectors:
                self.Insert(vector)
            self.vectors = None
    
dataset = np.random.normal(size=(10000,4))
hnsw = HNSW(2,16,16,dataset,False,None,False,False,0,0)
hnsw.graph_creation()
print(hnsw.graph)   
    
           