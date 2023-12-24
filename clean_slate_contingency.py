# ==================================================
# Authors: 1-Omar Badr                             |
#          2-Karim Hafez                           |
#          3-Abdulhameed                           |
#          4-Seif albaghdady                       |
# Subject: ADB                                     |
# Project: HNSW +                                  |
# ==================================================
import logging as logs
import math
import random
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from operator import itemgetter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, l

from IVF import IVFile, sort_vectors_by_cosine_similarity
from product_quantization import quantizer


class vector_db_IVFPQ:
    def __init__(self, vectors=None, partitions=3, segments=3, dim: int = 70, nbits: int = 32):
        self.vectors = vectors
        self.partitions = partitions
        self.segments = segments
        self.dim = dim
        self.nbits = nbits

    def l2_norm(self, vector: np.ndarray):
        return np.linalg.norm(vector)

    def vector_preparation(self):
        for vector in self.vectors:
            vector = self.l2_norm(vector)
        self.IVF = IVFile(self.partitions, self.vectors)
        self.clusters = self.IVF.clustering()
        self.cluster_headers = [np.fromstring(vector) for vector in self.clusters.keys()]
        self.vectors = None
        self.PQ = quantizer(self.dim, self.segments, self.nbits)
        self.PQ.train(self.cluster_headers)
        self.ids = {}
        for cluster in self.cluster_headers:
            self.ids[self.PQ.encode(cluster)] = cluster
        self.cluster_headers = self.PQ.codes

    def knn(self, X: np.ndarray, K: int):
        closest = self.PQ.search(X, K)
        original = []
        for quantized_centroid in closest:
            original.append(self.ids[quantized_centroid])
        data = self.IVF.get_K_closest_neighbors_given_centroids(original, X, K)
        return data
