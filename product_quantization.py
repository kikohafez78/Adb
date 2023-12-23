from __future__ import annotations

import logging as log

import numba
import numpy as np
from sklearn.cluster import KMeans as km
from sklearn.metrics.pairwise import cosine_similarity

BITS2DTYPE = {8: np.uint8}


class quantizer(object):
    def __init__(self, dimension: int, m: int, nbits: int, estimators_kwargs):
        if dimension % m != 0:
            raise ValueError("m must be a divisor of dimension")
        if nbits not in BITS2DTYPE:
            raise ValueError("nbits must be in ")
        self.m = m
        self.dim = dimension
        self.k = 2**nbits
        self.ds = dimension // self.k
        self.estimators = [km(n_clusters=self.k, **estimators_kwargs) for _ in range(m)]
        log.info("Creating following estimators:{self.estimators[0]!r}")
        self.is_trained = False
        self.dtype = BITS2DTYPE[nbits]
        self.dtype_orig = np.float32
        self.codes: np.ndarray | None = None

    # @numba.njit(fast_math = True)
    def train(self, X: np.ndarray):
        if self.is_trained:
            raise ValueError("Training has been done")
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:][i * self.ds : (i + 1) * self.ds]
            log.info("fitting KMaens")
            estimator.fit(X_i)
        self.is_trained = True

    def encode(self, X: np.ndarray):
        n = len(X)
        result = np.empty((n, self.m), dtype=self.dtype)
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:][i * self.ds : (i + 1) * self.ds]
            result[:][i] = estimator.predict(X_i)
        return result

    def add(self, X: np.ndarray):
        if not self.is_trained:
            raise ValueError("not trained yet")
        self.codes = self.encode(X)

    # @numba.njit(fast_math = True)
    def get_asymmetric_distances(self, X: np.ndarray):
        if not self.is_trained:
            raise ValueError("no trained")
        if self.codes is None:
            raise ValueError("no codes detected,you need to run add first")
        n_queries = len(X)
        n_codes = len(self.codes)
        distance_table = np.empty((n_queries, self.m, self.k), dtype=self.dtype_orig)
        for i in range(self.m):
            X_i = X[:][i * self.ds : (i + 1) * self.ds]
            centers = self.estimators[i].cluster_centers_
            distance_table[:, i, :] = cosine_similarity(X_i, centers, squared=True)
        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)
        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i]]
        return distances

    # @numba.njit(fast_math = True)
    def search(self, X: np.ndarray, k: int):
        n_queries = len(X)
        distances_all = self.get_asymmetric_distances(X)
        indices = np.argsort(distances_all, axis=1)[:, :k]
        distances = np.empty((n_queries, k), dtype=np.float32)
        for i in range(n_queries):
            distances[i] = distances_all[i][indices[i]]
        return distances, indices

    def get_encodings(self):
        return self.codes

    def getcentroids(self):
        return [estimator.cluster_centers_ for estimator in self.estimators]
