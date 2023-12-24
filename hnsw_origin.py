# -*- coding: utf-8 -*-

import pprint
import sys
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random

import numpy as np


class HNSW(object):
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def l2_distance(self, a, b):
        return np.linalg.norm(a - b)

    def cosine_distance(self, a, b):
        try:
            return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
        except ValueError:
            print(a)
            print(b)

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        pprint.pprint([self.distance_func(x, y) for y in ys])
        return [self.distance_func(x, y) for y in ys]

    def __init__(self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        self.data = []
        if distance_type == "l2":
            # l2 distance
            distance_func = self.l2_distance
        elif distance_type == "cosine":
            # cosine distance
            distance_func = self.cosine_distance
        else:
            raise TypeError("Please check your distance type!")

        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

        self._select = self._select_heuristic if heuristic else self._select_naive

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef
        # those are just "references" to the class attributes
        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # level at which the element will be inserted
        level = int(-log2(random()) * self._level_mult) + 1
        # print("level: %d" % level)

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):  # if level of vector > highest level in HNSW, list will be empty
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # at these levels we have to insert elem; ep is a heap of entry points.
            ep = [(-dist, point)]
            # pprint.pprint(ep)
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                # if level of vector > highest level in HNSW, list will be full and we will loop on all levels
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, layer, ef)
                # insert in g[idx] the best neighbors
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # assert len(layer_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx

    def balanced_add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
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
            # pprint.pprint(len(graphs))
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                # print('\n')
                # pprint.pprint(layer)
                level_m = m0 if level == 0 else m
                candidates = self._search_graph(elem, [(-dist, point)], layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                # add reverse edges
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

    def search(self, q, k=None, ef=None):
        """Find the k points closest to q."""

        distance = self.distance
        graphs = self._graphs
        point = self._enter_point

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = distance(q, self.data[point])
        # look for the closest neighbor from the top to the 2nd level
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        # look for ef neighbors in the bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

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

    def _search_graph_ef1(self, q, entry, dist, layer):
        """Equivalent to _search_graph when ef=1."""

        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        # set of visited nodes, to avoid loops
        visited = set([entry])
        # for each candidate, we look for the closest neighbor in the graph layer
        # if the distance is better than the best, we update best and best_dist
        # and add the neighbor to the candidates heap
        # and add the neighbor to the visited set
        # if dist > best_dist, we stop the search
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
                    # break

        return best, best_dist

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

    def _search_graph(self, q, ep, layer, ef):
        vectorized_distance = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            # pprint.pprint(layer[c])
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

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

    def _select_naive(self, d, to_insert, m, layer, heap=False):
        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)  # smallest m distances
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):
        nb_dicts = [g[idx] for idx in d]

        def prioritize(idx, dist):
            return any(nd.get(idx, float("inf")) < dist for nd in nb_dicts), dist, idx

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist) for mdist, idx in to_insert))

        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist) for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m

    def __getitem__(self, idx):
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return


if __name__ == "__main__":
    dim = 25
    num_elements = 1000

    import pickle
    import time

    import h5py

    # from progressbar import *

    f = h5py.File("glove-25-angular.hdf5", "r")
    distances = f["distances"]
    neighbors = f["neighbors"]
    test = f["test"]
    train = f["train"]
    pprint.pprint(list(f.keys()))
    pprint.pprint(train.shape)
    # pprint.pprint()

    # Generating sample data
    data = np.array(np.float32(np.random.random((num_elements, dim))))
    data_labels = np.arange(num_elements)

    hnsw = HNSW("cosine", m0=5, ef=10)

    # widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
    #     ' ', ETA()]
    # pbar = ProgressBar(widgets=widgets, maxval=train.shape[0]).start()
    # for i in range(train.shape[0]):
    #     # if i == 1000:
    #     #     break
    #     hnsw.balanced_add(train[i])
    #     pbar.update(i + 1)
    # pbar.finish()

    for index, i in enumerate(data):
        if index % 1000 == 0:
            pprint.pprint("train No.%d" % index)
        # hnsw.balanced_add(i)
        hnsw.add(i)

    # with open('glove-25-angular-balanced-128.ind', 'wb') as f:
    #     picklestring = pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)

    # add_point_time = time.time()
    # idx = hnsw.search(np.float32(np.random.random((1, 25))), 1)
    # search_time = time.time()
    # pprint.pprint(idx)
    # pprint.pprint("add point time: %f" % (add_point_time - time_start))
    # pprint.pprint("searchtime: %f" % (search_time - add_point_time))
    # print('\n')
    # # pprint.pprint(hnsw._graphs)
    # for n in hnsw._graphs:
    #     pprint.pprint(len(n))
    # pprint.pprint(len(hnsw._graphs))
    # print(hnsw.data)

# for index, i in enumerate(data):
#     idx = hnsw.search(i, 1)
#     pprint.pprint(idx[0][0])
#     pprint.pprint(i)
#     pprint.pprint(hnsw.data[idx[0][0]])

# pprint.pprint('------------------------------')
# pprint.pprint(hnsw.data)
# pprint.pprint('------------------------------')
# pprint.pprint(data)
# pprint.pprint('------------------------------')
# pprint.pprint(hnsw._graphs)
# pprint.pprint(len(hnsw._graphs))
