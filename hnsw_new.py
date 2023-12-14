import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import random


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
