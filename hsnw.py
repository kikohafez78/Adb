#==================================================
#Authors: 1-Omar Badr                             |
#         2-Karim Hafez                           |
#         3-Abdulhameed                           |
#         4-Seif albaghdady                       |
#Subject: ADB                                     |
#Project: HNSW +                                  |
#==================================================
from product_quantization import quantizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import threading
import numba as nm
import random
dimension = 70
segments = 14
nbits = 8

#random number generator
#I = float[-ln(uniform(0,1))*mL] mL is better selected as a non-zero element
#The authors of the paper propose choosing the optimal value of mL which is equal to 1 / ln(M). This value corresponds to the parameter p = 1 / M of the skip list being an average single element overlap between the layers.

class Node(object):
    
    def __init__(self,vector: np.ndarray,M: int,all_layers: int,M_MAX: int):
        self.vec = vector
        self.M = M
        self.layers = all_layers
        self.M_MAX = M_MAX
        self.friends_list = []
    
    @nm.set(fast_math = True)
    def get_distance_n_similarity(self, vector: np.ndarray):
        return cosine_distances(self.vec,vector),cosine_similarity(self.vec,vector)


#good values for M lie between 5 and 48, High M is better for high dimensionality and recall -> dim = 70 therefore M must be High
#Higher values of efConstruction imply a more profound search as more candidates are explored. However, it requires more computations. Authors recommend choosing such an efConstruction value that results at recall being close to 0.95–1 during training.
#M_MAX must be close to 2*M
#heuristic of selecting M out of efConstruction is both closest Nodes and connectivity of th graph by considering the connectivity distances between the candidates
class vector_db(object):
    
    def __init__(self,M: int,efSearch: int, efConstruction): #M is max number of links per node, M_MAX is max number of links connected to the node, efSearch is number of threads or number of nearest neighbors to use to find closest query result
        self.M = M
        self.M_MAX = 2*M
        self.efSearch = efSearch
        self.efConstruction = efConstruction
        self.Layers = -1
        self.max_layers = 0
    def insertion(self,q:Node,normalization_factor):
        pass    
    
    
    def graph_creation(self):
        pass      
        