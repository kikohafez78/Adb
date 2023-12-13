import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter


def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z

class IVFile(object):
    
    def __init__(self,partitions: int, vectors: np.ndarray):
        self.partitions = partitions
        self.vectors = vectors
        
    def clustering(self):
        (centroids,assignments)= kmeans2(self.vectors,self.partitions)
        self.data = (centroids, assignments)
        index= [[] for _ in range(self.partitions)]
        for n,k in enumerate(assignments):
            index[k].append(n)
        centroid_assignment = {}
        x = 0
        for k in index:
            byte_file = np.asarray(k)
            centroid_assignment[str(centroids[x])] = f"./data/file{x}.npy"
            file = open(f"./data/file{x}.npy",'a')
            np.save(f"./data/file{x}.npy", byte_file)
            file.close()
            x += 1
            k = None
        self.assigments =  centroid_assignment
        return self.assigments
    
    def get_closest_centroids(self, vector: np.ndarray, K: int):
        centroids = self.data[0]
        distances = np.absolute(cosine_similarity(vector,centroids))
        return sort_list(centroids,distances)
    
    def get_cluster_data(self, centroid: np.ndarray, vector: np.ndarray, K: int):
        data = np.load(self.assigments[str(centroid)])
        distances = cosine_similarity(vector,data)
        indices = np.argpartition(distances,K)[:K]
        sorted_indices = indices[np.argsort(data[indices])]
        return data[sorted_indices]
    
# K = 3
# num_partions = 16
# dataset = np.random.normal(size = (1000,10))
# Iv = IVFile(num_partions, dataset)
# a = Iv.clustering()
# test_vector = np.random.normal(size= (1,10))
# #===========================================================================
# print("test-vector:",test_vector)
# #===========================================================================
# # print("centroid to file: ",a,"\n")
# #===========================================================================
# closest= Iv.get_closest_centroids(test_vector, K)
# print(f"{K} closest clusters are: ",closest)
