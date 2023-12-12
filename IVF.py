import numpy as np
from scipy.cluster.vq import kmeans2



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
            print(centroids[x])
            centroid_assignment[str(centroids[x])] = f"./data/file{x}.npy"
            file = open(f"./data/file{x}.npy",'a')
            np.save(f"./data/file{x}.npy", byte_file)
            file.close()
            x += 1
            k = None
        self.assigments =  centroid_assignment
        return self.assigments 
    
    def get_cluster_data(centroid: np.ndarray):
        