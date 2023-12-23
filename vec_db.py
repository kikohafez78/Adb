from clean_slate import HNSW,IVFile,nlargest,itemgetter,cosine_similarity
import numpy as np

class vec_db(object):
    def __init__(self,partitions: int, vectors : np.ndarray):
        self.IVF = IVFile(partitions,vectors)
        self.partitions = partitions
        
    def build_index(self):
        self.assignments = self.IVF.clustering()
        x = 1
        for assignment in self.assignments.keys():
            data = self.IVF.cluster_data(np.fromstring(assignment))
            hnsw = HNSW(2,0,None,True,50,50,data,False,False,0,0)
            hnsw.graph_creation()
            graph = [hnsw.data,hnsw._graphs]
            array = np.asarray(graph)
            np.save(f"./data/hnsw{x}.npy",array)
            self.assignments[assignment] = f"./data/hnsw{x}.npy"
            x += 1
        for i in range(self.partitions):
            np.delete(f"./data/file{i}.npy")
        data = [np.formstring(assignment) for assignment in self.assignments.keys()]
        self.hnsw = HNSW(2,0,None,True,50,50,data,False,False,0,0)
        self.hnsw.graph_creation()
        graph = [self.hnsw.data,self.hnsw._graphs]
        array = np.asarray(graph)
        np.save("./data/hnsw0.npy",array)
        self.hnsw._graphs = None
        self.hnsw.data = None
        self.name = "./data/hnsw0.npy"
        
    def search_index(self, query: np.ndarray,K: int):
        graph = np.load(self.name)
        self.hnsw._graphs = graph[1]
        self.hnsw.data = graph[0]
        centroids = self.hnsw.search(query,K)
        for centroid in centroids:
            centroid[1] = self.hnsw.data[centroid[1]]
        files = [self.assignments[str(centroid[1])] for centroid in centroids]
        closest = []
        for file in files:
            graph = np.load(file)
            self.hnsw._graphs = graph[1]
            self.hnsw.data = graph[0]
            vectors = self.hnsw.search(query,K)
            for vector in vectors:
                vector[1] = self.hnsw.data[vector[1]]
            closest += vectors
        data = nlargest(K,closest,key = itemgetter(0))
        self.hnsw._graphs = None
        self.hnsw.data = None
        return data
    
dataset = np.random.normal(size=(100,3))
vecDB = vec_db(16,dataset)
vecDB.build_index()
vector = np.random.normal(size=(1,3))

            
        
        
            