import numpy as np
from numpy import ndarray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity,pairwise_distances
import csv
from itertools import islice
from heapq import nlargest
def sort_vectors_by_cosine_similarity(vectors, reference_vector):
    # Calculate cosine similarities
    cos_similarities = cosine_similarity(reference_vector, vectors)
    # Sort indices by cosine similarity
    sorted_indices = np.argsort(cos_similarities)

    return sorted_indices

# kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
# kmeans.fit(vectors)
# cluster_centroids = {}
# for cluster_index in range(n_clusters):
#    cluster_filename = f"cluster_{cluster_index}.npy"
#    vectors_in_cluster = vectors[kmeans.labels_ == cluster_index]
#    np.save(cluster_filename, vectors_in_cluster)
#    cluster_centroids[kmeans.cluster_centers_[cluster_index]] = cluster_filename
class IVFile_optimized(object):
    def __init__(self,batch_size: int,no_vectors: int):
        self.partitions = np.ceil(no_vectors / np.sqrt(no_vectors)) * 3
        self.Kmeans = MiniBatchKMeans(np.ceil(no_vectors / np.sqrt(no_vectors)) * 3,batch_size=batch_size,init="k-means++")
        self.clusters = {}
        self.batch_size = batch_size
    def transform(self, X) -> ndarray:
        return pairwise_distances(X,self.centroids,metric = cosine_similarity)    
        
    def build_index(self,file_name_s):
        for file_name in file_name_s:
            with open(file_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                while True:
                    batch = list(islice(reader, self.batch_size))
                    if not batch:
                        break
                    self.Kmeans.partial_fit(batch)
        self.centroids = self.Kmeans.cluster_centers_
        X = 0
        for centroid in self.centroids:
            with open(f'./data/data{X}.csv','a',newline='') as csvfile:
                pass
            self.clusters[str(centroid)] = f'data{X}.csv'
            X += 1 
        self.final_pass(file_name_s)
        return self.Kmeans.cluster_centers_
    
    def final_pass(self,file_name_s):
        for file_name in file_name_s:
            with open(file_name,'r') as csvfile:
                reader = csv.reader(csvfile)
                while True:
                    batch = list(islice(reader,self.batch_size))
                    if not batch:
                        break
                    labels = np.argmax(self.transform(batch),axis = 1)
                    grouped_vectors_dict = {i: [vector for vector, label in zip(batch, labels) if label == i] for i in range(self.partitions)}
                    for label,vectors in  grouped_vectors_dict.keys(),grouped_vectors_dict:
                        file_to_insert = self.clusters[self.centroids[label]]
                        with open(file_to_insert,'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(vectors)
                           

    def retrieve_k_closest(self,query: np.ndarray, K: int): #retrieval function
        closest = self.transform(query)
        closest_k_centroids = []
        for i in range(K):
            closest_centroid = np.argmax(closest)
            closest_k_centroids.append(self.centroids[closest_centroid])
            closest[closest_centroid] = -1
        vectors  = []
        for centroid in closest_k_centroids:
            vectors.extend(self.retrieve_K_closest_data_given_centroid(centroid,query,K))
        return sort_vectors_by_cosine_similarity(vectors,query)[:K]
        
    def retrieve_K_closest_data_given_centroid(self,centroid:np.ndarray,query:np.ndarray,K: int): #batch inspection
        file_to_inspect = self.clusters[centroid]
        closest_k = []
        with open(file_to_inspect,'r') as csvfile:
            reader = csv.reader(csvfile)
            while True:
                batch = list(islice(reader,K))
                if not batch:
                    break
                closest_k = sort_vectors_by_cosine_similarity(np.concatenate((closest_k,batch)))[:K]
        return closest_k
                    
    def retrieve_K_closest_data_given_centroid(self,centroid:np.ndarray,query:np.ndarray,K: int): #full inspection
        file_to_inspect = self.clusters[centroid]
        closest_k = []
        with open(file_to_inspect,'r') as csvfile:
            reader = csv.reader(csvfile)
            all_rows = list(reader)
            closest_k = sort_vectors_by_cosine_similarity(all_rows,query)[:K]
        return closest_k
                
test_vector = np.random.normal(size=(1,70)) 
IV = IVFile_optimized(10000,1000000)
IV.build_index("./random_vectors.csv")
IV.retrieve_k_closest(test_vector,3)