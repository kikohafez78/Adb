import pickle as pi

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from clean_slate import HNSW, IVFile, cosine_similarity, itemgetter, nlargest
import itertools
import pandas as pd
from sklearn.preprocessing import normalize
def sort_vectors_by_cosine_similarity(vectors, reference_vector):
    # Calculate cosine similarities
    cos_similarities = cosine_similarity(reference_vector, vectors)
    # Sort indices by cosine similarity
    print(cos_similarities)
    sorted_indices = np.argsort(cos_similarities)
    sorted_indices[0] = np.flip(sorted_indices, axis=1)
    return sorted_indices


def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


sizes = {
    "saved_db_100k": 100000,
    "saved_db_1m": 1000000,
    "saved_db_5m": 5000000,
    "saved_db_10m": 10000000,
    "saved_db_20m": 20000000,
}
batches = {
    "saved_db_100k": 100000,
    "saved_db_1m": 1000000,
    "saved_db_5m": 5000000,
    "saved_db_10m": 10000000,
    "saved_db_20m": 20000000,
}
n_files = {
        "saved_db_100k": 1,
        "saved_db_1m": 10,
        "saved_db_5m": 50,
        "saved_db_10m": 100,
        "saved_db_20m": 200,
}


class vec_db(object):
    def __init__(self, file_path: str):
        self.folder = file_path
        self.partitions = int(np.ceil(sizes[file_path] / np.sqrt(sizes[file_path])) * 3)
        self.batch_size = batches[file_path]
        self.no_of_files = n_files[file_path]
        self.kmeans = MiniBatchKMeans(n_clusters=self.partitions)
    def insert_records(self, vectors: list):
        self.vectors = vectors

    def build_index(self):
        batch = 0
        for i in range(self.no_of_files):
            data = np.load(f"./{self.folder}/batch{batch}.npy")
            data = normalize(data)
            #===================================================
            self.kmeans.partial_fit(data)
            #===================================================
            batch += 1
        self.assignments = self.kmeans.labels_
        self.clusters = self.kmeans.cluster_centers_
        X = 0
        for cluster in self.clusters:
            # file = open(f"./{self.folder}/file{X}.pickle","w")
            # np.save(f"./{self.folder}/file{X}.pickle",np.array([]))
            # file.close()
            with open(f"./{self.folder}/file{X}.pickle","ab") as file:
                pass
            X += 1
        X = 0
        batch = 0
        clusters = {x:[] for x in range(len(self.clusters))}
        for i in range(self.no_of_files):
            data = np.load(f"./{self.folder}/batch{batch}.npy")
            #===================================================
            for vector in data:
                clusters[self.assignments[X]].append(vector)
                X += 1
            #===================================================
            batch += 1
        X = 0
        for  cluster in clusters:
            with open(f"./{self.folder}/file{X}.pickle","ab") as file:
                pi.dump(clusters[cluster],file)
            X += 1
        X = 0
        assignments = {}
        for cluster in self.clusters:
            assignments[str(X)] = (f"./{self.folder}/file{X}.pickle",cluster)
            X += 1
        with open(f"./{self.folder}/clusters.pickle","ab") as file:
            pi.dump(assignments,file)
        self.clusters = None
        self.assignments = None
        return 

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):  # <===
        vector = normalize([vector])
        with open(f"./{self.folder}/clusters.pickle","rb") as file:
            self.clusters = pi.load(file)
        centroids = [self.clusters[str(centroid)][1] for centroid in range(self.partitions)]
        files = [self.clusters[str(centroid)][0] for centroid in range(self.partitions)]
        vectors = sort_vectors_by_cosine_similarity(centroids,vector)[0][:30]
        files_to_inspect = [files[x] for x in vectors]
        full_vectors = []
        for file in files_to_inspect:
            with open(file,"rb") as file:
                data = pi.load(file)
                data = normalize(data)
                vectors = sort_vectors_by_cosine_similarity(data[:][:70],vector)[0][:K+1]
                vectors = [data[vec] for vec in vectors]
                full_vectors.extend(vectors)
        full_vectors = [vector[:70] for vector in full_vectors]
        final_vectors = sort_vectors_by_cosine_similarity(full_vectors,vector.reshape(1,-1))[0][:K+1]
        final_vectors = [full_vectors[vec] for vec in final_vectors]
        return final_vectors
            

# Example usage:
vecDB = vec_db("saved_db_100k")   # Use the appropriate file path
vector = np.load("./saved_db_100k/batch0.npy")[0]
# print(vecDB.build_index())
# print(pd.read_pickle("./saved_db_100k/clusters.pickle"))
vectors = vecDB.get_closest_k_neighbors(vector,3)
vec_no_ids = [vector[:70] for vector in vectors]
for vector_no_id in vec_no_ids:
    print(cosine_similarity([vector_no_id],[vector]))