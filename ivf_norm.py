import pickle as pi

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from clean_slate import HNSW, IVFile, cosine_similarity, itemgetter, nlargest


def sort_vectors_by_cosine_similarity(vectors, reference_vector):
    # Calculate cosine similarities
    cos_similarities = cosine_similarity(reference_vector, vectors)
    # Sort indices by cosine similarity
    sorted_indices = np.argsort(cos_similarities)
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
    "saved_db_100k": 10,
    "saved_db_1m": 100,
    "saved_db_5m": 500,
    "saved_db_10m": 1000,
    "saved_db_20m": 2000,
}


class vec_db(object):
    def __init__(self, file_path: str):
        self.folder = file_path
        self.partitions = np.ceil(sizes[file_path] / np.sqrt(sizes[file_path])) * 3
        self.batch_size = batches[file_path]

    def insert_records(self, vectors: list):
        self.vectors = vectors

    def build_index(self):
        # ===================================================
        kmeans = MiniBatchKMeans(n_clusters=self.partitions)
        kmeans.partial_fit(self.vectors)
        # ===================================================
        assignments = kmeans.labels_
        centroids = kmeans.cluster_centers_
        self.data = (centroids, assignments)
        index = [[] for _ in range(self.partitions)]
        for n, k in enumerate(assignments):
            index[k].append(self.vectors[n])
        centroid_assignment = {}
        x = 0
        for k in index:
            byte_file = np.asarray(k)
            centroid_assignment[str(centroids[x])] = f"./{self.folder}/file{x}.npy"
            file = open(f"./{self.folder}/file{x}.npy", "a")
            np.save(f"./{self.folder}/file{x}.npy", byte_file)
            file.close()
            x += 1
        self.assigments = centroid_assignment
        self.vectors = None  # <===
        file = open(f"./{self.folder}/clusters.npy", "a")
        np.save(f"./{self.folder}/clusters.npy", np.asarray([self.assigments]))
        file.close()
        return self.assigments

    def get_closest_centroids(self, vector: np.ndarray, K: int):
        centroids = self.data[0]
        index = sort_vectors_by_cosine_similarity(centroids, vector)
        centroids = centroids[index]
        return centroids[0][len(centroids) - K - 1 :]

    def get_cluster_data(self, centroid: np.ndarray, vector: np.ndarray, K: int):
        data = np.load(self.assigments[str(centroid)])
        index = sort_vectors_by_cosine_similarity(data, vector)
        data = data[index]
        return data[0][len(data) - K - 1 :]

    def cluster_data(self, centroids: np.ndarray):
        return [np.load(self.assigments[str(centroid)]) for centroid in centroids]

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):  # <===
        centroids = self.get_closest_centroids(vector, K)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neighbors_given_centroids(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neigbors_inside_centroid_space(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, 1)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]


# Example usage:
dataset = np.random.normal(size=(200, 3))
vecDB = vec_db("saved_db_100k.npy", dataset)  # Use the appropriate file path
vecDB.build_index()
vector = np.random.normal(size=(1, 3))
