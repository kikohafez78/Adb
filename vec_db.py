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


class vec_db(object):
    def __init__(self, file_path: str, new_db=False):
        switch = {
            "saved_db_100k.csv": 200,
            "saved_db_1m.csv": 1500,
            "saved_db_5m.csv": 4000,
            "saved_db_10m.csv": 6000,
            "saved_db_20m.csv": 8000,
        }
        self.partitions = switch[file_path]
        self.vectors = vectors  # TODO: to be replaced with a file paths that we will load the vectors from (numpy arrays)

    def build_index(self):
        # TODO: make the clustering work with batches of vectors instead of all at once (memory issues)
        kmeans = MiniBatchKMeans(n_clusters=self.partitions)
        assignments = kmeans.fit_predict(self.vectors)
        centroids = kmeans.cluster_centers_
        # (centroids, assignments) = kmeans2(self.vectors, self.partitions)
        self.data = (centroids, assignments)
        index = [[] for _ in range(self.partitions)]
        for n, k in enumerate(assignments):
            index[k].append(self.vectors[n])
        centroid_assignment = {}
        x = 0
        for k in index:
            byte_file = np.asarray(k)
            centroid_assignment[str(centroids[x])] = f"./data/file{x}.npy"
            file = open(f"./data/file{x}.npy", "a")
            np.save(f"./data/file{x}.npy", byte_file)
            file.close()
            x += 1
            k = None
        self.assigments = centroid_assignment
        self.vectors = None  # <===
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

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):
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


dataset = np.random.normal(size=(100, 3))
vecDB = vec_db(16, dataset)
vecDB.build_index()
vector = np.random.normal(size=(1, 3))
