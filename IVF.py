from operator import itemgetter

import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


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


# clusters = ceil(len(vectors)/sqrt(len(vectors)) * 3
class IVFile(object):
    def __init__(self, partitions: int, vectors: np.ndarray):
        self.partitions = partitions
        self.vectors = vectors

    def clustering(self):
        (centroids, assignments) = kmeans2(self.vectors, self.partitions)
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


class IVFTree(object):
    def __init__(self, partitions: int, vectors: np.ndarray):
        self.partitions = partitions
        self.nlevels = np.floor(np.log2(partitions))
        self.vectors = vectors

    def create_tree(self):
        self.graph: list[IVFile] = []
        self.assignments: list[list[dict]] = []
        vectors = self.vectors
        for i in range(self.nlevels, -1, 1):
            iv = IVFile(i**2, vectors)
            cluster_assignment = iv.clustering()
            self.assignments.append(cluster_assignment)
            vectors = [np.fromstring(cluster) for cluster in cluster_assignment.keys()]
            self.graph.append(iv)

    def find(self, vector: np.ndarray, K: int):
        level0 = self.graph[0]
        clusters = vector
        for iv in reversed(self.graph):
            if iv != level0:
                clusters = iv.get_closest_k_neighbors(clusters, 1)
            else:
                clusters = iv.get_K_closest_neighbors_given_centroids(clusters, vector, K)
        return clusters


# clusters = ceil(len(vectors)/sqrt(len(vectors)) * 3
class IVFile_optimized(object):
    def __init__(self, vectors: np.ndarray):
        self.partitions = np.ceil(len(vectors) / np.sqrt(len(vectors))) * 3
        self.vectors = vectors

    def clustering(self):
        (centroids, assignments) = kmeans2(self.vectors, self.partitions)
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


# K = 3
# num_partions = 4096
# dataset = np.random.normal(size=(20000000, 70))
# Iv = IVFile_optimized(dataset)
# a = Iv.clustering()
# test_vector = np.random.normal(size=(1, 70))
# # ===========================================================================
# print("test-vector:", test_vector)
# # ===========================================================================
# print("centroid to file: ", a, "\n")
# # ===========================================================================
# # print(Iv.get_closest_centroids(test_vector,K),"\n",cosine_similarity(Iv.get_closest_centroids(test_vector,K),test_vector))
# # ===========================================================================
# closest = Iv.get_closest_k_neighbors(test_vector, K)
# print(f"{K} closest vectors are: ", closest, cosine_similarity(closest, test_vector))
