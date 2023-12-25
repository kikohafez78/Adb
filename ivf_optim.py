import csv
from heapq import nlargest
from itertools import islice

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


def sort_vectors_by_cosine_similarity(vectors, reference_vector):
    # Calculate cosine similarities
    cos_similarities = cosine_similarity(reference_vector, vectors)
    # Sort indices by cosine similarity
    sorted_indices = np.argsort(cos_similarities)
    # sorted_indices = sorted(sorted_indices,reverse=True)
    sorted_indices[0] = np.flip(sorted_indices, axis=1)
    # print("===========================")
    # print(cos_similarities)
    # print(sorted_indices)
    # print("===========================")
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
    def __init__(self, batch_size: int, no_vectors: int):
        self.partitions = np.ceil(no_vectors / np.sqrt(no_vectors)) * 3
        self.Kmeans = MiniBatchKMeans(int(np.ceil(no_vectors / np.sqrt(no_vectors)) * 3), batch_size=batch_size, init="k-means++")
        self.clusters = {}
        self.batch_size = batch_size

    def transform(self, X) -> ndarray:
        return pairwise_distances(X, self.centroids, metric=cosine_similarity)

    def similarity(self, query: np.ndarray, neighbor: np.ndarray):
        try:
            return np.dot(query, neighbor.T) / (np.linalg.norm(query) * np.linalg.norm(neighbor))
        except:
            print(query.shape)
            print(neighbor.shape)

    def vectorized_distance_(self, x, ys):
        return [self.similarity(x, y) for y in ys]

    def build_index(self, file_name_s, n: int):
        if n > 1:
            for file_name in file_name_s:
                with open(file_name, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    while True:
                        batch = list(islice(reader, self.batch_size))
                        if not batch:
                            break
                        self.Kmeans.partial_fit(batch)
            self.centroids = self.Kmeans.cluster_centers_
            X = 0
            for centroid in self.centroids:
                with open(f"./data/data{X}.csv", "a", newline="") as csvfile:
                    pass
                self.clusters[str(centroid)] = f"data{X}.csv"
                X += 1
            self.final_pass(file_name_s)
            return self.Kmeans.cluster_centers_
        else:
            with open(file_name_s, "r") as csvfile:
                reader = csv.reader(csvfile)
                while True:
                    batch = list(islice(reader, self.batch_size))
                    if not batch:
                        break
                    self.Kmeans.partial_fit(batch)
            self.centroids = self.Kmeans.cluster_centers_
            X = 0
            for centroid in self.centroids:
                with open(f"./data/data{X}.csv", "a", newline="") as csvfile:
                    pass
                self.clusters[str(centroid)] = f"./data/data{X}.csv"
                X += 1
            self.final_pass(file_name_s, n)
            with open(f"./clusters.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for centroid in self.centroids:
                    writer.writerow(centroid)
            return self.Kmeans.cluster_centers_

    def final_pass(self, file_name_s, n: int):
        if n > 1:
            for file_name in file_name_s:
                with open(file_name, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    while True:
                        batch = list(islice(reader, self.batch_size))
                        if not batch:
                            break
                        labels = np.argmax(self.similarity([batch], self.centroids), axis=1)
                        grouped_vectors_dict = {
                            i: [vector for vector, label in zip(batch, labels) if label == i] for i in range(self.partitions)
                        }
                        for label, vectors in grouped_vectors_dict.keys(), grouped_vectors_dict:
                            file_to_insert = self.clusters[self.centroids[label]]
                            with open(file_to_insert, "a", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(vectors)
        else:
            with open(file_name_s, "r") as csvfile:
                reader = csv.reader(csvfile)
                for vector in reader:
                    if not vector:
                        break
                    labels = sort_vectors_by_cosine_similarity(self.centroids, np.asarray(vector).reshape((1, -1)))[0][0]
                    file_to_insert = self.clusters[str(self.centroids[labels])]
                    with open(file_to_insert, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(vector)
                # while True:
                #     batch = list(islice(reader,self.batch_size))
                #     if not batch:
                #         break
                #     labels = np.argmax(self.transform([batch]),axis = 1)
                #     grouped_vectors_dict = {i: [vector for vector, label in zip(batch, labels) if label == i] for i in range(self.partitions)}
                #     for label,vectors in  grouped_vectors_dict.keys(),grouped_vectors_dict:
                #         file_to_insert = self.clusters[self.centroids[label]]
                #         with open(file_to_insert,'a', newline='') as csvfile:
                #             writer = csv.writer(csvfile)
                #             writer.writerows(vectors)

    def retrieve_k_closest(self, query: np.ndarray, K: int):  # retrieval function
        self.assignments = {}
        X = 0
        self.centroids = []
        with open("./clusters.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for centroid in reader:
                if not centroid:
                    break
                self.assignments[str(centroid)] = f"./data/data{X}.csv"
                self.centroids.append(centroid)
                X += 1
        closest_K = sort_vectors_by_cosine_similarity(self.centroids, query)[0][:30]
        K_cent = []
        for v in closest_K:
            K_cent.append(self.centroids[v])
        full_vectors = []
        for centroid in K_cent:
            file_to_inspect = self.assignments[str(centroid)]
            closest_k = []
            with open(file_to_inspect, "r") as csvfile:
                reader = csv.reader(csvfile)
                all_rows = list(reader)
                if all_rows == []:
                    continue
                closest_k = sort_vectors_by_cosine_similarity(all_rows, query)[0][: K + 1]
                closest_k = [all_rows[vector] for vector in closest_k]
                for vector in closest_k:
                    vector = np.array([float(vec) for vec in vector], dtype=float)
                    full_vectors.append(vector)
        # print(full_vectors)
        closest_indices = sort_vectors_by_cosine_similarity(full_vectors, query)[0][: K + 1]
        closest_vectors = []
        for vector, value in enumerate(closest_indices):
            # print(vector,value)
            closest_vectors.append(np.array(full_vectors[value], dtype=float))
        return closest_vectors
        #         for vector in closest_k:
        #             vector = all_rows[vector]
        #             vector = np.array([float(vec) for vec in vector],dtype = float)
        #             full_vectors.append(vector)
        # final = sort_vectors_by_cosine_similarity(full_vectors,query)[0][:K]
        # for vector in final:
        #     vector = full_vectors[vector]
        # return final

    def batch_retrieve_K_closest_data_given_centroid(self, centroid: np.ndarray, query: np.ndarray, K: int):  # batch inspection
        file_to_inspect = self.assignments[str(centroid)]
        closest_k = []
        with open(file_to_inspect, "r") as csvfile:
            reader = csv.reader(csvfile)
            while True:
                batch = list(islice(reader, K))
                if not batch:
                    break
                closest_k = sort_vectors_by_cosine_similarity(np.concatenate((closest_k, batch)))[:K]
        return closest_k

    def full_retrieve_K_closest_data_given_centroid(self, centroid: np.ndarray, query: np.ndarray, K: int):  # full inspection
        file_to_inspect = self.assignments[str(centroid)]
        closest_k = []
        with open(file_to_inspect, "r") as csvfile:
            reader = csv.reader(csvfile)
            all_rows = list(reader)
            closest_k = sort_vectors_by_cosine_similarity(all_rows, query)[0][:K]
            for vector in closest_k:
                vector = all_rows[vector]
        return closest_k


test_vector = [[-0.4665489192145053, -2.4149580061941975, -0.04247832948671558, 0.7563412531550323]]
IV = IVFile_optimized(1000, 100000)
IV.build_index("random_vectors.csv", 1)
vectors = IV.retrieve_k_closest(test_vector, 3)
print(vectors)
for vector in vectors:
    print(cosine_similarity(test_vector, vector.reshape(1, -1)))
