import csv
from heapq import nlargest
from itertools import islice

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import normalize


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
            return np.dot(query, neighbor.T) / (np.linalg.norm(query) * np.linalg.norm(neighbor))

    def vectorized_distance_(self, x, ys):
        return [self.similarity(x, y) for y in ys]

    def build_index(self, file_name_s, folder_path: str, n: int):
        if n > 1:
            for file_name in file_name_s:
                with open(file_name, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    while True:
                        batch = list(islice(reader, self.batch_size))
                        batch = normalize(batch)
                        if not batch:
                            break
                        self.Kmeans.partial_fit(batch)
            self.centroids = self.Kmeans.cluster_centers_
            X = 0
            for centroid in self.centroids:
                with open(f"./{folder_path}/data/data{X}.csv", "a", newline="") as csvfile:
                    pass
                self.clusters[str(centroid)] = f"./{folder_path}/data/data{X}.csv"
                X += 1
            self.final_pass(file_name_s)
            return self.Kmeans.cluster_centers_
        else:
            with open(f"./{folder_path}/{file_name_s}", "r") as csvfile:
                reader = csv.reader(csvfile)
                while True:
                    batch = list(islice(reader, self.batch_size))
                    if not batch:
                        break
                    batch = normalize(batch)
                    self.Kmeans.partial_fit(batch)
            self.centroids = self.Kmeans.cluster_centers_
            X = 0
            for centroid in self.centroids:
                with open(f"./{folder_path}/data/data{X}.csv", "a", newline="") as csvfile:
                    pass
                self.clusters[str(centroid)] = f"./{folder_path}/data/data{X}.csv"
                X += 1
            self.final_pass(f"./{folder_path}/{file_name_s}", n)
            with open(f"./{folder_path}/clusters.csv", "a", newline="") as csvfile:
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
                        values = normalize(batch)
                        labels = np.argmax(self.similarity([values], self.centroids), axis=1)
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
                    norm = normalize([vector])[0]
                    labels = sort_vectors_by_cosine_similarity(self.centroids, np.asarray(norm).reshape((1, -1)))[0][0]
                    file_to_insert = self.clusters[str(self.centroids[labels])]
                    with open(file_to_insert, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(vector)

    def retrieve_k_closest(self, folder_path: str, query: np.ndarray, K: int):  # retrieval function
        query = normalize([query])
        self.assignments = {}
        X = 0
        self.centroids = []
        with open(f"./{folder_path}/clusters.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for centroid in reader:
                if not centroid:
                    break
                self.assignments[str(centroid)] = f"./{folder_path}/data/data{X}.csv"
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

    # def batch_retrieve_K_closest_data_given_centroid(self,centroid:np.ndarray,query:np.ndarray,K: int): #batch inspection
    #     file_to_inspect = self.assignments[str(centroid)]
    #     closest_k = []
    #     with open(file_to_inspect,'r') as csvfile:
    #         reader = csv.reader(csvfile)
    #         while True:
    #             batch = list(islice(reader,K))
    #             if not batch:
    #                 break
    #             closest_k = sort_vectors_by_cosine_similarity(np.concatenate((closest_k,batch)))[:K]
    #     return closest_k

    # def full_retrieve_K_closest_data_given_centroid(self,centroid:np.ndarray,query:np.ndarray,K: int): #full inspection
    #     file_to_inspect = self.assignments[str(centroid)]
    #     closest_k = []
    #     with open(file_to_inspect,'r') as csvfile:
    #         reader = csv.reader(csvfile)
    #         all_rows = list(reader)
    #         closest_k = sort_vectors_by_cosine_similarity(all_rows,query)[0][:K]
    #         for vector in closest_k:
    #             vector = all_rows[vector]
    #     return closest_k


test_vector = [
    1.356549859046936035e-01,
    7.601705193519592285e-01,
    7.894873619079589844e-01,
    5.459748506546020508e-01,
    6.158747673034667969e-01,
    8.827945590019226074e-01,
    8.573454618453979492e-02,
    4.456452131271362305e-01,
    2.457318902015686035e-01,
    3.362176418304443359e-01,
    7.569709420204162598e-01,
    1.734935045242309570e-01,
    1.175599098205566406e-01,
    9.990936517715454102e-01,
    3.962355852127075195e-01,
    1.498378515243530273e-01,
    1.471777558326721191e-01,
    9.817693233489990234e-01,
    9.413446784019470215e-01,
    8.490877747535705566e-01,
    7.055155038833618164e-01,
    7.967842817306518555e-01,
    6.494476795196533203e-01,
    9.378776550292968750e-01,
    2.359203696250915527e-01,
    1.019475460052490234e-01,
    9.782267808914184570e-01,
    1.652941107749938965e-01,
    3.455436229705810547e-02,
    4.011875391006469727e-01,
    9.487032890319824219e-03,
    8.921121358871459961e-01,
    9.214249253273010254e-01,
    8.608156442642211914e-01,
    2.790191173553466797e-01,
    3.017135858535766602e-01,
    2.334207296371459961e-02,
    5.363611578941345215e-01,
    8.772153854370117188e-01,
    1.413692235946655273e-01,
    2.500416636466979980e-01,
    6.492682099342346191e-01,
    6.941686272621154785e-01,
    6.659759879112243652e-01,
    3.610870242118835449e-01,
    4.449421763420104980e-01,
    7.560720443725585938e-01,
    3.450798988342285156e-02,
    4.606603980064392090e-01,
    4.232681989669799805e-01,
    7.925334572792053223e-01,
    1.159440279006958008e-01,
    9.027293920516967773e-01,
    5.301722884178161621e-01,
    6.587631106376647949e-01,
    6.501644849777221680e-01,
    6.776183247566223145e-01,
    2.136875987052917480e-01,
    7.257928848266601562e-01,
    2.976732254028320312e-01,
    9.329499006271362305e-01,
    3.651674389839172363e-01,
    1.487530469894409180e-01,
    6.382648348808288574e-01,
    1.659295558929443359e-01,
    9.258993864059448242e-01,
    3.332807421684265137e-01,
    4.010143280029296875e-01,
    7.749438285827636719e-03,
    7.619243860244750977e-01,
]
IV = IVFile_optimized(10000, 1000000)
IV.build_index("saved_db_1m.csv", "1m", 1)
# vectors = IV.retrieve_k_closest("100k",test_vector,3)
# for vector in vectors:
#     print(cosine_similarity([test_vector],[vector]))
