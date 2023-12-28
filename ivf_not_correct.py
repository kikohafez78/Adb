import itertools
import pickle as pi

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import os
from clean_slate import HNSW, IVFile, cosine_similarity, itemgetter, nlargest
import glob as gl

def sort_vectors_by_cosine_similarity(vectors, reference_vector):
    # Calculate cosine similarities
    cos_similarities = cosine_similarity(reference_vector, vectors)
    # Sort indices by cosine similarity
    sorted_indices = np.argsort(cos_similarities)
    # sorted_indices[0] = np.flip(sorted_indices, axis=1)
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
    "saved_db_15m": 15000000,
    "saved_db_20m": 20000000,
    "saved_db": 0
}
batches = {
    "saved_db_100k": 100000,
    "saved_db_1m": 1000000,
    "saved_db_5m": 5000000,
    "saved_db_10m": 10000000,
    "saved_db_15m": 15000000,
    "saved_db_20m": 20000000,
    "saved_db": 0
}
n_files = {
    "saved_db_100k": 1,
    "saved_db_1m": 10,
    "saved_db_5m": 50,
    "saved_db_10m": 100,
    "saved_db_15m": 150,
    "saved_db_20m": 200,
    "saved_db": 0
}


class VecDB(object):
    def __init__(self, file_path: str = None, new_db=True):
        self.folder = file_path
        if file_path is None:
                self.folder = "saved_db"
        self.partitions = int(np.ceil(sizes[self.folder] / np.sqrt(sizes[self.folder])) * 3) if self.folder != "saved_db" else 0
        self.batch_size = 100000
        self.no_of_files = n_files[self.folder]
        self.kmeans = MiniBatchKMeans(n_clusters=self.partitions)

    def insert_records(self, vectors: list):
        vectors = [vector["embed"] for vector in vectors]
        if os.path.exists(f"./{self.folder}/clusters.pickle"):
            os.remove(f"./{self.folder}/clusters.pickle")
            for partition in range(self.partitions-1):
                os.remove(f"./{self.folder}/file{partition}.pickle")
        batch_number = len(gl.glob("batch*.npy")) - 1
        self.batch_size = 100000
        num_batches = int(np.ceil(len(vectors) / self.batch_size))
        self.no_of_files = len(gl.glob(f"./{self.folder}/batch*.npy"))
        number_of_original_vectors = self.no_of_files*self.batch_size
        for i in range(num_batches):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(vectors)) 
            batch = vectors[start_index:end_index]
            filename = f"./{self.folder}/batch{i + 1 + batch_number}.npy"
            np.save(filename, batch)
            print(f"Created batch file {filename} containing {len(batch)} vectors")
        self.partitions = int(np.ceil((len(vectors)+number_of_original_vectors) / np.sqrt((len(vectors)+number_of_original_vectors))) * 3)
        self.no_of_files = len(gl.glob(f"./{self.folder}/batch*.npy"))
        self.kmeans = MiniBatchKMeans(n_clusters=self.partitions)
        print(self.no_of_files)
        self.build_index()
        
        
        
        
    def build_index(self):
        batch = 0
        for i in range(self.no_of_files):
            data = np.load(f"./{self.folder}/batch{batch}.npy")
            data = normalize(data)
            # ===================================================
            self.kmeans.partial_fit(data)
            # ===================================================
            batch += 1
        self.clusters = self.kmeans.cluster_centers_
        self.assignments = self.kmeans.labels_
        print(len(self.kmeans.labels_))
        X = 0
        for cluster in self.clusters:
            with open(f"./{self.folder}/file{X}.pickle", "ab") as file:
                pass
            X += 1
        X = 0
        batch = 0
        clusters = {x: [] for x in range(len(self.clusters))}
        ids = 0
        for i in range(self.no_of_files):
            data = np.load(f"./{self.folder}/batch{batch}.npy")
            # ===================================================
            for vector in data:
                vector = normalize([vector])
                get_cluster = sort_vectors_by_cosine_similarity(self.clusters,vector)[0][0]
                vector = np.append(vector, ids)
                # print(vector)
                clusters[get_cluster].append(vector)
                print(ids)
                X += 1
                ids += 1
            # X = 0
            # ===================================================
            batch += 1
        X = 0
        for cluster in clusters:
            with open(f"./{self.folder}/file{X}.pickle", "ab") as file:
                pi.dump(clusters[cluster], file)
            X += 1
        X = 0
        assignments = {}
        for cluster in self.clusters:
            assignments[str(X)] = (f"./{self.folder}/file{X}.pickle", cluster)
            X += 1
        with open(f"./{self.folder}/clusters.pickle", "ab") as file:
            pi.dump(assignments, file)
        self.clusters = None
        self.assignments = None
        return

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):  # <===
        # vector = normalize([vector])
        # vector = vector.reshape(1, -1)
        vector = normalize(vector.reshape(1, -1))
        with open(f"./{self.folder}/clusters.pickle", "rb") as file:
            self.clusters = pi.load(file)
        centroids = [self.clusters[str(centroid)][1] for centroid in range(self.partitions)]
        print(self.partitions)
        files = [self.clusters[str(centroid)][0] for centroid in range(self.partitions)]
        vectors = sort_vectors_by_cosine_similarity(centroids, vector)[0][:50]
        files_to_inspect = [files[x] for x in vectors]
        full_vectors = []
        for file in files_to_inspect:
            with open(file, "rb") as file:
                data = np.array(pi.load(file))  # Data are with dimensions 70 + 1
                data = np.append(normalize(data[:, :70]), data[:, 70:], axis=1)
                vectors = sort_vectors_by_cosine_similarity(data[:, :70], vector)[0][: K]
                vectors = [data[vec] for vec in vectors]
                full_vectors.extend(vectors)
                # print(full_vectors)
        full_vectors = np.array([vector for vector in full_vectors])
        # print(full_vectors.shape)
        final_vectors = sort_vectors_by_cosine_similarity(full_vectors[:, :70], vector.reshape(1, -1))[0][: K]
        final_vectors = [full_vectors[vec] for vec in final_vectors]
        return final_vectors

    def retrive(self, vector: np.ndarray, K: int):  # <===
        vectors = self.get_closest_k_neighbors(vector, K)
        # print(vectors)
        vec_no_ids = [int(vector[-1]) for vector in vectors]
        return vec_no_ids


# # # Example usage:
# vecDB = VecDB("saved_db_100k")  # Use the appropriate file path
# # vector = np.load("./saved_db_100k/batch0.npy")[0]  # dimension 70
# print(vecDB.build_index())
# # # print(pd.read_pickle("./saved_db_100k/clusters.pickle"))
# # vectors = vecDB.get_closest_k_neighbors(vector, 3)
# ids = vecDB.retrive(vector, 3)
# print(ids)
# # vec_no_ids = [vector[:70] for vector in vectors]
# # for vector_no_id in vec_no_ids:
# #     print(cosine_similarity([vector_no_id], [vector]))
# rng = np.random.default_rng(20)
# vectors = rng.random((10**4, 70), dtype=np.float32)
# records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(vectors)]
# # vecDB = VecDB()
# # vecDB.insert_records(records_dict)
# test_vector = np.array([[0.5304362  ,0.1494149  ,0.4865445  ,0.13678914 ,0.24513423 ,0.24892092
#   ,0.02200389 ,0.38282466 ,0.7204832  ,0.649079   ,0.77705085 ,0.83756375
#   ,0.06120175 ,0.7760319  ,0.11445612 ,0.33951557 ,0.08761412 ,0.14856869
#   ,0.4816984  ,0.45701933 ,0.35744345 ,0.4378643  ,0.42141086 ,0.57421756
#   ,0.6229626  ,0.3732692  ,0.75553846 ,0.633825   ,0.48633796 ,0.11464435
#   ,0.83315873 ,0.23309046 ,0.33111197 ,0.767241   ,0.8557654  ,0.98712426
#   ,0.8222418  ,0.80800104 ,0.7386268  ,0.8429656  ,0.5602805  ,0.79568267
#   ,0.24378216 ,0.4568413 , 0.7938885 , 0.73867065, 0.42071766 ,0.578455
#   ,0.36776322 ,0.4507355 , 0.03630698 ,0.2710244,  0.7429225 ,0.8646031
#   ,0.80520135 ,0.06865561, 0.15775746 ,0.81673443, 0.65432876, 0.881835
#   ,0.4614241 , 0.4235164 , 0.24339634, 0.8332293,  0.15596849, 0.3410167
#   ,0.06857234, 0.5197915 , 0.6052311 , 0.54920644]])
# vecDB = VecDB("saved_db_100k")
# db_ids,vectors = vecDB.retrive(test_vector,5)
# print(db_ids)
# print(cosine_similarity(test_vector,vectors[:][:70]))