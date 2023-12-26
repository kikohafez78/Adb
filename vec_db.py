import csv

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from IVF import IVFile
from non_normal_IVF import IVFile_optimized

batch = {
    "saved_db_100k.csv": 100000,
    "saved_db_1m.csv": 1000000,
    "saved_db_5m.csv": 5000000,
    "saved_db_10m.csv": 10000000,
    "saved_db_15m.csv": 15000000,
    "saved_db_20m.csv": 20000000,
}
file_size = {
    "saved_db_100k.csv": 1000,
    "saved_db_1m.csv": 10000,
    "saved_db_5m.csv": 50000,
    "saved_db_10m.csv": 100000,
    "saved_db_15m.csv": 150000,
    "saved_db_20m.csv": 200000,
}
folders = {
    "saved_db_100k.csv": "100k",
    "saved_db_1m.csv": "1m",
    "saved_db_5m.csv": "5m",
    "saved_db_10m.csv": "10m",
    "saved_db_15m.csv": "15m",
    "saved_db_20m.csv": "20m",
}


class VecDB(object):
    def __init__(self, file_name: str):
        self.file = file_name
        self.folder_path = folders[file_name]
        self.Ivf = IVFile_optimized(batch[file_name], file_size[file_name])

    def set_type(self, file_name: str):
        self.file = file_name
        self.folder_path = folders[file_name]
        self.Ivf = IVFile_optimized(batch[file_name], file_size[file_name])

    def build_index(self):
        self.Ivf.build_index(self.file, self.folder_path, 1)

    def insert_records(self, records: list):
        pass

    def retreive(self, query: np.ndarray, K: int):
        return self.Ivf.retrieve_k_closest(query, K)


vecdb = VecDB()
