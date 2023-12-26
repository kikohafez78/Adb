import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from sklearn.metrics.pairwise import cosine_similarity
from IVF import IVFile


def read_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    vectors = df.values  # Assuming all columns are vector components
    return vectors


# # Main script
# if __name__ == "__main__":
#     # Specify the file path to your CSV file
#     csv_file_path = "saved_db.csv"

#     # Read data from CSV
#     dataset = read_data_from_csv(csv_file_path)

#     # Parameters
#     K = 3
#     num_partitions = 16

#     # Create IVFile object
#     Iv = IVFile(num_partitions, dataset)

#     # Perform clustering
#     assignments = Iv.clustering()

#     # Test vector
#     test_vector = np.random.normal(size=(1, dataset.shape[1]))

#     # Get closest centroids
#     closest_centroids = Iv.get_closest_centroids(test_vector, K)
#     print(f"{K} closest centroids are: {closest_centroids}")

#     # Get closest neighbors
#     closest_neighbors = Iv.get_closest_k_neighbors(test_vector, K)
#     print(f"{K} closest neighbors are: {closest_neighbors}")
#     print(f"Cosine similarities: {cosine_similarity(closest_neighbors, test_vector)}")
