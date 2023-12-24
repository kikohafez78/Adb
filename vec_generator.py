import numpy as np
import csv

outfile = 'random_vectors.csv'
num_vectors = 1000000
vector_dim = 70
X = 0
with open(outfile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(num_vectors):
        print(X)
        vector = np.random.normal(size=(1,vector_dim))[0]
        writer.writerow(vector)
        X += 1