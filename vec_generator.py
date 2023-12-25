import numpy as np
import csv
rng = np.random.default_rng(50)
with open("./5m/saved_db_5m.csv",'a',newline='') as csvfile:
    pass
vectors = rng.random((5000000, 70), dtype=np.float32)
np.savetxt("./5m/saved_db_5m.csv", vectors,delimiter=",")
del vectors
# vectors = rng.random((5000000, 70), dtype=np.float32)
# np.save("data/database_5M_2.npy", vectors)
# del vectors
# vectors = rng.random((5000000, 70), dtype=np.float32)
# np.save("data/database_5M_3.npy", vectors)
# del vectors
# vectors = rng.random((5000000, 70), dtype=np.float32)
# np.save("data/database_5M_4.npy", vectors)
# del vectors