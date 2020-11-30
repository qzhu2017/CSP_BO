import numpy as np

C1 = np.load("C_loop.npy")
C2 = np.load("C_nloop.npy")

shp = C1.shape

for i in range(shp[0]):
    for j in range(shp[1]):
        print(C1[i,j], C2[i,j])

truth = np.allclose(C1, C2)
print(truth)
