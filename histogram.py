import numpy as np
import matplotlib.pyplot as plt

random_sg194 = [25, 4, 4, 11, 6, 3, 11, 4, 12, 5, 4, 11, 1, 18, 1, 7, 7, 1, 5, 4, 16, 3, 13, 4, 3, 2, 1, 5, 3, 3, 12, 2, 3, 3, 4, 1, 15, 2, 16, 2, 1, 20, 10, 1, 1, 2, 8, 4, 3, 8, 12, 5, 2, 1, 4, 2, 13, 10, 7, 1, 4, 16, 7, 4, 1, 5, 3, 7, 7, 9, 1, 3, 4, 7, 13, 15, 3, 1, 3, 7, 18, 3, 10, 11, 4, 7, 3, 6, 6, 2, 8, 13, 7, 2, 2, 3, 8, 5, 2, 5, 5, 17, 15, 4, 18, 9, 11, 7, 7, 6, 2, 1, 6, 3, 3, 5, 2, 8, 1, 16, 4, 12, 7, 14, 1, 3, 2, 1, 1, 5, 2, 20, 4, 6, 3, 2, 12, 7, 3, 8, 8, 3, 30, 7, 5,  4, 27, 2, 4, 9, 8, 3, 5, 2, 1, 10, 3, 1, 2, 1, 11, 1, 5, 1, 6, 7, 3, 27, 2, 2, 19, 1, 8, 2, 9, 3, 7, 1, 8, 2, 18, 3, 3, 5, 9, 1, 5, 5, 6, 9, 1, 5, 4, 5, 9, 4, 4, 4, 2, 19]

random_sg = [5, 2, 2, 7, 1, 11, 21, 1, 4, 1, 9, 28, 2, 7, 5, 11, 18, 2, 2, 6, 2, 5, 4, 3, 10, 1, 2, 14, 9, 1, 10, 4, 1, 30, 22, 10, 33, 14, 1, 14, 6, 9, 5, 1, 5, 5, 2, 1, 28, 5, 7, 24, 16, 36, 7, 5, 34, 17, 13, 7, 15, 1, 21, 6, 4, 12, 6, 39, 6, 27, 1, 3, 2, 60, 62, 4, 14, 2, 3, 10, 3, 20, 4, 10, 1, 23, 23, 9, 5, 12, 15, 2, 9, 2, 22, 8, 9, 11, 23, 12, 3, 2, 6, 12, 2, 22, 21, 27, 4, 13, 19, 1, 1, 10, 4, 20, 14, 20, 12, 6, 3, 10, 2, 11, 5, 7, 45, 1, 3, 20, 21, 4, 7, 1, 7, 9, 5, 9, 5, 41, 8, 10, 7, 7, 27, 11, 20, 14, 26, 7, 2, 10, 5, 8, 19, 12, 10, 39, 27, 6, 3, 3, 18, 4, 1, 13, 25, 6, 44, 2, 6, 2, 31, 13, 2, 43, 9, 8, 14, 7, 30, 3, 7, 6, 4, 27, 70, 8, 3, 7, 3, 9, 28, 28, 1, 9, 14, 1, 25, 38]
random_sg = np.array(random_sg) - 1

# For every member of the bo must be substracted by one.
bo = [14, 11, 3, 2, 12, 9, 6, 15, 11, 11, 25, 21, 36, 8, 6, 1, 16, 1, 28, 8, 3, 27, 4, 2, 26, 8, 12, 17, 24, 9, 8, 9, 18, 2, 1, 4, 5, 8, 1, 27, 3, 3, 9, 6, 4, 19, 4, 52, 34, 10, 10, 5, 8, 4, 8, 10, 2, 27, 2, 5, 1, 6, 2, 1, 10, 17, 1, 13, 5, 4, 1, 2, 20, 16, 6, 3, 12, 21, 5, 1, 7, 23, 31, 6, 44, 3, 29, 1, 25, 5, 6, 9, 27, 19, 7, 7, 7, 7, 7, 11, 8, 6, 27, 18, 5, 6, 3, 12, 21, 32, 8, 14, 15, 3, 1, 3, 23, 4, 13, 1, 14, 10, 21, 13, 4, 4, 39, 7, 2, 20, 6, 2, 2, 4, 14, 17, 2, 17, 8, 10, 9, 11, 5, 13, 33, 3, 6, 40, 1, 25, 20, 3, 8, 2, 2, 5, 13, 17, 26, 19, 1, 8, 4, 2, 26, 10, 18, 40, 14, 39, 13, 9, 8, 1, 1, 17, 13, 6, 36, 13, 8, 7, 4, 2, 7, 6, 23, 2, 10, 18, 15, 8, 7, 1, 8, 4, 5, 1, 11, 19]
bo = np.array(bo) - 1



print("The number of PyXtal random_crystal calls for random CSP is ", sum(random_sg))
print("The number of PyXtal random_crystal calls for BO CSP is ", sum(bo)*10+10)

plt.hist(bo, label="Bayesian Optimization", bins=40)
plt.hist(random_sg, histtype="step", label="Random", bins=60)
plt.ylabel("Frequency")
plt.xlabel("The number of optimization steps")
plt.legend()
plt.title("Crystal Structure Prediction Schemes: BO vs Random")
plt.show()
