import numpy as np

py_KEE = np.load("kernel_EE.npy")
py_KEF = np.load("kernel_EF.npy")
#py_KFE = np.load("kernel_FE.npy")
py_KFF = np.load("kernel_FF.npy")

C_KEE = np.load("../kernel_EE.npy")
C_KEF = np.load("../kernel_EF.npy")
#C_KFE = np.load("../kernel_FE.npy")
C_KFF = np.load("../kernel_FF.npy")

print(np.allclose(py_KEE, C_KEE))
print(np.allclose(py_KEF, C_KEF))
#print(np.allclose(py_KFE, C_KFE))
print(np.allclose(py_KFF, C_KFF))

#shp = py_KFF.shape
#for i in range(shp[0]):
#    for j in range(shp[1]):
#        print(py_KFF[i,j], C_KFF[i,j])
