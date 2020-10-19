from copy import deepcopy
import numpy as np
import sys
from cspbo.utilities import metric_single, build_desc, get_data, plot, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb
from time import time

N1, ncpu = 500, 10
#db_file = "database/Si_5352_DFT.db"
#db_file = "database/Si_244_DFT.db"
db_file = sys.argv[1]
des = build_desc("SO3", lmax=4, nmax=4, rcut=5.0)
N_train = int(N1*0.8)

train_data = get_data(db_file, des, N_force=30, lists=range(0, N_train), ncpu=ncpu)
test_data = get_data(db_file, des, lists=range(N_train, N1), ncpu=ncpu)

t0 = time()
print("------------------Tranining Energy only--------------------------")
train_data1 = deepcopy(train_data)
train_data1["force"] = []
test_data1 = deepcopy(test_data)
test_data1["force"] = []


kernel = RBF_mb(para=[5.0, 0.2], ncpu=ncpu, zeta=1)
model = gpr(kernel=kernel, descriptor=des, noise_e=[3e-2, 3e-3, 1e-1], f_coef=10)
model.fit(train_data1)

train_E, train_E1, F, F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 

test_E, test_E1, F, F1 = model.validate_data(test_data1)
l2 = metric_single(test_E, test_E1, "Test  Energy") 

plot((train_E, test_E), (train_E1, test_E1), (l1, l2), "E.png")
#l2 = metric_single(F, F1, "Test  Forces") 

#print("------------------Tranining Force only--------------------------")
#train_data1 = deepcopy(train_data)
#train_data1["energy"] = []
#
#kernel = RBF_mb(para=[5.0, 0.2], ncpu=ncpu)
#model = gpr(kernel=kernel, descriptor=des, noise_e=[2e-2, 5e-3, 1e-1], f_coef=30)
#model.fit(train_data1)
#
#E, E1, F, F1 = model.validate_data()
#metric_single(F, F1, "Train Forces") 
#
#E, E1, F, F1 = model.validate_data(test_data)
#l1 = metric_single(E, E1, "Test  Energy") 
#l2 = metric_single(F, F1, "Test  Forces") 



#print("------------------Tranining both E and F--------------------------")
#train_data1 = deepcopy(train_data)
#
#kernel = RBF_mb(para=[5.0, 0.2], ncpu=ncpu)
#model = gpr(kernel=kernel, descriptor=des, noise_e=[2e-2, 2e-2, 1e-1], f_coef=30)
#model.fit(train_data1)
#
#E, E1, F, F1 = model.validate_data()
#metric_single(E, E1, "Train Energy") 
#metric_single(F, F1, "Train Forces") 
#
#E, E1, F, F1 = model.validate_data(test_data)
#l1 = metric_single(E, E1, "Test  Energy") 
#l2 = metric_single(F, F1, "Test  Forces") 

print('elapsed', time()-t0)


