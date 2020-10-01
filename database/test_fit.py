# Crystal Structure Prediction with Random Search
import sys
import numpy as np
from time import time
from cspbo.descriptors.rdf import RDF
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
import sklearn.gaussian_process as gp
from pymatgen.io.ase import AseAtomsAdaptor
from warnings import catch_warnings, simplefilter
from sklearn.preprocessing import StandardScaler  
from ase.db import connect
from sklearn.metrics import mean_absolute_error as mae, r2_score as r2, mean_squared_error as mse
scaler = StandardScaler()  

Rmax = 10.0
train_X = []
train_Y = []
test_X = []
test_Y = []

with connect(sys.argv[1]) as db:
    for row in db.select():
        s = db.get_atoms(id=row.id)
        pmg_struc = AseAtomsAdaptor().get_structure(s)
        des = RDF(pmg_struc, R_max=Rmax).RDF[1]
        train_X.append(des)
        train_Y.append(row.ff_energy)
        count = len(train_X)
        if count % 100 == 0:
            print("processing train", str(len(train_X)))

with connect(sys.argv[2]) as db:
    N = len(db)
    cut_N = int(N*0.6)
    for row in db.select():
        s = db.get_atoms(id=row.id)
        pmg_struc = AseAtomsAdaptor().get_structure(s)
        des = RDF(pmg_struc, R_max=Rmax).RDF[1]
        test_X.append(des)
        test_Y.append(row.ff_energy)
        count = len(test_X)
        if count % 100 == 0:
            print("processing test", str(len(test_X)))

for kernel in (1.0*gp.kernels.Matern(), 1.0*gp.kernels.RBF(length_scale=10), 1.0*gp.kernels.DotProduct()+gp.kernels.WhiteKernel()):
    model = gpr(kernel=kernel,  n_restarts_optimizer=10)
    model.fit(train_X, train_Y)
    print(model.kernel_)
    #pred = model.predict(train_X)
    #tmp = np.array(train_Y)
    #print("Train MAE : {:6.3f}".format(mae(pred, tmp)))
    #print("Train RMSE: {:6.3f}".format(np.sqrt(mse(pred, tmp))))
    #print("Train R2  : {:6.3f}".format(r2(pred, tmp)))
    
    pred = model.predict(test_X)
    tmp = np.array(test_Y)
    print("Test  MAE : {:6.3f}".format(mae(pred, tmp)))
    print("Test  RMSE: {:6.3f}".format(np.sqrt(mse(pred, tmp))))
    print("Test  R2  : {:6.3f}".format(r2(pred, tmp)))

print("Apply scaling")
tmp1 = np.array(train_X)
tmp2 = np.array(test_X)
scaler.fit(tmp1)
train_X = scaler.transform(tmp1) 
test_X = scaler.transform(tmp2) 

for kernel in (1.0*gp.kernels.Matern(), 1.0*gp.kernels.RBF(length_scale=1), 1.0*gp.kernels.DotProduct()+gp.kernels.WhiteKernel()):
    model = gpr(kernel=kernel,  n_restarts_optimizer=10)
    model.fit(train_X, train_Y)
    print(model.kernel_)
    #pred = model.predict(train_X)
    #tmp = np.array(train_Y)
    #print("Train MAE : {:6.3f}".format(mae(pred, tmp)))
    #print("Train RMSE: {:6.3f}".format(np.sqrt(mse(pred, tmp))))
    #print("Train R2  : {:6.3f}".format(r2(pred, tmp)))
    
    pred = model.predict(test_X)
    tmp = np.array(test_Y)
    print("Test  MAE : {:6.3f}".format(mae(pred, tmp)))
    print("Test  RMSE: {:6.3f}".format(np.sqrt(mse(pred, tmp))))
    print("Test  R2  : {:6.3f}".format(r2(pred, tmp)))
    
