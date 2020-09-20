import sys
import numpy as np
from ase.db import connect
from sklearn.metrics import mean_absolute_error as mae, r2_score as r2, mean_squared_error as mse
from pyxtal_ff.descriptors.SO3 import SO3
from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum as SO4
from cspbo.gaussianprocess import GaussianProcess as gpr

def decompose(descriptors, ids):
    X = []
    for i, id in enumerate(ids[:-1]):
        X.append(descriptors[ids[i]:ids[i+1],:])
    return X

nmax, lmax, rcut = 4, 4, 5.0
#des = SO3(nmax=nmax, lmax=lmax, rcut=rcut, derivative=False)
des = SO4(lmax=lmax, rcut=rcut, derivative=False)
train_Y = []
test_Y = []

ids = [0]
descriptors = [] 
with connect(sys.argv[1]) as db:
    for row in db.select():
        s = db.get_atoms(id=row.id)
        _des = des.calculate(s)['x']
        print(s)
        if len(ids) == 1:
            descriptors = _des #vstack
        else:
            descriptors = np.vstack((descriptors, _des))
        ids.append(ids[-1]+len(_des))
        train_Y.append(row.ff_energy*len(s))
        count = len(train_Y)
        if count % 100 == 50:
            print("processing train", str(count))
            break

m1=np.max(descriptors, axis=0)
m2=np.min(descriptors, axis=0)
descriptors = (descriptors-m2)/(m1-m2)
train_X = decompose(descriptors, ids)

ids = [0]
descriptors = [] 
with connect(sys.argv[2]) as db:
    for row in db.select():
        s = db.get_atoms(id=row.id)
        _des = des.calculate(s)['x']
        print(s)
        if len(ids) == 1:
            descriptors = _des #vstack
        else:
            descriptors = np.vstack((descriptors, _des))
        ids.append(ids[-1]+len(_des))
        test_Y.append(row.ff_energy*len(s))
        count = len(test_Y)
        if count == 20:
            print("processing test", str(count))
            break

descriptors = (descriptors-m2)/(m1-m2)
test_X = decompose(descriptors, ids)

# GP model
model = gpr()
model.fit((train_X, train_Y))
pred = model.predict(train_X)
tmp = np.array(train_Y)
print("Train MAE : {:6.3f}".format(mae(pred, tmp)))
print("Train RMSE: {:6.3f}".format(np.sqrt(mse(pred, tmp))))
print("Train R2  : {:6.3f}".format(r2(pred, tmp)))

pred = model.predict(test_X)
tmp = np.array(test_Y)
print("Test  MAE : {:6.3f}".format(mae(pred, tmp)))
print("Test  RMSE: {:6.3f}".format(np.sqrt(mse(pred, tmp))))
print("Test  R2  : {:6.3f}".format(r2(pred, tmp)))

