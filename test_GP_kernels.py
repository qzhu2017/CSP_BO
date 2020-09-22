import sys
from cspbo.descriptors.rdf import RDF
import numpy as np
from ase.db import connect
from sklearn.metrics import mean_absolute_error as mae, r2_score as r2, mean_squared_error as mse
from pyxtal_ff.descriptors.SO3 import SO3
from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum as SO4
from cspbo.gaussianprocess import GaussianProcess as gpr, RBF, Dot, Combo, RBF_2b
from pymatgen.io.ase import AseAtomsAdaptor

def decompose(descriptors, ids):
    X = []
    for i, id in enumerate(ids[:-1]):
        X.append(descriptors[ids[i]:ids[i+1],:])
    return X

#f_train, N_train, f_test, N_test = "database/Si_200_DFT.db", 200, "database/Si_44_DFT.db", 44
f_train, N_train, f_test, N_test = "database/Si-Tersoff-8atoms-1000.db", 250, "database/Si-Tersoff-8atoms-500.db", 100
nmax, lmax, rcut = 4, 3, 4.9
des1 = SO3(nmax=nmax, lmax=lmax, rcut=rcut, derivative=False)
des2 = SO4(lmax=lmax, rcut=rcut, derivative=False)

for des in [des1, des2]:
    train_Y = []
    test_Y = []
    
    ids = [0]
    descriptors = [] 
    descriptors2 = [] 
    with connect(f_train) as db:
        for row in db.select():
            s = db.get_atoms(id=row.id)
            pmg_struc = AseAtomsAdaptor().get_structure(s)
            _des1 = RDF(pmg_struc, R_max=rcut*2).RDF[1]
            descriptors2.append(_des1)

            _des = des.calculate(s)['x']
            if len(ids) == 1:
                descriptors = _des #vstack
            else:
                descriptors = np.vstack((descriptors, _des))
            ids.append(ids[-1]+len(_des))
            #train_Y.append(row.ff_energy*len(s))

            train_Y.append(row.ff_energy)
            count = len(train_Y)
            if count % 20 == 0:
                print("processing train", str(count))
            if count == N_train:
                break
    
    m1=1 #2*np.max(descriptors)
    train_X = (decompose(descriptors/m1, ids), descriptors2)
    #train_X = ([None]*len(descriptors2), descriptors2)
    
    ids = [0]
    descriptors = [] 
    descriptors2 = [] 
    with connect(f_test) as db:
        for row in db.select():
            s = db.get_atoms(id=row.id)
            pmg_struc = AseAtomsAdaptor().get_structure(s)
            _des1 = RDF(pmg_struc, R_max=rcut*2).RDF[1]
            descriptors2.append(_des1) 

            _des = des.calculate(s)['x']
            if len(ids) == 1:
                descriptors = _des #vstack
            else:
                descriptors = np.vstack((descriptors, _des))
            ids.append(ids[-1]+len(_des))
            #test_Y.append(row.ff_energy*len(s))

            test_Y.append(row.ff_energy)
            count = len(test_Y)
            if count % 10 == 0:
                print("processing test", str(count))
            if count == N_test:
                break
    
    test_X = (decompose(descriptors/m1, ids), descriptors2)
    #test_X = ([None]*len(descriptors2), descriptors2)
    
    # GP model
    #for kernel in [RBF_2b(para=[2.82, 7.7])]:
    for kernel in [RBF_2b(para=[2.82, 7.7]), 
                   #Combo(para=[2.82, 7.7, 0.2], coef=0.1),
                   #Combo(para=[2.82, 7.7, 0.5], coef=0.2), 
                   #Combo(para=[2.82, 7.7, 0.8], coef=0.5), 
                   Combo(para=[2.82, 7.7, 1.0], coef=1.0), 
                   #RBF(para=[2.82, 7.7]), 
                   Dot()]:
        model = gpr(kernel=kernel)
        model.fit((train_X, train_Y))
        N = 1 #np.array([len(s) for s in train_X[0]])
        pred = model.predict(train_X)/N
        tmp = np.array(train_Y)/N
        print("Train MAE : {:8.4f}".format(mae(pred, tmp)))
        print("Train RMSE: {:8.4f}".format(np.sqrt(mse(pred, tmp))))
        print("Train R2  : {:8.4f}".format(r2(pred, tmp)))
        
        #N = np.array([len(s) for s in test_X[0]])
        pred = model.predict(test_X)/N
        tmp = np.array(test_Y)/N
        print("Test  MAE : {:8.4f}".format(mae(pred, tmp)))
        print("Test  RMSE: {:8.4f}".format(np.sqrt(mse(pred, tmp))))
        print("Test  R2  : {:8.4f}".format(r2(pred, tmp)))


