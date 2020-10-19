import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single, build_desc, get_data, plot, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb
from scipy.spatial.distance import cdist

def new_pt(data, Refs, d_tol=1e-1):
    (X, ele) = data
    X = X/np.linalg.norm(X)
    for Ref in Refs:
        (X1, ele1) = Ref
        if ele1 == ele:
            X1 = X1/np.linalg.norm(X1)
            d = X@X1.T
            if 1-d**2 < d_tol:
                return False
    return True
        
#N_start, N_step, N_max, zeta, ncpu, fac = 4, 1, 2554, 2, 10, 1.2
N_start, N_step, N_max, zeta, ncpu, fac = 4, 1, 50, 2, 10, 1.2

des = build_desc("SO3", lmax=3, nmax=3, rcut=4.5)
kernel = RBF_mb(para=[1, 0.5], zeta=zeta, ncpu=ncpu)
model = gpr(kernel=kernel, descriptor=des, noise_e=[5e-3, 2e-3, 5e-3], f_coef=20)
db_file = sys.argv[1]

db_ids = range(N_start)
pool_ids = range(0, N_max, N_step)  

train_data = get_data(db_file, des, N_force=5, lists=db_ids, select=True, ncpu=ncpu)

model.fit(train_data)
E, E1, F, F1 = model.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
print("\n")


for id in pool_ids:
    ids = range(id, id+N_step)
    test_data = get_data(db_file, des, lists=ids)
    E, E1, E_std, F, F1, F_std = model.validate_data(test_data, return_std=True)
    l1 = metric_single(E, E1, "Test Energy") 
    l2 = metric_single(F, F1, "Test Forces") 

    pts_to_add = {"energy": [], "force": [], "db": []}
    F_count = 0
    for i in range(len(E)):
        Num = len(test_data['energy'][i][0])
        diff_E = E[i] - E1[i]
        _F = F[F_count:F_count+Num*3]
        _F1 = F1[F_count:F_count+Num*3]
        _std = F_std[F_count:F_count+Num*3]
        print("{:4d} E: {:8.3f} -> {:8.3f} {:8.3f}  Var-E: {:8.3f} F: {:8.3f}".format(\
                id+i, E[i], E1[i], diff_E, E_std[i], np.max(_std)))

        energy_in = False
        force_in = []

        if E_std[i] > 2*model.noise_e: 
            pts_to_add["energy"].append(test_data['energy'][i])
            energy_in = True

        _std = _std.reshape([Num, 3])
        xs_added = []
        for f_id in range(Num):
            if np.max(_std[f_id]) > fac*model.noise_f: 
                X = test_data['energy'][i][0][f_id]
                ele = test_data['energy'][i][2][f_id]
                if len(xs_added) == 0:
                    pts_to_add["force"].append(test_data["force"][int(F_count/3)+f_id])
                    force_in.append(f_id)
                    xs_added.append((X, ele))
                else:
                    if new_pt((X, ele), xs_added):
                        pts_to_add["force"].append(test_data["force"][int(F_count/3)+f_id])
                        force_in.append(f_id)
                        xs_added.append((X, ele))
                if len(xs_added) == 6:
                    break

        if energy_in or len(force_in)>0:
            (struc, energy, force, _, _) = test_data["db"][i]
            pts_to_add["db"].append((struc, energy, force, energy_in, force_in))

        F_count += Num*3

    # update the database
    if len(pts_to_add["db"])>0:
        model.set_train_pts(pts_to_add, mode='a+')
        strs = "========{:d} structures| {:d} energies |{:d} forces were added".format(\
        len(pts_to_add["db"]), len(pts_to_add["energy"]), len(pts_to_add["force"]))
        print(strs)

        model.fit()
        train_E, train_E1, train_F, train_F1 = model.validate_data()
        l1 = metric_single(train_E, train_E1, "Train Energy") 
        l2 = metric_single(train_F, train_F1, "Train Forces") 
        print(model)

model.save("models/test.json", "models/test.db")

