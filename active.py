import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single, build_desc, get_data, plot, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb

N_start, N_step, N_max, zeta = 50, 50, 200, 2

des = build_desc("SO3", lmax=4, nmax=4, rcut=4.5)
kernel = RBF_mb(para=[1, 0.5], zeta=zeta, ncpu=1)
model = gpr(kernel=kernel, descriptor=des, noise_e=[2e-2, 5e-3, 1e-1], f_coef=30)
db_file = sys.argv[1]

db_ids = range(N_start)
pool_ids = range(0, N_max, N_step)  

train_data = get_data(db_file, des, N_force=5, lists=db_ids, select=True)

model.fit(train_data)
E, E1, F, F1 = model.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
print("\n")


for id in pool_ids:
    ids = range(id, id+N_step)
    test_data = get_data(db_file, des, lists=ids, ncpu=1)
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
        for f_id in range(Num):
            if np.max(_std[f_id]) > 2*model.noise_f: 
                pts_to_add["force"].append(test_data["force"][int(F_count/3)+f_id])
                force_in.append(f_id)
                #print("add force data to GP model", _std[f_id])
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
        l1 = metric_single(train_E, train_E1, "Energy") 
        l2 = metric_single(train_F, train_F1, "Forces") 

model.save("1.json", "test.db")

test_E, test_E1, E_var, test_F, test_F1, F_var = None, None, None, None, None, None
for id in range(0, N_max, N_step):
    ids = range(id, id+N_step)
    test_data = get_data(db_file, des, lists=ids, ncpu=1)
    print("Testing-------", id, id+N_step)
    E, E1, E_std, F, F1, F_std = model.validate_data(test_data, return_std=True)
    metric_single(E, E1, "Test Energy") 
    metric_single(F, F1, "Test Forces") 
    if test_E is None:
        test_E = E
        test_E1 = E1
        test_F = F
        test_F1 = F1
        E_var = E_std
        F_var = F_std
    else:
        test_E = np.hstack((test_E, E))
        test_E1 = np.hstack((test_E1, E1))
        test_F = np.hstack((test_F, F))
        test_F1 = np.hstack((test_F1, F1))
        E_var = np.hstack((E_var, E_std))
        F_var = np.hstack((F_var, F_std))

l3 = metric_single(test_E, test_E1, "Energy") 
l4 = metric_single(test_F, test_F1, "Forces") 

plot((train_E, test_E), (train_E1, test_E1), (l1, l3), "E.png")
plot((train_F, test_F), (train_F1, test_F1), (l2, l4), "F.png")
plot([np.abs(test_E-test_E1)], [E_var], ["Energy: True .v.s Var"], "E_var.png", False)
plot([np.abs(test_F-test_F1)], [F_var], ["Forces: True .v.s Var"], "F_var.png", False)
plot_two_body(model, des, kernel, "2-body.png") 

model2 = gpr()
model2.load("1.json")
E, E1, F, F1 = model2.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
 
