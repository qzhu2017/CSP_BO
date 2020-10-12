import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single, build_desc, convert_struc, mae, plot, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb
from random import choice


def get_data(db_name, des, N_force=100000, lists=None, select=False, ncpu=1):
    """
    Nmax: Maximum number of force data
    """
    X, Y = convert_struc(db_name, des, lists, ncpu=ncpu)
    print('\n')
    energy_data = []
    force_data = []

    for id in range(len(X)):
        energy_data.append((X[id]['x'], Y["energy"][id]/len(X[id]['x']))) 
        if select:
            ids = [0] #[choice(range(len(X[id]['x'])))]
        else:
            ids = range(len(X[id]['x']))
        for i in ids:
            if len(force_data) < N_force:
                ids = np.argwhere(X[id]['seq'][:,1]==i).flatten()
                _i = X[id]['seq'][ids, 0] 
                force_data.append((X[id]['x'][_i,:], X[id]['dxdr'][ids], Y['forces'][id][i]))


    train_data = {"energy": energy_data, "force": force_data}
    return train_data


des = build_desc("SO3", lmax=4, nmax=4, rcut=4.5)
print(des)

t0 = time()
N_start, N_step, N_max, zeta = 50, 50, 5000, 2

kernel = RBF_mb(para=[1, 0.5], zeta=zeta)
model = gpr(kernel=kernel, noise_e=0.02, noise_f=0.10)

db_file = sys.argv[1]

db_ids = range(N_start)
pool_ids = range(0, N_max, N_step)  

train_data = get_data(db_file, des, N_force=15, lists=db_ids, select=True)

model.fit(train_data)
E, E1, F, F1 = model.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
print("\n")


E_tol, Ev_tol = 0.03, 0.04
F_tol, Fv_tol = 5e-1, 0.25

for id in pool_ids:
    ids = range(id, id+N_step)
    test_data = get_data(db_file, des, lists=ids, ncpu=1)
    E, E1, E_std, F, F1, F_std = model.validate_data(test_data, return_std=True)
    l1 = metric_single(E, E1, "Test Energy") 
    l2 = metric_single(F, F1, "Test Forces") 


    F_count = 0
    F_in_s = []
    for i in range(len(E)):
        Num = len(test_data['energy'][i][0])
        F_in_s.extend([i]*Num)
        diff_E = E[i] - E1[i]
        _F = F[F_count:F_count+Num]
        _F1 = F1[F_count:F_count+Num]
        print("{:d} ML Energy: {:8.3f} -> {:8.3f} {:8.3f}  Variance: {:8.3f}  F_mae: {:8.3f} ".format(\
                id+i, E[i], E1[i], diff_E, E_std[i], mae(_F, _F1)))
        F_count += Num

    update = False
    
    #diffs_E = np.abs(E-E1)
    for e_id, std in enumerate(E_std):
        if E_std[e_id] > Ev_tol:
            #print("add energy data to GP model", e_id)
            model.add_train_pts_energy(test_data['energy'][e_id])
            update=True

    diffs_F = np.abs(F-F1)
    F_std = F_std.reshape([len(F_in_s),3])
    s_ids = []
    for id in range(len(F_std)):
        s_id = F_in_s[id]
        if np.max(F_std[id]) > Fv_tol and s_id not in s_ids:
            print("add force data to GP model", F_std[id])
            update=True
            model.add_train_pts_force(test_data["force"][id])
            s_ids.append(s_id)

    #for id0, diff_Fa in enumerate(diffs_F):
    #    if diff_F > F_tol and f_id not in f_ids and s_id not in s_ids:
    #        #print("add force data to GP model", diffs_F[id0], F_std[id0])
    #        model.add_train_pts_force(test_data["force"][f_id])
    #        f_ids.append(f_id)
    #        #break

    if update:
        model.fit()
        E, E1, F, F1 = model.validate_data()
        metric_single(E, E1, "Train Energy") 
        metric_single(F, F1, "Train Forces") 
        print("\n")

train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Energy") 
l2 = metric_single(train_F, train_F1, "Forces") 

test_E, test_E1, E_var, test_F, test_F1, F_var = None, None, None, None, None, None

for id in range(0, N_max, N_step):
    ids = range(id, id+N_step)
    test_data = get_data(db_file, des, lists=ids, ncpu=1)
    print('\n')
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

print(model)
#print("elapsed time: ", time()-t0)
#np.savetxt('ids.txt', np.array(db_ids))



