import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single, build_desc, get_data, get_train_data, rmse, plot, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.calculator import GPR, LJ
from cspbo.RBF_mb import RBF_mb
from cspbo.Dot_mb import Dot_mb

#N_start, N_step, N_max, zeta, ncpu, fac = 4, 1, 2554, 2, 10, 1.2
N_start, N_step, N_max, zeta, device, fac, N_force = 1, 1, 50, 2, 'gpu', 1.2, 8

des = build_desc("SO3", lmax=3, nmax=3, rcut=4.0)
kernel = RBF_mb(para=[1, 0.5], zeta=zeta, device=device)
#kernel = Dot_mb(para=[2, 0.5], zeta=zeta, device=device)
lj = None #LJ(parameters={"rc": 5.0, "sigma": 2.13})

model = gpr(kernel=kernel, 
            descriptor=des, 
            base_potential=lj,
            noise_e=[5e-3, 2e-3, 5e-3], 
            f_coef=10,
           )

db_file = sys.argv[1]
db_ids = range(N_start)

train_data = get_data(db_file, des, N_force=5, lists=db_ids, select=True, ncpu=10)

model.fit(train_data)
E, E1, F, F1 = model.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
print("\n")

(strucs, energies, forces) = get_train_data(db_file)

for id in range(N_max):
    data = (strucs[id], energies[id], forces[id])
    pts, N_pts, (E, E1, E_std, F, F1, F_std) = model.add_structure(data)
    print("{:4d} E_True/Pred: {:8.4f} -> {:8.4f} Error_E/F: {:8.4f}[{:8.4f}] {:8.4f}[{:8.4f}]".format(\
        id, E, E1, E-E1, E_std, rmse(F, F1), np.max(F_std)))
    
    if N_pts > 0:
        model.set_train_pts(pts, mode="a+")
        model.fit()
        train_E, train_E1, train_F, train_F1 = model.validate_data()
        l1 = metric_single(train_E, train_E1, "Train Energy") 
        l2 = metric_single(train_F, train_F1, "Train Forces") 
        print(model)
model.save("models/test.json", "models/test.db")
plot_two_body(model, "2-body.png")
