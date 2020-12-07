import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single, build_desc, get_data, get_train_data, rmse
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.calculator import GPR, LJ
#from cspbo.kernels.RBF_mb import RBF_mb
from cspbo.kernels.Dot_mb import Dot_mb

#N_start, N_step, N_max, zeta, device, fac, N_force = 1, 1, None, 2, 'gpu', 1.2, 8
N_start, N_step, N_max, zeta, device, fac, N_force = 1, 1, None, 2, 'cpu', 1.2, 8

if len(sys.argv) == 2:
    des = build_desc("SO3", lmax=3, nmax=3, rcut=4.0)
    #kernel = RBF_mb(para=[1, 0.5], zeta=zeta, device=device)
    kernel = Dot_mb(para=[2, 0.5], zeta=zeta, device=device)
    lj = None #LJ(parameters={"rc": 6.0, "sigma": 2.13})
    
    model = gpr(kernel=kernel, 
                descriptor=des, 
                base_potential=lj,
                noise_e=[0.01, 0.01, 0.03], 
                f_coef=20,
               )
    db_file = sys.argv[1]
    db_ids = range(N_start)
    train_data = get_data(db_file, des, N_force=5, lists=db_ids, select=True)
    model.fit(train_data)
else: #pick a model from the previous calcs
    m_file = sys.argv[1]
    model = gpr()
    #model.load(m_file, N_max=2, opt=True, device=device)
    model.load(m_file, N_max=None, opt=True, device=device)
    model.sparsify()
    
    db_file = sys.argv[2]

E, E1, F, F1 = model.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
print("\n")

(strucs, energies, forces) = get_train_data(db_file)
if N_max is None:
    N_max = len(strucs)

#for id in range(1200, N_max):
for id in range(N_max):
    data = (strucs[id], energies[id], forces[id])
    N_atom = len(strucs[id])
    pts, N_pts, (E, E1, E_std, F, F1, F_std) = model.add_structure(data)
    print("{:4d} E_True/Pred: {:8.4f} -> {:8.4f} Error_E/F: {:8.4f}[{:8.4f}] {:8.4f}[{:8.4f}]".format(\
        id, E, E1, E-E1, E_std, rmse(F, F1), np.max(F_std)))
    if N_pts > 0:
        model.set_train_pts(pts, mode="a+")
        model.fit(opt=False, show=False)
    if id % 20 == 0:
        model.sparsify(e_tol=1e-5, f_tol=2e-3)
        #model.fit()
        train_E, train_E1, train_F, train_F1 = model.validate_data()
        l1 = metric_single(train_E, train_E1, "Train Energy") 
        l2 = metric_single(train_F, train_F1, "Train Forces") 
        print(model)
    if id % 100 == 0:
        model.save("models/sample.json", "models/sample.db")
model.save("models/test.json", "models/test.db")
