import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single, build_desc, get_data, get_train_data, rmse
from cspbo.utilities import convert_train_data
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.calculator import GPR, LJ
from cspbo.kernels.RBF_mb import RBF_mb
from cspbo.kernels.Dot_mb import Dot_mb
from ase.calculators.emt import EMT
from ase.db import connect

zeta = 2
device = 'cpu'
#N_start, N_step, N_max, zeta, device, fac, N_force = 1, 1, None, 2, 'cpu', 1.2, 8

des = build_desc("SO3", lmax=4, nmax=3, rcut=4.0)
#kernel = RBF_mb(para=[1, 0.5], zeta=zeta, device=device)
kernel = Dot_mb(para=[2, 0.5], zeta=zeta, device=device)
#lj = None #LJ(parameters={"rc": 6.0, "sigma": 2.13})
    
model = gpr(kernel=kernel, descriptor=des, noise_e=[0.01, 0.01, 0.03], f_coef=20)
db_file = sys.argv[1]
#db_ids = range(N_start)
#train_data = get_data(db_file, des) #, N_force=5, lists=db_ids, select=True)

with connect(db_file) as db:
    for row in db.select():
        struc = db.get_atoms(id=row.id)
        struc.calc = EMT()
        data = (struc, struc.get_potential_energy(), struc.get_forces())
        pts, N_pts, (E, E1, E_std, F, F1, F_std) = model.add_structure(data)
        print("{:4d} E_True/Pred: {:8.4f} -> {:8.4f} Error_E/F: {:8.4f}[{:8.4f}] {:8.4f}[{:8.4f}]".format(\
        row.id, E, E1, E-E1, E_std, rmse(F, F1), np.max(F_std)))
        if N_pts > 0:
            model.set_train_pts(pts, mode="a+")
            model.fit()
        
        my_data = convert_train_data([data], des)
        E, E1, E_std, F, F1, F_std = model.validate_data(my_data, return_std=True)
        print("{:4d} E_True/Pred: {:8.4f} -> {:8.4f} Error_E/F: {:8.4f}[{:8.4f}] {:8.4f}[{:8.4f}]".format(\
        row.id, E[0], E1[0], E[0]-E1[0], E_std[0], rmse(F, F1), np.max(F_std)))

#print('Final validatation')
#model.save("models/test.json", "models/test.db")
#model = gpr()
#model.load("models/test.json", N_max=None, opt=True)
#E, E_1, E_v, F, F_1, F_v = model.validate_data(return_std=True)
#print(E, E_1, E_v)
