import numpy as np
#from cspbo.RBF_mb import RBF_mb
from cspbo.Dot_mb import Dot_mb
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.calculator import GPR, LJ
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
from cspbo.utilities import PyXtal, metric_single, convert_train_data, build_desc
from ase.db import connect
from spglib import get_symmetry_dataset
from pyxtal.interface.gulp import GULP

   
zeta, ncpu, fac = 2, 4, 2.0
sgs = range(16, 231)
species = ["Si"]
numIons = [8]
ff = "edip_si.lib" 
des = build_desc("SO3", lmax=3, nmax=3, rcut=4.0)
kernel = Dot_mb(para=[1, 0.5], zeta=zeta, ncpu=ncpu)
lj = LJ(parameters={"rc": 5.0, "sigma": 2.13})
model = gpr(kernel=kernel, 
            descriptor=des, 
            base_potential=LJ(),
            noise_e=[5e-3, 2e-3, 2e-1], 
            f_coef=10)

# ================= Get the first step of training database
for i in range(5):
    struc = PyXtal(sgs, species, numIons) 

    # Relax the structure
    calc = GULP(struc, ff=ff, opt="conv")
    calc.run()
    struc = calc.to_ase()
    data = (calc.to_ase(), calc.energy, calc.forces)
    pts, N_pts, _ = model.add_structure(data)
    if N_pts > 0:
        model.set_train_pts(pts, mode="a+")
        model.fit()

    # Relax the structure
    calc = GULP(struc, ff=ff, opt="conp")
    calc.run()
    data = (calc.to_ase(), calc.energy, calc.forces)
    pts, N_pts, _ = model.add_structure(data)
    if N_pts > 0:
        model.set_train_pts(pts, mode="a+")
        model.fit()

train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 
print(model)

# ==== Generate the structures and relax them based on the surrogate model
for i in range(10):
    train_data = []

    for j in range(5):
        if i > 0 and j==0:
            struc = best_struc
        else:
            struc = PyXtal(sgs, species, numIons) 

        calc = GPR(ff=model, stress=True, return_std=True)
        struc.set_calculator(calc)
        dyn = FIRE(struc)
        dyn.run(fmax=0.05, steps=10)

        ecf = ExpCellFilter(struc)
        dyn = FIRE(ecf)
        dyn.run(fmax=0.05, steps=25)
        E = struc.get_potential_energy()/len(struc)
        E_var = struc._calc.get_var_e()

        try:
            spg = get_symmetry_dataset(struc, symprec=5e-2)['international']
        except:
            spg = 'N/A'

        calc2 = GULP(struc, ff=ff, opt="single")
        calc2.run()
        E1 = calc2.energy/len(struc)

        print("{:4d} {:8s} E: {:8.3f} -> {:8.3f} {:8.3f} E_var: {:8.3f}".format(\
        j, spg, E, E1, E-E1, E_var))
        struc.set_calculator()

        #Collect the train data
        if calc2.forces is not None:
            train_data.append((struc, calc2.energy, calc2.forces))

    print("Summary of the predictions:")
    my_data = convert_train_data(train_data, des)
    E, E1, E_std, F, F1, F_std = model.validate_data(my_data, return_std=True)
    l1 = metric_single(E, E1, "Test Energy") 
    l2 = metric_single(F, F1, "Test Forces") 
    F_count = 0
    for _i in range(len(E)):
        Num = len(my_data['energy'][_i][0])
        diff_E = E[_i] - E1[_i]
        _std = F_std[F_count:F_count+Num*3]
        print("{:4d} True-E: {:8.3f} -> Pred-E{:8.3f} {:8.3f}  Var-E: {:8.3f} F: {:8.3f}".format(\
            _i, E[_i], E1[_i], diff_E, E_std[_i], np.max(_std)))
        F_count += Num*3

    ids = np.argsort(E1)
    best_struc = train_data[ids[0]][0]
    for _i in ids[:2]:
        print("update the model with structure {:d}:".format(_i))
        pts, N_pts, _ = model.add_structure(train_data[_i])
        if N_pts > 0:
            model.set_train_pts(pts, mode="a+")
            model.fit()

    print("Summary of the predictions after model update:")
    E, E1, E_std, F, F1, F_std = model.validate_data(my_data, return_std=True)
    l1 = metric_single(E, E1, "Test Energy") 
    l2 = metric_single(F, F1, "Test Forces") 
    F_count = 0
    for _i in range(len(E)):
        Num = len(my_data['energy'][_i][0])
        diff_E = E[_i] - E1[_i]
        _std = F_std[F_count:F_count+Num*3]
        print("{:4d} True-E: {:8.3f} -> Pred-E{:8.3f} {:8.3f}  Var-E: {:8.3f} F: {:8.3f}".format(\
            _i, E[_i], E1[_i], diff_E, E_std[_i], np.max(_std)))
        F_count += Num*3


