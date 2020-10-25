import numpy as np
#from cspbo.RBF_mb import RBF_mb
from cspbo.Dot_mb import Dot_mb
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.calculator import GPR
from ase.optimize import BFGS, FIRE
from ase.constraints import ExpCellFilter, UnitCellFilter
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry
from cspbo.utilities import PyXtal, metric_single, convert_train_data, build_desc, new_pt
import os

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
model = gpr(kernel=kernel, 
            descriptor=des, 
            noise_e=[5e-3, 2e-3, 2e-1], 
            f_coef=10)

# ================= Get the first step of training database
for i in range(5):
    train_data = []
    struc = PyXtal(sgs, species, numIons) 
    calc = GULP(struc, ff=ff, opt="single")
    calc.run()
    train_data.append((struc, calc.energy, calc.forces))

    # Relax the structure
    calc = GULP(struc, ff=ff, opt="conv")
    calc.run()
    struc = calc.to_ase()
    train_data.append((struc, calc.energy, calc.forces))

    # Relax the structure
    calc = GULP(struc, ff=ff, opt="conp")
    calc.run()
    train_data.append((calc.to_ase(), calc.energy, calc.forces))
    #train_data[0][0].write("0.vasp", format='vasp', vasp5=True, direct=True)
    #train_data[1][0].write("1.vasp", format='vasp', vasp5=True, direct=True)
    #train_data[2][0].write("2.vasp", format='vasp', vasp5=True, direct=True)
    my_data = convert_train_data(train_data, des)
    #print("problem")
    #import sys
    #sys.exit()
    if i == 0:
        if len(my_data["force"])>10:
            my_data["force"] = my_data["force"][:10]
        model.fit(my_data)
        E, E1, E_std, F, F1, F_std = model.validate_data(my_data, return_std=True)
        l1 = metric_single(E, E1, "Test Energy") 
        l2 = metric_single(F, F1, "Test Forces") 
    else:
        E, E1, E_std, F, F1, F_std = model.validate_data(my_data, return_std=True)
        l1 = metric_single(E, E1, "Test Energy") 
        l2 = metric_single(F, F1, "Test Forces") 

        pts_to_add = {"energy": [], "force": [], "db": []}
        xs_added = []
        F_count = 0
        for _i in range(len(E)):
            Num = len(my_data['energy'][_i][0])
            diff_E = E[_i] - E1[_i]
            _F = F[F_count:F_count+Num*3]
            _F1 = F1[F_count:F_count+Num*3]
            _std = F_std[F_count:F_count+Num*3]
            print("{:4d} E: {:8.3f} -> {:8.3f} {:8.3f}  Var-E: {:8.3f} F: {:8.3f}".format(\
                _i, E[_i], E1[_i], diff_E, E_std[_i], np.max(_std)))

            energy_in = False
            force_in = []

            if E_std[_i] > 2*model.noise_e: 
                pts_to_add["energy"].append(my_data['energy'][_i])
                energy_in = True

            _std = _std.reshape([Num, 3])
            for f_id in range(Num):
                if np.max(_std[f_id]) > fac*model.noise_f: 
                    X = my_data['energy'][_i][0][f_id]
                    ele = my_data['energy'][_i][2][f_id]
                    if len(xs_added) == 0:
                        pts_to_add["force"].append(my_data["force"][int(F_count/3)+f_id])
                        force_in.append(f_id)
                        xs_added.append((X, ele))
                    else:
                        if new_pt((X, ele), xs_added):
                            pts_to_add["force"].append(my_data["force"][int(F_count/3)+f_id])
                            force_in.append(f_id)
                            xs_added.append((X, ele))
                    if len(xs_added) == 6:
                        break

            if energy_in or len(force_in)>0:
                (struc, energy, force, _, _) = my_data["db"][_i]
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

# ==== Generate the structures and relax them based on the surrogate model
for i in range(10):
    train_data = []
    Es = []
    E_stds = []

    for j in range(5):
        if i > 0 and j==0:
            struc = best_struc
        else:
            struc = PyXtal(sgs, species, numIons) 
        calc = GPR(ff=model, stress=True, return_std=True, lj={"rc": 2.5, "sigma": 0.7})
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

        if calc2.forces is not None:
            train_data.append((struc, calc2.energy, calc2.forces))
            Es.append(E)

    print("Summary of the predictions:")
    my_data = convert_train_data(train_data, des)
    E, E1, E_std, F, F1, F_std = model.validate_data(my_data, return_std=True)
    print(E, E1)
    l1 = metric_single(E, E1, "Test Energy") 
    l2 = metric_single(F, F1, "Test Forces") 
    for _i in range(len(E)):
        Num = len(my_data['energy'][_i][0])
        diff_E = E[_i] - E1[_i]
        _F = F[F_count:F_count+Num*3]
        _F1 = F1[F_count:F_count+Num*3]
        _std = F_std[F_count:F_count+Num*3]
        print("{:4d} True-E: {:8.3f} -> Pred-E{:8.3f} {:8.3f}  Var-E: {:8.3f} F: {:8.3f}".format(\
            _i, E[_i], E1[_i], diff_E, E_std[_i], np.max(_std)))

    print("update the model:")
    ids = np.argsort(E1)
    best_struc = train_data[ids[0]][0]
    for _i in ids[:2]:
        model.add_structure(train_data[_i])
        model.fit()
        train_E, train_E1, train_F, train_F1 = model.validate_data()
        l1 = metric_single(train_E, train_E1, "Train Energy") 
        l2 = metric_single(train_F, train_F1, "Train Forces") 
        print(model)

    print("Summary of the predictions after model update:")
    E, E1, E_std, F, F1, F_std = model.validate_data(my_data, return_std=True)
    l1 = metric_single(E, E1, "Test Energy") 
    l2 = metric_single(F, F1, "Test Forces") 
    for _i in range(len(E)):
        Num = len(my_data['energy'][_i][0])
        diff_E = E[_i] - E1[_i]
        _F = F[F_count:F_count+Num*3]
        _F1 = F1[F_count:F_count+Num*3]
        _std = F_std[F_count:F_count+Num*3]
        print("{:4d} True-E: {:8.3f} -> Pred-E{:8.3f} {:8.3f}  Var-E: {:8.3f} F: {:8.3f}".format(\
            _i, E[_i], E1[_i], diff_E, E_std[_i], np.max(_std)))


