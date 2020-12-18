import os
import numpy as np
from cspbo.kernels.RBF_mb import RBF_mb as KERNEL
from pymatgen.io.ase import AseAtomsAdaptor
#from cspbo.kernels.Dot_mb import Dot_mb as KERNEL
from cspbo.gaussianprocess import GaussianProcess as gp
from cspbo.utilities import list_to_tuple
from cspbo.calculator import GPR, LJ
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
from cspbo.utilities import PyXtal, metric_single, convert_train_data, build_desc
from ase.db import connect
from spglib import get_symmetry_dataset
from pyxtal.interface.gulp import GULP
from pyxtal.database.element import Element
from interface import process
import pymatgen.analysis.structure_matcher as sm
from time import time
from copy import deepcopy


# Parameters
n_init, n_trail_structures = 10, 10
n_repeat = 95
zeta, fac = 2, 2.0
sgs = range(16, 231)
species = ["Si"]
numIons = [4]
ff = "edip_si.lib"
alpha = 2

# Invoke model
kernel = KERNEL(para=[1, 0.5], zeta=zeta)
lj = LJ(parameters={"rc": 5.0, "sigma": 2.13})
model = gp(kernel=kernel,
            descriptor=build_desc("SO3", lmax=3, nmax=3, rcut=4.0),
            base_potential=lj, #=LJ(),
            noise_e=[5e-3, 2e-3, 2e-1],
            f_coef=10)

#compute the reference ground state
from ase.build import bulk
ref = bulk(species[0], 'diamond', a=5.459, cubic=True)
calc = GULP(ref, ff=ff, opt="conv")
calc.run()
ref_eng = calc.energy/len(ref)
ref_pmg = AseAtomsAdaptor().get_structure(ref)
spg = get_symmetry_dataset(ref, symprec=1e-1)['international']
print("The reference structure is {:8.5f} eV/atom in {:s}".format(ref_eng, spg))
print("\n")

# Get the first n_init of training database
counts, times = [], []
for _ in range(n_repeat):
    t0 = time()
    for i in range(n_init):
        struc = PyXtal(sgs, species, numIons)
        struc.set_constraint(FixSymmetry(struc))

        # Relax the structure
        #print("Relax structure with constant volume")
        calc = GULP(struc, ff=ff, opt="conv")
        calc.run()
        #data = (calc.to_ase(), calc.energy, calc.forces)
        #pts, N_pts, _ = model.add_structure(data, tol_e_var=1.2)
        #if N_pts > 0:
        #    model.set_train_pts(pts, mode="a+")
        #    model.fit()

        # Relax the structure
        #print("Relax structure with constant pressure")
        calc = GULP(calc.to_ase(), ff=ff, opt="conp")
        calc.run()
        data = (calc.to_ase(), calc.energy, calc.forces)
        pts, N_pts, _ = model.add_structure(data, tol_e_var=1.2)
        if N_pts > 0:
            model.set_train_pts(pts, mode="a+")
            model.fit()

    count = 0
    while True:
        print(f"Trial {count+1} begins.")
        structures = []
        data_x = {'energy': [], 'force': []}
        N_E, N_F = 0, 0
        
        # set-up trial structures and minimize them.
        j = 0
        while j < n_trail_structures:
            try:
                #for j in range(n_trail_structures):
                # randomly generating a structure.
                struc = PyXtal(sgs, species, numIons)
                ori_struc = deepcopy(struc)
                calc = GPR(ff=model, stress=True, return_std=True)
                struc.set_calculator(calc)
                E = struc.get_potential_energy()
                E_var = struc._calc.get_var_e()

                try:
                    spg = get_symmetry_dataset(struc, symprec=5e-2)['international']
                except:
                    spg = 'N/A'
                print("{:4d} {:8s} E: {:8.3f} E_var: {:8.3f}".format(j, spg, E, E_var))
                
                # GPR minimization scheme
                dyn = FIRE(struc, logfile='min')
                dyn.run(fmax=0.05, steps=10)
                ecf = ExpCellFilter(struc)
                dyn = FIRE(ecf, logfile='min')
                dyn.run(fmax=0.05, steps=10)
                E = struc.get_potential_energy()
                E_var = struc._calc.get_var_e()
                F = struc.get_forces()
                try:
                    spg = get_symmetry_dataset(struc, symprec=5e-2)['international']
                except:
                    spg = 'N/A'
                print("{:4d} {:8s} E: {:8.3f} E_var: {:8.3f}".format(j, spg, E, E_var))
                
                # Nullify the calculator
                struc.set_calculator()
                structures.append(ori_struc)

                _data = (struc, E, F)
                pts, N_pts, _ = model.add_structure(_data, N_max=100000, tol_e_var=-10, tol_f_var=10000)
                for key in pts.keys():
                    if key == 'energy':
                        if len(pts[key])>0:
                            (X, ELE, indices, E) = list_to_tuple(pts[key], include_value=True, mode='energy')
                            if len(data_x['energy']) == 3:
                                (_X, _ELE, _indices) = data_x['energy']
                                _X = np.concatenate((_X, X), axis=0)
                                _indices.extend(indices)
                                _ELE = np.concatenate((_ELE, ELE), axis=0)
                                data_x['energy'] = (_X, _ELE, _indices)
                            else:
                                data_x['energy'] = (X, ELE, indices)
                j += 1
            except:
                pass
                    
        mean, cov = model.predict(data_x, stress=False, return_cov=True)
        samples = np.random.multivariate_normal(mean, cov * alpha ** 2, 1)[0,:]
        ix = np.argmin(samples)
        print("{:4d}-th structure is picked".format(ix))

        best_struc = structures[ix]
        best_struc.set_constraint(FixSymmetry(best_struc))
        #opt_best_struc, opt_eng, _, opt_spg, _ = process(best_struc, "GULP", ff, 'label', "bo.db")
        #print("{:4d}-th structure: E/atom: {:8.3f} eV/atom sg: {:8s}".format(ix, opt_eng/len(opt_best_struc), opt_spg))

        #calc = GULP(best_struc, ff=ff, opt="conv")
        #calc.run()
        #data = (calc.to_ase(), calc.energy, calc.forces)
        #pts, N_pts, _ = model.add_structure(data, tol_e_var=1.2)
        #if N_pts > 0:
        #    model.set_train_pts(pts, mode="a+")
            #model.fit()
        
        # Optimize the best structure
        calc = GULP(best_struc, ff=ff, opt="conv")
        calc.run()
        calc = GULP(calc.to_ase(), ff=ff, opt="conp")
        calc.run()
        data = (calc.to_ase(), calc.energy, calc.forces)
        pts, N_pts, _ = model.add_structure(data, tol_e_var=1.2)
        if N_pts > 0:
            model.set_train_pts(pts, mode="a+")
            model.fit()

        opt_best_struc, opt_eng = calc.to_ase(), calc.energy
        try:
            opt_spg = get_symmetry_dataset(opt_best_struc, symprec=1e-1)['international']
        except:
            opt_spg = "N/A"
        print("{:4d}-th structure: E/atom: {:8.3f} eV/atom sg: {:8s}".format(ix, opt_eng/len(opt_best_struc), opt_spg))
        print("\n")

        count += 1

        if abs(opt_eng/len(opt_best_struc)-ref_eng) < 1e-2:
            s_pmg = AseAtomsAdaptor().get_structure(opt_best_struc)
            if sm.StructureMatcher().fit(s_pmg, ref_pmg):
                print("Ground State is found at trial {}\n".format(count))
                break
    t1 = time()
    counts.append(count)
    times.append(t1-t0)
    print("{:.2f} s in {:d} BO trials".format(t1-t0, count))
print(counts)
print(times)
print("\n")
