"""
This is a script to use the pretrained model to perform geometry relaxation
of the random crystals.
"""
import numpy as np
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.calculator import GPR
from ase.optimize import BFGS, FIRE
from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter

from spglib import get_symmetry_dataset
from pyxtal.interface.gulp import GULP
from cspbo.utilities import write_db_from_dict, PyXtal

import os
   
zeta, ncpu, fac = 2, 4, 2.0
sgs = range(16, 231)
species = ["Si"]
numIons = [4]
ff = "edip_si.lib" 
model = gpr()
model.load("models/Si.json")
calc_gp = GPR(ff=model, stress=True, return_std=True)
# ================= Get the first step of training database
data_dict = {"atoms": [], "ML_energy": [], "QM_energy": [], 
            "ML_variance": [], "spg": []}

for i in range(50):
    train_data = []

    # structure from Pyxtal
    struc = PyXtal(sgs, species, numIons) 
    struc.set_constraint(FixSymmetry(struc)) 
    struc.calc = calc_gp

    # opt1: relaxation with the constant cell
    dyn = FIRE(struc, logfile='results.log')
    dyn.run(fmax=0.05, steps=10)

    # opt2: relax both atomic coordinates + lattice vectors
    ecf = ExpCellFilter(struc)
    dyn = FIRE(ecf, logfile='results.log')
    dyn.run(fmax=0.005, steps=50)
    E = struc.get_potential_energy()/len(struc)
    E_var = struc._calc.get_var_e()
    #F = struc.get_forces()
    
    #
    calc2 = GULP(struc, ff=ff, opt="single")
    calc2.run()
    E1 = calc2.energy/len(struc)
 
    try:
        spg = get_symmetry_dataset(struc, symprec=5e-2)['international']
    except:
        spg = 'N/A'


    print("{:4d} {:8s} E: {:8.3f} -> {:8.3f} {:8.3f} E_var: {:8.3f}".format(\
        i, spg, E, E1, E-E1, E_var))
    struc.set_calculator()
    data_dict["atoms"].append(struc)
    data_dict["ML_energy"].append(E)
    data_dict["QM_energy"].append(E1)
    data_dict["spg"].append(spg)
    data_dict["ML_variance"].append(E_var)

write_db_from_dict(data_dict)

