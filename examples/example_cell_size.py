import os
import sys
import numpy as np
from time import time
from cspbo.utilities import rmse, metric_single, get_strucs, plot
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.calculator import GPR
from mpi4py import MPI
from copy import deepcopy
from ase import Atoms
from spglib import get_symmetry_dataset
np.set_printoptions(formatter={'float': '{: 5.2f}'.format})
eV2GPa = 160.21766
ncpu = 16
cmd = "mpirun -np " + str(ncpu) + " vasp_std"

#comm=MPI.COMM_WORLD 

N_max = 5
m_file = sys.argv[1]
db_file = sys.argv[2]
model = gpr()
model.load(m_file, N_max=None, opt=True)

train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 

# get _structures
strucs, values = get_strucs(db_file, N_max=N_max)
(E0, F0, S0) = values[0]
stress = True  

# set calculator
calc = GPR(ff=model, return_std=True, stress=stress)

def symmetrize(struc, sym_tol=1e-3):
    """
    symmetrize the ase atoms struc
    """
    s = (struc.get_cell(), struc.get_scaled_positions(), struc.get_atomic_numbers())
    dataset = get_symmetry_dataset(s, sym_tol)
    cell = dataset['std_lattice']
    pos = dataset['std_positions']
    numbers = dataset['std_types']

    return Atoms(numbers, scaled_positions=pos, cell=cell, pbc=[1,1,1])

from ase.calculators.vasp import Vasp

def set_vasp(level='opt', kspacing=0.5):
    """
    INCAR parameters for VASP calculation
    """
    para0 = {'xc': 'pbe',
            'npar': 8,
            'kgamma': True,
            'lcharg': False,
            'lwave': False,
            'ibrion': 2,
            }
    if level == 'single':
        para = {'prec': 'accurate',
                'encut': 500,
                'ediff': 1e-4,
                'nsw': 0,
                }
    else:
        para = {'prec': 'accurate',
                'encut': 650,
                'isif': 3,
                'ediff': 1e-6,
                'nsw': 100, # we don't need to fully relax it
                }
    dict_vasp = dict(para0, **para)
    return Vasp(kspacing=kspacing, **dict_vasp)

def dft_run(struc, path, max_time=3, clean=True):
    """
    perform dft calculation and get energy and forces
    """
    os.environ["VASP_COMMAND"] = "timeout " + str(max_time) + "m " + cmd
    cwd = os.getcwd()
    os.chdir(path)
    try:
        eng = struc.get_potential_energy()
        forces = struc.get_forces()
    except:
        eng = None
        forces = None
    if clean:
        os.system("rm POSCAR POTCAR INCAR OUTCAR")
    os.chdir(cwd)
    return eng, forces

from ase.db import connect
from ase.build import bulk
calc_folder = 'tmp_vasp'
db_file = 'database.db'
if not os.path.exists(calc_folder):
    os.makedirs(calc_folder)

if not os.path.exists(calc_folder+'/'+db_file):
    struc = bulk('Si', 'diamond', a=5.0, cubic=True)
    print("The initial structure: ", struc)
    struc.set_calculator(set_vasp('opt', 0.15))
    E0, F0 = dft_run(struc, path=calc_folder, max_time=30)
    print('The optimal energy: {:6.4f} eV/atom'.format(E0/len(struc)))
    struc.set_calculator()
    print("The optimal structure: ", struc)

    curdir = os.getcwd()
    os.chdir(calc_folder)
    with connect("database.db") as db:
        d1 = {'tag': 'diamond',
              'dft_energy': E0/len(struc),
              'dft_forces': F0}
        db.write(struc, data=d1)
    os.chdir(curdir)
    E0 /= len(struc)
else:
    with connect(calc_folder+'/'+db_file) as db:
        for row in db.select():
            struc = db.get_atoms(row.id)
            E0 = row.data['dft_energy']
            F0 = row.data['dft_forces'].copy()

total_F, total_F0 = None, None

struc0 = deepcopy(struc)
struc0.set_calculator(calc)
E = struc0.get_potential_energy()/len(struc0)
F = struc0.get_forces()
if total_F is None:
    total_F, total_F0 = F.flatten(), F0.flatten()
else:
    total_F = np.hstack((total_F, F.flatten()))
    total_F0 = np.hstack((total_F0, F0.flatten()))
F_mse = rmse(F.flatten(), F0.flatten())
E_var = struc0._calc.get_var_e()
E_var_total = struc0._calc.get_var_e(total=True)
vol = struc0.get_volume()/len(struc0)
print("GPR prediction of diamond structure before expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} E0: {:6.3f} eV/atom F_MSE: {:6.3f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, E0, F_mse, vol))

# Expand 20%
struc0.set_cell(1.063*struc0.cell)
struc0.set_scaled_positions(struc0.get_scaled_positions())
E = struc0.get_potential_energy()/len(struc0)
F = struc0.get_forces()
if total_F is None:
    total_F, total_F0 = F.flatten(), F0.flatten()
else:
    total_F = np.hstack((total_F, F.flatten()))
    total_F0 = np.hstack((total_F0, F0.flatten()))
F_mse = rmse(F.flatten(), F0.flatten())
E_var = struc0._calc.get_var_e()
E_var_total = struc0._calc.get_var_e(total=True)
vol = struc0.get_volume()/len(struc0)
print("GPR prediction of diamond structure after expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, vol))

struc2 = deepcopy(struc)*(2,2,2)
struc2.set_calculator(calc)
E = struc2.get_potential_energy()/len(struc2)
E_var = struc2._calc.get_var_e()
E_var_total = struc2._calc.get_var_e(total=True)
vol = struc2.get_volume()/len(struc2)
print("GPR prediction of diamond structure (2,2,2) before expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} E0: {:6.3f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, E0, vol))

struc2.set_cell(1.063*struc2.cell)
struc2.set_scaled_positions(struc2.get_scaled_positions())
E = struc2.get_potential_energy()/len(struc2)
E_var = struc2._calc.get_var_e()
E_var_total = struc2._calc.get_var_e(total=True)
vol = struc2.get_volume()/len(struc2)
print("GPR prediction of diamond structure (2,2,2) after expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, vol))
