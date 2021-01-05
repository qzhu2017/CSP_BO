from ase import Atoms
import numpy as np
import os
from time import time

#--------- Database related ------------------

from ase.db import connect

def add_dimers(dimers, db_file):
    """
    Backup the dimer data for fitting LJ
    """
    with connect(db_file) as db:
        for dimer in dimers:
            pos = dimer.positions
            kvp = {"tag": 'dimer',
                   "r": dimer.positions[1,0],
                   "dft_energy": dimer.get_potential_energy(),
                  }
            db.write(dimer, key_value_pairs=kvp)

def add_GP_train(data, db_file):
    """
    Backup the DFT data for GP training
    """
    with connect(db_file) as db:
        for d in data:
            struc, eng, force = data
            d1 = {'tag': 'GP',
                  'dft_energy': eng,
                  'dft_forces': force,
                  }
            db.write(struc, data=d1)

#--------- DFT related ------------------

from ase.calculators.vasp import Vasp

def dft_run(struc, path, clean=False):
    """
    perform dft calculation and get energy and forces
    """
    cwd = os.getcwd()
    os.chdir(path)
    eng = struc.get_potential_energy()
    forces = struc.get_forces()
    if clean:
        os.system("rm POSCAR POTCAR INCAR OUTCAR")
    os.chdir(cwd)
    return eng, forces

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
                'encut': 520,
                'ediff': 1e-4,
                'nsw': 0,
                }
    else:
        para = {'prec': 'accurate',
                'encut': 400,
                'isif': 3,
                'ediff': 1e-4,
                'nsw': 50,
                }
    dict_vasp = dict(para0, **para)
    return Vasp(kspacing=kspacing, **dict_vasp)


#--------- LJ fitting  ------------------
def LJ_fit(rs, engs, eng_cut=10.0, p1=12, p2=6):
    """
    Fit the Lennard-Jones potential
    """
    # to improve the fitting, remove very postive energies
    rs = np.array(rs)
    engs = np.array(engs)
    ids = engs < eng_cut
    rs = rs[ids]
    engs = engs[ids]
    from scipy.optimize import curve_fit
    def fun(x, eps, sigma):
        return 4*eps*((sigma/x)**p1 - (sigma/x)**p2)
    
    para, cov = curve_fit(fun, rs, np.array(engs), bounds=(0, [30, 4]))

    return para

#---------------------- Main Program -------------------
# --------- DFT calculator set up
calc_folder = 'vasp_tmp'
if not os.path.exists(calc_folder):
    os.makedirs(calc_folder)
species = ["Si"]

#---------- Get the LJ potential 
dimer_db = 'dimers.db'
engs = []
if not os.path.exists(dimer_db):
    rs = np.linspace(1, 4, 30)
    cell = 15*np.eye(3)
    dimers = []
    for r in rs:
        pos = [[0,0,0], [r,0,0]]
        dimer = Atoms(2*species, positions=pos, cell=cell, pbc=[1,1,1]) 
        dimer.set_calculator(set_vasp('single', 0.5))
        eng, _ = dft_run(dimer, path=calc_folder)
        dimers.append(dimer)
        engs.append(eng)
    add_dimers(dimers, dimer_db)
else:
    rs = []
    with connect(dimer_db) as db:
        for row in db.select():
            rs.append(row.r)
            engs.append(row.dft_energy)

para = LJ_fit(rs, engs)

from cspbo.calculator import LJ
lj = LJ(parameters={"rc": 5.0, "epsilon": para[0], "sigma": para[1]})

##----------- Get the initial training database
from ase.build import bulk
train_db = 'init_train.db'
data = []
if not os.path.exists(train_db):
    strucs = []
    strucs.append(bulk(species[0], 'fcc', a=3.6, cubic=True))
    strucs.append(bulk(species[0], 'bcc', a=3.6, cubic=True))
    strucs.append(bulk(species[0], 'sc', a=3.6, cubic=True))
    strucs.append(bulk(species[0], 'diamond', a=3.6, cubic=True)) 
    for struc in strucs:
        #opt
        struc.set_calculator(set_vasp('opt', 0.3))
        eng, forces = dft_run(struc, path=calc_folder)
        struc.set_calculator(set_vasp('single', 0.20))
        eng, forces = dft_run(struc, path=calc_folder)
        data.append((struc1, eng, forces))

        #expansion
        struc1 = struc.copy()
        struc1.set_cell(1.2*struc.cell)
        struc1.set_scaled_positions(struc.get_scaled_positions())
        struc1.set_calculator(set_vasp('single', 0.20))
        eng, forces = dft_run(struc1, path=calc_folder)
        data.append((struc1, eng, forces))

        #shrink
        struc2 = struc.copy()
        struc2.set_cell(0.8*struc.cell)
        struc2.set_scaled_positions(struc.get_scaled_positions())
        struc2.set_calculator(set_vasp('single', 0.20))
        eng, forces = dft_run(struc2, path=calc_folder)
        data.append((struc2, eng, forces))
    add_GP_train(data, train_db)
else:
    with connect(train_db) as db:
        for row in db.select():
            struc = db.get_atoms(row.id)
            data.append((struc, row.data['dft_energy'], row.data['dft_forces'].copy()))

##----------- Fit the GP model
from cspbo.kernels.RBF_mb import RBF_mb
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.utilities import build_desc

des = build_desc("SO3", lmax=3, nmax=3, rcut=4.0)
kernel = RBF_mb(para=[1, 0.5])
model = gpr(kernel=kernel, 
            descriptor=des, 
            base_potential=lj,
            noise_e=[5e-3, 2e-3, 2e-1], 
            f_coef=10)
for d in data:
    pts, N_pts, error = model.add_structure(d)
    if N_pts > 0:
        model.set_train_pts(pts, mode="a+")
        model.fit(show=False)
print(model)

## ----------- Structure generation/optimization based on the surrogate model
from cspbo.utilities import PyXtal
from cspbo.calculator import GPR
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
from spglib import get_symmetry_dataset
sgs = range(16, 231)
numIons = [8]
gen_max = 50
N_pop = 50

for gen in range(gen_max):
    for pop in range(N_pop):
        # generate and relax the structure
        t0 = time()
        struc = PyXtal(sgs, species, numIons) 
        calc = GPR(ff=model, stress=True, return_std=True)
        struc.set_calculator(calc)

        # fix cell opt
        dyn = FIRE(struc, logfile='opt.log')
        dyn.run(fmax=0.05, steps=100)

        # variable cell opt
        ecf = ExpCellFilter(struc)
        dyn = FIRE(ecf, logfile='opt.log')
        dyn.run(fmax=0.05, steps=100)
        E = struc.get_potential_energy()/len(struc)
        E_var = struc._calc.get_var_e()

        try:
            spg = get_symmetry_dataset(struc, symprec=5e-2)['international']
        except:
            spg = 'N/A'
        cputime = (time()-t0)/60
        strs = "{:3d} {:3d} {:6s} {:16s} {:8.3f}[{:8.3f}] {:6.2f}".format(\
        gen, pop, struc.get_chemical_formula(), spg, E, E_var, cputime)
        print(strs)
#
#    # select the structures
#
#    # perform single point DFT calculation
#
#    # update the GP model








