from ase import Atoms
import numpy as np
import os
from time import time
from scipy.stats import norm

# setup the actual command to run vasp
# define the walltime and number of cpus
cmd = "timeout 1m mpirun -np 8 vasp_std > vasp_log"
os.environ["VASP_COMMAND"] = cmd

#--------- Database related ------------------

from ase.db import connect
from cspbo.utilities import list_to_tuple
import pymatgen.analysis.structure_matcher as sm 
from pymatgen.io.ase import AseAtomsAdaptor

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
            struc, eng, force = d
            d1 = {'tag': 'GP',
                  'dft_energy': eng,
                  'dft_forces': force,
                  }
            db.write(struc, data=d1)

def add_structures(data, db_file):
    """
    Backup the structure data from the entire simulation
    """
    with connect(db_file) as db:
        for d in data:
            struc = d
            d1 = {'tag': 'all_structures',
                 }
            db.write(struc, data=d1)

def collect_data(gpr_model, data, struc, energy, force):
    """ Collect data for GPR.predict. """
    _data = (struc, energy, force)
    # High force value means to ignore force since bo only compares energies
    pts, N_pts, _ = gpr_model.add_structure(_data, N_max=100000, tol_e_var=-10, tol_f_var=10000)
    for key in pts.keys():
        if key == 'energy':
            (X, ELE, indices, E) = list_to_tuple(pts[key], include_value=True, mode='energy')
            if len(data["energy"]) == 3:
                (_X, _ELE, _indices) = data['energy']
                _X = np.concatenate((_X, X), axis=0)
                _indices.extend(indices) 
                _ELE = np.concatenate((_ELE, ELE), axis=0) 
                data['energy'] = (_X, _ELE, _indices)
            else:
                data['energy'] = (X, ELE, indices)

    return data
    
def new_struc(struc, ref_strucs):
    """
    check if this is a new structure

    Args:
        struc: input structure
        ref_strucs: reference structure

    Return:
        id: `None` or the id (int) of matched structure
    """
    vol1 = struc.get_volume()/len(struc)
    eng1 = struc.get_potential_energy()/len(struc)
    pmg_s1 = AseAtomsAdaptor.get_structure(struc)

    for i, ref in enumerate(ref_strucs):
        vol2 = ref.get_volume()/len(ref)
        eng2 = ref.get_potential_energy()/len(ref)
        if abs(vol1-vol2)/vol1<5e-2 and abs(eng1-eng2)<2e-3:
            pmg_s2 = AseAtomsAdaptor.get_structure(ref)
            if sm.StructureMatcher().fit(pmg_s1, pmg_s2):
                return i
    return None

#-------- Bayesian Optimization --------
# QZ: Probably, list other acquisition functions
# so that we can try to play with it

def BO_select(model, data, structures, min_E=None, alpha=0.5, n_indices=1, style='Thompson'):
    """ Return the index of the trial structures. """
    if style == 'Thompson':
        mean, cov = model.predict(data, total_E=True, stress=False, return_cov=True)
        if model.base_potential is not None:
            for i, struc in enumerate(structures):
                energy_off, _, _ = model.compute_base_potential(struc)
                mean[i] += energy_off
                mean[i] /= len(struc)
                # Covariance / atom**2
                cov[i,:] /= len(struc)
                cov[:,i] /= len(struc)
        samples = np.random.multivariate_normal(mean, cov * alpha ** 2, 1)[0,:]
    
    elif style == 'EI': # Expected Improvement
        if min_E is None:
            msg = "PI style needs to know the minimum energy"
            return ValueError(msg)
        
        mean, cov = model.predict(data, total_E=True, stress=False, return_cov=True)
        if model.base_potential is not None:
            for i, struc in enumerate(structures):
                energy_off, _, _ = model.compute_base_potential(struc)
                mean[i] += energy_off
                mean[i] /= len(struc)
                # Covariance / atom**2
                cov[i,:] /= len(struc)
                cov[:,i] /= len(struc)
        std_per_atom = np.sqrt(np.diag(cov))
        tmp1 = mean - min_E
        tmp2 = tmp1 / std_per_atom
        samples = tmp1 * norm.cdf(tmp2) + std_per_atom * norm.pdf(tmp2)

    elif style == 'PI': # Probability of Improvement
        if min_E is None:
            msg = "PI style needs to know the minimum energy"
            return ValueError(msg)
        
        mean, cov = model.predict(data, total_E=True, stress=False, return_cov=True)
        if model.base_potential is not None:
            for i, struc in enumerate(structures):
                energy_off, _, _ = model.compute_base_potential(struc)
                mean[i] += energy_off
                mean[i] /= len(struc)
                # Covariance / atom**2
                cov[i,:] /= len(struc)
                cov[:,i] /= len(struc)
        std_per_atom = np.sqrt(np.diag(cov))
        samples = norm.cdf((mean-min_E)/(std_per_atom+1E-9))

    else:
        msg = "The acquisition function style is not equipped."
        raise NotImplementedError(msg)
    
    #if n_indices == 1:
    #    ix = np.argmin(samples)
    #    indices = [ix]
    #else:
    #    indices = np.argsort(samples)[:n_indices]

    indices = np.argsort(samples)
    return indices

#--------- DFT related ------------------

from ase.calculators.vasp import Vasp

def dft_run(struc, path, clean=False):
    """
    perform dft calculation and get energy and forces
    """

    cwd = os.getcwd()
    os.chdir(path)
    try:
        eng = struc.get_potential_energy()
        forces = struc.get_forces()
    except:
        #print("VASP calculation is wrong!")
        eng = None
        forces = None
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
def LJ_fit(rs, engs, eng_cut=5.0, p1=12, p2=6):
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
calc_folder = 'vasp_tmp_Si'
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
min_E = 0
if not os.path.exists(train_db):
    strucs = []
    strucs.append(bulk(species[0], 'fcc', a=3.6, cubic=True))
    strucs.append(bulk(species[0], 'bcc', a=3.6, cubic=True))
    strucs.append(bulk(species[0], 'sc', a=3.6, cubic=True))
    #strucs.append(bulk(species[0], 'diamond', a=3.6, cubic=True)) 
    for struc in strucs:
        #opt
        struc.set_calculator(set_vasp('opt', 0.3))
        eng, forces = dft_run(struc, path=calc_folder)
        struc.set_calculator(set_vasp('single', 0.20))
        eng, forces = dft_run(struc, path=calc_folder)
        data.append((struc, eng, forces))
        if eng/len(struc) < min_E:
            min_E = eng/len(struc)

        #expansion
        struc1 = struc.copy()
        struc1.set_cell(1.2*struc.cell)
        struc1.set_scaled_positions(struc.get_scaled_positions())
        struc1.set_calculator(set_vasp('single', 0.20))
        eng, forces = dft_run(struc1, path=calc_folder)
        data.append((struc1, eng, forces))
        if eng/len(struc) < min_E:
            min_E = eng/len(struc)

        #shrink
        struc2 = struc.copy()
        struc2.set_cell(0.8*struc.cell)
        struc2.set_scaled_positions(struc.get_scaled_positions())
        struc2.set_calculator(set_vasp('single', 0.20))
        eng, forces = dft_run(struc2, path=calc_folder)
        data.append((struc2, eng, forces))
        if eng/len(struc) < min_E:
            min_E = eng/len(struc)

    add_GP_train(data, train_db)
else:
    with connect(train_db) as db:
        for row in db.select():
            struc = db.get_atoms(row.id)
            data.append((struc, row.data['dft_energy'], row.data['dft_forces'].copy()))
            if row.data['dft_energy']/len(struc) < min_E:
                min_E = row.data['dft_energy']/len(struc)

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
#print(model)
#print(model.base_potential)
## ----------- Structure generation/optimization based on the surrogate model

from cspbo.utilities import PyXtal
from cspbo.calculator import GPR
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
from spglib import get_symmetry_dataset
#from pyxtal import pyxtal
sgs = range(16, 231)
numIons = [8]
gen_max = 50
N_pop = 10
alpha = 1
n_bo_select = max([1,N_pop//5])
BO_style = 'Thompson'

logfile = 'opt.log'

for gen in range(gen_max):
    data = {'energy': [], 'force': []}
    structures = []
    Es, E_vars = [], []
    for pop in range(N_pop):
        # generate and relax the structure
        t0 = time()
        struc = PyXtal(sgs, species, numIons) 
        calc = GPR(ff=model, stress=True, return_std=True)
        struc.set_calculator(calc)

        # fix cell opt
        dyn = FIRE(struc, logfile=logfile)
        dyn.run(fmax=0.05, steps=100)

        # variable cell opt
        ecf = ExpCellFilter(struc)
        dyn = FIRE(ecf, logfile=logfile)
        dyn.run(fmax=0.05, steps=100)
        E = struc.get_potential_energy()/len(struc) #per atom
        E_var = struc._calc.get_var_e()

        if new_struc(struc, structures) is None:        
            try:
                spg = get_symmetry_dataset(struc, symprec=5e-2)['international']
            except:
                spg = 'N/A'
            cputime = (time()-t0)/60
            strs = "{:3d} {:3d} {:6s} {:16s} {:8.3f}[{:8.3f}] {:6.2f} {:6.2f}".format(\
            gen, pop, struc.get_chemical_formula(), spg, E, E_var, 
            struc.get_volume()/len(struc), cputime)
            print(strs)
            
            #struc.set_calculator()
            structures.append(struc)
            Es.append(E)
            E_vars.append(E_var)
            
            # QZ: This should be outside the for loop
            F = struc.get_forces()
            data = collect_data(model, data, struc, E, F)
            # save the structures to a db
            # we probably need to re-evaluate the structures after the GP model is converged
            # add_structures(structures, 'all.db')
        else:
            print("skip the duplicate structures")

    # remove unnecessary logfile
    os.remove(logfile)
    #------------------- BO selection ------------------------------
    # The following loop should have only one python process

    # The idea of BO selection is to let us quickly find low-energy structures with low uncertainties
    # With this, we will need to add many new points to GP training database
    # then we hope to see less appearing low energy structures when the generation evloves
    # An overall pattern is that the overall uncertainties will decrease for the low-energy structures
    # We don't need to worry about appearing high energy structures with high uncertainties
    # However, the current scheme seems to select only structures with high uncertainties
    # Therefore, each generation will still generate many appearing low energy structures
    # Perhaps, we need to play with alpha, or revise the selection rule
    
    indices = BO_select(model, data, structures, min_E=min_E, alpha=alpha, n_indices=n_bo_select, style=BO_style)
    total_pts = 0
    bo_count, fail_count = 0, 0
    while bo_count < n_bo_select:
        if bo_count+fail_count == len(indices):
            print("Breaking the while loop.")
            break
        ix = indices[bo_count+fail_count]
        
        if E_vars[ix] > 1e-6:
            best_struc = structures[ix]
            best_struc.set_calculator(set_vasp('single', 0.20))
            strs = "Struc {:4d} is picked: {:8.3f}[{:8.3f}]".format(ix, Es[ix], E_vars[ix])

            # perform single point DFT
            t0 = time()
            best_eng, best_forces = dft_run(best_struc, path=calc_folder)
            cputime = (time() - t0)/60
            # sometimes the vasp calculation will fail
            if best_eng is not None:
                if best_eng/len(best_struc) < min_E:
                    min_E = best_eng/len(best_struc)
                strs += " -> DFT energy: {:8.3f} eV/atom ".format(best_eng/len(best_struc))
                strs += "in {:6.2f} minutes".format(cputime)
                # update GPR model
                pts, N_pts, _ = model.add_structure((best_struc, best_eng, best_forces), tol_e_var=1.2)
                if N_pts > 0:
                    model.set_train_pts(pts, mode="a+")
                    model.fit(show=False)
                    total_pts += N_pts
                bo_count += 1
            else:
                strs += " !!!skipped due to error in vasp calculation"
                fail_count += 1
            print(strs)
        else:
            print("Structure is skipped due to small variance.")
            fail_count += 1

    # Let's do update once a generation
    print("The minimum energy: {:8.3f} eV/atom ".format(min_E)) 
    model.sparsify()
    print(model)

    # some metrics here to quickly show the improvement, e.g?
    # the average uncertainties for low energy structures?
    # the list of best structures (i.e., both low E and E_var structure)?
    # or some metrics from typical BO optimization?

    # Resort the energy based on the updated ranking
    # For the first %20 of structures (with sigma>some value), 
    # we will continue to relax them in the new generation
    # otherwise, the structures will be generated from the scratch
    # This way, we can focus on some true low-energy structures
