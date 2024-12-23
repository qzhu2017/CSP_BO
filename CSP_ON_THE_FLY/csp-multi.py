from ase import Atoms
from copy import deepcopy
import numpy as np
import os, sys
from time import time
from scipy.stats import norm
from spglib import get_symmetry_dataset
from random import choice

# setup the actual command to run vasp
# define the walltime and number of cpus
ncpu = 16
cmd = "mpirun -np " + str(ncpu) + " vasp_std"
logfile = 'opt.log2'  #ase dyn log file
sym_tol = 1e-3

# parallel processing
from multiprocessing import current_process, Pool
from functools import partial

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

def opt_struc(struc, calcs, sgs, species, numIons):
    """
    Prepare and perform the structural relaxation for each individual

    Args:
        struc: ase structure
        calcs: [raw, high] calculator
        sgs: 
        species:
        numIons:
    """
    t0 = time()
    if struc is None:
        if isinstance(numIons, list):
            numIons = [choice(numIons)]
        else:
            numIons = [numIons]
        struc = PyXtal(sgs, species, numIons, conventional=False) 
        struc.relax = "normal"

    calc1 = calcs[0]
    calc2 = calcs[1]
    steps = [0, 0]
    if struc.relax == "normal":
        steps = [100, 25]
    elif struc.relax == "medium":
        # single point by DFT
        steps = [10, 5]
        calc1 = calcs[1]
    elif struc.relax == "Light":
        # already relaxed by DFT
        steps = [3, 1]
        calc1 = calcs[1]

    # fix cell opt
    struc.set_calculator(calc1) # set calculator
    if steps[0]>0:
        struc.set_constraint(FixSymmetry(struc))
        dyn = FIRE(struc, logfile=logfile)
        dyn.run(fmax=0.01, steps=steps[0])
        #print("Fix cell", time()-t0)

    # variable cell opt
    struc.set_calculator(calc2) # set calculator
    if steps[1]>0:
        ecf = ExpCellFilter(struc)
        dyn = FIRE(ecf, logfile=logfile)
        dyn.run(fmax=0.01, steps=steps[1])
        #print("var cell", time()-t0)

    fmax = np.max(np.abs(struc.get_forces()).flatten())
    smax = np.max(np.abs(struc.get_stress()).flatten())
    if max([fmax, smax])<0.25:
        dyn.run(fmax=0.01, steps=25)
    ## symmetrize the final struc, 
    # useful for later dft calculation
    # also save some space for force points in GPR
    # struc.set_calculator(calc) # set calculator
    struc = symmetrize(struc)
    cpu_time = (time() - t0)/60
    return (struc, cpu_time)


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
            kvp = {'dft_energy': eng/len(force),
                   'dft_fmax': np.max(np.abs(force.flatten())),
                  }
            db.write(struc, data=d1, key_value_pairs=kvp)

def add_structures(data, db_file):
    """
    Backup the structure data from the entire simulation
    """
    with connect(db_file) as db:
        struc, eng, spg, e_var, gen = data
        kvp = {'tag': 'good_structures',
               'dft_energy': eng,
               'spg': spg,
               'e_var': e_var,
               'gen': gen,
             }
        db.write(struc, key_value_pairs=kvp)

def collect_data(gpr_model, data, structures):
    """ Collect data for GPR.predict. """
    for i, struc in enumerate(structures):
        #energy, force = energies[i], forces[i]
        energy, force = struc.get_potential_energy(), struc.get_forces()
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

def BO_select(model, data, structures, min_E=None, alpha=0.5, zeta=0.01, style='Thompson'):
    """ Return the index of the trial structures. """
    if style == 'Thompson':
        mean, cov = model.predict(data, total_E=True, stress=False, return_cov=True) # cov is in eV^2/atom^2
    else:
        mean, std = model.predict(data, total_E=True, stress=False, return_std=True) # std is in eV/atom

    if model.base_potential is not None:
        for i, struc in enumerate(structures):
            energy_off, _, _ = model.compute_base_potential(struc)
            mean[i] += energy_off
            mean[i] /= len(struc)

    if style == 'Thompson':
        samples = np.random.multivariate_normal(mean, cov * alpha ** 2, 1)[0,:]
    
    elif style == 'EI': # Expected Improvement
        if min_E is None:
            msg = "PI style needs to know the minimum energy"
            return ValueError(msg)
        
        tmp1 = mean - min_E + zeta
        tmp2 = tmp1 / std
        samples = tmp1 * norm.cdf(tmp2) + std * norm.pdf(tmp2)

    elif style == 'PI': # Probability of Improvement
        if min_E is None:
            msg = "PI style needs to know the minimum energy"
            return ValueError(msg)

        if zeta < 0:
            msg = "Please insert positive number for zeta."
            return ValueError(msg)
        
        samples = norm.cdf((mean-min_E)/(std+1E-9))
    
    elif style== 'LCB':
        if zeta < 0:
            msg = "Please insert positive number for zeta."
            return ValueError(msg)
        
        samples = mean - zeta * std

    else:
        msg = "The acquisition function style is not equipped."
        raise NotImplementedError(msg)
    
    indices = np.argsort(samples)
    return indices

#--------- DFT related ------------------

from ase.calculators.vasp import Vasp

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
        #print("VASP calculation is wrong!")
        #os.system(os.environ["VASP_COMMAND"])
        eng = None
        forces = None
        #import sys; sys.exit()
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
                'encut': 500,
                'ediff': 1e-4,
                'nsw': 0,
                #'symprec': 1e-8,
                #'isym': 0,
                }
    else:
        para = {'prec': 'accurate',
                'encut': 400,
                'isif': 3,
                'ediff': 1e-4,
                'nsw': 20, # we don't need to fully relax it
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
calc_folder = 'vasp_B'
if not os.path.exists(calc_folder):
    os.makedirs(calc_folder)
species = ["B"]

# create the model
from cspbo.gaussianprocess import GaussianProcess as gpr
if os.path.exists("models/test.json"):
    model = gpr()
    model.load('models/test.json', opt=True)
    Es = []
    with connect('models/test.db') as db:
        for row in db.select():
            atoms = db.get_atoms(id=row.id)
            energy = row.data.energy
            if model.base_potential is not None:  
                energy_off, _, _ = model.compute_base_potential(atoms)
                energy += energy_off
            Es.append(energy/len(atoms))
    min_E = min(Es)
else:
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
            eng, forces = dft_run(struc, path=calc_folder, max_time=4)
            struc.set_calculator(set_vasp('single', 0.20))
            eng, forces = dft_run(struc, path=calc_folder)
            data.append((struc.copy(), eng, forces))
            #deformation to learn the energy basin
            for fac in [0.85, 0.90, 0.95, 1.05, 1.1, 1.15]:   
                struc1 = struc.copy()
                pos = struc1.get_scaled_positions().copy()
                struc1.set_cell(fac*struc.cell)
                struc1.set_scaled_positions(pos)
                struc1.set_calculator(set_vasp('single', 0.20))
                eng, forces = dft_run(struc1, path=calc_folder)
                data.append((struc1.copy(), eng, forces))
    
        add_GP_train(data, train_db)
    else:
        with connect(train_db) as db:
            for row in db.select():
                struc = db.get_atoms(row.id)
                data.append((struc, row.data['dft_energy'], row.data['dft_forces'].copy()))

    # The current minimum energy from DFT calc
    min_E = min([d[1]/len(d[0]) for d in data])

    ##----------- Fit the GP model
    from cspbo.kernels.RBF_mb import RBF_mb
    from cspbo.utilities import build_desc
    
    des = build_desc("SO3", lmax=3, nmax=3, rcut=4.0)
    kernel = RBF_mb(para=[1, 0.5])
    model = gpr(kernel=kernel, 
                descriptor=des, 
                base_potential=lj,
                noise_e=[1e-2, 1e-2, 5e-2], 
                f_coef=20)
    for d in data:
        pts, N_pts, error = model.add_structure(d)
        if N_pts > 0:
            model.set_train_pts(pts, mode="a+")
            model.fit(show=False)

print("\nThe minimum energy in DFT is {:6.3f} eV/atom".format(min_E))
print(model)
#model.save('1.json', '1.db')
#import sys; sys.exit()
#print(model.base_potential)
## ----------- Structure generation/optimization based on the surrogate model

from cspbo.utilities import PyXtal
from cspbo.calculator import GPR
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
#from pyxtal import pyxtal
#sgs = range(16, 231)
sgs = [166]
#numIons = [4, 6, 8, 9, 10, 12]
if len(sys.argv) >1:
    numIons = int(sys.argv[1])
    if len(sys.argv) >2:
        # 16-231
        strs = sys.argv[2]
        if strs.find('-') > -1:
            tmp = strs.split('-')
            sgs = range(int(tmp[0]), int(tmp[1]))
        elif strs.find(',') > -1:
            tmp = strs.split(',')
            sgs = [int(t) for t in tmp]
        else:
            sgs = [int(strs)]
    else:
        sgs = range(16, 231)
else:
    numIons = 12

print("sgs", sgs)
print("numIons", numIons)

gen_max = 10
N_pop = 32
alpha, zeta = 2, 0.1
n_bo_select = max([1,N_pop//10])
BO_style = 'LCB' #'EI' #'Thompson'

Current_data = {"struc": [None] * N_pop,
                "E": 100000*np.ones(N_pop),
                "fmax": 100000*np.ones(N_pop),
                "smax": 100000*np.ones(N_pop),
                "E_var": np.zeros(N_pop),
                "E_DFT": [None] *N_pop,
                "spg": ['P1'] *N_pop,
                "relax": ["normal"] *N_pop,
                "survival": [0] *N_pop,
               }
calc_raw =  GPR(ff=model, stress=True, f_tol=1e-0)
calc_high = GPR(ff=model, stress=True, f_tol=1e-4)
calc_std =  GPR(ff=model, stress=True, return_std=True, f_tol=1e-6)

for gen in range(gen_max):

    print("\n================ GPR Relaxation ===============")
    total_gpr_time = 0
    t0_gen = time()
    if ncpu > 1:
        with Pool(ncpu) as p:
            func = partial(opt_struc, calcs=[calc_raw, calc_high], sgs=sgs, species=species, numIons=numIons)
            res = p.map(func, Current_data['struc'])
    else:
        res = []
        for i, struc in enumerate(Current_data['struc']):
            print(i)
            res.append(opt_struc(struc, [calc_raw, calc_high], sgs, species, numIons))
    total_gpr_time += (time()-t0_gen)/60
    os.remove(logfile) # remove unnecessary logfile

    # unpack the results
    structures = []
    ids = []
    
    for pop, d in enumerate(res):
        (struc, cputime) = d
        struc.set_calculator(calc_std)
        E = struc.get_potential_energy()/len(struc) #per atom
        E_var = struc._calc.get_var_e()
        fmax = np.max(np.abs(struc.get_forces()).flatten())
        smax = np.max(np.abs(struc.get_stress()).flatten())
        vol = struc.get_volume()/len(struc)
        try:
            spg = get_symmetry_dataset(struc, symprec=sym_tol)['international']
        except:
            spg = 'N/A'
        formula = struc.get_chemical_formula()
        strs = "{:3d} {:2d}[{:2d}] {:4s} {:9s} {:8.3f}[{:6.3f}] {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:6s}".format( 
        gen, pop, Current_data["survival"][pop], formula, spg, E, E_var, vol, fmax, smax, cputime, Current_data['relax'][pop])

        if Current_data['E_DFT'][pop] is not None:
            strs += "{:8.3f}".format(Current_data['E_DFT'][pop])

        if vol > 50:
            strs += " discarded"
            Current_data["struc"][pop] = None
            Current_data['relax'][pop] = "normal"
            Current_data['survival'][pop] = 0
        else: 
            if new_struc(struc, structures) is None:        
                structures.append(struc)
                ids.append(pop)
                Current_data["struc"][pop] = struc
                Current_data["E"][pop] = E
                Current_data["fmax"][pop] = fmax
                Current_data["smax"][pop] = smax
                Current_data["E_var"][pop] = E_var
                Current_data['survival'][pop] += 1
            else:
                strs += " duplicate"
                Current_data["struc"][pop] = None
                Current_data['E_DFT'][pop] = None
                Current_data['relax'][pop] = "normal"
                Current_data['survival'][pop] = 0

        Current_data['spg'][pop] = spg

        if E < min_E:
            strs += ' +++++'
        print(strs)

    print("\n================ BO selection ================")
    t0 = time()
    data = {'energy': [], 'force': []}
    data = collect_data(model, data, structures)
    indices = BO_select(model, data, structures, min_E, alpha=alpha, style=BO_style)
    total_gpr_time += (time()-t0)/60

    total_dft_pts = 0
    total_dft_time = 0
    add_dft_pts = []
    for ix in indices[:n_bo_select]:
        struc = structures[ix]
        formula = struc.get_chemical_formula()
        E = struc._calc.get_e()
        E_var = struc._calc.get_var_e()
        N_pts = None
        if E_var > model.noise_e/2:
            spg = Current_data['spg'][ids[ix]]
            struc.set_constraint()
            struc.set_calculator(set_vasp('single', 0.20))
            strs = "{:2d}[{:8s} {:9s}]: {:7.3f}[{:7.3f}]".format(ids[ix], formula, spg, E, E_var)

            # perform single point DFT
            t0 = time()
            eng, forces = dft_run(struc, path=calc_folder)
            cputime = (time() - t0)/60
            total_dft_time += cputime
            # sometimes the vasp calculation will fail
            if eng is not None:
                E = eng/len(struc)
                strs += " -> DFT energy: {:6.3f} eV/atom ".format(E)
                strs += "in {:5.2f} minutes".format(cputime)
                # update GPR model
                if E > min_E + 1.0:
                    add_force = False
                else:
                    add_force = True
                pts, N_pts, _ = model.add_structure((struc.copy(), eng, forces), 
                                            tol_e_var=1.2, add_force=add_force)
                if N_pts > 0:
                    add_dft_pts.append(pts)
                    total_dft_pts += N_pts
                Current_data['E_DFT'][ids[ix]] = E
                Current_data['relax'][ids[ix]] = "medium"
            else:
                E = 100000
                strs += " !!!skipped due to error in vasp calculation"
            print(strs)
            #print(struc)
            #print(struc.get_scaled_positions())

            if N_pts is not None and E < min_E + 0.75:
                strs = "Switch to DFT relaxation, energy: "
                t0 = time()
                struc.set_calculator(set_vasp('opt', 0.3))
                eng, forces = dft_run(struc, path=calc_folder, max_time=10)
                struc.set_calculator(set_vasp('single', 0.20))
                eng, forces = dft_run(struc, path=calc_folder)
                E = eng/len(struc)
                #update data
                Current_data['E_DFT'][ids[ix]] = E
                Current_data['relax'][ids[ix]] = "light"
                pts, N_pts, _ = model.add_structure((struc.copy(), eng, forces))
                cputime = (time() - t0)/60
                total_dft_time += cputime
                #print(struc)
                #print(struc.get_scaled_positions())
                if N_pts > 0:
                    add_dft_pts.append(pts)
                    total_dft_pts += N_pts
                    #deformation to learn the energy basin
                    for fac in [0.80, 0.90, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2]:   
                        t0 = time()
                        struc1 = struc.copy()
                        pos = struc1.get_scaled_positions().copy()
                        struc1.set_cell(fac*struc.cell)
                        struc1.set_scaled_positions(pos)
                        struc1.set_calculator(set_vasp('single', 0.20))
                        eng, forces = dft_run(struc1, path=calc_folder)
                        pts, N_pts, _ = model.add_structure((struc1.copy(), eng, forces))
                        cputime = (time() - t0)/60
                        total_dft_time += cputime
                        print("Deformation: {:6.2f}, DFT {:7.3f} eV/atom in {:5.2f} minutes".format(fac, eng/len(struc1), cputime))
                        #print(struc1)
                        #print(struc1.get_scaled_positions())
                        if N_pts > 0:
                            add_dft_pts.append(deepcopy(pts))
                            total_dft_pts += deepcopy(N_pts)

                    strs += "{:8.3f} eV/atom in {:5.2f} minutes".format(E, cputime)
                else:
                    Current_data['relax'][ids[ix]] = "freeze"
                    strs += ' freeze, no update'
                print(strs)

                if E < min_E:
                    min_E = E
                Current_data['struc'][ids[ix]] = struc
            #strs += "{:8.3f}".format(Current_data['E_DFT'][pop])

        # update energy and make sure E-var is large
        #Current_data["E_var"][ids[ix]] = model.noise_e*1.5 
        Current_data['E'][ids[ix]] = E

    print("\n================ Postprocessing ===============")
    for pop in range(N_pop):
        struc = Current_data['struc'][pop]
        e_var = Current_data['E_var'][pop]
        eng = Current_data['E'][pop]
        fmax = Current_data['fmax'][pop]
        smax = Current_data['smax'][pop]
        spg = Current_data['spg'][pop]
        if struc is not None:
            deposit = False
            reset = False
            dft_calc = False
            formula = struc.get_chemical_formula()
            strs = "{:2d}[{:6s} {:9s}] {:7.3f}[{:7.3f}] {:7.3f} {:7.3f}".format(\
            pop, formula, spg, eng, e_var, fmax, smax)

            if fmax<5e-2 and smax<2e-2:
                if e_var < 1.2*model.noise_e:
                    deposit = True
                    reset = True
                    print(strs, " Deposited")
                else:
                    if eng < min_E + 2.5:
                        dft_calc = True
                    else:
                        reset = True
                        print(strs, " discard due to high energy")
                    
            else:
                if e_var > 10:
                    Current_data['struc'][pop].relax = "light"
                elif e_var > 5:
                    Current_data['struc'][pop].relax = "medium"
                else:
                    Current_data['struc'][pop].relax = Current_data['relax'][pop]

            if dft_calc:
                t0 = time()
                struc.set_constraint()
                struc.set_calculator(set_vasp('single', 0.20))
                eng, forces = dft_run(struc, path=calc_folder)
                cputime = (time() - t0)/60
                total_dft_time += cputime
                # sometimes the vasp calculation will fail
                if eng is not None:
                    E = eng/len(struc)
                    strs += " -> DFT energy: {:6.3f} eV/atom ".format(E)
                    strs += "in {:5.2f} minutes".format(cputime)
                    if E > min_E + 0.5:
                        add_force = False
                    else:
                        add_force = True
                    pts, N_pts, _ = model.add_structure((struc.copy(), eng, forces), add_force=add_force)
                    if N_pts > 0:
                        add_dft_pts.append(pts)
                        total_dft_pts += N_pts
                    Current_data['E_DFT'][pop] = E
                    Current_data['relax'][pop] = "medium"
                    Current_data['struc'][pop].relax = "medium"
                    print(strs)
                else:
                    E = 100000
                    strs += " !!!skipped due to error in vasp calculation"
                    print(strs)

            if deposit:
                struc.set_calculator()
                struc.set_constraint()
                add_structures((struc, eng, spg, e_var, gen), 'all.db')

            if reset:
                Current_data['struc'][pop] = None
                Current_data['relax'][pop] = "normal"
                Current_data['E_DFT'][pop] = None
                Current_data['survival'][pop] = 0


    if total_dft_pts > 0:
        for pt in add_dft_pts:
            model.set_train_pts(pt, mode="a+")
        if gen % 2 == 0:
            model.fit(show=False)
            model.sparsify(e_tol=1e-5, f_tol=1e-2)
        else: #we don't update the model too frequently since it becomes expansive
            model.fit(show=False, opt=False)
        model.save("models/test.json", "models/test.db")
        print(model)
    else:
        print("No updates on the GP model in gen {:d}".format(gen))
    
    print("================ Summary ===================")
    print("Total GPR time {:6.3f} minutes in gen {:d}".format(total_gpr_time, gen))
    print("Total DFT time {:6.3f} minutes in gen {:d}".format(total_dft_time, gen))
    print("Total Run time {:6.3f} minutes in gen {:d}".format((time()-t0_gen)/60, gen))
    print("The minimum DFT energy is {:6.3f} eV/atom in gen {:d}".format(min_E, gen))

