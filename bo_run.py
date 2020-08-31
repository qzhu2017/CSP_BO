import os
import numpy as np

from time import time
from scipy.stats import norm
import random
from ase import Atoms
from ase.db import connect
from ase.geometry import Cell
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from cspbo.descriptors.rdf import RDF
from cspbo.interface.pyxtal import PyXtal, Calculator

from spglib import get_symmetry_dataset
from warnings import catch_warnings, simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

def opt_acquisition(descriptors, model, directory, filename, trial=10):
    print(f"Searching for the next best candidate out of {trial} trial structures.")
    sg = "random"
    species = ["C"]
    numIons = [4]
    factor = 1.0
    potential = "tersoff.lib"

    PyXtal(n=trial, sg=sg, species=species, numIons=numIons,
           factor=factor, potential=potential, optimization="single",
           directory=directory, filename=filename, restart=True)

    structures = []
    db = connect(directory+filename)
    for i, row in enumerate(db.select()):
        struc = db.get_atoms(row.id)
        structures.append(struc)
        struc = AseAtomsAdaptor().get_structure(db.get_atoms(row.id))

        _des = RDF(struc, R_max=10).RDF[1]
        
        if i == 0:
            columns = _des.shape[0]
            _descriptors = np.zeros([trial, columns])
            _descriptors[0] = _des
        else:
            _descriptors[i] = _des

    ix = acquisition(descriptors, _descriptors, model, style="Thompson")

    info = get_symmetry_dataset(structures[ix], symprec=1e-1)
    print("  Raw structure information:")
    print("   ", structures[ix])
    print("    space group: {:s}".format(info['international']))

    return structures[ix], _descriptors[ix]

def acquisition(descriptors, _descriptors, model, style="PI"):
    if style == "PI":
        # Probabilistic Improvement
        _e, _ = surrogate(model, descriptors)
        best = min(_e)
        mu, std = surrogate(model, _descriptors)
        mu = mu[:, 0]
        probs = norm.cdf((mu-best)/(np.sqrt(std)+1E-9))
        ix = np.argmax(probs)

    elif style == "Thompson":
        alpha = 1
        mu, cov = surrogate(model, _descriptors, std=False)
        mu_1d = mu.ravel()
        score = np.random.multivariate_normal(mu_1d, cov * alpha ** 2, 1)
        score = score[0, :]
        ix = np.argmax(score)   
        
    return ix
       
def surrogate(model, descriptors, std=True):
    with catch_warnings():
        simplefilter("ignore")
        if std:
            return model.predict(descriptors, return_std=True)
        else:
            return model.predict(descriptors, return_cov=True)

n = 10
sg = "random"
species = ["C"]
numIons = [4]
factor = 1.0
potential = "tersoff.lib"

counts, times = [], []
for a in range(191, 200):
    t0 = time()
    # Random structure search
    print(f"\nRunning random structure search {a}... \n")

    # This will provide initial training data set for BO.
    # The initial training structures will be saved in BayesCSP/initial_structures.db
    # The ASE database is used.
    directory, filename = f"BayesCSP{a}/", "initial_structures.db"
    PyXtal(n, sg=sg, species=species, numIons=numIons, 
           factor=factor, potential=potential, optimization="conp",
           directory=directory, filename=filename)

    ############## Bayesian Optimization Scheme ################

    # Randomly select candidate for BO training

    db = connect(directory+filename)
    total_structures = len(db)
    print(f"Total initial random structures: {total_structures}\n")

    structures, energies = [], []
    for i, row in enumerate(db.select()):
        structure = db.get_atoms(row.id)
        calc = Calculator("GULP", structure, potential, optimization="conp")
        calc.run()
        energies.append([calc.energy/len(structure)])

        opt_structure = AseAtomsAdaptor().get_structure(
                        Atoms(structure.symbols, scaled_positions=calc.positions, cell=calc.cell, pbc=True))
        _des = RDF(opt_structure, R_max=10).RDF[1]
        if i == 0:
            columns = _des.shape[0]
            descriptors = np.zeros([total_structures, columns])
            descriptors[0] = _des
        else:
            descriptors[i] = _des

    # Define the surrogate model
    model = GaussianProcessRegressor()
    model.fit(descriptors, energies)

    count, ground_state = 0, False
    while not ground_state:
        best_structure, best_descriptor = opt_acquisition(descriptors, model, directory=directory, filename=f"trial_{count}.db")
        calc = Calculator("GULP", best_structure, potential, optimization="conp")
        calc.run()
        optimal_energy = calc.energy/len(best_structure)
        optimal_descriptor = RDF(AseAtomsAdaptor().get_structure(Atoms(calc.sites, scaled_positions=calc.positions, cell=calc.cell, pbc=True)), 
                                 R_max=10).RDF[1]
        
        info = get_symmetry_dataset(best_structure, symprec=1e-1)
        print('  Optimized structures information:')
        print("    {:4d} {:8.6f} {:s}".format(count, optimal_energy, info['international']))

        est, _ = surrogate(model, [optimal_descriptor])
        print('    predicted energy = %3f, actual energy = %.3f \n' % (est[0][0], optimal_energy))

        descriptors = np.vstack((descriptors, [optimal_descriptor]))
        energies.append([optimal_energy])

        model.fit(descriptors, energies)
        count += 1

        if info['international'] == 'P6_3/mmc' and round(optimal_energy,5) == -7.39552:
            ground_state = True
            print("Ground state is found.")
            print("{:4d} {:8.6f} {:s}".format(count, optimal_energy, info['international']))
            #counts.append(count)
    t1 = time()
    counts.append(count)
    times.append(t1-t0)
    print(t1-t0, " s")
    print(count, " BO trials")
print(counts)
