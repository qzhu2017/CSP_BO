import os
import numpy as np

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

# Random structure search
print("\nRunning random structure search... \n")

n = 10
sg = "random"
species = ["C"]
numIons = [4]
factor = 1.0
potential = "tersoff.lib"

# This will provide initial training data set for BO.
# The initial training structures will be saved in BayesCSP/initial_structures.db
# The ASE database is used.
directory, filename = "BayesCSP/", "initial_structures.db"
PyXtal(n, sg=sg, species=species, numIons=numIons, 
       factor=factor, potential=potential, optimization="conp",
       directory=directory, filename=filename)

############## Bayesian Optimization Scheme ################

# Note:
# To get structure: db.get_atoms(unique_id=ids[0]))
# To get energy: db.get(unique_id=ids[0]).data["initial_energy"]

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

    #scores = acquisition(descriptors, _descriptors, model)
    #scores = acquisition(descriptors, _descriptors, model, style="Thompson")
    #ix = np.argmax(scores)

    #ix = acquisition(descriptors, _descriptors, model)
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
        #print(mu, min(mu), best)
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

count, counts, ground_state = 0, [], False
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
        counts.append(count)

print(count)
            
    

#ids, selected_id = [], []

#for r, row in enumerate(db.select()):
#    ids.append(row["unique_id"])

#N_initial_selection = 50
#rng = np.random.default_rng()
#random_initial_selection = rng.choice(len(db), size=[N_initial_selection], replace=False)

#structures, energies = [], []
#for i, r in enumerate(random_initial_selection):
#    struc = db.get_atoms(unique_id=ids[r])
#    calc = Calculator("GULP", struc, "tersoff.lib", optimization='conp')
#    calc.run()
#    
#    energies.append([calc.energy/len(struc)])
#    opt_struc = AseAtomsAdaptor().get_structure(
#                Atoms(struc.symbols, scaled_positions=calc.positions, cell=calc.cell, pbc=True))
#
#    _des = RDF(opt_struc, R_max=10).RDF[1]
#    if i == 0:
#        columns = _des.shape[0]
#        descriptors = np.zeros([len(random_initial_selection), columns])
#        descriptors[0] = _des
#    else:
#        descriptors[i] = _des
#
#    selected_id.append(r)
#
#print(f"Initial random energy: {energies[np.argmin(energies)][0]}")
#
## up to this point
## we have descriptors act as the initial training data set.
## we save the ids for the selected structures.
#
## Define the surrogate model
#model = GaussianProcessRegressor()
#model.fit(descriptors, energies)
#for j in range(200):
#    best_des, best_id, selected_id = opt_acquisition(descriptors, model, selected_id)
#
#    # optimize the structure belonging to the best descriptors
#    struc = db.get_atoms(unique_id=ids[best_id])
#    calc = Calculator("GULP", struc, "tersoff.lib", optimization='conp')
#    calc.run()
#    actual_energy = calc.energy/len(struc)
#
#    opt_struc = AseAtomsAdaptor().get_structure(
#                Atoms(struc.symbols, scaled_positions=calc.positions, cell=calc.cell, pbc=True))
#    opt_des = RDF(opt_struc, R_max=10).RDF[1]
#
#    est, _ = surrogate(model, [opt_des])
#    print('predicted=%3f, actual=%.3f' % (est[0][0], actual_energy))
#    
#    descriptors = np.vstack((descriptors, [opt_des]))
#    energies.append([actual_energy])
#    model.fit(descriptors, energies)
#
#ix = np.argmin(energies)
#print(f"Best energies: {energies[ix][0]}")

# Generate random numbers
#rans, count = [], 0
#while count < trial:
#    if len(selected_id) == total_structures:
#        break
#    ran = random.randrange(total_structures)
#    if ran not in selected_id:
#        selected_id.append(ran)
#        rans.append(ran)
#        count += 1
