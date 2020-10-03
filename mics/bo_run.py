import os
import numpy as np
from time import time
from scipy.stats import norm
from pymatgen.io.ase import AseAtomsAdaptor
from cspbo.descriptors.rdf import RDF
from cspbo.interface.pyxtal import PyXtal, process
from warnings import catch_warnings, simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process as gp
import pymatgen.analysis.structure_matcher as sm

def opt_acquisition(descriptors, model, sg, species, numIons, trial=10):
    
    # Here we only generate 10 structures, no optimization
    strucs = []
    for i in range(trial):
        struc = PyXtal(sg, species, numIons)
        pmg_struc = struc.to_pymatgen()
        _des = RDF(pmg_struc, R_max=Rmax).RDF[1]
        strucs.append(struc)
        if i == 0:
            columns = _des.shape[0]
            _descriptors = np.zeros([trial, columns])
        _descriptors[i] = _des

    ix = acquisition(descriptors, _descriptors, model, style="Thompson")

    return strucs[ix], _descriptors[ix]

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
        alpha = 0 #1
        mu, cov = surrogate(model, _descriptors, std=False)
        mu_1d = mu.ravel()
        print(mu_1d)
        score = np.random.multivariate_normal(mu_1d, cov * alpha ** 2, 1)
        print(score)
        alpha = 1
        score = np.random.multivariate_normal(mu_1d, cov * alpha ** 2, 1)
        print(score)
        import sys
        sys.exit()
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
N = 100 # number of BO runs
Rmax = 10
sgs = range(1,231)
species = ["Si"]
numIons = [8]
ff = "edip_si.lib" 

#compute the reference ground state
from ase.lattice import bulk
ref = bulk(species[0], 'diamond', a=5.459, cubic=True)
struc, eng, _, spg = process(ref, "GULP", ff) 
ref_eng = eng/len(struc)
ref_pmg = AseAtomsAdaptor().get_structure(struc)
print("The reference structure is {:8.5f} eV/atom in {:s}".format(ref_eng, spg))

kernel = gp.kernels.Matern()

counts, times = [], []
for p in range(N):
    t0 = time()
    # Random structure search to build the first surrogate model
    print(f"\nRunning random structure search {p}... \n")
    energies = []
    for i in range(n):
        struc = PyXtal(sgs, species, numIons)
        struc, eng, _, spg = process(struc, "GULP", ff, p, "bo.db")
        pmg_struc = AseAtomsAdaptor().get_structure(struc)
        _des = RDF(pmg_struc, R_max=Rmax).RDF[1]

        print("{:4d} {:12s} {:8.5f}".format(i, spg, eng/len(struc)))
        if i == 0:
            columns = _des.shape[0]
            descriptors = np.zeros([n, columns])

        descriptors[i] = _des
        energies.append(eng/len(struc))

    # Define the surrogate model
    model = GaussianProcessRegressor(kernel)
    model.fit(descriptors, energies)

    count = 0
    while True:
        _struc, _ = opt_acquisition(descriptors, model, sgs, species, numIons)
        struc, eng, _, spg = process(_struc, "GULP", ff, p, "bo.db") 
        opt_eng = eng/len(struc)
        pmg_struc = AseAtomsAdaptor().get_structure(struc)
        opt_des = RDF(pmg_struc, R_max=Rmax).RDF[1]
        est, _ = surrogate(model, [opt_des])
        print("{:4d} {:12s} {:8.5f} -> {:8.5f} {:8.5f} ".format(count, spg, est[0], opt_eng, opt_eng-est[0]))

        descriptors = np.vstack((descriptors, opt_des))
        energies.append(opt_eng)
        model.fit(descriptors, energies)
        count += 1

        if abs(eng/len(struc) - ref_eng) < 1e-2: 
            s_pmg = AseAtomsAdaptor().get_structure(struc)
            if sm.StructureMatcher().fit(s_pmg, ref_pmg):
                print("Ground State is found at trial {}\n".format(count))
                break

    t1 = time()
    counts.append(count)
    times.append(t1-t0)
    print("{:.2f} s in {:d} BO trials".format(t1-t0, count))
print(counts)
