# Crystal Structure Prediction with Random Search

import os
import numpy as np
from time import time
from cspbo.descriptors.rdf import RDF
import matplotlib.pyplot as plt
from cspbo.interface.pyxtal import PyXtal, process
import pymatgen.analysis.structure_matcher as sm
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process as gp
from scipy.spatial.distance import cdist
from pyxtal.interface.gulp import GULP
from warnings import catch_warnings, simplefilter

def parse_ref(struc, eng, ref, ref_eng):
    if abs(eng - ref_eng) < 1e-2: 
        s_pmg = AseAtomsAdaptor().get_structure(struc)
        if sm.StructureMatcher().fit(s_pmg, ref_pmg):
            return True

    return False

def opt_acquisition(descriptors, model, sg, species, numIons, ff, trial=5):
    
    # Here we only generate 10 structures, no optimization
    strucs = []
    energies = np.zeros(trial)
    for i in range(trial):
        while True:
            struc = PyXtal(sg, species, numIons)
            calc = GULP(struc, ff=ff, steps=20)
            calc.run()
            if not calc.error:
                break
            else:
                print("GULP is wrong, try another structure")
        pmg_struc = calc.to_pymatgen()
        _des = RDF(pmg_struc, R_max=Rmax).RDF[1]
        strucs.append(struc)
        energies[i] += calc.energy/len(calc.frac_coords)
        if i == 0:
            columns = _des.shape[0]
            _descriptors = np.zeros([trial, columns])
        _descriptors[i] = _des
    #print("True energies:", energies)

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
        alpha = 2.0
        mu, cov = surrogate(model, _descriptors, std=False)
        mu_1d = mu.ravel()
        score = np.random.multivariate_normal(mu_1d, cov * alpha ** 2, 1)
        score = score[0, :]
        ix = np.argmin(score)   
        #print("Predicted:    ", mu_1d)
        #print("Score:        ", score)
        
    return ix
       
def surrogate(model, descriptors, std=True):
    with catch_warnings():
        simplefilter("ignore")
        if std:
            return model.predict(descriptors, return_std=True)
        else:
            return model.predict(descriptors, return_cov=True)



N = 2000
n = 10
sgs = range(1,231)
species = ["Si"]
numIons = [8]
Rmax = 10
ff = "edip_si.lib" #"tersoff.lib"
kernel = gp.kernels.Matern()

#compute the reference ground state
from ase.lattice import bulk
ref = bulk('Si', 'diamond', a=5.459, cubic=True)
#from ase.io import read
#ref = read("0.cif", format='cif')
struc, eng, _, spg, error = process(ref, "GULP", ff) 
if error:
    print("Reference structure cannot be calculated properly")
    sys.exit()
else:
    ref_eng = eng/len(struc)
    ref_pmg = AseAtomsAdaptor().get_structure(struc)
    print("The reference structure is {:8.5f} eV/atom in {:s}".format(ref_eng, spg))

#Random search
t0 = time()
engs_random = []
ground_rand = 0
for count in range(N):
    struc = PyXtal(sgs, species, numIons)
    struc, eng, runtime, spg, error = process(struc, "GULP", ff)
    if not error:
        eng = eng/len(struc)
        engs_random.append(eng)
        strs = "{:4d} {:8.4f} {:8.2f} seconds {:12s} ".format(count, eng, runtime, spg)
        res = parse_ref(struc, eng, ref, ref_eng)
        if res:
            print(strs+'+++++++')
            ground_rand += 1
        else:
            print(strs)
t1 = time()
print("Complete in {:.3f}s".format(t1-t0))


#BO
t0 = time()
engs_bo = []
ground_bo = 0

print(f"\nRunning BO\n")
energies = []
descriptors = []
for i in range(n):
    struc = PyXtal(sgs, species, numIons)
    struc, eng, runtime, spg, error = process(struc, "GULP", ff)
    if not error:
        eng = eng/len(struc)
        strs = "{:4d} {:8.4f} {:8.2f} seconds {:12s} ".format(i, eng, runtime, spg)
        strs += "create GP model"
        res = parse_ref(struc, eng, ref, ref_eng)
        if res:
            print(strs+'+++++++')
            ground_bo += 1
        else:
            print(strs)
            
        pmg_struc = AseAtomsAdaptor().get_structure(struc)
        _des = RDF(pmg_struc, R_max=Rmax).RDF[1]
        descriptors.append(_des)
        energies.append(eng)
        engs_bo.append(eng)

# Define the surrogate model
descriptors = np.array(descriptors)
model = GaussianProcessRegressor(kernel)
model.fit(descriptors, energies)

for i in range(n, N):
    _struc, _ = opt_acquisition(descriptors, model, sgs, species, numIons, ff)
    struc, eng, runtime, spg, error = process(_struc, "GULP", ff) 
    if not error:
        eng = eng/len(struc)
        engs_bo.append(eng)
        pmg_struc = AseAtomsAdaptor().get_structure(struc)
        opt_des = RDF(pmg_struc, R_max=Rmax).RDF[1]
        res = parse_ref(struc, eng, ref, ref_eng)
        est, _ = surrogate(model, [opt_des])
        strs = "{:4d}/{:4d} {:8.4f} -> {:8.4f} {:8.2f} seconds {:12s} ".format(i, len(energies), est[0], eng, runtime, spg)
        if res:
            print(strs+'+++++++')
            ground_bo += 1

        # only update the model if the descriptor is new!
        # dist = cdist([opt_des], descriptors, 'cosine')
        #if len(dist[dist < 0.01]) == 0:
        if abs(est[0]-eng) > 0.1: 
            descriptors = np.vstack((descriptors, opt_des))
            energies.append(eng)
            model.fit(descriptors, energies)
            print(strs + "update GP model")
        else:
            print(strs)
t1 = time()
print("Complete in {:.3f}s".format(t1-t0))

#print(engs_bo)
#print(engs_random)
bins = np.linspace(ref_eng - 0.1, ref_eng + 3.0, 150)
label1 = "Random: {:d}/{:d}".format(ground_rand, len(engs_random))
label2 = "BO: {:d}/{:d}".format(ground_bo, len(engs_bo))
plt.hist(engs_random, bins, alpha=0.5, label=label1)
plt.hist(engs_bo, bins, alpha=0.5, label=label2)
plt.xlabel("Energy (eV/atom)")
plt.ylabel("Frequency")
plt.legend()
plt.title("Crystal Structure Prediction: Random .v.s BO")
plt.savefig("comp.png", dpi=300)
