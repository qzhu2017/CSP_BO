# Crystal Structure Prediction with Random Search

import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
from cspbo.interface.pyxtal import PyXtal, process
import pymatgen.analysis.structure_matcher as sm
from pymatgen.io.ase import AseAtomsAdaptor

N, counts = 200, []
sgs = range(1,231)
species = ["Si"]
numIons = [16]
ff = "edip_si.lib" #"tersoff.lib"

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
for p in range(N):
    print(f"Random search #{p+1}")
    count = 0
    while True:
        count += 1
        struc = PyXtal(sgs, species, numIons)
        struc, eng, runtime, spg, error = process(struc, "GULP", ff, p, filename='random.db')
        if not error:
            print("{:4d} {:8.5f} {:8.2f} seconds {:s}".format(count, eng/len(struc), runtime, spg))
            if abs(eng/len(struc) - ref_eng) < 1e-2: 
                s_pmg = AseAtomsAdaptor().get_structure(struc)
                if sm.StructureMatcher().fit(s_pmg, ref_pmg):
                    print("Ground State is found at trial {}\n".format(count))
                    break

    counts.append(count)

t1 = time()
print(t1-t0, " s")
print(counts)

plt.hist(counts, bins=50)
plt.xlabel("Trials")
plt.ylabel("Frequency")
plt.title("Random Crystal Structure Prediction")
plt.savefig("RCSP.png", dpi=300)
