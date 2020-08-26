# Crystal Structure Prediction with Random Search

import os
import numpy as np
from time import time
from ase.db import connect
from spglib import get_symmetry_dataset
import matplotlib.pyplot as plt
from cspbo.interface.pyxtal import PyXtal, Calculator

ground_state = False

N, counts = 200, []

n = 1
sg = "random"
species = ["C"]
numIons = [4]
factor = 1.0
potential = "tersoff.lib"

t0 = time()
for p in range(N):
    count = 0
    print(f"Random search #{p+1}")
    while not ground_state:
        PyXtal(n, sg=sg, species=species, numIons=numIons, 
               factor=factor, potential=potential, optimization="conp",
               verbose=False)

        db = connect("OUTPUTs/PyXtal.db")
        for i, row in enumerate(db.select()):
            structure = db.get_atoms(row.id)
            energy = row.data["energy"]
            info = get_symmetry_dataset(structure, symprec=1e-1)
            structure.write("OUTPUTs/"+f"{i}.vasp", format='vasp', vasp5=True, direct=True)
            print("{:4d} {:8.6f} {:s}".format(count, energy/len(structure), info['international']))
            count += 1

            if info['international'] == 'P6_3/mmc' and round(energy/len(structure),5) == -7.39552:
                ground_state = True
                print(f" Ground State is found at trial #{count}.\n")
    counts.append(count)
    ground_state = False

t1 = time()
print(t1-t0, " s")
print(counts)

plt.hist(counts)
plt.xlabel("Trials")
plt.ylabel("Frequency")
plt.title("Random Crystal Structure Prediction")
plt.savefig("RCSP.png", dpi=300)
