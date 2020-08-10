from cspbo.interface.pyxtal import PyXtal

n = 10
sg = "random"
species = ["C"]
numIons = [16]
factor = 1.0
potential = "tersoff.lib"

PyXtal(n, sg=sg, species=species, numIons=numIons, factor=factor,
        potential=potential)
