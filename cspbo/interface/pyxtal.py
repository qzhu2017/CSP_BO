import numpy as np
import warnings
from ase.db import connect
from spglib import get_symmetry_dataset
from pyxtal.crystal import random_crystal
from random import choice
from ase import Atoms
warnings.filterwarnings("ignore")

def process(struc, calculator="GULP", potential="reaxff.lib", label=None, filename=None):
    """ 
    database interface for the followings,
        1, optimize the geometry for each structure
        2, write the optimized geometries to the ase database

    Parameters
    ----------
        gen_id: the id for the current calculation (int)
        calculator: string ('GULP' or 'VASP')
        filename: the filename for the structure database

    Return:
        the best structure and energy
    """
    if not isinstance(struc, Atoms):
        init_struc = struc.to_ase()
    s, energy, time = Calculator(calculator, struc, potential)
    spg = get_symmetry_dataset(s, symprec=1e-1)['international']

    #save structures
    if filename is not None:
        with connect(filename) as db:
            db.write(s, data={'energy': energy, 
                              'symmetry': spg,
                              'init_struc': init_struc,
                              'gen_id': label,
                              'time': time,
                              })
        
    return s, energy, time, spg
 

def PyXtal(sgs, species, numIons):
    """ 
    PyXtal interface for the followings,

    Parameters
    ----------
        sg: a list of allowed space groups, e.g., range(2, 231)
        species: a list of chemical species, e.g., ["Na", "Cl"]
        numIons: a list to denote the number of atoms for each speice, [4, 4]
    Return:
        the pyxtal structure
    """
    while True:
        struc = random_crystal(choice(sgs), species, numIons)
        if struc.valid:
            return struc
    
def Calculator(calculator, struc, potential):
    """ The calculator for energy minimization. """
    if calculator == "GULP":
        from pyxtal.interface.gulp import optimize
        if not isinstance(struc, Atoms):
            struc = struc.to_ase()
        return optimize(struc, ff=potential, exe='timeout -k 10 120 gulp')
    else:
        raise NotImplementedError("The package {calculator} is not implemented.")           
    


