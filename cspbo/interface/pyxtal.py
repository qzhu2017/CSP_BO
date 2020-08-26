import os
import shutil
import numpy as np
import warnings
from warnings import warn
from random import randint

from ase import Atoms
from ase.db import connect
from spglib import get_symmetry_dataset
from pyxtal.interface.gulp import GULP
from pyxtal.crystal import random_crystal
warnings.filterwarnings("ignore")


def PyXtal(n, sg="random", species=["C"], numIons=[16], factor=1.0, calculator="GULP",
           potential="tersoff.lib", optimization="conp", directory="OUTPUTs/",
           filename="PyXtal.db", restart=False, verbose=True):
    """ Parameters """
    if not restart:
        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)
            os.mkdir(directory)
        else:
            os.mkdir(directory)
    db = connect(directory+filename)

    count, energies = 0, []
    while count < n:
        _sg = spacegroup_generator(sg)
        struc = random_crystal(_sg, species, numIons, factor)

        if struc.valid:
            calc = Calculator(calculator, struc, potential, optimization=optimization)
            calc.run()

            if optimization == "single":
                s = struc.to_ase()
            else:
                s = Atoms(calc.sites, scaled_positions=calc.positions, cell=calc.cell, pbc=True)
            #print(s)
            info = get_symmetry_dataset(s, symprec=1e-1)
            energy = calc.energy

            db.write(s, data={'energy': energy})
            count += 1
            if verbose:
                print("{:4d} {:8.5f} {:s}".format(count, energy/len(s), info['international']))

        #else:
        #    print("Structure is invalid.")

    
def Calculator(calculator, structure, potential, optimization='conp'):
    """ The calculator for energy minimization. """
    if calculator == "GULP":
        return GULP(structure, ff=potential, opt=optimization, dump='struc.cif')
    else:
        raise NotImplementedError("The package {calculator} is not implemented.")           
    

def spacegroup_generator(sg):
    """ Generate a random space group for a random crystal structure.

    Parameters
    ----------
    sg: str, list of int
        Users have 2 ways to generate a space group for a crystal structure.
        - "random": 
            Randomly generate space group from 2 to 230.
        - a list of int:
            Randomly pick a member from the list.
            Each member of the list needs to be in between 1 and 231.

    Returns
    -------
    spacegroup: int
        The randomly generated space group.
    """
    if isinstance(sg, str):
        assert sg == "random", \
        "sg takes only 'random' as the input value for string type."
        spacegroup = randint(2, 230)
    
    elif isinstance(sg, list):
        length = len(sg)
        temp = randint(0, length-1) # randint includes end
        if 1 < sg[temp] <= 230:
            spacegroup = sg[temp]
        else:
            msg = "A member in space group list is not in between 1 and 231."
            raise NotImplementedError(msg)
    
    else:
        msg = "user-defined space group is not recognized. " + \
              "Random space group will be generated instead."
        warn(msg)
        spacegroup = randint(2, 230)
    
    return spacegroup
