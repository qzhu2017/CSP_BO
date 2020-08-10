import os
from warnings import warn
from random import randint

from ase import Atoms
from spglib import get_symmetry_dataset
from pyxtal.interface.gulp import GULP
from pyxtal.crystal import random_crystal


def PyXtal(n, sg="random", species=["C"], numIons=[16], factor=1.0, calculator="GULP", potential="tersoff.lib"):
    """ Parameters """
    file = "POSCARs"
    if os.path.exists(file):
        os.remove(file)

    for i in range(n):
        _sg = spacegroup_generator(sg)
        struc = random_crystal(_sg, species, numIons, factor)
        if struc.valid:
            calc = Calculator(calculator, struc, potential)
            calc.run()
            s = Atoms(struc.sites, scaled_positions=calc.positions, cell=calc.cell)
            info = get_symmetry_dataset(s, symprec=1e-1)
            s.write("1.vasp", format='vasp', vasp5=True, direct=True)
            os.system("cat 1.vasp >> " + file)

            print(calc.energy)
            print("{:4d} {:8.3f} {:s}".format(i, calc.energy, info['international']))
            print("\n")

        else:
            print("Structure is invalid.")
            print("\n")


def Calculator(calculator, structure, potential):
    if calculator == "GULP":
        return GULP(structure, ff=potential)
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
