import os
import sys
import numpy as np
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.calculator import GPR
from ase.build import bulk
np.set_printoptions(formatter={'float': '{: 5.2f}'.format})

N_max = 5
m_file = sys.argv[1]
db_file = sys.argv[2]
model = gpr()
model.load(m_file, N_max=None, opt=False)

# set calculator
calc = GPR(ff=model, return_std=True)

struc = bulk('Si', 'diamond', a=5.0, cubic=True)

struc0 = struc.copy() 
struc0.calc = calc

E = struc0.get_potential_energy()/len(struc0)
E_var = struc0._calc.get_var_e()
E_var_total = struc0._calc.get_var_e(total=True)
vol = struc0.get_volume()/len(struc0)
print("GPR prediction of diamond structure before expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, vol))

# Expand 20%
pos = struc0.get_scaled_positions().copy()
struc0.set_cell(1.063*struc0.cell)
struc0.set_scaled_positions(pos)
E = struc0.get_potential_energy()/len(struc0)
E_var = struc0._calc.get_var_e()
E_var_total = struc0._calc.get_var_e(total=True)
vol = struc0.get_volume()/len(struc0)
#struc0.write("1.vasp", format='vasp', vasp5=True, direct=True)
print("GPR prediction of diamond structure after expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, vol))

struc2 = struc.copy()*(1,1,2)
struc2.calc = calc
E = struc2.get_potential_energy()/len(struc2)
E_var = struc2._calc.get_var_e()
E_var_total = struc2._calc.get_var_e(total=True)
vol = struc2.get_volume()/len(struc2)
print("GPR prediction of diamond structure (2,2,2) before expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, vol))

pos = struc2.get_scaled_positions().copy()
struc2.set_cell(1.063*struc2.cell)
struc2.set_scaled_positions(pos)
E = struc2.get_potential_energy()/len(struc2)
E_var = struc2._calc.get_var_e()
E_var_total = struc2._calc.get_var_e(total=True)
vol = struc2.get_volume()/len(struc2)
#struc2.write("2.vasp", format='vasp', vasp5=True, direct=True)
print("GPR prediction of diamond structure (2,2,2) after expansion ")
print("    E: {:6.3f} eV/atom E_var: {:13.10f} E_var_total: {:13.10f} V: {:6.2f} A^3/atom".format(E, E_var, E_var_total, vol))
