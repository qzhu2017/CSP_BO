import numpy as np
from ase import units
from optparse import OptionParser
from ase.build import bulk
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb
from cspbo.utilities import metric_single, build_desc, get_data
from cspbo.calculator import GPR

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE

from time import time

def printenergy(a, t):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    var_e = a._calc.get_var_e()
    var_f = np.max(a._calc.get_var_f())
    print('Step: %4d: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV E_var = %.3feV F_var = %.3feV/A' % (\
          t, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, var_e, var_f))

parser = OptionParser()
parser.add_option("-f", "--file", dest="file",
                  help="dbfile, REQUIRED",
                  metavar="file")

(options, args) = parser.parse_args()
des = build_desc("SO3", lmax=3, nmax=3, rcut=4.0)
print(des)

N_start, zeta = 50, 2

kernel = RBF_mb(para=[1, 0.5], zeta=zeta, ncpu=1)
model = gpr(kernel=kernel, noise_e=[0.02, 0.01, 0.1], f_coef=30)

db_file = options.file
train_data = get_data(db_file, des, N_force=5, lists=range(0,N_start), select=True)

model.fit(train_data)
E, E1, F, F1 = model.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
print("\n")

calc = GPR(ff=model, descriptor=des)
si = bulk('Si', 'diamond', a=5.459, cubic=True)
si = si*2
si.set_calculator(calc)
pos = si.positions
pos[0] += 0.1
si.set_positions(pos)

print(si.get_potential_energy())
dyn = FIRE(si)
dyn.run(fmax=0.05, steps=10)

MaxwellBoltzmannDistribution(si, 300*units.kB)
dyn = VelocityVerlet(si, 1*units.fs)  # 2 fs time step.
t0 = time()
for i in range(10):
    dyn.run(steps=1)
    printenergy(si, i)
    print(time()-t0)

