import numpy as np
from optparse import OptionParser
from ase.build import bulk
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.calculator import GPR
from ase.optimize import BFGS, FIRE
from cspbo.mushybox import mushybox
from cspbo.utilities import rmse, metric_single, get_strucs, plot


parser = OptionParser()
parser.add_option("-f", "--file", dest="file",
                  help="gp model file, REQUIRED",
                  metavar="file")
parser.add_option("-n", "--ncpu", dest="ncpu", default=4, type=int,
                  help="gp model file, REQUIRED",
                  metavar="ncpu")


(options, args) = parser.parse_args()

model = gpr()
model.load(options.file, N_max=None)
model.kernel.ncpu = options.ncpu

train_E, train_E1, train_F, train_F1, _, _ = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 


calc = GPR(ff=model, stress=True, return_std=False)
si = bulk('Si', 'diamond', a=5.459, cubic=True)

# --------------------------- Example of Geometry Optimization
si.set_calculator(calc)
pos = si.positions
pos[0] += 0.1
si.set_positions(pos)

print(si.get_potential_energy())
box = mushybox(si)
dyn = FIRE(box)
dyn.run(fmax=0.05, steps=50)

print(si.get_potential_energy())


# --------------------------- Example of MD
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from time import time

def printenergy(a, it, t0):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    var_e = a._calc.get_var_e()
    var_f = np.max(a._calc.get_var_f())
    t_now = time()
    print('Step: %4d [%6.2f]: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV E_var = %.3feV F_var = %.3feV/A' % (\
          it, t_now-t0, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, var_e, var_f))
    return t_now


calc = GPR(ff=model, stress=False, return_std=True)
si = si*2
si.set_calculator(calc)

MaxwellBoltzmannDistribution(si, 300*units.kB)
dyn = VelocityVerlet(si, 1*units.fs)  # 2 fs time step.
t0 = time()
for i in range(10):
    dyn.run(steps=1)
    t_now = printenergy(si, i, t0)
    t0 = t_now

