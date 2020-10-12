from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.calculators.emt import EMT
from ase.db import connect
from cspbo.utilities import build_desc, convert_struc, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb
import os
import numpy as np
from cspbo.utilities import metric_single, plot
from time import time

def save_db(db_filename, strucs):
    if os.path.exists(db_filename):
        os.remove(db_filename)
    with connect(db_filename) as db:
        for s in strucs:
            #s.calc = LennardJones()
            s.calc = EMT()
            data = {"energy": s.get_potential_energy(),
                    "force": s.get_forces(),
                  }
            db.write(s, data=data)

def get_data(db_filename, des, Nmax=None, lists=None, ncpu=8):
    """
    Nmax: Maximum number of force data
    """
    X, Y = convert_struc(db_name, des, ncpu=ncpu)
    energy_data = []
    force_data = []
    if Nmax is None:
        Nmax = 1000000 # infinite

    if lists is None:
        lists = range(len(X))

    for id in lists:
        energy_data.append((X[id]['x'], Y["energy"][id]/len(X[id]['x']))) 
        for i in range(len(X[id]['x'])):
            if len(force_data) == Nmax:
                break
            ids = np.argwhere(X[id]['seq'][:,1]==i).flatten()
            _i = X[id]['seq'][ids, 0] 
            force_data.append((X[id]['x'][_i,:], X[id]['dxdr'][ids], Y['forces'][id][i]))

    train_data = {"energy": energy_data, "force": force_data}
    train_pt_E = {"energy": [data[0] for data in train_data['energy']]}
    train_pt_F = {"force": [(data[0], data[1]) for data in train_data['force']]}
    train_Y_E = np.array([data[1] for data in train_data['energy']])
    train_Y_F = np.array([data[2] for data in train_data['force']]).flatten()

    return train_data, train_pt_E, train_pt_F, train_Y_E, train_Y_F

def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


des = build_desc("SO3", lmax=3, nmax=3, rcut=4.0)
print(des)

size = 2
atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          symbol='Cu',
                          size=(size, size, size),
                          pbc=True)
atoms.set_calculator(EMT())
MaxwellBoltzmannDistribution(atoms, 1000*units.kB)
dyn = VelocityVerlet(atoms, 1*units.fs)  # 2 fs time step.

# Collect the training data from the 1st 2000 steps
strucs = []
for i in range(10):
    dyn.run(steps=10)
    printenergy(atoms)
    strucs.append(atoms.copy())

db_name = "train.db"
save_db(db_name, strucs)
train_data, train_pt_E, train_pt_F, train_Y_E, train_Y_F = get_data(db_name, des, Nmax=5)


kernel = RBF_mb(para=[1.0, 1.0])
model = gpr(kernel=kernel, noise_e=1e-3, noise_f=5e-2)
model.fit(train_data)
E, E1, F, F1 = model.validate_data()
metric_single(E, E1, "Train Energy") 
metric_single(F, F1, "Train Forces") 
            
E_tol = 1e-2
Ev_tol = 1e-1
F_tol = 1e-1
Fv_tol = 0.05
steps = 10

for i in range(1000):
    dyn.run(steps=steps)
    printenergy(atoms)
    d = des.calculate(atoms)
    energy_data = [(d['x'], atoms.get_potential_energy()/len(atoms))]
    force_data = []
    forces = atoms.get_forces()
    for i in range(len(d['x'])):
        ids = np.argwhere(d['seq'][:,1]==i).flatten()
        _i = d['seq'][ids, 0] 
        force_data.append((d['x'][_i,:], d['dxdr'][ids], forces[i]))

    test_data = {"energy": energy_data, "force": force_data}
    E, E1, E_std, F, F1, F_std = model.validate_data(test_data, return_std=True)

    diff_E = E[0] - E1[0]
    print("\nML Energy: {:6.3f} -> {:6.3f} ======== diff: {:6.3f}  Variance E: {:6.3f}  F: {:6.3f}".format(E[0], E1[0], diff_E, E_std[0], np.max(F_std)))
    metric_single(F, F1, "ML Forces", True)
    print("\n")

    update = False
    #if abs(diff_E) > E_tol:
    if E_std[0] > Ev_tol or abs(diff_E) > E_tol:
        print("add energy data to GP model")
        model.add_train_pts_energy(energy_data[0])
        update=True

    #if np.max(diffs_F) > F_tol:
    F_std = F_std.reshape([len(atoms),3])
    for id in range(len(F_std)):
        if np.max(F_std[id]) > Fv_tol:
            print("add force data to GP model", F_std[id])
            model.add_train_pts_force(test_data["force"][id])
            update=True
    if update:
        model.fit()
        E, E1, F, F1 = model.validate_data()
        metric_single(E, E1, "Train Energy") 
        metric_single(F, F1, "Train Forces") 
        print("\n")
            

print(model)
