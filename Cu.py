from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.cluster.icosahedron import Icosahedron
from ase.db import connect
from cspbo.utilities import build_desc, convert_struc, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb
import os
import numpy as np
from cspbo.utilities import metric_single, plot

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
        energy_data.append((X[id]['x'], Y["energy"][id])) 
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


des = build_desc("SO3", lmax=4, nmax=4, rcut=5.0)
print(des)

#atoms = Icosahedron('Ar', noshells=2, latticeconstant=3)
#atoms.set_calculator(LennardJones())
size = 2
atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          symbol='Cu',
                          size=(size, size, size),
                          pbc=True)
atoms.set_calculator(EMT())
MaxwellBoltzmannDistribution(atoms, 300*units.kB)
dyn = VelocityVerlet(atoms, 2*units.fs)  # 2 fs time step.

# Collect the training data from the 1st 2000 steps
strucs = []
for i in range(20):
    dyn.run(steps=10)
    printenergy(atoms)
    strucs.append(atoms.copy())

db_name = "train.db"
save_db(db_name, strucs)
train_data, train_pt_E, train_pt_F, train_Y_E, train_Y_F = get_data(db_name, des, Nmax=30)


kernel = RBF_mb(para=[1.0, 1.0])
model = gpr(kernel=kernel)
model.fit(train_data)

train_pred = model.predict(train_pt_E)
metric_single(train_Y_E, train_pred, "Train Energy")
train_pred = model.predict(train_pt_F)
metric_single(train_Y_F, train_pred, "Train Forces")


for i in range(20):
    strucs = []
    for i in range(20):
        dyn.run(steps=10)
        printenergy(atoms)
        strucs.append(atoms.copy())
    db_name = "test.db"
    save_db(db_name, strucs)
    _, test_pt_E, test_pt_F, test_Y_E, test_Y_F = get_data(db_name, des)

    test_pred = model.predict(test_pt_E)
    metric_single(test_Y_E, test_pred, "Test Energy")
    test_pred = model.predict(test_pt_F)
    metric_single(test_Y_F, test_pred, "Test Forces")

