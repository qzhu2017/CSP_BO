import sys
import numpy as np
from time import time
from cspbo.utilities import rmse, metric_single, get_strucs, plot
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.calculator import GPR
from mpi4py import MPI

eV2GPa = 160.21766

comm=MPI.COMM_WORLD 

np.set_printoptions(formatter={'float': '{: 5.2f}'.format})

#N_max, ncpu = 100, 24

N_max = 5
m_file = sys.argv[1]
db_file = sys.argv[2]
model = gpr()
model.load(m_file, N_max=None, opt=True)
#model.load(m_file, N_max=None, opt=False, device=device)
#model.load(m_file, N_max=2, opt=True, device=device)

train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 

# get _structures
strucs, values = get_strucs(db_file, N_max=N_max)
(E0, F0, S0) = values[0]
stress = False  #False

# set calculator
calc = GPR(ff=model, return_std=False, stress=stress)

total_E0, total_E = [], []
total_F0, total_F = None, None
total_S0, total_S = None, None

t0 = time()
count = 0
for struc, val in zip(strucs, values):
    count += 1
    (E0, F0, S0) = val
    struc.calc = calc
    E = struc.get_potential_energy()/len(struc)
    F = struc.get_forces()

    if stress:
        S = struc.get_stress()*eV2GPa

    E0 /= len(struc)
    if total_F is None:
        total_F, total_F0 = F.flatten(), F0.flatten()
        if stress:
            total_S, total_S0 = S.flatten(), S0.flatten()
    else:
        total_F = np.hstack((total_F, F.flatten()))
        total_F0 = np.hstack((total_F0, F0.flatten()))
        if stress:
            total_S = np.hstack((total_S, S.flatten()))
            total_S0 = np.hstack((total_S0, S0.flatten()))
    total_E.append(E)
    total_E0.append(E0)
    F_mse = rmse(F.flatten(), F0.flatten())
    strs = "{:4d} E: {:6.3f} -> {:6.3f}  F_MSE: {:6.3f} ".format(count, E, E0, F_mse)
    if stress:
        S_mse = rmse(S.flatten(), S0.flatten())
        strs += "S_MSE: {:6.3f}".format(S_mse)
    if comm.rank == 0:
        print(strs)

if comm.rank == 0:
    total_E = np.array(total_E)
    total_E0 = np.array(total_E0)
    l3 = metric_single(total_E, total_E0, "Test Energy") 
    l4 = metric_single(total_F, total_F0, "Test Forces") 

    print("{:.3f} seconds elapsed".format(time()-t0))
    plot((train_E, total_E), (train_E1, total_E0), (l1, l3), "E.png")
    plot((train_F, total_F), (train_F1, total_F0), (l2, l4), "F.png", type="Force")

    if stress:
        l5 = metric_single(total_S, total_S0, "Test Stress") 
        plot([total_S], [total_S0], [l5], "S.png", type="Stress")
