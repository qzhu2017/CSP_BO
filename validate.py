import sys
import numpy as np
from time import time
from cspbo.utilities import rmse, metric_single, get_strucs, plot
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.calculator import GPR


t0 = time()
N_max, ncpu = None, 10
m_file = sys.argv[1]
db_file = sys.argv[2]
model = gpr()
model.load(m_file, N_max=None)
model.kernel.ncpu = ncpu

train_E, train_E1, train_F, train_F1, _, _ = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 

# set calculator
calc = GPR(ff=model, return_std=False, stress=True)

# get _structures
strucs, values = get_strucs(db_file, N_max=N_max)


total_E0, total_E = [], []
total_F0, total_F = None, None
total_S0, total_S = None, None

for struc, val in zip(strucs, values):
    #E, F, S = model.predict_structure(struc)
    struc.set_calculator(calc)
    E = struc.get_potential_energy()/len(struc)
    F = struc.get_forces()
    S = struc.get_stress()

    (E0, F0, S0) = val
    E0 /= len(struc)
    if total_F is None:
        total_F, total_F0 = F.flatten(), F0.flatten()
        total_S, total_S0 = S.flatten(), S0.flatten()
    else:
        total_F = np.hstack((total_F, F.flatten()))
        total_F0 = np.hstack((total_F0, F0.flatten()))
        total_S = np.hstack((total_S, S.flatten()))
        total_S0 = np.hstack((total_S0, S0.flatten()))
    total_E.append(E)
    total_E0.append(E0)
    F_mse = rmse(F.flatten(), F0.flatten())
    S_mse = rmse(S.flatten(), S0.flatten())
    print("E: {:6.3f} -> {:6.3f}  F_MSE: {:6.3f} S_MSE: {:6.3f}".format(E, E0, F_mse, S_mse))

total_E = np.array(total_E)
total_E0 = np.array(total_E0)
l3 = metric_single(total_E, total_E0, "Test Energy") 
l4 = metric_single(total_F, total_F0, "Test Forces") 
l5 = metric_single(total_S, total_S0, "Test Stress") 

print("{:.3f} seconds elapsed".format(time()-t0))
plot((train_E, total_E), (train_E1, total_E0), (l1, l3), "E.png")
plot((train_F, total_F), (train_F1, total_F0), (l2, l4), "F.png", type="Force")
plot([total_S], [total_S0], [l5], "S.png", type="Stress")
#
