import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single, build_desc, get_data, plot, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb

N_start, N_step, N_max, zeta, ncpu = 10, 10, 10, 2, 10
m_file = sys.argv[1]
db_file = sys.argv[2]
model = gpr()
model.load(m_file)
#plot_two_body(model, des, kernel, "2-body.png") 
model.kernel.ncpu = ncpu
train_E, train_E1, train_F, train_F1, _, _ = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 

test_E, test_E1, E_var, test_F, test_F1, F_var, test_S, test_S1, S_var = \
None, None, None, None, None, None, None, None, None

for id in range(0, N_max, N_step):
    ids = range(id, id+N_step)
    #test_data = get_data(db_file, model.descriptor, lists=ids, ncpu=ncpu, stress=True)
    test_data = get_data(db_file, model.descriptor, lists=ids, ncpu=ncpu)
    print("Testing-------", id, id+N_step)
    #test_data["force"] = []
    #E, E1, E_std, F, F1, F_std, S, S1, S_std = model.validate_data(test_data, return_std=True)
    E, E1, F, F1, S, S1 = model.validate_data(test_data)
    metric_single(E, E1, "Test Energy") 
    metric_single(F, F1, "Test Forces") 
    metric_single(S, S1, "Test Stress") 
    if test_E is None:
        test_E = E
        test_E1 = E1
        test_F = F
        test_F1 = F1
        test_S = S
        test_S1 = S1
        #E_var = E_std
        #F_var = F_std
        #S_var = S_std
    else:
        test_E = np.hstack((test_E, E))
        test_E1 = np.hstack((test_E1, E1))
        test_F = np.hstack((test_F, F))
        test_F1 = np.hstack((test_F1, F1))
        test_S = np.hstack((test_S, S))
        test_S1 = np.hstack((test_S1, S1))
        #E_var = np.hstack((E_var, E_std))
        #F_var = np.hstack((F_var, F_std))
        #S_var = np.hstack((S_var, S_std))

l3 = metric_single(test_E, test_E1, "Test Energy") 
l4 = metric_single(test_F, test_F1, "Test Forces") 
l5 = metric_single(test_S, test_S1, "Test Stress") 

plot((train_E, test_E), (train_E1, test_E1), (l1, l3), "E.png")
plot((train_F, test_F), (train_F1, test_F1), (l2, l4), "F.png")
#plot([np.abs(test_E-test_E1)], [E_var], ["Energy: True .v.s Var"], "E_var.png", False)
#plot([np.abs(test_F-test_F1)], [F_var], ["Forces: True .v.s Var"], "F_var.png", False, "Force")
