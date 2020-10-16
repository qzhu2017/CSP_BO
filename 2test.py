from copy import deepcopy
import numpy as np
import sys
from cspbo.utilities import metric_single, plot
from cspbo.utilities import build_desc, get_data, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb
eV2GPa = 160.21766


def parse_data(data, stress=False):
    pt_E = {"energy": [(dat[0], dat[2]) for dat in data["energy"]]}
    pt_F = {"force": [(dat[0], dat[1], dat[3]) for dat in data["force"]]}

    Y_E = np.array([dat[1] for dat in data["energy"]])
    Y_F = np.array([dat[2] for dat in data["force"]]).flatten()
    if stress:
        pt_S = {"stress": [(dat[0], dat[1])  for dat in data["stress"]]}
        Y_S = np.array([dat[2] for dat in data["stress"]]).flatten()
        return data, pt_E, pt_F, pt_S, Y_E, Y_F, Y_S
    else:
        return data, pt_E, pt_F, None, Y_E, Y_F, None

total_pts = 244
N1, N2, cpu = 219, 25, 8
des = build_desc("SO3", lmax=4, nmax=4, rcut=5.0)
train_data = get_data(sys.argv[1], des, ncpu=cpu, lists=list(range(0, N1)), select=True, force_mod=10)
test_data = get_data(sys.argv[1], des, ncpu=cpu, lists=list(range(total_pts-25, N2+(total_pts-25))), select=True, force_mod=2, stress=True)

train_data, train_pt_E, train_pt_F, train_pt_S, train_Y_E, train_Y_F, train_Y_S = parse_data(train_data)
test_data, test_pt_E, test_pt_F, test_pt_S, test_Y_E, test_Y_F, test_Y_S = parse_data(test_data, stress=True)

print("------------------Tranining both E and F--------------------------")
train_data1 = deepcopy(train_data)

kernel = RBF_mb(para=[1.0, 1.0])
model = gpr(kernel=kernel)
model.fit(train_data1)

train_pred = model.predict(train_pt_E)
metric_single(train_Y_E, train_pred, "Train Energy")

test_pred = model.predict(test_pt_E)
metric_single(test_Y_E, test_pred, "Test  Energy")

train_pred = model.predict(train_pt_F)
metric_single(train_Y_F, train_pred, "Train Forces")

test_pred = model.predict(test_pt_F)
metric_single(test_Y_F, test_pred, "Test  Forces")

test_pred = model.predict(test_pt_S)*eV2GPa
metric_single(test_Y_S, test_pred, "Test  Stress")

for i in range(len(test_pred)):
    print(test_Y_S[i], test_pred[i])

#print("------------------Tranining Energy only--------------------------")
#train_data1 = deepcopy(train_data)
#train_data1["force"] = []
#
#kernel = RBF_mb(para=[5.0, 0.2])
#model = gpr(kernel=kernel)
#model.fit(train_data1)
#
#train_pred = model.predict(train_pt_E)
#metric_single(train_Y_E, train_pred, "Train Energy")
#test_pred = model.predict(test_pt_E)
#metric_single(test_Y_E, test_pred, "Test  Energy")
#test_pred = model.predict(test_pt_F)
#metric_single(test_Y_F, test_pred, "Test  Forces")
#
#test_pred = model.predict(test_pt_S)*eV2GPa
#metric_single(test_Y_S, test_pred, "Test  Stress")
#
#for i in range(len(test_pred)):
#    print(test_Y_S[i], test_pred[i])
