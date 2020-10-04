from copy import deepcopy
import numpy as np
import sys
from cspbo.utilities import metric_single, plot
from cspbo.utilities import build_desc, convert_struc, plot_two_body
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb

def get_data(X, Y, lists, Nmax=None):
    energy_data = []
    force_data = []
    if Nmax is None:
        Nmax = len(X)

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

N1, N2, cpu = None, None, 8
des = build_desc("SO3", lmax=4, nmax=4, rcut=5.0)
print(des)
X, Y = convert_struc(sys.argv[1], des, N=N1, ncpu=cpu)
N_train = int(len(X)*0.8)

train_data, train_pt_E, train_pt_F, train_Y_E, train_Y_F = get_data(X, Y, list(range(0, N_train)), Nmax=10)
test_data, test_pt_E, test_pt_F, test_Y_E, test_Y_F = get_data(X, Y, list(range(N_train, len(X))))

print("------------------Tranining Energy only--------------------------")
train_data1 = deepcopy(train_data)
train_data1["force"] = []

kernel = RBF_mb(para=[1.0, 1.0])
model = gpr(kernel=kernel)
model.fit(train_data1)

train_pred = model.predict(train_pt_E)
metric_single(train_Y_E, train_pred, "Train Energy")
test_pred = model.predict(test_pt_E)
metric_single(test_Y_E, test_pred, "Test  Energy")
test_pred = model.predict(test_pt_F)
metric_single(test_Y_F, test_pred, "Test  Forces")

print("------------------Tranining Force only--------------------------")
train_data1 = deepcopy(train_data)
train_data1["energy"] = []

kernel = RBF_mb(para=[1.0, 1.0])
model = gpr(kernel=kernel)
model.fit(train_data1)

train_pred = model.predict(train_pt_F)
metric_single(train_Y_F, train_pred, "Train Forces")

test_pred = model.predict(test_pt_E)
metric_single(test_Y_E, test_pred, "Test  Energy")

test_pred = model.predict(test_pt_F)
metric_single(test_Y_F, test_pred, "Test  Forces")


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

