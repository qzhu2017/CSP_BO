import sys
from time import time
from cspbo.utilities import convert_rdf, metrics, plot
from cspbo.utilities import build_desc, convert_struc, plot_two_body
import numpy as np

#N1, N2, cpu = 50, 50, 8
N1, N2, cpu = None, None, 8
des = build_desc("SO3")
print(des)
X, Y = convert_struc(sys.argv[1], des, N=N1, ncpu=cpu)
#test_X, test_Y = convert_struc(sys.argv[2], des, N=N2, ncpu=cpu)
N_train = 200


train_data = {"energy": [(x['x'], y) for (x, y) in zip(X[:N_train], Y["energy"][:N_train])]}

train_pt  = {"energy": [x['x'] for x in X[:N_train]]}
train_Y1  = np.array(Y["energy"][:N_train])

test_pt1  = {"energy": [x['x'] for x in X[N_train:]]}
test_Y1  = np.array(Y["energy"][N_train:])

#build the test pts for forces
test_Y2 = None
force_data = []
for (x, y) in zip(X[N_train:], Y["forces"][N_train:]):
    # sample the force for atom 1
    for i in range(4,5):
        ids = np.argwhere(x['seq'][:,1]==i).flatten()
        _i = x['seq'][ids, 0] #.flatten()
        force_data.append((x['x'][_i,:], x['dxdr'][ids]))
        if test_Y2 is None:
            test_Y2 = y[i]
        else:
            test_Y2 = np.hstack((test_Y2, y[i]))
test_pt2 = {"force": force_data}


from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb

t0 = time()

kernel = RBF_mb(para=[0.8, 0.5])
model = gpr(kernel=kernel)
model.fit(train_data)
train_pred = model.predict(train_pt)
test_pred = model.predict(test_pt1)
labels = metrics(train_Y1, test_Y1, train_pred, test_pred, "MB")
test_pred = model.predict(test_pt2)
labels = metrics(train_Y1, test_Y2, train_pred, test_pred, "MB")

print("elapsed time: ", time()-t0)
#plot((train_Y, test_Y), (train_pred, test_pred), labels, kernel.name+'-fit.png')
#plot_two_body(model, des, kernel, kernel.name+'-2-body.png')
