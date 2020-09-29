import sys
from time import time
from cspbo.utilities import convert_rdf, metrics, plot
from cspbo.utilities import build_desc, convert_struc, plot_two_body
import numpy as np

N1, N2 = 20, 20 #None, None
des = build_desc("SO4")
print(des)
train_X, train_Y = convert_struc(sys.argv[1], des, N=N1, ncpu=4)
test_X, test_Y = convert_struc(sys.argv[2], des, N=N2, ncpu=4)
train_data = {"energy": [(x['x'], y) for (x, y) in zip(train_X, train_Y)]}

train_pt  = {"energy": [x['x'] for x in train_X]}
test_pt1   = {"energy": [x['x'] for x in test_X]}

#build the test pts for forces
force_data = []
for x in test_X[:3]:
    # sample the force for atom 1
    ids = np.argwhere(x['seq'][:,1]==1)
    _i = x['seq'][ids, 0]
    #print("sampling atom 1")
    #print(_i.T)
    #print(ids.T)
    force_data.append((x['x'][_i], x['dxdr'][ids]))
test_pt2   = {"force": force_data}


from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb

t0 = time()

kernel = RBF_mb(para=[0.8, 0.5])
model = gpr(kernel=kernel)
model.fit(train_data)
train_pred = model.predict(train_pt)
test_pred = model.predict(test_pt1)
#test_pred = model.predict(test_pt2)

print("elapsed time: ", time()-t0)
labels = metrics(train_Y, test_Y, train_pred, test_pred, "MB")
plot((train_Y, test_Y), (train_pred, test_pred), labels, kernel.name+'-fit.png')
#plot_two_body(model, des, kernel, kernel.name+'-2-body.png')
