import sys
import numpy as np
from time import time
from cspbo.utilities import metrics, build_desc, convert_struc, plot, plot_two_body

def extract(train_X, train_Y, db_ids):
    x1, x2, y = [], [], []
    for id in db_ids:
        x1.append(train_X[0][id])
        x2.append(train_X[1][id])
        y.append(train_Y[id])
    return (x1, x2), np.array(y)


des = build_desc()
print(des)
train_X, train_Y = convert_struc(sys.argv[1], des, N=5000, ncpu=4)
test_X, test_Y = convert_struc(sys.argv[2], des, N=None, ncpu=4)
t0 = time()

from cspbo.gaussianprocess_np import GaussianProcess as gpr
from cspbo.gaussianprocess_np import Dot, RBF

kernel = RBF(para=[1, 0.5])
model = gpr(kernel=kernel)

N = 25
db_ids = [id for id in range(N)]
pool_ids = range(N, len(train_Y), N)  
train_X0, train_Y0 = extract(train_X, train_Y, db_ids)
model.fit((train_X0, train_Y0), show=False)

for id in pool_ids:
    if id+N > len(train_Y):
        ids = range(id, len(train_Y))
    else:
        ids = range(id, id+N)
    trial_X, trail_Y = extract(train_X, train_Y, ids)
    pred = model.predict(trial_X)
    to_add = []
    for m, y0 in enumerate(trail_Y):
        y1 = pred[m]
        strs = "{:4d} {:6.3f} -> {:6.3f} {:6.3f}  ".format(id+m, y0, y1, y0-y1)
        if abs(y0-y1) > 0.05:
            try:
                to_add.append(ids[m])
            except:
                print(ids, m)
        print(strs)
    
    if len(to_add) > 0:
        db_ids.extend(to_add)
        train_X0, train_Y0 = extract(train_X, train_Y, db_ids)
        model.fit((train_X0, train_Y0), show=False)
        strs += " update GP [{:d}] {:s}".format(len(db_ids), str(model.kernel))
        print(strs)

    if id%100 == 0:
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)
        labels = metrics(train_Y, test_Y, train_pred, test_pred, kernel.name)
        fig = "out/MB-{:d}.png".format(id)
        fig1 = "out/2-body-{:d}.png".format(id)
        plot((train_Y, test_Y), (train_pred, test_pred), labels, fig)
        plot_two_body(model, des, fig1)

print("elapsed time: ", time()-t0)

