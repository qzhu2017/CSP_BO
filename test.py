import sys
from time import time
from cspbo.utilities import convert_rdf, metrics, plot
from cspbo.utilities import build_desc, convert_struc, plot_two_body

N1, N2 = 50, 50 #None, None
des = build_desc("SO4")
print(des)
train_X, train_Y = convert_struc(sys.argv[1], des, N=N1, ncpu=4)
test_X, test_Y = convert_struc(sys.argv[2], des, N=N2, ncpu=4)
train_data = {"energy": [(x, y) for (x, y) in zip(train_X, train_Y)]}
train_pt  = {"energy": [x for x in train_X]}
test_pt   = {"energy": [x for x in test_X]}

from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.RBF_mb import RBF_mb

for kernel in [
               RBF_mb(para=[0.8, 0.5]), 
              ]:
    t0 = time()
    model = gpr(kernel=kernel)
    model.fit(train_data)
    train_pred = model.predict(train_pt)
    test_pred = model.predict(test_pt)
    print("elapsed time: ", time()-t0)
    labels = metrics(train_Y, test_Y, train_pred, test_pred, "MB")
    plot((train_Y, test_Y), (train_pred, test_pred), labels, kernel.name+'-fit.png')
    #plot_two_body(model, des, kernel, kernel.name+'-2-body.png')
