import sys
from time import time
from cspbo.utilities import convert_rdf, metrics, plot
from cspbo.utilities import build_desc, convert_struc, plot_two_body

N1, N2 = None, None
des = build_desc("SO4")
print(des)
train_X, train_Y = convert_struc(sys.argv[1], des, N=N1, ncpu=4)
test_X, test_Y = convert_struc(sys.argv[2], des, N=N2, ncpu=4)

from cspbo.gaussianprocess_np import GaussianProcess as gpr
from cspbo.kernels import Dot_mb, RBF_mb, RBF_2b, Combo

for kernel in [
               RBF_mb(para=[0.8, 0.5]), 
               RBF_2b(para=[1.1, 0.5]), 
               #Combo(para=[0.8, 0.5, 1.1, 0.5]),
               #Combo(para=[5.0, 0.5, 1.1, 0.5]),
              ]:
    t0 = time()
    model = gpr(kernel=kernel)
    model.fit((train_X, train_Y))
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    print("elapsed time: ", time()-t0)
    labels = metrics(train_Y, test_Y, train_pred, test_pred, "MB")
    plot((train_Y, test_Y), (train_pred, test_pred), labels, kernel.name+'-fit.png')
    plot_two_body(model, des, kernel, kernel.name+'-2-body.png')
