import sys
from time import time
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
import sklearn.gaussian_process as gp
from cspbo.utilities import convert_rdf, metrics, plot
from cspbo.utilities import build_desc, convert_struc, plot_two_body

N1, N2 = 2000, 200
#train_X, train_Y = convert_rdf(sys.argv[1], N=N1)
#test_X, test_Y = convert_rdf(sys.argv[2], N=N2)

#t0 = time()
#kernel = 1.0*gp.kernels.RBF()
#model = gpr(kernel=kernel,  n_restarts_optimizer=10)
#model.fit(train_X, train_Y)
#print(model.kernel_)
#train_pred = model.predict(train_X)
#test_pred = model.predict(test_X)
#labels = metrics(train_Y, test_Y, train_pred, test_pred, "RDF")
#plot((train_Y, test_Y), (train_pred, test_pred), labels, 'RDF.png')
#print("elapsed time: ", time()-t0)

des = build_desc()
print(des)
train_X, train_Y = convert_struc(sys.argv[1], des, N=N1, ncpu=4)
test_X, test_Y = convert_struc(sys.argv[2], des, N=N2, ncpu=4)


t0 = time()
from cspbo.gaussianprocess_np import GaussianProcess as gpr
from cspbo.gaussianprocess_np import Dot, RBF
kernel = RBF(para=[1, 0.5])
model = gpr(kernel=kernel)

model.fit((train_X, train_Y))
train_pred = model.predict(train_X)
test_pred = model.predict(test_X)
print("elapsed time: ", time()-t0)
labels = metrics(train_Y, test_Y, train_pred, test_pred, "MB")
plot((train_Y, test_Y), (train_pred, test_pred), labels, 'MB.png')
plot_two_body(model, des, '2-body.png')


