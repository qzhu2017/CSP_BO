import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single
from cspbo.gaussianprocess import GaussianProcess as gpr

np.set_printoptions(formatter={'float': '{: 5.2f}'.format})

# 
m_file = sys.argv[1]
model = gpr()
model.load(m_file, N_max=None, opt=False)

t0 = time()
train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 
print("1st run: {:6.3f}".format(time()-t0))

model.sparsify(e_tol=1e-4, f_tol=1e-3)
t0 = time()
train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 
print("2nd run: {:6.3f}".format(time()-t0))

model.save("models/sparse.json", "models/sparse.db")
#model.load("models/sparse.json")
