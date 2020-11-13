import sys
import numpy as np
from time import time
from cspbo.utilities import metric_single
from cspbo.gaussianprocess_ef import GaussianProcess as gpr

np.set_printoptions(formatter={'float': '{: 5.2f}'.format})

device = 'gpu'
m_file = sys.argv[1]
model = gpr()
model.load(m_file, N_max=None, device=device)

t0 = time()
train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 
print("First run: {:6.3f}".format(time()-t0))

model.sparsify()
t0 = time()
train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 
print("First run: {:6.3f}".format(time()-t0))


#model.save("models/test.json", "models/test.db")
#model.load("models/test.json")
#train_E, train_E1, train_F, train_F1 = model.validate_data()
#l1 = metric_single(train_E, train_E1, "Train Energy") 
#l2 = metric_single(train_F, train_F1, "Train Forces") 


