import sys
import numpy as np
from time import time
from cspbo.utilities import rmse, metric_single, get_strucs, plot
from cspbo.gaussianprocess_ef import GaussianProcess as gpr
from cspbo.calculator import GPR

np.set_printoptions(formatter={'float': '{: 5.2f}'.format})

device = 'gpu'
m_file = sys.argv[1]
model = gpr()
model.load(m_file, N_max=4, device= device)

train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 

model.sparsify()
train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 


model.save("models/test.json", "models/test.db")
model.load("models/test.json", N_max=200)
model.kernel.ncpu = ncpu
train_E, train_E1, train_F, train_F1 = model.validate_data()
l1 = metric_single(train_E, train_E1, "Train Energy") 
l2 = metric_single(train_F, train_F1, "Train Forces") 


