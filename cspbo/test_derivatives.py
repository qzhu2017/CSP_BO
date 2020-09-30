"""
Here is the code to test 
- dDdX2
- dkdX2
"""
import numpy as np

def fun(x1, x2, x1_norm, x2_norm, eps=0):
    d = x1@x2.T/(eps+np.outer(x1_norm, x2_norm))
    D = 1 - d**2
    return D

def dD_dx1(x1, x2, x1_norm, x2_norm, D):
    # x1: m,d
    # x2: n,d
    tmp1 = np.einsum("ij,k->kij", x2, x1_norm) # [n,d] x [m] -> [m, n, d]
    tmp2 = (x1@x2.T)[:,:,None] * (x1/x1_norm[:, None])[:,None,:] #[m, n, d]
    tmp3 = (x1_norm**2)[:, None, None] * x2_norm[None, :, None]  #[m, n, d]
    out = (tmp1-tmp2)/tmp3  # m,n,d
    return np.einsum("ij, ijk->ik", -2*D, out) #m,n;  m,n,d -> m,d

def dD_dx2(x1, x2, x1_norm, x2_norm, D):
    tmp1 = np.einsum("ij,k->ikj", x1, x2_norm) # [m,d] x [n] -> [m, n, d]
    tmp2 = (x1@x2.T)[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp3 = x1_norm[:, None, None] * (x2_norm**2)[None, :, None]
    out = (tmp1-tmp2)/tmp3  # m,n,d
    return np.einsum("ij, ijk->jk", -2*D, out) # -> n,d

m, n, d = 9, 5, 2 #20, 10, 6
x1 = 0.1+np.random.random([m, d])
x2 = np.random.random([n, d])
x1_norm = np.linalg.norm(x1, axis=1)
x2_norm = np.linalg.norm(x2, axis=1)
D1 = fun(x1, x2, x1_norm, x2_norm)
print("dD_dx1")
print(dD_dx1(x1, x2, x1_norm, x2_norm, D1))
print("dD_dx2")
print(dD_dx2(x1, x2, x1_norm, x2_norm, D1))

import torch
print("numerical")
def distance_torch(x1, x2):
    d=x1@x2.T/(torch.ger(torch.norm(x1, dim=1), torch.norm(x2, dim=1)))
    return (1-d**2).sum()

x1 = torch.tensor(x1, requires_grad=True)
x2 = torch.tensor(x2, requires_grad=True)
print("dD_dx1")
print(torch.autograd.grad(distance_torch(x1, x2), x1))
print("dD_dx2")
print(torch.autograd.grad(distance_torch(x1, x2), x2))

