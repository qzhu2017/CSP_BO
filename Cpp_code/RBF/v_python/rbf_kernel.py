import numpy as np
from kernel_base import *
from time import time

def kee_many(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False):
    sigma2, l2, zeta = sigma**2, l**2, zeta

    (x1, ele1, x1_indices) = X1
    (x_all, ele_all, x2_indices) = X2
        
    # num of X1, num of X2, num of big X2
    m1, m2, m2p = 1, len(x2_indices), len(x_all)

    x2_inds = [(0, x2_indices[0])]
    for i in range(1, len(x2_indices)):
        ID = x2_inds[i-1][1]
        x2_inds.append( (ID, ID+x2_indices[i]) )

    if grad:
        C = np.zeros([m1, m2p, 3])
    else:
        C = np.zeros([m1, m2p, 1])

    for i in range(m1):
        #if i%1000 == 0: print("Kee", i)
        #_x1, _ele1 = x1, ele1
        mask = get_mask(ele1, ele_all)
        C[i] = K_ee(x1, x_all, sigma2, l2, zeta, grad, mask)
    
    _C = np.zeros([m1, m2])
    if grad:
        _C_s = np.zeros([m1, m2])
        _C_l = np.zeros([m1, m2])

    for j, ind in enumerate(x2_inds):
        tmp = C[:, ind[0]:ind[1], :].sum(axis=1)/x2_indices[j]
        _C[:, j]  = tmp[:, 0]
        if grad:
            _C_s[:, j]  = tmp[:, 1]
            _C_l[:, j]  = tmp[:, 2]
    if grad:
        return _C, _C_s, _C_l
    else:
        return _C

def K_ee(x1, x2, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8, wrap=False):
    """
    Compute the Kee between two structures
    Args:
        x1: [M, D] 2d array
        x2: [N, D] 2d array
        sigma2: float
        l2: float
        zeta: power term, float
        mask: to set the kernel zero if the chemical species are different
    """
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D = d**zeta

    k = sigma2*np.exp(-(0.5/l2)*(1-D))
    if mask is not None:
        k[mask] = 0
    dk_dD = (-0.5/l2)*k

    Kee = k.sum(axis=0)
    m = len(x1)

    if grad:
        l3 = np.sqrt(l2)*l2
        dKee_dsigma = 2*Kee/np.sqrt(sigma2)
        dKee_dl = (k*(1-D)).sum(axis=0)/l3
        #ans = np.stack((Kee/m, dKee_dsigma/m, dKee_dl/m), axis=-1)
        #print(Kee.shape, ans.shape)
        return np.stack((Kee/m, dKee_dsigma/m, dKee_dl/m), axis=-1)
    else:
        n = len(x2)
        if wrap:
            return Kee.sum()/(m*n)
        else:
            return Kee.reshape([n, 1])/m

X1_EE = np.load('../X1_EE.npy', allow_pickle=True)
X2_EE = np.load('../X2_EE.npy', allow_pickle=True)
X1_FF = np.load('../X1_FF.npy', allow_pickle=True)
X2_FF = np.load('../X2_FF.npy', allow_pickle=True)
sigma = 9.55544058601137

t0 = time()
C_EE = kee_many(X1_EE, X2_EE, sigma=sigma)
print(C_EE.shape)

print(C_EE)

print("Elapsed time: ", time()-t0)
np.save("kernel_EE.npy", C_EE)
