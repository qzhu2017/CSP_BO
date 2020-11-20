import numpy as np
from time import time

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
        dKee_dl = (k*(1.-D)).sum(axis=0)/l3
        return np.stack((Kee/m, dKee_dsigma/m, dKee_dl/m), axis=-1)
    else:
        n = len(x2)
        if wrap:
            return Kee.sum()/(m*n)
        else:
            return Kee.reshape([n, 1])/m


x1 = np.loadtxt("x1.txt")
x2 = np.loadtxt("x2.txt")
x2_indices = [2, 4, 6, 3, 5]

#print(x1)
#print(x2)

t0 = time()
kee = K_ee(x1, x2, sigma2=28.835*28.835, l2=1)
Kee = []
count = 0
for ind in x2_indices:
    Kee.append(kee[count:count+ind].sum())
    count += ind
t1 = time()
print(Kee)
print("time in Python: ", t1-t0)
