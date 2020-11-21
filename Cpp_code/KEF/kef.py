import numpy as np
from time import time
np.set_printoptions(linewidth=100000)

def K_ef(x1, x2, dx2dr, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8, device='cpu'):
    m = len(x1)
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x2_norm2 = x2_norm**2
    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1


    k = sigma2*np.exp(-(0.5/l2)*(1-D))
    if mask is not None:
        k[mask] = 0
    dk_dD = (-0.5/l2)*k

    if device == 'gpu':
        x1 = cp.array(x1)
        x2 = cp.array(x2)
        x1_norm = cp.array(x1_norm)
        x2_norm = cp.array(x2_norm)
        x2_norm2 = cp.array(x2_norm2)
        x1x2_dot = cp.array(x1x2_dot)
        D1 = cp.array(D1)
        dk_dD = cp.array(dk_dD)

    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None] 
    dd_dx2 = (tmp21-tmp22)/tmp23
    
    zd1 = zeta * D1
    dD_dx2 = -zd1[:,:,None] * dd_dx2

    kef1 = (-dD_dx2[:,:,:,None]*dx2dr[None,:,:,:]).sum(axis=2) #[m, n, 9]
    Kef = (kef1*dk_dD[:,:,None]).sum(axis=0) #[n, 9]
    return Kef/m

x1 = np.loadtxt("x1.txt")
x2 = np.loadtxt("x2.txt")
dx2dr = np.loadtxt("dx2dr.txt")
dx2dr = np.reshape(dx2dr, [20, 5, 3])
x2_indices = [2, 4, 6, 3, 5]

t0 = time()
kef = K_ef(x1, x2, dx2dr, sigma2=28.835*28.835, l2=1)
Kef = np.zeros([len(x2_indices), 3])
count = 0
for i, ind in enumerate(x2_indices):
    Kef[i,:] += np.sum(kef[count:count+ind, :3], axis=0)
    count += ind
t1 = time()
print(Kef)
print("time in Python: ", t1-t0)
