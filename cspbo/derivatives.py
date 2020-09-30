"""
Here is the code to test 
- dDdX2
- dkdX2
"""
import numpy as np

def fun_k(x1, x2, x1_norm, x2_norm, sigma2, l2):
    D, d = fun_D(x1, x2, x1_norm, x2_norm)
    _k = sigma2*np.exp(-0.5*D/l2)
    return _k, _k.sum()

def fun_D(x1, x2, x1_norm, x2_norm, eps=0):
    d = x1@x2.T/(eps+np.outer(x1_norm, x2_norm))
    D = 1 - d**2
    return D, d

def fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2):
    _k, _ksum = fun_k(x1, x2, x1_norm, x2_norm, sigma2, l2)
    return -0.5*_k/l2

def fun_dk_dx1(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    dD_dx1, _ = fun_dD_dx1(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    return np.einsum("ij,ijk->ik", dk_dD,dD_dx1)

def fun_dk_dx2(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    dD_dx2, _ = fun_dD_dx2(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    return np.einsum("ij,ijk->jk", dk_dD, dD_dx2)


def fun_dD_dx1(x1, x2, x1_norm, x2_norm, d):
    # x1: m,d
    # x2: n,d
    tmp1 = np.einsum("ij,k->kij", x2, x1_norm) # [n,d] x [m] -> [m, n, d]
    tmp2 = (x1@x2.T)[:,:,None] * (x1/x1_norm[:, None])[:,None,:] #[m, n, d]
    tmp3 = (x1_norm**2)[:, None, None] * x2_norm[None, :, None]  #[m, n, d]
    out = (tmp1-tmp2)/tmp3  # m,n,d
    dD_dx1 = np.einsum("ij, ijk->ijk", -2*d, out) #m,n;  m,n,d -> m,d
    return dD_dx1, dD_dx1.sum(axis=1)

def fun_dD_dx2(x1, x2, x1_norm, x2_norm, d):
    tmp1 = np.einsum("ij,k->ikj", x1, x2_norm) # [m,d] x [n] -> [m, n, d]
    tmp2 = (x1@x2.T)[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp3 = x1_norm[:, None, None] * (x2_norm**2)[None, :, None]
    out = (tmp1-tmp2)/tmp3  # m,n,d
    dD_dx2 = np.einsum("ij, ijk->ijk", -2*d, out) # -> n,d
    return dD_dx2, dD_dx2.sum(axis=0)


if __name__ == "__main__":
    m, n, k, sigma2, l2 = 9, 5, 2, 1.1, 0.9
    x1 = 0.1+np.random.random([m, k])
    x2 = np.random.random([n, k])
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    D1, d1 = fun_D(x1, x2, x1_norm, x2_norm)
    print("dD_dx1")
    _, tmp = fun_dD_dx1(x1, x2, x1_norm, x2_norm, d1)
    print(tmp)
    print("dD_dx2")
    _, tmp = fun_dD_dx2(x1, x2, x1_norm, x2_norm, d1)
    print(tmp)
    print("dk_dx1")
    print(fun_dk_dx1(x1, x2, x1_norm, x2_norm, d1, sigma2, l2))
    print("dk_dx2")
    print(fun_dk_dx2(x1, x2, x1_norm, x2_norm, d1, sigma2, l2))
    
    
    import torch
    print("numerical")
    def distance_torch(x1, x2):
        d=x1@x2.T/(torch.ger(torch.norm(x1, dim=1), torch.norm(x2, dim=1)))
        return 1-d**2 #.sum()
    
    def k_torch(x1, x2, sigma2, l2):
        D = distance_torch(x1, x2)
        k = sigma2*(torch.exp(-0.5*D/l2).sum())
        return k 
    
    
    x1 = torch.tensor(x1, requires_grad=True)
    x2 = torch.tensor(x2, requires_grad=True)
    print("dD_dx1")
    print(torch.autograd.grad(distance_torch(x1, x2).sum(), x1))
    print("dD_dx2")
    print(torch.autograd.grad(distance_torch(x1, x2).sum(), x2))
    
    print("dk_dx1")
    print(torch.autograd.grad(k_torch(x1, x2, sigma2, l2), x1))
    print("dk_dx2")
    print(torch.autograd.grad(k_torch(x1, x2, sigma2, l2), x2))
