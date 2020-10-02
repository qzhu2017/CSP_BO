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


def fun_dd_dx1(x1, x2, x1_norm, x2_norm):  
    # x1: m,d
    # x2: n,d
    tmp1 = np.einsum("ij,k->kij", x2, x1_norm) # [n,d] x [m] -> [m, n, d]
    tmp2 = (x1@x2.T)[:,:,None] * (x1/x1_norm[:, None])[:,None,:] #[m, n, d]
    tmp3 = (x1_norm**2)[:, None, None] * x2_norm[None, :, None]  #[m, n, d]
    out = (tmp1-tmp2)/tmp3  # m,n,d
    return out, out.sum(axis=1)


def fun_dd_dx2(x1, x2, x1_norm, x2_norm):
    tmp1 = np.einsum("ij,k->ikj", x1, x2_norm) # [m,d] x [n] -> [m, n, d]
    tmp2 = (x1@x2.T)[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp3 = x1_norm[:, None, None] * (x2_norm**2)[None, :, None]
    out = (tmp1-tmp2)/tmp3  # m,n,d
    return out, out.sum(axis=0)

def fun_d2d_dx1dx2(x1, x2, x1_norm, x2_norm):

    x1_norm3 = x1_norm**3        
    x2_norm3 = x2_norm**3      
    x2_norm2 = x2_norm**2      

    tmp0 = np.ones(x2.shape)
    x1x1 = np.einsum("ijl,ik->ijkl", x1[:,None,:]*tmp0[None,:,:], x1)
    tmp1 = x1x1/x2_norm[None,:,None,None]
    x1x2 = np.einsum("ik,jl->ijkl", x1, x2/x2_norm3[:,None])
    tmp2 = np.einsum("ijkl,ij->ijkl", x1x2, x1@x2.T)
    out1 = (tmp1-tmp2)/(x1_norm3[:,None,None,None])

    tmp3 = np.eye(x2.shape[1])[None,:,:] - np.einsum('ij,ik->ijk',x2,x2)/x2_norm2[:,None,None] # n*d1*d2
    out2 = tmp3[None, :, :, :]/x1_norm[:,None, None,None]/x2_norm[None,:,None,None]
    #out2 = tmp3[None, :, :]/x1_norm[:,None,None]/x2_norm[None,:,None] #[m,n,d2]

    return out2 - out1

def fun_dD_dx1(x1, x2, x1_norm, x2_norm, d):
    out, _ = fun_dd_dx1(x1, x2, x1_norm, x2_norm)
    dD_dx1 = -2*np.einsum("ij, ijk->ijk", d, out) #m,n;  m,n,d -> m,d
    return dD_dx1, dD_dx1.sum(axis=1)

def fun_dD_dx2(x1, x2, x1_norm, x2_norm, d):
    out, _ = fun_dd_dx2(x1, x2, x1_norm, x2_norm)
    dD_dx2 = np.einsum("ij, ijk->ijk", -2*d, out) # -> n,d
    return dD_dx2, dD_dx2.sum(axis=0)

def fun_d2D_dx1dx2(x1, x2, x1_norm, x2_norm, d):

    d2d_dx1dx2 = fun_d2d_dx1dx2(x1, x2, x1_norm, x2_norm)
    _, dd_dx1 = fun_dd_dx1(x1, x2, x1_norm, x2_norm) # [m, d1]
    _, dd_dx2 = fun_dd_dx2(x1, x2, x1_norm, x2_norm) # [n, d2]
    d2D_dx1dx2 = np.einsum('ik, jl->ijkl', dd_dx1, dd_dx2)
    d2D_dx1dx2 += np.einsum("ij, ijkl->ijkl", d, d2d_dx1dx2)
    return -2*d2D_dx1dx2

def fun_dk_dx1(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    dD_dx1, _ = fun_dD_dx1(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    return np.einsum("ij,ijk->ik", dk_dD,dD_dx1)

def fun_dk_dx2(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    dD_dx2, _ = fun_dD_dx2(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    return np.einsum("ij,ijk->jk", dk_dD, dD_dx2)
    
def fun_d2k_dx1x2(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    dD_dx1, _ = fun_dD_dx1(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    dD_dx2, _ = fun_dD_dx2(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    #d2D_dx1dx2 = 
    #d2k_dD = 
    tmp1 = dk_dD*d2D_dx1dx2 + d2k_dD * (dD_dx1, dD_dx2)
    return np.einsum("ij,ijk->jk", dk_dD, dD_dx2)



if __name__ == "__main__":
    m, n, k, sigma2, l2 = 2, 2, 3, 1.1, 0.9
    x1 = np.random.random([m, k])
    x2 = np.random.random([n, k])
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    D1, d1 = fun_D(x1, x2, x1_norm, x2_norm)
    _, dd_dx1_np = fun_dd_dx1(x1, x2, x1_norm, x2_norm)
    _, dd_dx2_np = fun_dd_dx2(x1, x2, x1_norm, x2_norm)
    d2d_dx1dx2_np = fun_d2d_dx1dx2(x1, x2, x1_norm, x2_norm)
    print(d2d_dx1dx2_np.shape)
    print(d2d_dx1dx2_np)
    #_, dd_dx1_np = fun_dd_dx1(x1, x2, x1_norm, x2_norm)
    #_, dd_dx1_np = fun_dd_dx1(x1, x2, x1_norm, x2_norm)
    #d2d_dx1dx2_np = fun_d2d_dx1dx2(x1, x2, x1_norm, x2_norm)
    
    import torch
    print("numerical")
    def d_torch(x1, x2):
        d=x1@x2.T/(torch.ger(torch.norm(x1, dim=1), torch.norm(x2, dim=1)))
        return d
    def D_torch(x1, x2):
        return 1-d_torch(x1, x2)**2
    
    def k_torch(x1, x2, sigma2, l2):
        D = D_torch(x1, x2)
        k = sigma2*(torch.exp(-0.5*D/l2).sum())
        return k 
 
    def t_torch(x1, x2):
        d=x1@x2.T/(torch.ger(torch.norm(x1, dim=1)**3, torch.norm(x2, dim=1)))
        return torch.einsum("ij,jk", d, x2)
 
    t_x1 = torch.tensor(x1, requires_grad=True)
    t_x2 = torch.tensor(x2, requires_grad=True)
    d = d_torch(t_x1, t_x2).sum()
    print("dd_dx1")
    print(torch.autograd.grad(d, t_x1, retain_graph=True))    
    print("dd_dx2")
    print(torch.autograd.grad(d, t_x2))    
    print("--------------")
    t_x1 = torch.tensor(x1, requires_grad=True)
    t_x2 = torch.tensor(x2, requires_grad=True)
    print("d2d_dx1dx2")
    eps = 1e-6
    for i in range(t_x2.size()[0]):
        for j in range(t_x2.size()[1]):
            d1 = d_torch(t_x1, t_x2).sum()
            grad1 = torch.autograd.grad(d1, t_x1)
            tmp = t_x2.clone()
            tmp[i, j] += eps
            d2 = d_torch(t_x1, tmp).sum()
            grad2 = torch.autograd.grad(d2, t_x1)
            print(i, j, (grad2[0]-grad1[0])/eps)


    #print("dk_dx1")
    #print(fun_dk_dx1(x1, x2, x1_norm, x2_norm, d1, sigma2, l2))
    #print("dk_dx2")
    #print(fun_dk_dx2(x1, x2, x1_norm, x2_norm, d1, sigma2, l2))
 
    #print("dk_dx1")
    #print(torch.autograd.grad(k_torch(x1, x2, sigma2, l2), x1))
    #print("dk_dx2")
    #print(torch.autograd.grad(k_torch(x1, x2, sigma2, l2), x2))
    #print("d2k_dx1x2")

