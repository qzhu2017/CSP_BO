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
    _k, _ = fun_k(x1, x2, x1_norm, x2_norm, sigma2, l2)
    return -0.5*_k/l2

def fun_d2k_dDdsigma(x1, x2, x1_norm, x2_norm, sigma, l):
    _k, _ = fun_k(x1, x2, x1_norm, x2_norm, sigma**2, l**2)
    return -_k/sigma/l**2

def fun_d2k_dDdl(x1, x2, x1_norm, x2_norm, sigma, l):
    D, _ = fun_D(x1, x2, x1_norm, x2_norm)
    _k, _ = fun_k(x1, x2, x1_norm, x2_norm, sigma**2, l**2)
    return _k*(-0.5*D/l**2 + 1)/l**3

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

    d2d_dx1dx2 = fun_d2d_dx1dx2(x1, x2, x1_norm, x2_norm) #[m,n,d1,d2]
    dd_dx1, _ = fun_dd_dx1(x1, x2, x1_norm, x2_norm) # [m, n, d1]
    dd_dx2, _ = fun_dd_dx2(x1, x2, x1_norm, x2_norm) # [m, n, d2]
    d2D_dx1dx2 = np.einsum('ijk, ijl->ijkl', dd_dx1, dd_dx2)
    d2D_dx1dx2 += np.einsum("ij, ijkl->ijkl", d, d2d_dx1dx2)
    return -2*d2D_dx1dx2

def fun_dk_dx1(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    dD_dx1, _ = fun_dD_dx1(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    out = np.einsum("ij,ijk->ik", dk_dD,dD_dx1)         #m, d1
    return out, dD_dx1

def fun_dk_dx2(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    dD_dx2, _ = fun_dD_dx2(x1, x2, x1_norm, x2_norm, d)        #m, n, d
    out = np.einsum("ij,ijk->jk", dk_dD, dD_dx2)        #n, d2
    return out, dD_dx2
    
def fun_d2k_dx1dx2(x1, x2, x1_norm, x2_norm, d, sigma2, l2):
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, l2) #m, n
    
    d2D_dx1dx2 = fun_d2D_dx1dx2(x1, x2, x1_norm, x2_norm, d) #m, n, d, d
    dD_dx1, _ = fun_dD_dx1(x1, x2, x1_norm, x2_norm, d)        #m, n, d1
    dD_dx2, _ = fun_dD_dx2(x1, x2, x1_norm, x2_norm, d)        #m, n, d2
    tmp1 = np.einsum('ijk,ijl->ijkl', dD_dx1, dD_dx2)          #m, n, d1, d2
    #_, dD_dx1 = fun_dD_dx1(x1, x2, x1_norm, x2_norm, d)        #m, d1
    #_, dD_dx2 = fun_dD_dx2(x1, x2, x1_norm, x2_norm, d)        #m, d2
    #tmp2 = np.einsum('ik,jl->ijkl', dD_dx1, dD_dx2)  #m, n, d1, d2

    tmp = d2D_dx1dx2 - 0.5/l2*tmp1 #m, n, d1, d2
    return tmp*dk_dD[:,:,None,None], tmp



if __name__ == "__main__":
   
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
 
    def test_fun(target="d"):
        m, n, k, sigma2, l2 = 2, 2, 3, 1.1, 0.9
        x1 = np.random.random([m, k])
        x2 = np.random.random([n, k])
        x1_norm = np.linalg.norm(x1, axis=1)
        x2_norm = np.linalg.norm(x2, axis=1)
        D1, d1 = fun_D(x1, x2, x1_norm, x2_norm)


        if target == "d":
            fun_df_dx1 = fun_dd_dx1
            fun_df_dx2 = fun_dd_dx2
            fun_d2f_dx1dx2 = fun_d2d_dx1dx2
            func = d_torch
            _, df_dx1_np = fun_df_dx1(x1, x2, x1_norm, x2_norm)
            _, df_dx2_np = fun_df_dx2(x1, x2, x1_norm, x2_norm)
            d2f_dx1dx2_np = fun_d2f_dx1dx2(x1, x2, x1_norm, x2_norm)

        elif target == 'D':
            fun_df_dx1 = fun_dD_dx1
            fun_df_dx2 = fun_dD_dx2
            fun_d2f_dx1dx2 = fun_d2D_dx1dx2
            func = D_torch
            _, df_dx1_np = fun_df_dx1(x1, x2, x1_norm, x2_norm, d1)
            _, df_dx2_np = fun_df_dx2(x1, x2, x1_norm, x2_norm, d1)
            d2f_dx1dx2_np = fun_d2f_dx1dx2(x1, x2, x1_norm, x2_norm, d1)

        elif target == 'k':
            fun_df_dx1 = fun_dk_dx1
            fun_df_dx2 = fun_dk_dx2
            fun_d2f_dx1dx2 = fun_d2k_dx1dx2
            func = k_torch
            df_dx1_np, _ = fun_df_dx1(x1, x2, x1_norm, x2_norm, d1, sigma2, l2)
            df_dx2_np, _ = fun_df_dx2(x1, x2, x1_norm, x2_norm, d1, sigma2, l2)
            d2f_dx1dx2_np, _ = fun_d2f_dx1dx2(x1, x2, x1_norm, x2_norm, d1, sigma2, l2)

        print("testing ", target)
           
        print("df_dx1 from np")
        print(df_dx1_np)
        print("df_dx2 from np")
        print(df_dx2_np)
        print("d2f_dx1dx2 from np")
        print(np.transpose(d2f_dx1dx2_np, axes=(0,1,3,2)))
 
        t_x1 = torch.tensor(x1, requires_grad=True)
        t_x2 = torch.tensor(x2, requires_grad=True)
        if target in ['d', 'D']:
            d = func(t_x1, t_x2).sum()
        else:
            d = func(t_x1, t_x2, sigma2, l2).sum()

        print("df_dx1")
        print(torch.autograd.grad(d, t_x1, retain_graph=True)[0].numpy())    
        print("df_dx2")
        print(torch.autograd.grad(d, t_x2)[0].numpy())    
        t_x1 = torch.tensor(x1, requires_grad=True)
        t_x2 = torch.tensor(x2, requires_grad=True)
        print("d2f_dx1dx2")
        eps = 1e-6
        for i in range(t_x2.size()[0]):
            for j in range(t_x2.size()[1]):
                tmp = t_x2.clone()
                tmp[i, j] += eps
                if target in ['d', 'D']:
                    d1 = func(t_x1, t_x2).sum()
                    d2 = func(t_x1, tmp).sum()
                else:
                    d1 = func(t_x1, t_x2, sigma2, l2).sum()
                    d2 = func(t_x1, tmp, sigma2, l2).sum()
                    
                grad1 = torch.autograd.grad(d1, t_x1)
                grad2 = torch.autograd.grad(d2, t_x1)
                print(((grad2[0]-grad1[0])/eps).numpy())

    test_fun("d")
    test_fun("D")
    test_fun("k")

