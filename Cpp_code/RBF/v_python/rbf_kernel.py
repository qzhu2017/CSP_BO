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


def kef_many(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False, stress=False):  
    sigma2, l2, zeta = sigma**2, l**2, zeta
    (x1, ele1, x1_indices) = X1
    (x_all, dxdr_all, ele_all, x2_indices) = X2

    m1, m2, m2p = 1, len(x2_indices), len(x_all)
   
    x2_inds = [(0, x2_indices[0])]
    for i in range(1, len(x2_indices)):
        ID = x2_inds[i-1][1]
        x2_inds.append( (ID, ID+x2_indices[i]) )
   
    if stress or grad:
        C = np.zeros([m1, m2p, 9])
    else:
        C = np.zeros([m1, m2p, 3])
    
    for i in range(m1):
        #if i%1000 == 0: print("Kef", i, time()-t0)
        mask = get_mask(ele1, ele_all)
        C[i] = K_ef(x1, x_all, dxdr_all, sigma2, l2, zeta, grad, mask)

    _C = np.zeros([m1, m2*3])
    if grad:
        _C_s = np.zeros([m1, m2*3])
        _C_l = np.zeros([m1, m2*3])
    elif stress:
        _C1 = np.zeros([m1, m2*6])

    for j, ind in enumerate(x2_inds):
        tmp = C[:, ind[0]:ind[1], :].sum(axis=1) 
        _C[:, j*3:(j+1)*3] =  tmp[:, :3]
        if stress:
            _C1[:, j*6:(j+1)*6] =  tmp[:, 3:]
        elif grad:
            _C_s[:, j*3:(j+1)*3] =  tmp[:, 3:6]
            _C_l[:, j*3:(j+1)*3] =  tmp[:, 6:9]

    if grad:
        return _C, _C_s, _C_l                  
    elif stress:
        return _C, _C1
    else:
        return _C

def kff_many(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False, stress=False):
    sigma2, l2, zeta = sigma**2, l**2, zeta
    
    (x1, dx1dr, ele1, x1_indices) = X1
    (x_all, dxdr_all, ele_all, x2_indices) = X2

    m1, m2, m2p = len(x1_indices), len(x2_indices), len(x_all)
    x2_inds = [(0, x2_indices[0])]
    for i in range(1, len(x2_indices)):
        ID = x2_inds[i-1][1]
        x2_inds.append( (ID, ID+x2_indices[i]) )

    if grad:
        C = np.zeros([m1, m2p, 3, 9])
    elif stress:
        C = np.zeros([m1, m2p, 9, 3])
    else:
        C = np.zeros([m1, m2p, 3, 3])

    count = 0
    print(m1)
    for i in range(m1):
        print(i)
        indices1 = x1_indices[i]
        #mask = get_mask(ele1[count:count+indices1], ele_all)
        batch = 1000
        if m2p > batch:
            for j in range(int(np.ceil(m2p/batch))):
                start = j*batch
                end = min([(j+1)*batch, m2p])
                mask = get_mask(ele1[count:count+indices1], ele_all[start:end])
                C[i, start:end, :, :] = K_ff(x1[count:count+indices1], x_all[start:end], dx1dr[count:count+indices1], dxdr_all[start:end], sigma2, l2, zeta, grad, mask, device='cpu')
        else:
            mask = get_mask(ele1[count:count+indices1], ele_all)
            C[i] = K_ff(x1[count:count+indices1], x_all, dx1dr[count:count+indices1], dxdr_all, sigma2, l2, zeta, grad, mask, device='cpu')
        count += indices1

    _C = np.zeros([m1*3, m2*3])
    if grad:
        _C_s = np.zeros([m1*3, m2*3])
        _C_l = np.zeros([m1*3, m2*3])
    elif stress:
        _C1 = np.zeros([m1*6, m2*3])

    for j, ind in enumerate(x2_inds):
        tmp = C[:, ind[0]:ind[1], :, :].sum(axis=1)
        for i in range(m1):
            _C[i*3:(i+1)*3, j*3:(j+1)*3]  = tmp[i, :3, :3]
            if stress:
                _C1[i*6:(i+1)*6, j*3:(j+1)*3]  = tmp[i, 3:, :3]
            elif grad:
                _C_s[i*3:(i+1)*3, j*3:(j+1)*3]  = tmp[i, :3, 3:6]
                _C_l[i*3:(i+1)*3, j*3:(j+1)*3]  = tmp[i, :3, 6:9]

    if grad:
        return _C, _C_s, _C_l
    elif stress:
        return _C, _C1
    else:
        return _C


def K_ee(x1, x2, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8, wrap=False):
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
        return np.stack((Kee/m, dKee_dsigma/m, dKee_dl/m), axis=-1)
    else:
        n = len(x2)
        if wrap:
            return Kee.sum()/(m*n)
        else:
            return Kee.reshape([n, 1])/m

def K_ef(x1, x2, dx2dr, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8, device='cpu'):
    """
    Compute the Kef between one structure and many atomic configurations
    """
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
    
    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None] 
    dd_dx2 = (tmp21-tmp22)/tmp23

    zd1 = zeta * D1
    dD_dx2 = -zd1[:,:,None] * dd_dx2
    kef1 = (-dD_dx2[:,:,:,None]*dx2dr[None,:,:,:]).sum(axis=2) #[m, n, 9]
    Kef = (kef1*dk_dD[:,:,None]).sum(axis=0) #[n, 9]
    if grad:
        l = np.sqrt(l2)
        l3 = l2*l
        d2k_dDdsigma = 2*dk_dD/np.sqrt(sigma2)
        d2k_dDdl = -((D-1)/l3 + 2/l)*dk_dD

        dKef_dsigma = (kef1*d2k_dDdsigma[:,:,None]).sum(axis=0) 
        dKef_dl     = (kef1*d2k_dDdl[:,:,None]).sum(axis=0)
        return np.concatenate((Kef/m, dKef_dsigma/m, dKef_dl/m), axis=-1)
    else:
        return Kef/m

def K_ff(x1, x2, dx1dr, dx2dr, sigma2, l2, zeta=2, grad=False, mask=None, eps=0., device='cpu', wrap=False):
    l = np.sqrt(l2)
    l3 = l*l2

    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x1_norm2 = x1_norm**2
    x1_norm3 = x1_norm**3
    x1x2_dot = x1@x2.T
    x1_x1_norm3 = x1/x1_norm3[:,None]

    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x2_norm2 = x2_norm**2
    tmp30 = np.ones(x2.shape)/x2_norm[:,None]
    tmp33 = np.eye(x2.shape[1])[None,:,:] - x2[:,:,None] * (x2/x2_norm2[:,None])[:,None,:]

    x2_norm3 = x2_norm**3
    x1x2_norm = x1_norm[:,None]*x2_norm[None,:]
    x2_x2_norm3 = x2/x2_norm3[:,None]

    d = x1x2_dot/(x1x2_norm)
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1
    k = sigma2*np.exp(-(0.5/l2)*(1-D))

    if mask is not None:
        k[mask] = 0

    dk_dD = (0.5/l2)*k
    zd2 = 0.5/l2*zeta*zeta*(D1**2)

    tmp31 = x1[:,None,:] * tmp30[None,:,:]

    tmp11 = x2[None, :, :] * x1_norm[:, None, None]
    tmp12 = x1x2_dot[:,:,None] * (x1/x1_norm[:, None])[:,None,:]
    tmp13 = x1_norm2[:, None, None] * x2_norm[None, :, None]
    dd_dx1 = (tmp11-tmp12)/tmp13

    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None]
    dd_dx2 = (tmp21-tmp22)/tmp23  # (29, 1435, 24)


    tmp31 = tmp31[:,:,None,:] * x1_x1_norm3[:,None,:,None]
    tmp32 = x1_x1_norm3[:,None,:,None] * x2_x2_norm3[None,:,None,:] * x1x2_dot[:,:,None,None]
    out1 = tmp31-tmp32
    out2 = tmp33[None,:,:,:]/x1x2_norm[:,:,None,None]
    d2d_dx1dx2 = out2 - out1

    dd_dx1_dd_dx2 = dd_dx1[:,:,:,None] * dd_dx2[:,:,None,:]
    dD_dx1_dD_dx2 = zd2[:,:,None,None] * dd_dx1_dd_dx2

    d2D_dx1dx2 = dd_dx1_dd_dx2 * D2[:,:,None,None] * (zeta-1)
    d2D_dx1dx2 += D1[:,:,None,None]*d2d_dx1dx2
    d2D_dx1dx2 *= zeta
    d2k_dx1dx2 = d2D_dx1dx2 + dD_dx1_dD_dx2 # m, n, d1, d2

    if grad:

        K_ff_0 = (d2k_dx1dx2[:,:,:,:,None] * dx1dr[:,None,:,None,:]).sum(axis=2)
        K_ff_0 = (K_ff_0[:,:,:,:,None] * dx2dr[None,:,:,None,:]).sum(axis=2)
        Kff = (K_ff_0 * dk_dD[:,:,None,None]).sum(axis=0)

        d2k_dDdsigma = 2*dk_dD/np.sqrt(sigma2)
        d2k_dDdl = ((D-1)/l3 + 2/l)*dk_dD

        dKff_dsigma = (K_ff_0 * d2k_dDdsigma[:,:,None,None]).sum(axis=0)
        dKff_dl = (-K_ff_0 * d2k_dDdl[:,:,None,None]).sum(axis=0)

        tmp = -dD_dx1_dD_dx2/l*2
        K_ff_1 = (tmp[:,:,:,:,None] * dx1dr[:,None,:,None,:]).sum(axis=2)
        K_ff_1 = (K_ff_1[:,:,:,:,None] * dx2dr[None,:,:,None,:]).sum(axis=2)
        dKff_dl += (K_ff_1 * dk_dD[:,:,None,None]).sum(axis=0) #/l*2
        return np.concatenate((Kff, dKff_dsigma, dKff_dl), axis=-1)
    else:
        tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None] #n1, n2, d, d
        _kff1 = (dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None]).sum(axis=(0,2)) # n1,n2,3
        kff = (_kff1[:,:,:,None] * dx2dr[:,:,None,:]).sum(axis=1)  # n2, 3, 9
        return kff

    
X1_EE = np.load('../X1_EE.npy', allow_pickle=True)
X2_EE = np.load('../X2_EE.npy', allow_pickle=True)
X1_FF = np.load('../X1_FF.npy', allow_pickle=True)
X2_FF = np.load('../X2_FF.npy', allow_pickle=True)
sigma = 9.55544058601137
l = 0.5

t0 = time()
# No grad
#C_EE = kee_many(X1_EE, X2_EE, sigma=sigma, l=l)
#C_EF = kef_many(X1_EE, X2_FF, sigma=sigma, l=l)
#C_FF = kff_many(X1_FF, X2_FF, sigma=sigma, l=l)


# Grad
C_EE, C_s_EE, C_l_EE = kee_many(X1_EE, X2_EE, sigma=sigma, l=l, grad=True)
C_EF, C_s_EF, C_l_EF = kef_many(X1_EE, X2_FF, sigma=sigma, l=l, grad=True)
C_FF, C_s_FF, C_l_FF = kff_many(X1_FF, X2_FF, sigma=sigma, l=l, grad=True)

print("Elapsed time: ", time()-t0)

# No Grad
#np.save("kernel_EF.npy", C_EE)
#np.save("kernel_EF.npy", C_EF)
#np.save("kernel_FF.npy", C_FF)

# Grad
np.save("kernel_EE.npy", C_EE)
np.save("kernel_EE_l.npy", C_l_EE)
np.save("kernel_EE_s.npy", C_s_EE)
np.save("kernel_EF.npy", C_EF)
np.save("kernel_EF_l.npy", C_l_EF)
np.save("kernel_EF_s.npy", C_s_EF)
np.save("kernel_FF.npy", C_FF)
np.save("kernel_FF_l.npy", C_l_FF)
np.save("kernel_FF_s.npy", C_s_FF)
