import numpy as np

def kff_many(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0):
    """
    Compute the energy-force kernel between structures and atoms
    The simplist version
    Args:
        X1: list of tuples ([X, dXdR, ele])
        X2: stacked ([X, dXdR, ele])
    Returns:
        C
    """

    sigma2, sigma02 = sigma**2, sigma0**2
    (x_all, dxdr_all, ele_all, x2_indices) = X2

    # num of X1, num of X2, num of big X2
    m1, m2, m2p = len(X1), len(x2_indices), len(x_all)

    # A dummy array to do the following transformation
    # [10, 10, 20] -> [[0, 10], [10, 20], [20, 40]

    x2_inds = [(0, x2_indices[0])]
    for i in range(1, len(x2_indices)):
        ID = x2_inds[i-1][1]
        x2_inds.append( (ID, ID+x2_indices[i]) )

    # Compute the big C array
    C = np.zeros([m1, m2p, 3, 3])
    for i in range(m1):
        (x1, dx1dr, ele1) = X1[i]
        # reset dk_dD[i,j]=0 if ele[i] != ele[j]
        dk_dD = sigma2*np.ones([len(x1), len(x_all)])
        ans = ele1[:,None] - ele_all[None,:]
        ids = np.where(ans!=0)
        ids = np.where( (ele1[:,None]-ele_all[None,:]) != 0)
        if len(ids[0]) > 0:
            dk_dD[ids] = 0
        C[i] = K_ff(x1, x_all, dx1dr, dxdr_all, sigma2, sigma02, zeta, dk_dD)

    # Collapse the big C array to a small _C
    _C = np.zeros([m1*3, m2*3])
    for j, ind in enumerate(x2_inds):
        tmp = C[:, ind[0]:ind[1], :, :].sum(axis=1)
        for i in range(m1):
            _C[i*3:(i+1)*3, j*3:(j+1)*3]  = tmp[i, :3, :3]
                
    return _C

def K_ff(x1, x2, dx1dr, dx2dr, sigma2, sigma02, zeta, dk_dD, eps=1e-8):
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x1_norm2 = x1_norm**2       
    x1_norm3 = x1_norm**3        
    x1_x1_norm3 = x1/x1_norm3[:,None]
    x1x2_dot = x1@x2.T

    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x2_norm2 = x2_norm**2      
    tmp30 = np.ones(x2.shape)/x2_norm[:,None]
    tmp33 = np.eye(x2.shape[1])[None,:,:] - x2[:,:,None] * (x2/x2_norm2[:,None])[:,None,:]

    x2_norm3 = x2_norm**3    
    x1x2_norm = x1_norm[:,None]*x2_norm[None,:]    
    x2_x2_norm3 = x2/x2_norm3[:,None]

    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1

    tmp11 = x2[None, :, :] * x1_norm[:, None, None]
    tmp12 = x1x2_dot[:,:,None] * (x1/x1_norm[:, None])[:,None,:] 
    tmp13 = x1_norm2[:, None, None] * x2_norm[None, :, None] 
    dd_dx1 = (tmp11-tmp12)/tmp13 
    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None] 
    dd_dx2 = (tmp21-tmp22)/tmp23 
    tmp31 = x1[:,None,:] * tmp30[None,:,:]


    tmp31 = tmp31[:,:,None,:] * x1_x1_norm3[:,None,:,None]
    tmp32 = x1_x1_norm3[:,None,:,None] * x2_x2_norm3[None,:,None,:] * x1x2_dot[:,:,None,None]
    out1 = tmp31-tmp32
    out2 = tmp33[None,:,:,:]/x1x2_norm[:,:,None,None]
    d2d_dx1dx2 = out2 - out1
    #print("d2d_dx1dx2", d2d_dx1dx2[0,0,:3,:3])

    dd_dx1_dd_dx2 = dd_dx1[:,:,:,None] * dd_dx2[:,:,None,:]
    d2D_dx1dx2 = dd_dx1_dd_dx2 * D2[:,:,None,None] * (zeta-1)
    d2D_dx1dx2 += D1[:,:,None,None]*d2d_dx1dx2
    d2D_dx1dx2 *= zeta
    d2k_dx1dx2 = d2D_dx1dx2

    tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None] #n1, n2, d, d
    #print("tmp0", tmp0[tmp0>1e-8])

    _kff1 = (dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None]).sum(axis=(0,2)) # n1,n2,3
    kff = (_kff1[:,:,:,None] * dx2dr[:,:,None,:]).sum(axis=1)  # n2, 3, 3
    #print(kff)
    #import sys; sys.exit()
    return kff

def kff_many_v2(X1, X2, sigma=1.0, zeta=2.0, eps=1e-8):
    """
    Compute the energy-force kernel between structures and atoms
    The simplist version
    Args:
        X1: stacked ([X, dXdR, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    sigma2 = sigma**2
    (x1, dx1dr, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2
    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2 = len(x1_indices), len(x2_indices)

    # Compute the C array
    C = np.zeros([m1*3, m2*3])

    for i in range(len(x1)):
        _x1, _dx1dr, _ele1, _i = x1[i], dx1dr[i], ele1[i], x1_inds[i]
        x1_norm = np.linalg.norm(_x1) #+ eps

        if x1_norm > eps:
            x1_norm2 = x1_norm**2       
            x1_norm3 = x1_norm**3        

            for j in range(len(x2)):
                _x2, _dx2dr, _ele2, _j = x2[j], dx2dr[j], ele2[j], x2_inds[j]
                x2_norm = np.linalg.norm(_x2) #+ eps

                if _ele1 == _ele2 and x2_norm > eps:
                    x2_norm2 = x2_norm**2      
                    x2_norm3 = x2_norm**3    

                    x1x2_norm = x1_norm*x2_norm #+ eps
                    x1x2_dot = _x1@_x2.T

                    # d, D, dk_dD
                    d = x1x2_dot/x1x2_norm

                    D2 = d**(zeta-2)
                    D1 = d*D2
                    D = d*D1
                    dk_dD = sigma2

                    # d2d_dx1dx2
                    dd_dx1 = _x2/x1x2_norm - x1x2_dot/x1_norm3/x2_norm * _x1 # [d]
                    dd_dx2 = _x1/x1x2_norm - x1x2_dot/x1_norm/x2_norm3 * _x2 # [d]

                    tmp1 = (np.eye(len(_x1)) - _x2[:,None] * _x2[None, :] /x2_norm2)/x1x2_norm #[d, d]
                    tmp2 = (_x1[:,None]*_x2[None,:]*x1x2_dot/x2_norm2 - _x1[:,None]*_x1[None,:])/x1_norm3/x2_norm
                    d2d_dx1dx2 = tmp1 + tmp2

                    # d2D_dx1dx2 and d2k_dx1dx2
                    dd_dx1_dd_dx2 = dd_dx1[:,None] * dd_dx2[None,:] # [d, d]
                    d2D_dx1dx2 = dd_dx1_dd_dx2 * D2 * (zeta-1) + D1 * d2d_dx1dx2 # [d, d]
                    d2D_dx1dx2 *= zeta # [d, d]
                    d2k_dx1dx2 = d2D_dx1dx2 * dk_dD # [d, d]

                    # Fill the 3*3 matrix to C between atoms i and j
                    ans = np.einsum("ik,ij,jl->kl", _dx1dr, d2k_dx1dx2, _dx2dr)
                    C[_i*3:(_i+1)*3, _j*3:(_j+1)*3] += ans
                    #import sys; sys.exit()
    return C


from time import time
X1 = np.load('X1.npy', allow_pickle=True)
X2 = np.load('X2.npy', allow_pickle=True)

t0 = time()
C1 = kff_many(X1, X2, sigma=28.835, sigma0=0.01)
print(C1[:3, :3])
print(time()-t0)

t0 = time()
C2 = kff_many_v2(X2, X2, sigma=28.835)
print(C2[:3, :3])
print(time()-t0)

print(np.allclose(C1, C2))
