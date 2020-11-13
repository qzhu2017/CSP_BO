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
        #print(C[0].sum(axis=0))
        #import sys; sys.exit()
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

    dd_dx1_dd_dx2 = dd_dx1[:,:,:,None] * dd_dx2[:,:,None,:]
    d2D_dx1dx2 = dd_dx1_dd_dx2 * D2[:,:,None,None] * (zeta-1)
    d2D_dx1dx2 += D1[:,:,None,None]*d2d_dx1dx2
    d2D_dx1dx2 *= zeta
    d2k_dx1dx2 = d2D_dx1dx2

    tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None] #n1, n2, d, d
    _kff1 = (dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None]).sum(axis=(0,2)) # n1,n2,3
    kff = (_kff1[:,:,:,None] * dx2dr[:,:,None,:]).sum(axis=1)  # n2, 3, 3
    return kff


from time import time
t0 = time()
X1 = np.load('X1.npy', allow_pickle=True)
X2 = np.load('X2.npy', allow_pickle=True)

#X1 = np.load('X1_big.npy')
#X2 = np.load('X2.big.npy')

C = kff_many(X1, X2, sigma=28.835, sigma0=0.01)
print(C[:9, :9])
print(time()-t0)
