from _dot_kernel import lib
from time import time
from cffi import FFI
import numpy as np

ffi = FFI()
def kee_C(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0):
    """
    Compute the energy-force kernel between structures and atoms
    Args:
        X1: stacked ([X, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    sigma2, sigma02 = sigma**2, sigma0**2
    (x1, ele1, x1_indices) = X1
    (x2, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])

    totalsize = m1p*d
    cstr='double['+str(totalsize)+']'
    pdat_x1=ffi.new(cstr)
    for i in range(m1p):
        for j in range(d):
            pdat_x1[i*d+j] = x1[i,j]

    totalsize = m2p*d
    cstr='double['+str(totalsize)+']'
    pdat_x2=ffi.new(cstr)
    for i in range(m2p):
        for j in range(d):
            pdat_x2[i*d+j] = x2[i,j]

    totalsize=m1p
    cstr='int['+str(totalsize)+']'
    pdat_ele1=ffi.new(cstr)               
    for i in range(m1p):
        pdat_ele1[i] = ele1[i]

    totalsize=m2p
    cstr='int['+str(totalsize)+']'
    pdat_ele2=ffi.new(cstr)               
    for i in range(m2p):
        pdat_ele2[i] = ele2[i]

    totalsize = m1p
    cstr='int['+str(totalsize)+']'
    pdat_x1_inds = ffi.new(cstr)                 
    for i in range(m1p):
        pdat_x1_inds[i] = x1_inds[i]

    totalsize = m2p
    cstr='int['+str(totalsize)+']'
    pdat_x2_inds = ffi.new(cstr)                 
    for i in range(m2p):
        pdat_x2_inds[i] = x2_inds[i]

    totalsize = m1*m2
    cstr='double['+str(totalsize)+']'    
    pout=ffi.new(cstr)
    
    #t0 = time()
    #print("before call")
    lib.kee_many(m1p, m2p, d, m2, zeta, sigma2, sigma02,
                 pdat_x1, pdat_ele1, pdat_x1_inds, 
                 pdat_x2, pdat_ele2, pdat_x2_inds, 
                 pout) 

    #print("after call", time()-t0)
    C = np.zeros([m1, m2])
    for i in range(m1):
        for j in range(m2):
            C[i, j]=pout[i*m2+j]/(x1_indices[i]*x2_indices[j])
    
    ffi.release(pdat_x1)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)    

    return C


def kef_C(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0):
    """
    Compute the energy-force kernel between structures and atoms
    Args:
        X1: stacked ([X, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    (x1, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])

    totalsize = m1p*d
    cstr='double['+str(totalsize)+']'
    pdat_x1=ffi.new(cstr)
    for i in range(m1p):
        for j in range(d):
            pdat_x1[i*d+j] = x1[i,j]

    totalsize = m2p*d
    cstr='double['+str(totalsize)+']'
    pdat_x2=ffi.new(cstr)
    for i in range(m2p):
        for j in range(d):
            pdat_x2[i*d+j] = x2[i,j]

    totalsize=m2p*d*3
    cstr='double['+str(totalsize)+']'
    pdat_dx2dr=ffi.new(cstr)           
    for i in range(m2p):
        for j in range(d):
            for k in range(3):
                pdat_dx2dr[(i*d+j)*3+k] = dx2dr[i,j,k]

    totalsize=m1p
    cstr='int['+str(totalsize)+']'
    pdat_ele1=ffi.new(cstr)               
    for i in range(m1p):
        pdat_ele1[i] = ele1[i]

    totalsize=m2p
    cstr='int['+str(totalsize)+']'
    pdat_ele2=ffi.new(cstr)               
    for i in range(m2p):
        pdat_ele2[i] = ele2[i]

    totalsize = m1p
    cstr='int['+str(totalsize)+']'
    pdat_x1_inds = ffi.new(cstr)                 
    for i in range(m1p):
        pdat_x1_inds[i] = x1_inds[i]

    totalsize = m2p
    cstr='int['+str(totalsize)+']'
    pdat_x2_inds = ffi.new(cstr)                 
    for i in range(m2p):
        pdat_x2_inds[i] = x2_inds[i]

    totalsize = m1*m2*3
    cstr='double['+str(totalsize)+']'    
    pout=ffi.new(cstr)
    
    lib.kef_many(m1p, m2p, d, m2, zeta,
                 pdat_x1, pdat_ele1, pdat_x1_inds, 
                 pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                 pout) 
    C = np.zeros([m1, m2*3])
    for i in range(m1):
        for j in range(m2*3):
            C[i, j]=pout[i*m2*3+j]/x1_indices[i]
    C *= -sigma*sigma
    
    ffi.release(pdat_x1)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)    

    return C




def kff_C(X1, X2, sigma=1.0, zeta=2.0):
    """
    Compute the energy-force kernel between structures and atoms
    Args:
        X1: stacked ([X, dXdR, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    (x1, dx1dr, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])

    totalsize = m1p*d
    cstr='double['+str(totalsize)+']'
    pdat_x1=ffi.new(cstr)
    for i in range(m1p):
        for j in range(d):
            pdat_x1[i*d+j] = x1[i,j]

    totalsize = m2p*d
    cstr='double['+str(totalsize)+']'
    pdat_x2=ffi.new(cstr)
    for i in range(m2p):
        for j in range(d):
            pdat_x2[i*d+j] = x2[i,j]

    totalsize=m1p*d*3
    cstr='double['+str(totalsize)+']'
    pdat_dx1dr=ffi.new(cstr)           
    for i in range(m1p):
        for j in range(d):
            for k in range(3):
                pdat_dx1dr[(i*d+j)*3+k] = dx1dr[i,j,k]

    totalsize=m2p*d*3
    cstr='double['+str(totalsize)+']'
    pdat_dx2dr=ffi.new(cstr)           
    for i in range(m2p):
        for j in range(d):
            for k in range(3):
                pdat_dx2dr[(i*d+j)*3+k] = dx2dr[i,j,k]

    totalsize=m1p
    cstr='int['+str(totalsize)+']'
    pdat_ele1=ffi.new(cstr)               
    for i in range(m1p):
        pdat_ele1[i] = ele1[i]

    totalsize=m2p
    cstr='int['+str(totalsize)+']'
    pdat_ele2=ffi.new(cstr)               
    for i in range(m2p):
        pdat_ele2[i] = ele2[i]

    totalsize = m1p
    cstr='int['+str(totalsize)+']'
    pdat_x1_inds = ffi.new(cstr)                 
    for i in range(m1p):
        pdat_x1_inds[i] = x1_inds[i]

    totalsize = m2p
    cstr='int['+str(totalsize)+']'
    pdat_x2_inds = ffi.new(cstr)                 
    for i in range(m2p):
        pdat_x2_inds[i] = x2_inds[i]

    totalsize = m1*3*m2*3
    cstr='double['+str(totalsize)+']'    
    pout=ffi.new(cstr)
    
    t0 = time()
    print("before call")
    lib.kff_many(m1p, m2p, d, m2, zeta,
                 pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds, 
                 pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                 pout) 

    print("after call", time()-t0)
    C = np.zeros([m1*3, m2*3])
    for i in range(m1*3):
        for j in range(m2*3):
            C[i, j]=pout[i*m2*3+j] 
    C *= sigma*sigma*zeta
    
    ffi.release(pdat_x1)
    ffi.release(pdat_dx1dr)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)    

    return C


#X1 = np.load('../X2.npy', allow_pickle=True)
#X2 = np.load('../X2.npy', allow_pickle=True)
X1_EE = np.load('X1_EE.npy', allow_pickle=True)
X2_EE = np.load('X2_EE.npy', allow_pickle=True)
X1_FF = np.load('X1_FF.npy', allow_pickle=True)
X2_FF = np.load('X2_FF.npy', allow_pickle=True)
sigma = 18.55544058601137
sigma0 = 0.01

t0 = time()
C_EE = kee_C(X1_EE, X2_EE, sigma=sigma, sigma0=sigma0)
print("KEE:", C_EE.shape)
print(C_EE)
C_EF = kef_C(X1_EE, X2_FF, sigma=sigma)
print("KEF:", C_EF.shape)
print(C_EF[0, :6])
C_FE = kef_C(X2_EE, X1_FF, sigma=sigma)
print("KFE:", C_FE.shape)
print(C_FE.T[:6, :3])
C_FF = kff_C(X1_FF, X2_FF, sigma=sigma)
print("KFF:", C_FF.shape)
print(C_FF[:6, :3])
print("Elapsed time: ", time()-t0)
exit()


