from _rbf_kernel import lib
from time import time
from cffi import FFI
import numpy as np

ffi = FFI()
def kee_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False):
    """
    Compute the energy-energy relation through RBF kernel.
    Args:
        X1: stack of ([X, ele, indices])
        X2: stack of ([X, ele, indices])
        sigma: 
        l:
        zeta:
        grad: if True, compute gradient w.r.t. hyperparameters
    Returns:
        C: the energy-energy kernel
        C_s: the energy-energy kernel derivative w.r.t. sigma
        C_l: the energy-energy kernel derivative w.r.t. l
    """
    sigma2, l2 = sigma*sigma, l*l

    (x1, ele1, x1_indices) = X1
    (x2, ele2, x2_indices) = X2

    x1_inds, x2_inds = [], []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), x1.shape[1]

    pdat_x1 = ffi.new('double['+str(m1p*d)+']', x1.ravel().tolist())
    pdat_x2 = ffi.new('double['+str(m2p*d)+']', x2.ravel().tolist())
    pdat_ele1 = ffi.new('int['+str(m1p)+']', ele1.tolist())
    pdat_ele2 = ffi.new('int['+str(m2p)+']', ele2.tolist())
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pout=ffi.new('double['+str(m1*m2)+']')

    lib.kee_many(m1p, m2p, d, m2, zeta, sigma2, l2,
                 pdat_x1, pdat_ele1, pdat_x1_inds,
                 pdat_x2, pdat_ele2, pdat_x2_inds,
                 pout)

    C = np.frombuffer(ffi.buffer(pout, m1*m2*8), dtype=np.float64)
    C.shape = (m1, m2)
    C /= (np.array(x1_indices)[:,None] * np.array(x2_indices)[None,:])

    ffi.release(pdat_x1)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)
    ffi.release(pdat_x2)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)
    ffi.release(pout)

    if grad:
        C_s = 2*C/sigma
        C_l = np.zeros([m1, m2]) # Need to change this for RBF
        return C, C_s, C_l
    else:
        return C

def kef_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False, stress=False, transpose=False):
    """
    Compute the energy-force relation through RBF kernel.
    Args:
        X1: stack of ([X, ele, indices])
        X2: stack of ([X, dXdR, ele, indices])
        sigma: 
        l:
        zeta:
        grad: if True, compute gradient w.r.t. hyperparameters
        stress: if True, compute energy-stress relation
        transpose: if True, get the kfe
    Returns:
        C: the energy-force kernel
        C_s: the energy-force kernel derivative w.r.t. sigma
        C_l: the energy-force kernel derivative w.r.t. l
    """
    sigma2, l2 = sigma*sigma, l*l

    (x1, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds, x2_inds = [], []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), x1.shape[1]
    
    pdat_x1 = ffi.new('double['+str(m1p*d)+']', x1.ravel().tolist())
    pdat_x2 = ffi.new('double['+str(m2p*d)+']', x2.ravel().tolist())
    pdat_ele1 = ffi.new('int['+str(m1p)+']', ele1.tolist())
    pdat_ele2 = ffi.new('int['+str(m2p)+']', ele2.tolist())
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)

    if stress:
        pdat_dx2dr=ffi.new('double['+str(m2p*d*9)+']', list(dx2dr.ravel()))
        pout=ffi.new('double['+str(m1*m2*9)+']')
        lib.kef_many_stress(m1p, m2p, d, m2, zeta,
                     pdat_x1, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        d2 = 9
    else:
        pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', list(dx2dr.ravel()))
        pout=ffi.new('double['+str(m1*m2*3)+']')
        lib.kef_many(m1p, m2p, d, m2, zeta, sigma2, l2,
                     pdat_x1, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        d2 = 3
    
    # convert cdata to np.array
    out = np.frombuffer(ffi.buffer(pout, m1*m2*d2*8), dtype=np.float64)
    out.shape = (m1, m2, d2)
    out /= np.array(x1_indices)[:,None,None]
    out *= -1

    C = out[:, :, :3].reshape([m1, m2*3])
    if stress:
        Cs = out[:, :, 3:].reshape([m1, m2*6])
    else:
        Cs = np.zeros([m1, m2*6])

    ffi.release(pdat_x1)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)
    ffi.release(pout)

    if transpose:
        C = C.T
        Cs = Cs.T

    if grad:
        C_s = 2*C/sigma
        C_l = np.zeros([m1, m2*3])
        return C, C_s, C_l
    elif stress:
        return C, Cs
    else:
        return C

def kff_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False, stress=False):
    """
    Compute the force-force relation through RBF kernel.
    Args:
        X1: stack of ([X, ele, indices])
        X2: stack of ([X, dXdR, ele, indices])
        sigma: 
        l:
        zeta:
        grad: if True, compute gradient w.r.t. hyperparameters
        stress: if True, compute force-stress relation
    Returns:
        C: the force-force kernel
        C_s: the force-force kernel derivative w.r.t. sigma
        C_l: the force-force kernel derivative w.r.t. l
    """
    sigma2, l2 = sigma*sigma, l*l

    (x1, dx1dr, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds, x2_inds = [], []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), x1.shape[1]
    
    pdat_x1=ffi.new('double['+str(m1p*d)+']', x1.ravel().tolist())
    pdat_x2=ffi.new('double['+str(m2p*d)+']', x2.ravel().tolist())
    pdat_ele1=ffi.new('int['+str(m1p)+']', ele1.tolist())
    pdat_ele2=ffi.new('int['+str(m2p)+']', ele2.tolist())
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', dx2dr.ravel().tolist())

    m2p_start, m2p_end = int(0), int(0); # Need to fix this for MPI

    if stress:
        pdat_dx1dr=ffi.new('double['+str(m1p*d*9)+']', list(dx1dr.ravel()))
        pout=ffi.new('double['+str(m1*9*m2*3)+']')
        lib.kff_many_stress(m1p, m2p, m2p_start, m2p_end, d, m2, zeta, sigma2, l2, 
                     pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        d1 = 9
    else:
        pdat_dx1dr=ffi.new('double['+str(m1p*d*3)+']', dx1dr.ravel().tolist())
        pout=ffi.new('double['+str(m1*3*m2*3)+']')
        lib.kff_many(m1p, m2p, m2p_start, m2p_end, d, m2, zeta, sigma2, l2,
                     pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        d1 = 3

    out = np.frombuffer(ffi.buffer(pout, m1*d1*m2*3*8), dtype=np.float64)
    out.shape = (m1, d1, m2*3)
    
    Cout = np.zeros([m1, d1, m2*3])
    Cout = out

    ffi.release(pdat_x1)
    ffi.release(pdat_dx1dr)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)

    #Cout *= (sigma*sigma*zeta)
    C = Cout[:, :3, :].reshape([m1*3, m2*3])
    if stress:
        Cs = Cout[:, 3:, :].reshape([m1*6, m2*3])
    if grad:
        C_l = 2*C/sigma
        C_s = np.zeros([m1*3, m2*3])
        return C, C_l, C_s
    elif stress:
        return C, Cs
    else:
        return C

X1_EE = np.load('X1_EE.npy', allow_pickle=True)
X2_EE = np.load('X2_EE.npy', allow_pickle=True)
X1_FF = np.load('X1_FF.npy', allow_pickle=True)
X2_FF = np.load('X2_FF.npy', allow_pickle=True)
sigma = 9.55544058601137

t0 = time()
C_EE = kee_C(X1_EE, X2_EE, sigma=sigma)
C_EF = kef_C(X1_EE, X2_FF, sigma=sigma)
C_FE = kef_C(X2_EE, X1_FF, sigma=sigma)
C_FF = kff_C(X1_FF, X2_FF, sigma=sigma)
print("Elapsed time: ", time()-t0)

np.save("kernel_EE.npy", C_EE)
np.save("kernel_EF.npy", C_EF)
np.save("kernel_FE.npy", C_FE)
np.save("kernel_FF.npy", C_FF)
