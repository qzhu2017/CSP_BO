from _kff import lib
from time import time
from cffi import FFI
import numpy as np

ffi = FFI()

def mykff_many(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0):
    """
    Compute the energy-force kernel between structures and atoms
    The simplist version
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
    pdat_x1=ffi.new(cstr, list(x1.ravel()))
    
    totalsize = m2p*d
    cstr='double['+str(totalsize)+']'
    pdat_x2=ffi.new(cstr)#, list(x2.ravel()))
    #print(np.array(list(pdat_x2)))
    #import sys; sys.exit()
    #print(np.allclose(np.array(list(pdat_x2)), x2.ravel()))
    #import sys; sys.exit()

    for i in range(m2p):
        for j in range(d):
            pdat_x2[i*d+j] = x2[i,j]

    totalsize=m1p*d*3
    cstr='double['+str(totalsize)+']'
    pdat_dx1dr=ffi.new(cstr, list(dx1dr.ravel()))
    
    totalsize=m2p*d*3
    cstr='double['+str(totalsize)+']'
    pdat_dx2dr=ffi.new(cstr, list(dx2dr.ravel()))
    
    totalsize=m1p
    cstr='int['+str(totalsize)+']'
    pdat_ele1=ffi.new(cstr, list(ele1))
    
    totalsize=m2p
    cstr='int['+str(totalsize)+']'
    pdat_ele2=ffi.new(cstr, list(ele2))
    
    totalsize = m1p
    cstr='int['+str(totalsize)+']'
    pdat_x1_inds = ffi.new(cstr, x1_inds)
    
    totalsize = m2p
    cstr='int['+str(totalsize)+']'
    pdat_x2_inds = ffi.new(cstr, x2_inds)        
    
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
    C = np.array(list(pout))
    C = np.reshape(C, [m1*3, m2*3])
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


X1 = np.load('../X1_big.npy', allow_pickle=True)
X2 = np.load('../X2_big.npy', allow_pickle=True)

t0 = time()
C = mykff_many(X1, X2, sigma=28.835)
np.save("C_nloop", C)
#print(C[:3, :3])
print(time()-t0)
exit()
