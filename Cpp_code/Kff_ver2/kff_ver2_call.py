from _kff import lib
from time import time
from cffi import FFI
import struct
import numpy as np

ffi = FFI()

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

    (x_all, dxdr_all, ele_all, x2_indices) = X2

    # num of X1, num of X2, num of big X2
    m1, m2, m2p = len(X1), len(x2_indices), len(x_all)

    # A dummy array to do the following transformation
    # [10, 10, 20] -> [[0, 10], [10, 20], [20, 40]

    x2_inds = [(0, x2_indices[0])]
    for i in range(1, len(x2_indices)):
        ID = x2_inds[i-1][1]
        x2_inds.append( (ID, ID+x2_indices[i]) )

    d=len(x_all[0])

    sized=8
    sizei=4
    totalsize=m2p*d*sized+m2p*d*3*sized+m2p*sizei+m2*sizei
    cstr='char['+str(totalsize)+']'

    pdat=ffi.new(cstr)
    mempoint=0
    for i in range(m2p):
        for j in range(d):
            ffi.memmove(pdat+mempoint,struct.pack('d',x_all[i][j]),sized)
            mempoint=mempoint+sized
    for i in range(m2p):
        for j in range(d):
            for k in range(3):
              ffi.memmove(pdat+mempoint,struct.pack('d',dxdr_all[i][j][k]),sized)
              mempoint=mempoint+sized
    for i in range(m2p):
        ffi.memmove(pdat+mempoint,struct.pack('i',ele_all[i]),sizei)
        mempoint=mempoint+sizei
    for i in range(m2):
        ffi.memmove(pdat+mempoint,struct.pack('i',x2_indices[i]),sizei)
        mempoint=mempoint+sizei

    totalsize=m2*3*m2*3*sized
    cstr='char['+str(totalsize)+']'    
    pout=ffi.new(cstr)
    lib.kff_many(m2p,d,m2,sigma,pdat,pout) #n,d,x2i  
        # Compute the big C array
    C = np.zeros([m2*3, m2*3])
    for i in range(m2*3):
        for j in range(m2*3):
           C[i][j]=(struct.unpack('d', ffi.unpack(pout+(i*m2*3+j)*sized,sized) ))[0]    
    
    ffi.release(pdat)  
    ffi.release(pout)    

    return C


from time import time
X1 = np.load('X1.npy', allow_pickle=True)
X2 = np.load('X2.npy', allow_pickle=True)

t0 = time()
C2 = kff_many(X2, X2, sigma=28.835)
#print(C2[:9, :9])
print(C2)
print(time()-t0)
exit()


