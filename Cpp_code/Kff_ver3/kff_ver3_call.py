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
#    totalsize=m2p*d*sized+m2p*d*3*sized+m2p*sizei+m2*sizei

    totalsize=m2p*d
    cstr='double['+str(totalsize)+']'
    pdat_x1=ffi.new(cstr)
    for i in range(m2p):
        for j in range(d):
            ffi.memmove(pdat_x1+i*d+j,struct.pack('d',x_all[i][j]),sized)
    totalsize=m2p*d*3
    cstr='double['+str(totalsize)+']'
    pdat_dxdr_all=ffi.new(cstr)           
    for i in range(m2p):
        for j in range(d):
            for k in range(3):
              ffi.memmove(pdat_dxdr_all+(i*d+j)*3+k,struct.pack('d',dxdr_all[i][j][k]),sized)
    totalsize=m2p
    cstr='int['+str(totalsize)+']'
    pdat_ele_all=ffi.new(cstr)               
    for i in range(m2p):
        ffi.memmove(pdat_ele_all+i,struct.pack('i',ele_all[i]),sizei)
    totalsize=m2
    cstr='int['+str(totalsize)+']'
    pdat_x2_indices=ffi.new(cstr)                 
    for i in range(m2):
        ffi.memmove(pdat_x2_indices+i,struct.pack('i',x2_indices[i]),sizei)

    totalsize=m2*3*m2*3
    cstr='double['+str(totalsize)+']'    
    pout=ffi.new(cstr)
    
#    print("before call")
#    print(pdat_x1,pdat_dxdr_all,pdat_ele_all,pdat_x2_indices,pout)
#    print(x2_indices)
#    exit()
    lib.kff_many(m2p,d,m2,sigma,pdat_x1,pdat_dxdr_all,pdat_ele_all,pdat_x2_indices,pout) #n,d,x2i 

#    exit()
#    print("after call")
#void kff_many(int n, int d, int x2i, double sigma, double* x1, double* dx1dr, int* ele1, int* x1_indices, double* pout){
        # Compute the big C array
    C = np.zeros([m2*3, m2*3])
    for i in range(m2*3):
        for j in range(m2*3):
#           C[i][j]=(struct.unpack('d', ffi.unpack(pout+(i*m2*3+j),1) ))[0]   
           C[i][j]=pout[i*m2*3+j] 
    
    ffi.release(pdat_x1)
    ffi.release(pdat_dxdr_all)
    ffi.release(pdat_ele_all)
    ffi.release(pdat_x2_indices)  
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


