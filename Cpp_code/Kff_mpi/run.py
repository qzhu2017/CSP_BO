from _kff import lib
from time import time
from cffi import FFI
import numpy as np
from mpi4py import MPI

ffi = FFI()

comm=MPI.COMM_WORLD
rank=comm.Get_rank()

def mykff_many(X1, X2, sigma=1.0, zeta=2.0):
    """
    Compute the energy-force kernel between structures and atoms
    The simplist version
    Args:
        X1: stacked ([X, dXdR, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    rank=comm.Get_rank()

    (x1, dx1dr, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])
    
    nprocs = comm.Get_size()
    ish=int(m2p/nprocs)
    irest=m2p-ish*nprocs
    ndim_pr = np.zeros([nprocs],dtype = int)
    n_mpi_pr = np.zeros([nprocs],dtype = int)
    for i in range(irest):
      ndim_pr[i]=ish+1
    for i in range(irest,nprocs):
      ndim_pr[i]=ish

    ind=0
    for i in range(nprocs):
       n_mpi_pr[i]=ind
       ind=ind+ndim_pr[i]
    m2p_start=n_mpi_pr[comm.rank]
    m2p_end=m2p_start+ndim_pr[comm.rank]

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
#    print("before call")
    lib.kff_many(m1p, m2p, m2p_start, m2p_end, d, m2, zeta,
                 pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds, 
                 pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                 pout) 

#    print("after call", time()-t0)
    C = np.zeros([m1*3, m2*3])
    for i in range(m1*3):
        for j in range(m2*3):
            C[i, j]=pout[i*m2*3+j] 
    
    
    Cout = np.zeros([m1*3, m2*3]) 
    comm.Barrier()
    comm.Reduce(
       [C, MPI.DOUBLE],
       [Cout, MPI.DOUBLE],
       op = MPI.SUM,
       root = 0
    )
    ffi.release(pdat_x1)
    ffi.release(pdat_dx1dr)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)    

    Cout *= sigma*sigma*zeta
    return Cout


X1 = np.load('../X1_big.npy', allow_pickle=True)
X2 = np.load('../X2_big.npy', allow_pickle=True)
#X1 = np.load('../X2.npy', allow_pickle=True)
#X2 = np.load('../X2.npy', allow_pickle=True)
t0 = time()
C = mykff_many(X1, X2, sigma=28.835)
if comm.rank==0:
  print(C)
  print(time()-t0)
exit()
