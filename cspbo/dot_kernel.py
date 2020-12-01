from cffi import FFI
import numpy as np
from .utilities import tuple_to_list, list_to_tuple
from ._dot_kernel import lib
from mpi4py import MPI

ffi = FFI()
def kee_C(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0, grad=False):
    """
    Compute the energy-force kernel between structures and atoms
    Args:
        X1: stacked ([X, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    sigma2, sigma02 = sigma**2, sigma0**2
    if isinstance(X1, list):
        X1 = list_to_tuple(X1, mode='energy')
    (x1, ele1, x1_indices) = X1
    (x2, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])
    pdat_x1=ffi.new('double['+str(m1p*d)+']', list(x1.ravel()))
    pdat_x2=ffi.new('double['+str(m2p*d)+']', list(x2.ravel()))
    pdat_ele1=ffi.new('int['+str(m1p)+']', list(ele1))
    pdat_ele2=ffi.new('int['+str(m2p)+']', list(ele2))
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pout=ffi.new('double['+str(m1*m2)+']')
 
    lib.kee_many(m1p, m2p, d, m2, zeta, sigma2, sigma02,
                 pdat_x1, pdat_ele1, pdat_x1_inds, 
                 pdat_x2, pdat_ele2, pdat_x2_inds, 
                 pout) 

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

    if grad:
        C_s = 2*C/sigma
        C_l = np.zeros([m1, m2])
        return C, C_s, C_l
    else:
        return C


def kef_C(X1, X2, sigma=1.0, zeta=2.0, grad=False, transpose=False):
    """
    Compute the energy-force kernel between structures and atoms

    Args:
        X1: stacked ([X, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
        sigma:
        zeta:
        grad:
        transpose:

    Returns:
        C
    """
    if isinstance(X1, list):
        X1 = list_to_tuple(X1, mode='energy')

    if isinstance(X2, list):
        X2 = list_to_tuple(X2)


    (x1, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])

    # copy the arrays to memory
    pdat_x1=ffi.new('double['+str(m1p*d)+']', list(x1.ravel()))
    pdat_x2=ffi.new('double['+str(m2p*d)+']', list(x2.ravel()))
    pdat_ele1=ffi.new('int['+str(m1p)+']', list(ele1))
    pdat_ele2=ffi.new('int['+str(m2p)+']', list(ele2))
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', list(dx2dr.ravel()))
    pout=ffi.new('double['+str(m1*m2*3)+']')

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

    if transpose:
        C = C.T

    if grad:
        C_s = 2*C/sigma
        C_l = np.zeros([m1, m2*3])
        
        return C, C_s, C_l
    else:
        return C

def kff_C(X1, X2, sigma=1.0, zeta=2.0, grad=False):
    """
    Compute the energy-force kernel between structures and atoms
    Args:
        X1: stacked ([X, dXdR, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    comm=MPI.COMM_WORLD 
    rank=comm.Get_rank()

    if isinstance(X1, list):
        X1 = list_to_tuple(X1)


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


    # copy the arrays to memory
    pdat_x1=ffi.new('double['+str(m1p*d)+']', list(x1.ravel()))
    pdat_x2=ffi.new('double['+str(m2p*d)+']', list(x2.ravel()))
    pdat_ele1=ffi.new('int['+str(m1p)+']', list(ele1))
    pdat_ele2=ffi.new('int['+str(m2p)+']', list(ele2))
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pdat_dx1dr=ffi.new('double['+str(m1p*d*3)+']', list(dx1dr.ravel()))
    pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', list(dx2dr.ravel()))
    pout=ffi.new('double['+str(m1*3*m2*3)+']')
    
    lib.kff_many(m1p, m2p, m2p_start, m2p_end, d, m2, zeta,
                 pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds, 
                 pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                 pout) 

    C = np.zeros([m1*3, m2*3])
    for i in range(m1*3):
        for j in range(m2*3):
            C[i, j]=pout[i*m2*3+j] 
    C *= sigma*sigma*zeta
    
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

    if grad:
        C_s = 2*C/sigma
        C_l = np.zeros([m1*3, m2*3])
        return C, C_s, C_l
    else:
        return C


#X1 = np.load('../X2.npy', allow_pickle=True)
#X2 = np.load('../X2.npy', allow_pickle=True)
#X1_EE = np.load('X1_EE.npy', allow_pickle=True)
#X2_EE = np.load('X2_EE.npy', allow_pickle=True)
#X1_FF = np.load('X1_FF.npy', allow_pickle=True)
#X2_FF = np.load('X2_FF.npy', allow_pickle=True)
#sigma = 18.55544058601137
#sigma0 = 0.01
#
#t0 = time()
#C_EE = kee_C(X1_EE, X2_EE, sigma=sigma, sigma0=sigma0)
#print("KEE:", C_EE.shape)
#print(C_EE)
#C_EF = kef_C(X1_EE, X2_FF, sigma=sigma)
#print("KEF:", C_EF.shape)
#print(C_EF[0, :6])
#C_FE = kef_C(X2_EE, X1_FF, sigma=sigma)
#print("KFE:", C_FE.shape)
#print(C_FE.T[:6, :3])
#C_FF = kff_C(X1_FF, X2_FF, sigma=sigma)
#print("KFF:", C_FF.shape)
#print(C_FF[:6, :3])
#print("Elapsed time: ", time()-t0)
