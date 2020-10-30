import numpy as np
from .kernel_base import *
from .gkernel_base import *
import cupy as cp
from functools import partial
from multiprocessing import Pool, cpu_count

class RBF_mb():
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 5e+1], [1e-1, 1e+1]], zeta=3, ncpu=1):
        self.name = 'RBF_mb'
        self.bounds = bounds
        self.update(para)
        self.zeta = zeta
        self.ncpu = ncpu
            
    def __str__(self):
        return "{:.3f}**2 *RBF(length={:.3f})".format(self.sigma, self.l)

    def load_from_dict(self, dict0):
        self.sigma = dict0["sigma"]
        self.l = dict0["l"]
        self.zeta = dict0["zeta"]
        self.bounds = dict0["bounds"]
        self.name = dict0["name"]
        
    def save_dict(self):
        """
        save the model as a dictionary in json
        """
        dict = {"name": self.name,
                "sigma": self.sigma,
                "l": self.l,
                "zeta": self.zeta,
                "bounds": self.bounds
               }
        return dict

    def parameters(self):
        return [self.sigma, self.l]
 
    def update(self, para):
        self.sigma, self.l = para[0], para[1]

    def diag(self, data):
        """
        Returns the diagonal of the kernel k(X, X)
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        C_ee, C_ff = None, None
               
        if "energy" in data:
            NE = len(data["energy"])
            C_ee = np.zeros(NE)
            for i in range(NE):
                (x1, ele1) = data["energy"][i]
                mask = get_mask(ele1, ele1)
                C_ee[i] = kee_single(x1, x1, sigma2, l2, zeta, False, mask) 

        if "force" in data:
            NF = len(data["force"])
            C_ff = np.zeros(3*NF)
            for i in range(NF):
                (x1, dx1dr, _, ele1) = data["force"][i]
                mask = get_mask(ele1, ele1)
                C_ff[i*3:(i+1)*3] = np.diag(kff_single(x1, x1, dx1dr, dx1dr, None, sigma2, l2, zeta, False, mask))

        if C_ff is None:
            return C_ee
        elif C_ee is None:
            return C_ff
        else:
            return np.hstack((C_ee, C_ff))

        #return np.ones(N)*sigma2

    def k_total(self, data1, data2=None):
        if data2 is None:
            data2 = data1
            same = True
        else:
            same = False
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        for key1 in data1.keys():
            for key2 in data2.keys():
                if len(data1[key1])>0 and len(data2[key2])>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee = self.kee_many(data1[key1], data2[key2], same=same)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = self.kef_many(data1[key1], data2[key2])
                    elif key1 == 'force' and key2 == 'energy':
                        if not same:
                            C_fe = self.kef_many(data2[key2], data1[key1]).T
                        else:
                            C_fe = C_ef.T 
                    elif key1 == 'force' and key2 == 'force':
                        C_ff = self.kff_many(data1[key1], data2[key2], same=same)

        return build_covariance(C_ee, C_ef, C_fe, C_ff)
        

    def k_total_with_grad(self, data1):
        """
        Compute the covairance for train data
        Both C and its gradient will be returned
        """
        data2 = data1   
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        C_ee_s, C_ef_s, C_fe_s, C_ff_s = None, None, None, None
        C_ee_l, C_ef_l, C_fe_l, C_ff_l = None, None, None, None
        for key1 in data1.keys():
            for key2 in data2.keys():
                if len(data1[key1])>0 and len(data2[key2])>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee, C_ee_s, C_ee_l = self.kee_many(data1[key1], data2[key2], True, True)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef, C_ef_s, C_ef_l = self.kef_many(data1[key1], data2[key2], True)
                        C_fe, C_fe_s, C_fe_l = C_ef.T, C_ef_s.T, C_ef_l.T

                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_ff_s, C_ff_l = self.kff_many(data1[key1], data2[key2], True, True)
        C = build_covariance(C_ee, C_ef, C_fe, C_ff, None, None)
        C_s = build_covariance(C_ee_s, C_ef_s, C_fe_s, C_ff_s, None, None)
        C_l = build_covariance(C_ee_l, C_ef_l, C_fe_l, C_ff_l, None, None)
        return C, np.dstack((C_s, C_l))
    
    def kee_many(self, X1, X2, same=False, grad=False):
        """
        Compute the energy-energy kernel for many structures
        Args:
            X1: list of 2D arrays
            X2: list of 2D arrays
            same: avoid double counting if true
            grad: output gradient if true
        Returns:
            C: M*N 2D array
            C_grad:
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])
        C_s = np.zeros([m1, m2])
        C_l = np.zeros([m1, m2])

        # This loop is rather expansive, a simple way is to do multi-process
        if same:
            indices = np.triu_indices(m1)
            (_is, _js) = indices
        else:
            indices = np.indices((m1, m2))
            _is = indices[0].flatten()
            _js = indices[1].flatten()

        if self.ncpu == 1 or self.ncpu == 'gpu':
            results = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kee_single(x1, x2, sigma2, l2, zeta, grad, mask))
        else:
            fun_vars = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                fun_vars.append((x1, x2, mask))

            with Pool(self.ncpu) as p:
                func = partial(kee_para, (sigma2, l2, zeta, grad))
                results = p.map(func, fun_vars)
                p.close()
                p.join()

        # unpack the results
        for i, j, res in zip(_is, _js, results):
            if grad:
                Kee, dKee_sigma, dKee_l = res
                C[i, j] = Kee
                C_s[i, j] = dKee_sigma
                C_s[j, i] = dKee_sigma
                C_l[i, j] = dKee_l
                C_l[j, i] = dKee_l
            else:
                C[i, j] = res
            if same and (i != j):
                C[j, i] = C[i, j]

        #print(C[:5,:5])
        if grad:
            return C, C_s, C_l
        else:
            return C

    def kff_many(self, X1, X2, same=False, grad=False):
        """
        Compute the energy-force kernel between structures and atoms
        Args:
            X1: list of tuples ([X, dXdR, rdXdR, ele])
            X2: list of tuples ([X, dXdR, rdXdR, ele])
            same: avoid double counting if true
            grad: output gradient if true
        Returns:
            C: M*N 2D array
            C_grad:
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        m1, m2 = len(X1), len(X2)
        C = np.zeros([3*m1, 3*m2])
        C_s = np.zeros([3*m1, 3*m2])
        C_l = np.zeros([3*m1, 3*m2])
        a, b, c = X1[0][1], np.einsum("ik,jl->ijkl", X1[0][0], X2[0][0]), X2[0][1]
        path = np.einsum_path('ikm,ijkl,jln->mn', a, b, c, optimize='greedy')[0]


        # This loop is rather expansive, a simple way is to do multi-process
        if same:
            indices = np.triu_indices(m1)
            (_is, _js) = indices
        else:
            indices = np.indices((m1, m2))
            _is = indices[0].flatten()
            _js = indices[1].flatten()

        if self.ncpu == 1:
            results = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, _, ele1) = X1[i]
                (x2, dx2dr, _, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kff_single(x1, x2, dx1dr, dx2dr, None, sigma2, l2, zeta, grad, mask, path))
        elif self.ncpu == 'gpu':
            results = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, _, ele1) = X1[i]
                (x2, dx2dr, _, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kff_single(x1, x2, dx1dr, dx2dr, None, sigma2, l2, zeta, grad, mask, path, device='gpu'))
        else:
            #print("Parallel version is on: ")
            fun_vars = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, _, ele1) = X1[i]
                (x2, dx2dr, _, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                fun_vars.append((x1, x2, dx1dr, dx2dr, None, mask))

            with Pool(self.ncpu) as p:
                func = partial(kff_para, (sigma2, l2, zeta, grad, path))
                results = p.map(func, fun_vars)
                p.close()
                p.join()

        # unpack the results
        for i, j, res in zip(_is, _js, results):
            if grad:
                Kff, dKff_sigma, dKff_l = res
                C[i*3:(i+1)*3, j*3:(j+1)*3] = Kff
                C_s[i*3:(i+1)*3, j*3:(j+1)*3] = dKff_sigma
                C_s[j*3:(j+1)*3, i*3:(i+1)*3] = dKff_sigma.T
                C_l[i*3:(i+1)*3, j*3:(j+1)*3] = dKff_l
                C_l[j*3:(j+1)*3, i*3:(i+1)*3] = dKff_l.T
            else:
                C[i*3:(i+1)*3, j*3:(j+1)*3] = res
            if same and (i != j):
                C[j*3:(j+1)*3, i*3:(i+1)*3] = C[i*3:(i+1)*3, j*3:(j+1)*3].T
        #print(C[:5,:5])
        #import sys
        #sys.exit()

        if grad:
            return C, C_s, C_l                  
        else:
            return C

    def kef_many(self, X1, X2, grad=False):
        """
        Compute the energy-force kernel between structures and atoms
        Args:
            X1: list of 2D arrays (each N*d)
            X2: list of tuples ([X, dXdR])
            grad: output gradient if true
        Returns:
            C: M*N 2D array
            C_grad:
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, 3*m2])
        C_s = np.zeros([m1, 3*m2])
        C_l = np.zeros([m1, 3*m2])
        
        # This loop is rather expansive, a simple way is to do multi-process
        indices = np.indices((m1, m2))
        _is = indices[0].flatten()
        _js = indices[1].flatten()
        a = np.einsum("ik,jk->ijk", X1[0][0], X2[0][0])
        b = X2[0][1]
        c = np.einsum("ik,jk->ij", X1[0][0], X2[0][0])
        path = np.einsum_path('ijk,jkl,ij->l', a, b, c, optimize='greedy')[0]


        if self.ncpu == 1 or self.ncpu == 'gpu':
            results = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, dx2dr, _, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kef_single(x1, x2, dx2dr, None, sigma2, l2, zeta, grad, mask, path))
        else:
            #print("Parallel version is on: ")
            fun_vars = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, dx2dr, _, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                fun_vars.append((x1, x2, dx2dr, None, mask))

            with Pool(self.ncpu) as p:
                func = partial(kef_para, (sigma2, l2, zeta, grad, path))
                results = p.map(func, fun_vars)
                p.close()
                p.join()

        # unpack the results
        for i, j, res in zip(_is, _js, results):
            if grad:
                Kef, dKef_sigma, dKef_l = res
                C[i, j*3:(j+1)*3] = Kef
                C_s[i, j*3:(j+1)*3] = dKef_sigma
                C_l[i, j*3:(j+1)*3] = dKef_l
            else:
                C[i, j*3:(j+1)*3] = res

        if grad:
            return C, C_s, C_l
        else:
            return C

    def k_total_with_stress(self, data1, data2):
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        for key1 in data1.keys():
            for key2 in data2.keys():
                if len(data1[key1])>0 and len(data2[key2])>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee = self.kee_many(data1[key1], data2[key2])
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = self.kef_many(data1[key1], data2[key2])
                    elif key1 == 'force' and key2 == 'energy':
                        C_fe, C_se = self.kef_many_with_stress(data2[key2], data1[key1])
                        C_fe, C_se = C_fe.T, C_se.T
                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_sf = self.kff_many_with_stress(data1[key1], data2[key2])

        return build_covariance(C_ee, C_ef, C_fe, C_ff), build_covariance(None, None, C_se, C_sf)
 
    def kef_many_with_stress(self, X1, X2):
        """
        Compute the energy-force kernel between structures and atoms
        Args:
            X1: list of 2D arrays (each N*d)
            X2: list of tuples ([X, dXdR])
            grad: output gradient if true
        Returns:
            C: M*9N 2D array
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, 3*m2])
        C1 = np.zeros([m1, 6*m2])
        a = np.einsum("ik,jk->ijk", X1[0][0], X2[0][0])
        b = X2[0][1]
        c = np.einsum("ik,jk->ij", X1[0][0], X2[0][0])
        path = np.einsum_path('ijk,jkl,ij->l', a, b, c, optimize='greedy')[0]

       
        # This loop is rather expansive, a simple way is to do multi-process
        indices = np.indices((m1, m2))
        _is = indices[0].flatten()
        _js = indices[1].flatten()

        if self.ncpu == 1 or self.ncpu == 'gpu':
            results = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, dx2dr, rdx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kef_single(x1, x2, dx2dr, rdx2dr, sigma2, l2, zeta, False, mask, path))
        else:
            #print("Parallel version is on: ")
            fun_vars = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, dx2dr, rdx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                fun_vars.append((x1, x2, dx2dr, rdx2dr, mask))

            with Pool(self.ncpu) as p:
                func = partial(kef_para, (sigma2, l2, zeta, False, path))
                results = p.map(func, fun_vars)
                p.close()
                p.join()

        # unpack the results
        for i, j, res in zip(_is, _js, results):
            (Kef, Kes) = res
            C[i, j*3:(j+1)*3] = Kef
            C1[i, j*6:(j+1)*6] = Kes
        return C, C1


    def kff_many_with_stress(self, X1, X2):
        """
        Compute the force-force kernel between structures and atoms
        with the output of stress, for prediction only, no grad
        Args:
            X1: list of tuples ([X, dXdR, rdXdR, ele])
            X2: list of tuples ([X, dXdR, rdXdR, ele])
            same: avoid double counting if true
            grad: output gradient if true
        Returns:
            C: M*N 2D array
            C_grad:
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        m1, m2 = len(X1), len(X2)
        C = np.zeros([3*m1, 3*m2])
        C1 = np.zeros([6*m1, 3*m2])
        a, b, c = X1[0][1], np.einsum("ik,jl->ijkl", X1[0][0], X2[0][0]), X2[0][1]
        path = np.einsum_path('ikm,ijkl,jln->mn', a, b, c, optimize='greedy')[0]

        # This loop is rather expansive, a simple way is to do multi-process
        indices = np.indices((m1, m2))
        _is = indices[0].flatten()
        _js = indices[1].flatten()

        if self.ncpu == 1:
            results = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, rdx1dr, ele1) = X1[i]
                (x2, dx2dr, rdx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kff_single(x1, x2, dx1dr, dx2dr, rdx1dr, sigma2, l2, zeta, False, mask, path))
        elif self.ncpu == 'gpu':
            results = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, rdx1dr, ele1) = X1[i]
                (x2, dx2dr, rdx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kff_single(x1, x2, dx1dr, dx2dr, rdx1dr, sigma2, l2, zeta, False, mask, path, device='gpu'))
        else:
            #print("Parallel version is on: ")
            fun_vars = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, rdx1dr, ele1) = X1[i]
                (x2, dx2dr, rdx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                fun_vars.append((x1, x2, dx1dr, dx2dr, rdx1dr, mask))

            with Pool(self.ncpu) as p:
                func = partial(kff_para, (sigma2, l2, zeta, False, path))
                results = p.map(func, fun_vars)
                p.close()
                p.join()

        # unpack the results
        for i, j, res in zip(_is, _js, results):
            (Kff, Ksf) = res
            C[i*3:(i+1)*3, j*3:(j+1)*3] = Kff
            C1[i*6:(i+1)*6, j*3:(j+1)*3] = Ksf

        return C, C1

def kee_single(x1, x2, sigma2, l2, zeta, grad=False, mask=None):
    """
    Compute the energy-energy kernel between two structures
    Args:
        x1: N1*d array
        x2: N2*d array
        l: length
    Returns:
        C: M*N 2D array
    """
    return K_ee(x1, x2, sigma2, l2, zeta, grad, mask)

def kee_para(args, data): 
    """
    para version
    """
    (x1, x2, mask) = data
    (sigma2, l2, zeta, grad) = args
    return K_ee(x1, x2, sigma2, l2, zeta, grad, mask)
 

def kef_single(x1, x2, dx2dr, rdx2dr, sigma2, l2, zeta, grad=False, mask=None, path=None): 
    """
    Args:
        x1: N*d
        x2: M*d
        dx2dr: M*d*3
    Returns:
        Kef: 3
    """
    return K_ef(x1, x2, dx2dr, rdx2dr, sigma2, l2, zeta, grad, mask, path)

def kef_para(args, data): 
    """
    para version
    """
    (x1, x2, dx2dr, rdx2dr, mask) = data
    (sigma2, l2, zeta, grad, path) = args
    return K_ef(x1, x2, dx2dr, rdx2dr, sigma2, l2, zeta, grad, mask, path)
 

def kff_single(x1, x2, dx1dr, dx2dr, rdx1dr, sigma2, l2, zeta, grad=False, mask=None, path=None, device='cpu'):
    """
    Compute the energy-energy kernel between two structures
    Args:
        x1: m*d1
        x2: n*d2
        dx1dr: m*d1*3
        dx2dr: n*d2*3
    Returns:
        Kff: 3*3 array
    """
    return K_ff(x1, x2, dx1dr, dx2dr, rdx1dr, sigma2, l2, zeta, grad, mask, path, device=device) 


def kff_para(args, data): 
    """
    para version
    """
    (x1, x2, dx1dr, dx2dr, rdx1dr, mask) = data
    (sigma2, l2, zeta, grad, path) = args
    return K_ff(x1, x2, dx1dr, dx2dr, rdx1dr, sigma2, l2, zeta, grad, mask, path) 

def K_ee(x1, x2, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8):
    """
    Compute the Kee between two structures
    Args:
        x1: [M, D] 2d array
        x2: [N, D] 2d array
        sigma2: float
        l2: float
        zeta: power term, float
        mask: to set the kernel zero if the chemical species are different
    """
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D = d**zeta

    #k = sigma2*cp.exp(-(0.5/l2)*(1-D))
    k = sigma2*np.exp(-(0.5/l2)*(1-D))
    if mask is not None:
        k[mask] = 0
    dk_dD = (-0.5/l2)*k

    Kee = np.sum(k)
    mn = len(x1)*len(x2)

    if grad:
        l3 = np.sqrt(l2)*l2
        dKee_dsigma = 2*Kee/np.sqrt(sigma2)
        dKee_dl = np.sum(k*(1-D))/l3
        return Kee/mn, dKee_dsigma/mn, dKee_dl/mn
    else:
        return Kee/mn

def K_ff(x1, x2, dx1dr, dx2dr, rdx1dr, sigma2, l2, zeta=2, grad=False, mask=None, path=None, eps=1e-8, device='cpu'):
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x1_norm2 = x1_norm**2       
    x1_norm3 = x1_norm**3        
    x2_norm3 = x2_norm**3    
    x2_norm2 = x2_norm**2      

    x1x2_norm = x1_norm[:,None]*x2_norm[None,:]

    l = np.sqrt(l2)
    l3 = l*l2

    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1x2_norm)
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1
    k = sigma2*np.exp(-(0.5/l2)*(1-D))
    if mask is not None:
        k[mask] = 0
    dk_dD = (-0.5/l2)*k
    zd2 = -0.5/l2*zeta*zeta*(D1**2)

    tmp11 = x2[None, :, :] * x1_norm[:, None, None]
    tmp12 = x1x2_dot[:,:,None] * (x1/x1_norm[:, None])[:,None,:] 
    tmp13 = x1_norm2[:, None, None] * x2_norm[None, :, None] 
    dd_dx1 = (tmp11-tmp12)/tmp13 

    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None] 
    dd_dx2 = (tmp21-tmp22)/tmp23 

    tmp30 = np.ones(x2.shape)/x2_norm[:,None]
    tmp31 = x1[:,None,:] * tmp30[None,:,:]
    tmp33 = np.eye(x2.shape[1])[None,:,:] - x2[:,:,None] * (x2/x2_norm2[:,None])[:,None,:]

    x1_x1_norm3 = x1/x1_norm3[:,None]
    x2_x2_norm3 = x2/x2_norm3[:,None]

    if device == 'gpu':
        dx1dr = cp.array(dx1dr)
        dx2dr = cp.array(dx2dr)
        x1x2_norm = cp.array(x1x2_norm)
        x1_x1_norm3 = cp.array(x1_x1_norm3)
        x2_x2_norm3 = cp.array(x2_x2_norm3)

        dk_dD = cp.array(dk_dD)
        D1 = cp.array(D1)
        D2 = cp.array(D2)
        zd2 = cp.array(zd2)
        x1x2_dot = cp.array(x1x2_dot)
        dd_dx1 = cp.array(dd_dx1)
        dd_dx2 = cp.array(dd_dx2)

        tmp31 = cp.array(tmp31)
        tmp33 = cp.array(tmp33)

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
    d2k_dx1dx2 = -d2D_dx1dx2 + dD_dx1_dD_dx2 # m, n, d1, d2

    if grad:
        if device == 'gpu':
            D = cp.array(D)
            K_ff_0 = (d2k_dx1dx2[:,:,:,:,None] * dx1dr[:,None,:,None,:]).sum(axis=2) 
            K_ff_0 = (K_ff_0[:,:,:,:,None] * dx2dr[None,:,:,None,:]).sum(axis=2) 
            
            Kff = (K_ff_0 * dk_dD[:,:,None,None]).sum(axis=(0,1))

            d2k_dDdsigma = 2*dkdD/np.sqrt(sigma2)
            d2k_dDdl = (D/l3)*dkdD + (2/l)*dkdD

            dKff_dsigma = (K_ff_0 * d2k_dDdsigma[:,:,None,None]).sum(axis=(0,1))
            
            dKff_dl = (K_ff_0 * d2k_dDdl[:,:,None,None]).sum(axis=(0,1))
            tmp1 = dD_dx1[:,:,:,None]*dD_dx2[:,:,None,:]

            K_ff_1 = cp.sum(tmp1[:,:,:,:,None] * dx1dr[:,None,:,None,:], axis=2)
            K_ff_1 = cp.sum(K_ff_1[:,:,:,:,None] * dx2dr[None,:,:,None,:], axis=2)
            dKff_dl += cp.sum(K_ff_1 * dk_dD[:,:,None,None], axis=(0,1))/(l2*np.sqrt(l2))

            return cp.asnumpy(Kff),  cp.asnumpy(dKff_dsigma),  cp.asnumpy(dKff_dl)
        else:
            K_ff_0 = np.einsum("ijkl,ikm->ijlm", d2k_dx1dx2, dx1dr) # m, n, d2, 3
            K_ff_0 = np.einsum("ijkl,jkm->ijlm", K_ff_0, dx2dr) # m, n, 3, 3

            Kff = np.einsum("ijkl,ij->kl", K_ff_0, dk_dD) # 3, 3

            d2k_dDdsigma = 2*dkdD/np.sqrt(sigma2)
            d2k_dDdl = (D/l3)*dkdD + (2/l)*dkdD
            
            dKff_dsigma = np.einsum("ijkl,ij->kl", K_ff_0, d2k_dDdsigma) 
            dKff_dl = np.einsum("ijkl,ij->kl", K_ff_0, d2k_dDdl)
            tmp1 = dD_dx1[:,:,:,None]*dD_dx2[:,:,None,:]
            K_ff_1 = np.einsum("ijkl,ikm->ijlm", tmp1, dx1dr)
            K_ff_1 = np.einsum("ijkl,jkm->ijlm", K_ff_1, dx2dr)

            dKff_dl += np.einsum("ijkl,ij->kl", K_ff_1, dk_dD)/l2/np.sqrt(l2)

            return Kff, dKff_dsigma, dKff_dl

    else:
        if device == 'gpu':
            tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None]
            _kff1 = cp.sum(dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None], axis=(0,2))
            Kff = cp.sum(_kff1[:,:,:,None] * dx2dr[:,:,None,:], axis=(0,1))
            #Kff = dx1dr[:,None,:,None,:,None] * tmp0[:,:,:,:,None,None] * dx2dr[None,:,None,:,None,:]
            #Kff = Kff.sum(axis=(0,1,2,3))

            if rdx1dr is None:
                return  cp.asnumpy(Kff)
            else:
                rdx1dr = cp.asarray(rdx1dr)
                _Ksf = cp.sum(rdx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None], axis=(0,2))
                Ksf = cp.sum(_Ksf[:,:,:,None] * dx2dr[:,:,None,:], axis=(0,1))
                #Ksf = rdx1dr[:,None,:,None,:,None] * tmp0[:,:,:,:,None,None] * dx2dr[None,:,None,:,None,:]
                #Ksf = Ksf.sum(axis=(0,1,2,3))
                return cp.asnumpy(Kff),  cp.asnumpy(Ksf)

        else:
            tmp0 = np.einsum("ijkl,ij->ijkl", d2k_dx1dx2, dk_dD) #m,n,d1,d2
            Kff = np.einsum("ikm,ijkl,jln->mn", dx1dr, tmp0, dx2dr, optimize=path)
            if rdx1dr is None:
                return Kff
            else:
                Ksf = np.einsum("ikm,ijkl,jln->mn", rdx1dr, tmp0, dx2dr, optimize=path) #[6,3]
                return Kff, Ksf

def K_ef(x1, x2, dx2dr, rdx2dr, sigma2, l2, zeta=2, grad=False, mask=None, path=None, eps=1e-8):

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
    m = len(x1)

    if grad:
        K_ef_0 = -np.einsum("ijk,jkl->ijl", dD_dx2, dx2dr) # [m, n, d2] [n, d2, 3] -> [m,n,3]
        Kef = np.einsum("ijk,ij->k", K_ef_0, dk_dD) # [m, n, 3] [m, n] -> 3

        d2k_dDdsigma = 2*dkdD/np.sqrt(sigma2)
        d2k_dDdl = (D/l3)*dkdD + (2/l)*dkdD

        dKef_dsigma = np.einsum("ijk,ij->k", K_ef_0, d2k_dDdsigma) 
        dKef_dl     = np.einsum("ijk,ij->k", K_ef_0, d2k_dDdl)
        return Kef/m, dKef_dsigma/m, dKef_dl/m
    else:
        Kef = -np.einsum("ijk,jkl,ij->l", dD_dx2, dx2dr, dk_dD, optimize=path) #[6]
        if rdx2dr is None:
            return Kef/m
        else:
            Kse = -np.einsum("ijk,jkl,ij->l", dD_dx2, rdx2dr, dk_dD, optimize=path) #[6]
            return Kef/m, Kse/m

