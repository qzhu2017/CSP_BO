import numpy as np
from .derivatives import *
#from .derivatives_many import K_ff_multi
from functools import partial
from multiprocessing import Pool, cpu_count

class RBF_mb():
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 2e+1], [1e-1, 1e+1]], zeta=3, ncpu=1):
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
                (x1, dx1dr, ele1) = data["force"][i]
                mask = get_mask(ele1, ele1)
                C_ff[i*3:(i+1)*3] = np.diag(kff_single(x1, x1, dx1dr, dx1dr, sigma2, l2, zeta, False, mask))

        if C_ff is None:
            return C_ee
        elif C_ee is None:
            return C_ff
        else:
            return np.hstack((C_ee, C_ff))

        #return np.ones(N)*sigma2

    def k_total(self, data1, data2=None, kff_quick=False):
        if data2 is None:
            data2 = data1
            same = True
        else:
            same = False
        C_ee, C_ef, C_fe, C_ff, C_se, C_sf = None, None, None, None, None, None
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
                        if kff_quick:
                            C_ff = self.kff_quick(data1[key1], data2[key2])
                        else:
                            C_ff = self.kff_many(data1[key1], data2[key2], same=same)
                    elif key1 == 'stress' and key2 == 'force':
                        C_sf = self.ksf_many(data1[key1], data2[key2])
                    elif key1 == 'stress' and key2 == 'energy':
                        C_se = self.kse_many(data1[key1], data2[key2])

        return build_covariance(C_ee, C_ef, C_fe, C_ff, C_se, C_sf)
        
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
                        #print(C_ee)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef, C_ef_s, C_ef_l = self.kef_many(data1[key1], data2[key2], True)
                        C_fe, C_fe_s, C_fe_l = C_ef.T, C_ef_s.T, C_ef_l.T

                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_ff_s, C_ff_l = self.kff_many(data1[key1], data2[key2], True, True)
        C = build_covariance(C_ee, C_ef, C_fe, C_ff, None, None)
        C_s = build_covariance(C_ee_s, C_ef_s, C_fe_s, C_ff_s, None, None)
        C_l = build_covariance(C_ee_l, C_ef_l, C_fe_l, C_ff_l, None, None)
        return C, np.dstack((C_s, C_l))

    def ksf_many(self, X1, X2, same=False, grad=False):
        """
        Compute the stress-force kernel for many structures
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
        C = np.zeros([m1*6, m2*3])

        for i, (x1, rdxdr) in enumerate(X1):
            for j, (x2, dxdr, ele2) in enumerate(X2):
                C[i*6:(i+1)*6, j*3:(j+1)*3] = ksf_single(x1, x2, rdxdr, dxdr, sigma2, l2, zeta)
        return C

    def kse_many(self, X1, X2, same=False, grad=False):
        """
        Compute the stress-energy kernel for many structures
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
        C = np.zeros([m1*6, m2])

        for i, (x1, rdxdr) in enumerate(X1):
            for j, (x2, ele2) in enumerate(X2):
                C[i*6:(i+1)*6, j] = kse_single(x1, x2, rdxdr, sigma2, l2, zeta, grad)
        return C

    
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

        if self.ncpu == 1:
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

        if grad:
            return C, C_s, C_l
        else:
            return C

    def kff_many(self, X1, X2, same=False, grad=False):
        """
        Compute the energy-force kernel between structures and atoms
        Args:
            X1: list of tuples ([X, dXdR])
            X2: list of tuples ([X, dXdR])
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

        # This loop is rather expansive, a simple way is to do multi-process
        if same:
            indices = np.triu_indices(m1)
            (_is, _js) = indices
        else:
            indices = np.indices((m1, m2))
            _is = indices[0].flatten()
            _js = indices[1].flatten()

        #from time import time
        #t0 = time()

        if self.ncpu == 1:
            results = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, ele1) = X1[i]
                (x2, dx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kff_single(x1, x2, dx1dr, dx2dr, sigma2, l2, zeta, grad, mask))
        else:
            #print("Parallel version is on: ")
            fun_vars = []
            for i, j in zip(_is, _js):
                (x1, dx1dr, ele1) = X1[i]
                (x2, dx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                fun_vars.append((x1, x2, dx1dr, dx2dr, mask))

            with Pool(self.ncpu) as p:
                func = partial(kff_para, (sigma2, l2, zeta, grad))
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

        #print("================", time()-t0, self.ncpu, len(_is))
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

        if self.ncpu == 1:
            results = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, dx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                results.append(kef_single(x1, x2, dx2dr, sigma2, l2, zeta, grad, mask))
        else:
            #print("Parallel version is on: ")
            fun_vars = []
            for i, j in zip(_is, _js):
                (x1, ele1) = X1[i]
                (x2, dx2dr, ele2) = X2[j]
                mask = get_mask(ele1, ele2)
                fun_vars.append((x1, x2, dx2dr, mask))

            with Pool(self.ncpu) as p:
                func = partial(kef_para, (sigma2, l2, zeta, grad))
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
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    return K_ee(x1, x2, x1_norm, x2_norm, sigma2, l2, zeta, grad, mask)

def kef_single(x1, x2, dx2dr, sigma2, l2, zeta, grad=False, mask=None):
    """
    Args:
        x1: N*d
        x2: M*d
        dx2dr: M*d*3
    Returns:
        Kef: 3
    """
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    _, d1 = fun_D(x1, x2, x1_norm, x2_norm, zeta)
    return K_ef(x1, x2, x1_norm, x2_norm, dx2dr, d1, sigma2, l2, zeta, grad, mask)

def kff_single(x1, x2, dx1dr, dx2dr, sigma2, l2, zeta, grad=False, mask=None):
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
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    D1, d1 = fun_D(x1, x2, x1_norm, x2_norm, zeta)
    return K_ff(x1, x2, x1_norm, x2_norm, dx1dr, dx2dr, d1, sigma2, l2, zeta, grad, mask) 

def kse_single(x1, x2, rdx1dr, sigma2, l2, zeta, grad=False):
    """
    Compute the stress-energy kernel between two structures
    Args:
        x1: m*d1
        x2: n*d2
        rdx1dr: m*d1*6
    Returns:
        Kse: 6*1 array
    """

    x1_norm = np.linalg.norm(x1, axis=1)

    x2_norm = np.linalg.norm(x2, axis=1)
    D1, d1 = fun_D(x1, x2, x1_norm, x2_norm, zeta)
    return K_se(x1, x2, x1_norm, x2_norm, rdx1dr, d1, sigma2, l2, zeta, grad)

def ksf_single(x1, x2, rdx1dr, dx2dr, sigma2, l2, zeta, grad=False):
    """
    Compute the stress-force kernel between two structures
    Args:
        x1: m*d1
        x2: n*d2
        rdx1dr: m*d1*6
        rdxdr: n*d2*3
    Returns:
        Ksf: 6*3 array
    """
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    D1, d1 = fun_D(x1, x2, x1_norm, x2_norm, zeta)
    return K_sf(x1, x2, x1_norm, x2_norm, rdx1dr, dx2dr, d1, sigma2, l2, zeta, grad)

    
def kee_para(args, data): 
    """
    para version
    """
    (x1, x2, mask) = data
    (sigma2, l2, zeta, grad) = args
    return kee_single(x1, x2, sigma2, l2, zeta, grad, mask)
 
def kff_para(args, data): 
    """
    para version
    """
    (x1, x2, dx1dr, dx2dr, mask) = data
    (sigma2, l2, zeta, grad) = args
    return kff_single(x1, x2, dx1dr, dx2dr, sigma2, l2, zeta, grad, mask)
 
def kef_para(args, data): 
    """
    para version
    """
    (x1, x2, dx2dr, mask) = data
    (sigma2, l2, zeta, grad) = args
    return kef_single(x1, x2, dx2dr, sigma2, l2, zeta, grad, mask)
 
def build_covariance(c_ee, c_ef, c_fe, c_ff, c_se, c_sf):
    exist = []
    for x in (c_ee, c_ef, c_fe, c_ff, c_se, c_sf):
        if x is None:
            exist.append(False)
        else:
            exist.append(True)
    #if False not in exist:
    if exist == [True, True, True, True, False, False]:
        return np.block([[c_ee, c_ef], [c_fe, c_ff]])
    elif exist == [False, False, True, True, False, False]: # F in train, E/F in predict
        #print(c_fe.shape, c_ff.shape)
        return np.hstack((c_fe, c_ff))
    elif exist == [True, True, False, False, False, False]: # E in train, E/F in predict
        return np.hstack((c_ee, c_ef))
    elif exist == [False, True, False, False, False, False]: # E in train, F in predict
        return c_ef
    elif exist == [True, False, False, False, False, False]: # E in train, E in predict
        return c_ee
    elif exist == [False, False, False, True, False, False]: # F in train, F in predict 
        return c_ff
    elif exist == [False, False, True, False, False, False]: # F in train, E in predict 
        return c_fe
    elif exist == [False, False, False, False, True, False]: # E in train, S in predict
        return c_se
    elif exist == [False, False, False, False, False, True]: # F in train, S in predict
        return c_sf
    elif exist == [False, False, False, False, True, True]: # E&F in train, S in predict
        return np.hstack((c_se, c_sf))

def get_mask(ele1, ele2):
    ans = ele1[:,None] - ele2[None,:]
    ids = np.where(ans!=0)
    if len(ids[0]) == 0:
        return None
    else:
        return ids
