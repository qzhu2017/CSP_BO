import numpy as np
from .derivatives import *

class RBF_mb():
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 2e+1], [1e-1, 1e+1]]):
        self.name = 'RBF_mb'
        self.bounds = bounds
        self.update(para)

    def __str__(self):
        return "{:.3f}**2 *RBF(length={:.3f})".format(self.sigma, self.l)
 
    def parameters(self):
        return [self.sigma, self.l]
 
    def update(self, para):
        self.sigma, self.l = para[0], para[1]

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
                        #if not same:
                        C_fe = self.kfe_many(data1[key1], data2[key2])
                        #C_fe = self.kfe_many(data2[key2], data1[key1])
                        #else:
                        #    C_fe = C_ef.T 
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
        C_grad_ee, C_grad_ef, C_grad_fe, C_grad_ff = None, None, None, None
            
        for key1 in data1.keys():
            for key2 in data2.keys():
                if len(data1[key1])>0 and len(data2[key2])>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee, C_grad_ee = self.kee_many(data1[key1], data2[key2], True, True)

                    elif key1 == 'energy' and key2 == 'force':
                        C_ef, C_grad_ef = self.kef_many(data1[key1], data2[key2])
                        C_fe = C_ef.T
                        C_grad_ef = np.transpose(C_grad_ef, axis=(1,0,2)) 

                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_grad_ff = self.kff_many(data1[key1], data2[key2], True, True)

        C = build_covariance(C_ee, C_ef, C_fe, C_ff)
        C_grad = build_covariance(C_grad_ee, C_grad_ef, C_grad_fe, C_grad_ff)
        return C, C_grad

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
        sigma, sigma2, l2, l3 = self.sigma, self.sigma**2, self.l**2, self.l**3
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])
        kd = np.zeros([m1, m2])

        for i, x1 in enumerate(X1):
            if same:
                start = i
            else:
                start = 0
            for j in range(start, len(X2)):
                x2 = X2[j]
                if grad:
                    C[i, j], kd[i, j] = kee_single_grad(x1, x2, sigma2, l2)
                    kd[j, i] = kd[i, j]
                else:
                    C[i, j] = kee_single(x1, x2, sigma2, l2)
                if same:
                    C[j, i] = C[i, j]

        if grad:
            C_grad = np.zeros([m1, m2, 2])
            C_grad[:,:,0] = 2*C/sigma
            C_grad[:,:,1] = kd/l3

            return C, C_grad                   
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
        sigma, sigma2, l2, l3 = self.sigma, self.sigma**2, self.l**2, self.l**3
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, 3*m2])
        kd = np.zeros([m1, m2])

        for i, x1 in enumerate(X1):
            for j, data in enumerate(X2):
                (x2, dx2dr) = data
                if grad:
                    C_grad = np.zeros([m1, 3*m2])
                    C[i, j*3:(j+1)*3] = kef_single(x1, x2, dx2dr, sigma2, l2)
                    C_grad[i, j*3:(j+1)*3, 0] = (2/sigma) * C
                    C_grad[i, j*3:(j+1)*3, 1] = kef_single_grad(x1, x2, dx2dr, sigma2, self.l)
                    
                    #C[i, j], kd[i, j] = kef_single_grad(x1, x2, dx2dr, sigma2, l2)
                else:
                    C[i, j*3:(j+1)*3] = kef_single(x1, x2, dx2dr, sigma2, l2)
        
        if grad:
            #C_grad = np.zeros([m1, m2, 2])
            #C_grad[:,:,0] = 2*C/sigma
            #C_grad[:,:,1] = kd/l3
            return C, C_grad                   
        else:
            return C


    def kfe_many(self, X1, X2, grad=False):
            sigma, sigma2, l2, l3 = self.sigma, self.sigma**2, self.l**2, self.l**3
            m1, m2 = len(X1), len(X2)
            C = np.zeros([m1*3, m2])
            kd = np.zeros([m1, m2])

            for i, data in enumerate(X1):
                for j, x2 in enumerate(X2):
                    (x1, dx1dr) = data
                    if grad:
                        C_grad = np.zeros([m1*3, m2])
                        C[i*3:(i+1)*3, j] = kfe_single(x1, x2, dx1dr, sigma2, l2)
                        C_grad[i*3:(i+1)*3, j, 0] = (2/sigma) * C
                        C_grad[i*3:(i+1)*3, j, 1] = kfe_single_grad(x1, x2, dx1dr, sigma2, self.l)

                        #C[i, j], kd[i, j] = kfe_single_grad(x1, x2, dx1dr, sigma2, l2)
                    else:
                        C[i*3:(i+1)*3, j] = kfe_single(x1, x2, dx1dr, sigma2, l2)

            if grad:
                #C_grad = np.zeros([m1, m2, 2])
                #C_grad[:,:,0] = 2*C/sigma
                #C_grad[:,:,1] = kd/l3
                return C, C_grad
            else:
                return C

def distance(x1, x2):
    """
    Args:
        X1: N1*M
        X2: N2*M

    Returns:
        distance: N1*N2
    """
    return x1@x2.T/(1e-4+np.outer(np.linalg.norm(x1, axis=1), np.linalg.norm(x2, axis=1)))

def kee_single(x1, x2, sigma2, l2):
    """
    Compute the energy-energy kernel between two structures
    Args:
        x1: N1*d array
        x2: N2*d array
        l: length
    Returns:
        C: M*N 2D array
    """
    D = 1-distance(x1, x2)**2
    k = np.exp(-0.5*D/l2)
    #return sigma2*np.mean(k)
    return sigma2*np.sum(k)

def kee_single_grad(x1, x2, sigma2, l2):
    D = 1-distance(x1, x2)**2
    k = np.exp(-0.5*D/l2)
    kd = k*D
    #return sigma2*np.mean(k), sigma2*np.mean(kd)
    return sigma2*np.sum(k), sigma2*np.sum(kd)

def kef_single(x1, x2, dx2dr, sigma2, l2):
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
    #_, d1 = fun_D(x1, x2, x1_norm, x2_norm)
    #dk_dx2 = fun_dk_dx2(x1, x2, x1_norm, x2_norm, d1, sigma2, l2)  
    #Kef = -np.einsum("ij, ijk->k", dk_dx2, dx2dr) # [m,d], [m,d,3] -> 3
    
    d = distance(x1, x2)
    d2 = d**2
    k = sigma2 * np.exp(-(0.5/l2)*(1-d2))
    kd = k * d / l2
    dD_dx2_1 = np.einsum("ij,k->ikj", x1, x2_norm)
    dD_dx2_2 = (x1@x2.T)[:,:,None] * (x2 / x2_norm[:, None])[None, :, :]
    dD_dx2_3 = x1_norm[:, None, None] * (x2_norm**2)[None,:,None]
    dD_dx2 = (dD_dx2_1 - dD_dx2_2) / dD_dx2_3
    kd_dD_dx2 = kd[:,:,None] * dD_dx2
    Kef = -np.einsum("ijk, jkl->l", kd_dD_dx2, dx2dr)

    return Kef

def kef_single_grad(x1, x2, dx2dr, sigma2, l):
    """ Get the derivative of Kef with respect to sigma. """ 
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    d = distance(x1, x2)
    d2 = d**2
    k = sigma2 * np.exp(-(0.5/l**2)*(1-d2))

    dKef_dl_1 = k * ((1-d2)/l**5 - (2/l**3)) * D

    dD_dx2_1 = np.einsum("ij,k->ikj", x1, x2_norm)
    dD_dx2_2 = (x1@x2.T)[:,:,None] * (x2 / x2_norm[:, None])[None, :, :]
    dD_dx2_3 = x1_norm[:, None, None] * (x2_norm**2)[None,:,None]
    dD_dx2 = (dD_dx2_1 - dD_dx2_2) / dD_dx2_3

    dKef_dl_2 = dKef_dl_1[:, :, None] * dD_dx2
    dKef_dl = -np.einsum("ijk, jkl->l", dKef_dl_2, dx2dr)

    return dKef_dl


def kff_single(x1, x2, dx1dr, dx2dr, sigma2, l2):


def kfe_single(x1, x2, dx1dr, sigma2, l2):
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)

    d = distance(x1, x2)
    d2 = d**2
    k = sigma2 * np.exp(-(0.5/l2)*(1-d2))
    kd = k * d / l2
    
    dD_dx1_1 = np.einsum("ij,k->kij", x2, x1_norm)
    dD_dx1_2 = (x1@x2.T)[:, :, None] * (x1 / x1_norm[:, None])[:, None, :]
    dD_dx1_3 = (x1_norm ** 2)[:, None, None] * x2_norm[None, :, None]
    dD_dx1 = (dD_dx1_1 - dD_dx1_2) / dD_dx1_3

    kd_dD_dx1 = kd[:, :, None] * dD_dx1
    Kfe = -np.einsum("ijk,ikl->l", kd_dD_dx1, dx1dr)
    return Kfe


def kfe_single_grad(x1, x2, dx1dr, sigma2, l):
    """ Get the derivative of Kfe with respect to sigma. """
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    d = distance(x1, x2)
    d2 = d**2
    k = sigma2 * np.exp(-(0.5/l**2)*(1-d2))

    dKfe_dl_1 = k * ((1-d2)/l**5 - (2/l**3)) * D

    dD_dx1_1 = np.einsum("ij,k->kij", x2, x1_norm)
    dD_dx1_2 = (x1@x2.T)[:, :, None] * (x1 / x1_norm[:, None])[:, None, :]
    dD_dx1_3 = (x1_norm ** 2)[:, None, None] * x2_norm[None, :, None]
    dD_dx1 = (dD_dx1_1 - dD_dx1_2) / dD_dx1_3

    dKfe_dl_2 = dKfe_dl_1[:, :, None] * dD_dx1
    dKfe_dl = -np.einsum("ijk, ikl->l", dKfe_dl_2, dx1dr)
    return dKfe_dl


def build_covariance(c_ee, c_ef, c_fe, c_ff):
    exist = []
    for x in (c_ee, c_ef, c_fe, c_ff):
        if x is None:
            exist.append(False)
        else:
            exist.append(True)
    if False not in exist:
        return np.block([[c_ee, c_ef], [c_fe, c_ff]])
    elif exist == [False, False, True, True]: # F in train, E/F in predict
        return np.vstack((c_fe, c_ff))
    elif exist == [True, True, False, False]: # E in train, E/F in predict
        return np.hstack((c_ee, c_ef))
    elif exist == [False, True, False, False]: # E in train, F in predict
        return c_ef
    elif exist == [True, False, False, False]: # E in train, E in predict
        return c_ee
    elif exist == [False, False, False, True]: # F in train, F in predict 
        return c_ff
    elif exist == [False, False, True, False]: # F in train, E in predict 
        return c_fe
