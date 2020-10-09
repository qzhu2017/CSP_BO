import numpy as np
from .derivatives import *
from .derivatives_many import K_ff_multi

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

    def diag(self, data):
        """
        Returns the diagonal of the kernel k(X, X)
        used foor the prediction of std
        """
        sigma2, l2= self.sigma**2, self.l**2
        
        C_ee, C_ff = None, None

        if "energy" in data:
            N_E = len(data["energy"])
            C_ee = np.zeros(N_E)
            for i, x1 in enumerate(data["energy"]):
                C_ee[i] += kee_single(x1, x1, sigma2, l2)

        if "force" in data:
            N_F = len(data["force"])
            C_ff = np.zeros(3*N_F)
            for i, data1 in enumerate(data["force"]):
                (x1, dx1dr) = data1
                C_ff[i*3:(i+1)*3] += np.diag(kff_single(x1, x1, dx1dr, dx1dr, sigma2, l2))
        if C_ee is None:
            return C_ff
        elif C_ff is None:
            return C_ee
        else:
            return np.hstack((C_ee, C_ff))


    def k_total(self, data1, data2=None, kff_quick=False):
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
                        if kff_quick:
                            C_ff = self.kff_quick(data1[key1], data2[key2])
                        else:
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
                        #print(C_ee)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef, C_ef_s, C_ef_l = self.kef_many(data1[key1], data2[key2], True)
                        C_fe, C_fe_s, C_fe_l = C_ef.T, C_ef_s.T, C_ef_l.T

                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_ff_s, C_ff_l = self.kff_many(data1[key1], data2[key2], True, True)
        C = build_covariance(C_ee, C_ef, C_fe, C_ff)
        C_s = build_covariance(C_ee_s, C_ef_s, C_fe_s, C_ff_s)
        C_l = build_covariance(C_ee_l, C_ef_l, C_fe_l, C_ff_l)
        return C, np.dstack((C_s, C_l))

    def kee_many(self, X1, X2, same=False, grad=False, diag=False):
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
        sigma2, l2= self.sigma**2, self.l**2
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])
        C_s = np.zeros([m1, m2])
        C_l = np.zeros([m1, m2])

        for i, x1 in enumerate(X1):
            if same:
                start = i
            else:
                start = 0
            for j in range(start, len(X2)):
                x2 = X2[j]
                if grad:
                    Kee, Kee_sigma, Kee_l = kee_single(x1, x2, sigma2, l2, True)
                    C[i, j] = Kee
                    C_s[i, j], C_s[j, i] = Kee_sigma, Kee_sigma
                    C_l[i, j], C_l[j, i] = Kee_l, Kee_l
                else:
                    C[i, j] = kee_single(x1, x2, sigma2, l2)
                if same:
                    C[j, i] = C[i, j]
        if grad:
            return C, C_s, C_l
        else:
            return C

    def kff_many(self, X1, X2, same=False, grad=False, diag=False):
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
        sigma2, l2 = self.sigma**2, self.l**2
        m1, m2 = len(X1), len(X2)
        C = np.zeros([3*m1, 3*m2])
        C_s = np.zeros([3*m1, 3*m2])
        C_l = np.zeros([3*m1, 3*m2])

        for i, data1 in enumerate(X1):
            (x1, dx1dr) = data1
            if same:
                start = i
            else:
                start = 0
            for j in range(start, len(X2)):
                (x2, dx2dr) = X2[j]
                if grad:
                    Kff, dKff_sigma, dKff_l = kff_single(x1, x2, dx1dr, dx2dr, sigma2, l2, True)
                    C[i*3:(i+1)*3, j*3:(j+1)*3] = Kff
                    C_s[i*3:(i+1)*3, j*3:(j+1)*3] = dKff_sigma
                    C_s[j*3:(j+1)*3, i*3:(i+1)*3] = dKff_sigma.T
                    C_l[i*3:(i+1)*3, j*3:(j+1)*3] = dKff_l
                    C_l[j*3:(j+1)*3, i*3:(i+1)*3] = dKff_l.T
                else:
                    C[i*3:(i+1)*3, j*3:(j+1)*3] = kff_single(x1, x2, dx1dr, dx2dr, sigma2, l2)
                    #if i%100==0 and j%10==0:
                    #    print("kff", i, j)
                if same:
                    C[j*3:(j+1)*3, i*3:(i+1)*3] = C[i*3:(i+1)*3, j*3:(j+1)*3].T
        if grad:
            return C, C_s, C_l                  
        else:
            return C


    def kff_quick(self, _X1, _X2):
        """
        Compute the force-force kernel between structures and atoms
        Args:
            X1: list of tuples ([X, dXdR])
            X2: list of tuples ([X, dXdR])
        Returns:
            C: M*N 2D array
        """
        sigma2, l2 = self.sigma**2, self.l**2
        m1, m2, n, d = len(_X1), len(_X2), 20, _X1[0][0].shape[1]
        X1 = np.zeros([m1, n, d])
        X2 = np.zeros([m2, n, d])
        dX1dr = np.zeros([m1, n, d, 3]) 
        dX2dr = np.zeros([m2, n, d, 3]) 

        for i, data in enumerate(_X1):
            (x1, dx1dr) = data
            N_atom = len(x1)
            X1[i,:N_atom, :] += x1
            dX1dr[i,:N_atom, :, :] += dx1dr
            
        for i, data in enumerate(_X2):
            (x1, dx1dr) = data
            N_atom = len(x1)
            X2[i,:N_atom, :] += x1
            dX2dr[i,:N_atom, :, :] += dx1dr

        ans = K_ff_multi(X1, X2, dX1dr, dX2dr, sigma2, l2)
        Kff = np.zeros([3*m1, 3*m2])
        for i in range(m1):
            for j in range(m2):
                Kff[i*3:(i+1)*3,j*3:(j+1)*3] = ans[i,j,:,:]
        return Kff

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
        sigma2, l2 = self.sigma**2, self.l**2
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, 3*m2])
        C_s = np.zeros([m1, 3*m2])
        C_l = np.zeros([m1, 3*m2])

        for i, x1 in enumerate(X1):
            for j, data in enumerate(X2):
                (x2, dx2dr) = data
                if grad:
                    Kef, Kef_sigma, Kef_l = kef_single(x1, x2, dx2dr, sigma2, l2, True)
                    C[i, j*3:(j+1)*3] = Kef
                    C_s[i, j*3:(j+1)*3] = Kef_sigma
                    C_l[i, j*3:(j+1)*3] = Kef_l
                else:
                    C[i, j*3:(j+1)*3] = kef_single(x1, x2, dx2dr, sigma2, l2)
        if grad:
            return C, C_s, C_l
        else:
            return C

def kee_single(x1, x2, sigma2, l2, grad=False):
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
    return K_ee(x1, x2, x1_norm, x2_norm, sigma2, l2, grad)

def kef_single(x1, x2, dx2dr, sigma2, l2, grad=False):
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
    _, d1 = fun_D(x1, x2, x1_norm, x2_norm)
    return K_ef(x1, x2, x1_norm, x2_norm, dx2dr, d1, sigma2, l2, grad)

def kff_single(x1, x2, dx1dr, dx2dr, sigma2, l2, grad=False):
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
    D1, d1 = fun_D(x1, x2, x1_norm, x2_norm)
    return K_ff(x1, x2, x1_norm, x2_norm, dx1dr, dx2dr, d1, sigma2, l2, grad) 
    
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
        #print(c_fe.shape, c_ff.shape)
        return np.hstack((c_fe, c_ff))
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
