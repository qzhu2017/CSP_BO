import numpy as np
import cupy as cp
from .kernel_base import *

class Dot_mb():
    """
    .. math::
        k(x_i, x_j) = \sigma ^2 * (\sigma_0 ^ 2 + x_i \cdot x_j)
    """
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 5e+1], [1e-2, 1e+1]], zeta=3, device="cpu"):
        self.name = 'Dot_mb'
        self.bounds = bounds
        self.update(para)
        self.zeta = zeta
        self.device = device
    def __str__(self):
        return "{:.3f}**2 *Dot(length={:.3f})".format(self.sigma, self.sigma0)

    def load_from_dict(self, dict0):
        self.sigma = dict0["sigma"]
        self.sigma0 = dict0["sigma0"]
        self.zeta = dict0["zeta"]
        self.bounds = dict0["bounds"]
        self.name = dict0["name"]
        
    def save_dict(self):
        """
        save the model as a dictionary in json
        """
        dict = {"name": self.name,
                "sigma": self.sigma,
                "sigma0": self.sigma0,
                "zeta": self.zeta,
                "bounds": self.bounds
               }
        return dict

    def parameters(self):
        return [self.sigma, self.sigma0]
 
    def update(self, para):
        self.sigma, self.sigma0 = para[0], para[1]

    def diag(self, data):
        """
        Returns the diagonal of the kernel k(X, X)
        """
        sigma2, sigma02, zeta = self.sigma**2, self.sigma0**2, self.zeta
        C_ee, C_ff = None, None
               
        if "energy" in data:
            NE = len(data["energy"])
            C_ee = np.zeros(NE)
            for i in range(NE):
                (x1, ele1) = data["energy"][i]
                mask = get_mask(ele1, ele1)
                C_ee[i] = kee_single(x1, x1, sigma2, sigma02, zeta, False, mask) 

        if "force" in data:
            NF = len(data["force"])
            C_ff = np.zeros(3*NF)
            for i in range(NF):
                (x1, dx1dr, _, ele1) = data["force"][i]
                mask = get_mask(ele1, ele1)
                C_ff[i*3:(i+1)*3] = np.diag(kff_single(x1, x1, dx1dr, dx1dr, None, None, sigma2, sigma02, zeta, False, mask))

        if C_ff is None:
            return C_ee
        elif C_ee is None:
            return C_ff
        else:
            return np.hstack((C_ee, C_ff))

    def k_total(self, data1, data2=None, same=False):
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
                        C_ff = self.kff_many(data1[key1], data2[key2])
        #print(C_ee)
        #print(C_ef)
        #print(C_ff[:5,:5])
        #import sys; sys.exit()
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

    def k_total_with_stress(self, data1, data2):
        """
        Compute the covairance
        Used for energy/force/stress prediction
        """
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        for key1 in data1.keys():
            for key2 in data2.keys():
                if len(data1[key1])>0 and len(data2[key2])>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee = self.kee_many(data1[key1], data2[key2])
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = self.kef_many(data1[key1], data2[key2])
                    elif key1 == 'force' and key2 == 'energy':
                        C_fe, C_se = self.kef_many(data2[key2], data1[key1], stress=True)
                        C_fe, C_se = C_fe.T, C_se.T
                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_sf = self.kff_many(data1[key1], data2[key2], stress=True)

        return build_covariance(C_ee, C_ef, C_fe, C_ff), build_covariance(None, None, C_se, C_sf)
        
    
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
        sigma2, sigma02, zeta = self.sigma**2, self.sigma0**2, self.zeta
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])
        C1 = np.zeros([m1, m2])
        C2 = np.zeros([m1, m2])

        # This loop is rather expansive, a simple way is to do multi-process
        if same:
            indices = np.triu_indices(m1)
            (_is, _js) = indices
        else:
            indices = np.indices((m1, m2))
            _is = indices[0].flatten()
            _js = indices[1].flatten()

        for i, j in zip(_is, _js):
            (x1, ele1) = X1[i]
            (x2, ele2) = X2[j]
            mask = get_mask(ele1, ele2)
            res = K_ee(x1, x2, sigma2, sigma02, zeta, grad, mask)

            if grad:
                Kee, dKee_sigma, dKee_l = res
                C[i, j] = Kee
                C1[i, j] = dKee_sigma
                C1[j, i] = dKee_sigma
                C2[i, j] = dKee_l
                C2[j, i] = dKee_l
            else:
                C[i, j] = res
            if same and (i != j):
                C[j, i] = C[i, j]

        if grad:
            return C, C1, C2
        else:
            return C

    def kff_many(self, X1, X2, same=False, grad=False, stress=False):
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
        sigma2, sigma02, zeta = self.sigma**2, self.sigma0**2, self.zeta
        x_all, dxdr_all, _, ele_all, x2_indices = X2[0]
        if len(X1) == 1: #unpack X1, used for training
            X, dXdR, _, ELE, indices = X1[0]
            X1 = []
            c = 0
            for ind in indices:
                X1.append((X[c:c+ind], dXdR[c:c+ind], ELE[c:c+ind]))
                c += ind
        # num of X1, num of X2, num of big X2
        m1, m2, m2p = len(X1), len(x2_indices), len(x_all)

        x2_inds = [(0, x2_indices[0])]
        for i in range(1, len(x2_indices)):
            ID = x2_inds[i-1][1]
            x2_inds.append( (ID, ID+x2_indices[i]) )

        if self.device == 'cpu': # Work on the cpu
            if grad:
                C = np.zeros([m1, m2p, 3, 9])
            elif stress:
                C = np.zeros([m1, m2p, 9, 3])
            else:
                C = np.zeros([m1, m2p, 3, 3])
        else:
            if grad:
                C = cp.zeros([m1, m2p, 3, 9])
            elif stress:
                C = cp.zeros([m1, m2p, 9, 3])
            else:
                C = cp.zeros([m1, m2p, 3, 3])
        
        for i in range(m1):
            (x1, dx1dr, ele1) = X1[i]
            mask = get_mask(ele1, ele_all)
            if self.device == 'gpu':
                dx1dr = cp.array(dx1dr)

            C[i] = K_ff(x1, x_all, dx1dr, dxdr_all, sigma2, sigma02, zeta, grad, mask, device=self.device)

        if self.device == 'gpu':
            C = cp.asnumpy(C)

        _C = np.zeros([m1*3, m2*3])
        if grad:
            _C_s = np.zeros([m1*3, m2*3])
            _C_l = np.zeros([m1*3, m2*3])
        elif stress:
            _C1 = np.zeros([m1*6, m2*3])

        for j, ind in enumerate(x2_inds):
            tmp = C[:, ind[0]:ind[1], :, :].sum(axis=1)
            #print(tmp.shape)
            for i in range(m1):
                _C[i*3:(i+1)*3, j*3:(j+1)*3]  = tmp[i, :3, :3]
                if stress:
                    _C1[i*6:(i+1)*6, j*3:(j+1)*3]  = tmp[i, 3:, :3]
                elif grad:
                    _C_s[i*3:(i+1)*3, j*3:(j+1)*3]  = tmp[i, :3, 3:6]
                    _C_l[i*3:(i+1)*3, j*3:(j+1)*3]  = tmp[i, :3, 6:9]
                    
        if grad:
            return _C, _C_s, _C_l                  
        elif stress:
            return _C, _C1
        else:
            return _C

    def kef_many(self, X1, X2, grad=False, stress=False):
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
        sigma2, sigma02, zeta = self.sigma**2, self.sigma0**2, self.zeta
        if len(X2) > 1:  #pack X2 to big array
            X2 = pack_x2(X2, stress)

        x_all, dxdr_all, _, ele_all, x2_indices = X2[0]
        m1, m2, m2p = len(X1), len(x2_indices), len(x_all)
       
        x2_inds = [(0, x2_indices[0])]
        for i in range(1, len(x2_indices)):
            ID = x2_inds[i-1][1]
            x2_inds.append( (ID, ID+x2_indices[i]) )
       
        if self.device == 'cpu':
            if stress or grad:
                C = np.zeros([m1, m2p, 9])
            else:
                C = np.zeros([m1, m2p, 3])
        else:
            if stress or grad:
                C = cp.zeros([m1, m2p, 9])
            else:
                C = cp.zeros([m1, m2p, 3])
            dxdr_all = cp.array(dxdr_all)

        for i in range(m1):
            (x1, ele1) = X1[i]
            mask = get_mask(ele1, ele_all)
            
            C[i] = K_ef(x1, x_all, dxdr_all, sigma2, sigma02, zeta, grad, mask, device=self.device)

        if self.device == 'gpu':
            C = cp.asnumpy(C)

        _C = np.zeros([m1, m2*3])
        if grad:
            _C_s = np.zeros([m1, m2*3])
            _C_l = np.zeros([m1, m2*3])
        elif stress:
            _C1 = np.zeros([m1, m2*6])

        for j, ind in enumerate(x2_inds):
            tmp = C[:, ind[0]:ind[1], :].sum(axis=1) 
            _C[:, j*3:(j+1)*3] =  tmp[:, :3]
            if stress:
                _C1[:, j*6:(j+1)*6] =  tmp[:, 3:]
            elif grad:
                _C_s[:, j*3:(j+1)*3] =  tmp[:, 3:6]
                _C_l[:, j*3:(j+1)*3] =  tmp[:, 6:9]

        if grad:
            return _C, _C_s, _C_l                  
        elif stress:
            return _C, _C1
        else:
            return _C



def K_ee(x1, x2, sigma2, sigma02, zeta=2, grad=False, mask=None, eps=1e-8):
    """
    Compute the Kee between two structures
    Args:
        x1: [M, D] 2d array
        x2: [N, D] 2d array
        sigma2: float
        sigma02: float
        zeta: power term, float
        mask: to set the kernel zero if the chemical species are different
    """
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps

    D, _ = fun_D(x1, x2, x1_norm, x2_norm, zeta) #
    dk_dD = fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, sigma02, zeta, mask) #m, n
    Kee0 = dk_dD*(D+sigma02) # [m, n] * [m, n]
    Kee = np.sum(Kee0)
    mn = len(x1)*len(x2)

    if grad:
        dKee_dsigma = 2*np.sum(Kee0)/np.sqrt(sigma2)
        dKee_dsigma0 = 2*dk_dD.sum()*np.sqrt(sigma02)
        return Kee/mn, dKee_dsigma/mn, dKee_dsigma0/mn
    else:
        return Kee/mn

def K_ff(x1, x2, dx1dr, dx2dr, sigma2, sigma02, zeta=2, grad=False, mask=None, eps=1e-8, device='cpu'):
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x1_norm2 = x1_norm**2       
    x1_norm3 = x1_norm**3        
    x2_norm3 = x2_norm**3    
    x2_norm2 = x2_norm**2      

    x1x2_norm = x1_norm[:,None]*x2_norm[None,:]    
    x1_x1_norm3 = x1/x1_norm3[:,None]
    x2_x2_norm3 = x2/x2_norm3[:,None]

    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1
    dk_dD = sigma2*np.ones([len(x1), len(x2)])
    if mask is not None:
        dk_dD[mask] = 0

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

    if device == 'gpu':
        dx1dr = cp.array(dx1dr)
        dx2dr = cp.array(dx2dr)
        x1x2_norm = cp.array(x1x2_norm)
        x1_x1_norm3 = cp.array(x1_x1_norm3)
        x2_x2_norm3 = cp.array(x2_x2_norm3)

        dk_dD = cp.array(dk_dD)
        D1 = cp.array(D1)
        D2 = cp.array(D2)
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
    d2D_dx1dx2 = dd_dx1_dd_dx2 * D2[:,:,None,None] * (zeta-1)
    d2D_dx1dx2 += D1[:,:,None,None]*d2d_dx1dx2
    d2D_dx1dx2 *= zeta
    d2k_dx1dx2 = d2D_dx1dx2

    if not grad:
        tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None] #n1, n2, d, d
        _kff1 = (dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None]).sum(axis=(0,2)) # n1,n2,3
        kff = (_kff1[:,:,:,None] * dx2dr[:,:,None,:]).sum(axis=1)  # n2, 3, 3
        return kff

def K_ef(x1, x2, dx2dr, sigma2, sigma02, zeta=2, grad=False, mask=None, eps=1e-8, device='cpu'):
    """
    Compute the Kef between one structure and many atomic configurations
    """

    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x2_norm2 = x2_norm**2
    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1

    dk_dD = sigma2*np.ones([len(x1), len(x2)])
    if mask is not None:
        dk_dD[mask] = 0
    
    if device == 'gpu':
        x1 = cp.array(x1)
        x2 = cp.array(x2)
        x1_norm = cp.array(x1_norm)
        x2_norm = cp.array(x2_norm)
        x2_norm2 = cp.array(x2_norm2)
        x1x2_dot = cp.array(x1x2_dot)
        D1 = cp.array(D1)
        dk_dD = cp.array(dk_dD)

    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None] 
    dd_dx2 = (tmp21-tmp22)/tmp23

    zd1 = zeta * D1
    dD_dx2 = zd1[:,:,None] * dd_dx2
    m = len(x1)

    kef1 = (-dD_dx2[:,:,:,None]*dx2dr[None,:,:,:]).sum(axis=2) #[m, n, 9]
    Kef = (kef1*dk_dD[:,:,None]).sum(axis=0) #[n, 9]
    if grad:
        if device == 'gpu':
            D = cp.array(D)
        l = np.sqrt(l2)
        l3 = l2*l
        d2k_dDdsigma = 2*dk_dD/np.sqrt(sigma2)
        d2k_dDdl = (D/l3)*dk_dD + (2/l)*dk_dD

        dKef_dsigma = (kef1*d2k_dDdsigma[:,:,None]).sum(axis=0) 
        dKef_dl     = (kef1*d2k_dDdl[:,:,None]).sum(axis=0)
        if device == 'gpu':
            return cp.concatenate((Kef/m, dKef_dsigma/m, dKef_dl/m), axis=-1)
        else:
            return np.concatenate((Kef/m, dKef_dsigma/m, dKef_dl/m), axis=-1)
    else:
        return Kef/m

# =================== Algebras for k, dkdD, d2kdDdsigma ==========================
def fun_dk_dD(x1, x2, x1_norm, x2_norm, sigma2, sigma02, zeta=2, mask=None):
    D, _ = fun_D(x1, x2, x1_norm, x2_norm, zeta)
    dkdD = sigma2*np.ones([len(x1), len(x2)])
    if mask is not None:
        dkdD[mask] = 0
    return dkdD

def fun_d2k_dDdsigma(dkdD, sigma2):
    return 2*dkdD/np.sqrt(sigma2)
