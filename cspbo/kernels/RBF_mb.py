import numpy as np
from .kernel_base import *
from .utilities import tuple_to_list, list_to_tuple
from time import time
class RBF_mb():
    """
    .. math::
        k(x_i, x_j) = \sigma ^2 * \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)
    """
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 5e+1], [1e-1, 1e+1]], zeta=2, device='cpu'):
        self.name = 'RBF_mb'
        self.bounds = bounds
        self.update(para)
        self.zeta = zeta
        self.device = device
            
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
                C_ee[i] = K_ee(x1, x1, sigma2, l2, zeta, False, mask, wrap=True) 

        if "force" in data:
            NF = len(data["force"])
            C_ff = np.zeros(3*NF)
            for i in range(NF):
                (x1, dx1dr, ele1) = data["force"][i]
                mask = get_mask(ele1, ele1)
                tmp = K_ff(x1, x1, dx1dr, dx1dr, sigma2, l2, zeta, False, mask, wrap=True)
                C_ff[i*3:(i+1)*3] = np.diag(tmp)

        if C_ff is None:
            return C_ee
        elif C_ee is None:
            return C_ff
        else:
            return np.hstack((C_ee, C_ff))

    def k_total(self, data1, data2=None, same=False):
        """
        Compute the covairance for train data
        Used for energy/force prediction
        """

        if data2 is None:
            data2 = data1

        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        for key1 in data1.keys():
            for key2 in data2.keys():
                if len(data1[key1])>0 and len(data2[key2])>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee = self.kee_many(data1[key1], data2[key2])
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = self.kef_many(data1[key1], data2[key2])
                    elif key1 == 'force' and key2 == 'energy':
                        if not same:
                            C_fe = self.kef_many(data2[key2], data1[key1]).T
                        else:
                            C_fe = C_ef.T 
                    elif key1 == 'force' and key2 == 'force':
                        C_ff = self.kff_many(data1[key1], data2[key2])

        return build_covariance(C_ee, C_ef, C_fe, C_ff)
        
    def k_total_with_grad(self, data1):
        """
        Compute the covairance for train data
        Used for energy/force training
        """
        data2 = data1   
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        C_ee_s, C_ef_s, C_fe_s, C_ff_s = None, None, None, None
        C_ee_l, C_ef_l, C_fe_l, C_ff_l = None, None, None, None
        for key1 in data1.keys():
            for key2 in data2.keys():
                if len(data1[key1])>0 and len(data2[key2])>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee, C_ee_s, C_ee_l = self.kee_many(data1[key1], data2[key2], True)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef, C_ef_s, C_ef_l = self.kef_many(data1[key1], data2[key2], True)
                        C_fe, C_fe_s, C_fe_l = C_ef.T, C_ef_s.T, C_ef_l.T
                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_ff_s, C_ff_l = self.kff_many(data1[key1], data2[key2], True)
        C = build_covariance(C_ee, C_ef, C_fe, C_ff, None, None)
        C_s = build_covariance(C_ee_s, C_ef_s, C_fe_s, C_ff_s, None, None)
        C_l = build_covariance(C_ee_l, C_ef_l, C_fe_l, C_ff_l, None, None)

        #import sys; sys.exit()
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
 
    
    def kee_many(self, X1, X2, grad=False):
        """
        Compute the energy-energy kernel for many structures
        Args:
            X1: list of 2D arrays
            X2: list of 2D arrays
            grad: output gradient if true
        Returns:
            C: M*N 2D array
            C_grad:
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        x_all, ele_all, x2_indices = X2

        if isinstance(X1, tuple): #unpack X1, used for training
            X1 = tuple_to_list(X1, mode='energy')
            
        # num of X1, num of X2, num of big X2
        m1, m2, m2p = len(X1), len(x2_indices), len(x_all)

        x2_inds = [(0, x2_indices[0])]
        for i in range(1, len(x2_indices)):
            ID = x2_inds[i-1][1]
            x2_inds.append( (ID, ID+x2_indices[i]) )

        if grad:
            C = np.zeros([m1, m2p, 3])
        else:
            C = np.zeros([m1, m2p, 1])
 
        for i in range(m1):
            #if i%1000 == 0: print("Kee", i)
            (x1, ele1) = X1[i]
            mask = get_mask(ele1, ele_all)
            C[i] = K_ee(x1, x_all, sigma2, l2, zeta, grad, mask)
        
        _C = np.zeros([m1, m2])
        if grad:
            _C_s = np.zeros([m1, m2])
            _C_l = np.zeros([m1, m2])

        for j, ind in enumerate(x2_inds):
            tmp = C[:, ind[0]:ind[1], :].sum(axis=1)/x2_indices[j]
            _C[:, j]  = tmp[:, 0]
            if grad:
                _C_s[:, j]  = tmp[:, 1]
                _C_l[:, j]  = tmp[:, 2]
        if grad:
            return _C, _C_s, _C_l
        else:
            return _C

    def kff_many(self, X1, X2, grad=False, stress=False):
        """
        Compute the energy-force kernel between structures and atoms
        dXdR is a stacked array if stress is True
        Args:
            X1: list of tuples ([X, dXdR, ele])
            X2: stacked ([X, dXdR, ele])
            grad: compute gradient if true
            stress: compute stress if true
        Returns:
            C:
            C_grad:
        """

        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        x_all, dxdr_all, ele_all, x2_indices = X2

        if isinstance(X1, tuple): #unpack X1, used for training
            X1 = tuple_to_list(X1)

        # num of X1, num of X2, num of big X2
        m1, m2, m2p = len(X1), len(x2_indices), len(x_all)

        x2_inds = [(0, x2_indices[0])]
        for i in range(1, len(x2_indices)):
            ID = x2_inds[i-1][1]
            x2_inds.append( (ID, ID+x2_indices[i]) )

        if grad:
            C = np.zeros([m1, m2p, 3, 9])
        elif stress:
            C = np.zeros([m1, m2p, 9, 3])
        else:
            C = np.zeros([m1, m2p, 3, 3])
        #t0 = time()
        for i in range(m1):
            #if i%500 == 0: print("Kff", i, time()-t0)
            (x1, dx1dr, ele1) = X1[i]
            mask = get_mask(ele1, ele_all)
            batch = 500
            # split the big array to smaller size
            if m2p > batch:
                for j in range(int(np.ceil(m2p/batch))):
                    start = j*batch
                    end = min([(j+1)*batch, m2p])
                    mask = get_mask(ele1, ele_all[start:end])
                    C[i, start:end, :, :] = K_ff(x1, x_all[start:end], dx1dr, dxdr_all[start:end], sigma2, l2, zeta, grad, mask, device=self.device)
            else:
                mask = get_mask(ele1, ele_all)
                C[i] = K_ff(x1, x_all, dx1dr, dxdr_all, sigma2, l2, zeta, grad, mask, device=self.device)

        _C = np.zeros([m1*3, m2*3])
        if grad:
            _C_s = np.zeros([m1*3, m2*3])
            _C_l = np.zeros([m1*3, m2*3])
        elif stress:
            _C1 = np.zeros([m1*6, m2*3])

        for j, ind in enumerate(x2_inds):
            tmp = C[:, ind[0]:ind[1], :, :].sum(axis=1)
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
        
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta

        if isinstance(X1, tuple):  #pack X2 to big array in tuple
            X1 = tuple_to_list(X1, mode='energy')

        if isinstance(X2, list):  #pack X2 to big array in tuple
            X2 = list_to_tuple(X2, stress)

        x_all, dxdr_all, ele_all, x2_indices = X2
        m1, m2, m2p = len(X1), len(x2_indices), len(x_all)
       
        x2_inds = [(0, x2_indices[0])]
        for i in range(1, len(x2_indices)):
            ID = x2_inds[i-1][1]
            x2_inds.append( (ID, ID+x2_indices[i]) )
       
        if stress or grad:
            C = np.zeros([m1, m2p, 9])
        else:
            C = np.zeros([m1, m2p, 3])
        t0 = time()
        for i in range(m1):
            #if i%1000 == 0: print("Kef", i, time()-t0)
            (x1, ele1) = X1[i]
            mask = get_mask(ele1, ele_all)
            C[i] = K_ef(x1, x_all, dxdr_all, sigma2, l2, zeta, grad, mask, device=self.device)

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


# ===================== Standalone functions to compute K_ee, K_ef, K_ff

def K_ee(x1, x2, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8, wrap=False):
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

    k = sigma2*np.exp(-(0.5/l2)*(1-D))
    if mask is not None:
        k[mask] = 0
    dk_dD = (-0.5/l2)*k

    Kee = k.sum(axis=0)
    m = len(x1)

    if grad:
        l3 = np.sqrt(l2)*l2
        dKee_dsigma = 2*Kee/np.sqrt(sigma2)
        dKee_dl = (k*(1-D)).sum(axis=0)/l3
        #ans = np.stack((Kee/m, dKee_dsigma/m, dKee_dl/m), axis=-1)
        #print(Kee.shape, ans.shape)
        return np.stack((Kee/m, dKee_dsigma/m, dKee_dl/m), axis=-1)
    else:
        n = len(x2)
        if wrap:
            return Kee.sum()/(m*n)
        else:
            return Kee.reshape([n, 1])/m

def K_ff(x1, x2, dx1dr, dx2dr, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8, device='cpu', wrap=False):
    """
    Compute the Kff between one and many configurations
    x2, dx1dr, dx2dr will be called from the cuda device in the GPU mode
    """
    l = np.sqrt(l2)
    l3 = l*l2

    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x1_norm2 = x1_norm**2       
    x1_norm3 = x1_norm**3        
    x1x2_dot = x1@x2.T
    x1_x1_norm3 = x1/x1_norm3[:,None]

    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x2_norm2 = x2_norm**2      
    tmp30 = np.ones(x2.shape)/x2_norm[:,None]
    tmp33 = np.eye(x2.shape[1])[None,:,:] - x2[:,:,None] * (x2/x2_norm2[:,None])[:,None,:]


    x2_norm3 = x2_norm**3    
    x1x2_norm = x1_norm[:,None]*x2_norm[None,:]
    x2_x2_norm3 = x2/x2_norm3[:,None]

    d = x1x2_dot/(eps+x1x2_norm)
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1
    k = sigma2*np.exp(-(0.5/l2)*(1-D))

    if mask is not None: 
        k[mask] = 0
    
    dk_dD = (-0.5/l2)*k
    zd2 = -0.5/l2*zeta*zeta*(D1**2)

    tmp31 = x1[:,None,:] * tmp30[None,:,:]

    #t0 = time()
    tmp11 = x2[None, :, :] * x1_norm[:, None, None]
    tmp12 = x1x2_dot[:,:,None] * (x1/x1_norm[:, None])[:,None,:] 
    tmp13 = x1_norm2[:, None, None] * x2_norm[None, :, None] 
    dd_dx1 = (tmp11-tmp12)/tmp13

    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None]
    dd_dx2 = (tmp21-tmp22)/tmp23  # (29, 1435, 24)


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

        K_ff_0 = (d2k_dx1dx2[:,:,:,:,None] * dx1dr[:,None,:,None,:]).sum(axis=2) 
        K_ff_0 = (K_ff_0[:,:,:,:,None] * dx2dr[None,:,:,None,:]).sum(axis=2) 
        Kff = (K_ff_0 * dk_dD[:,:,None,None]).sum(axis=0)

        d2k_dDdsigma = 2*dk_dD/np.sqrt(sigma2)
        d2k_dDdl = ((D-1)/l3 + 2/l)*dk_dD
        #d2k_dDdl = (D/l3 + 2/l)*dk_dD

        dKff_dsigma = (K_ff_0 * d2k_dDdsigma[:,:,None,None]).sum(axis=0)
        dKff_dl = (-K_ff_0 * d2k_dDdl[:,:,None,None]).sum(axis=0)
        
        tmp = -dD_dx1_dD_dx2/l*2
        K_ff_1 = (tmp[:,:,:,:,None] * dx1dr[:,None,:,None,:]).sum(axis=2)
        K_ff_1 = (K_ff_1[:,:,:,:,None] * dx2dr[None,:,:,None,:]).sum(axis=2)
        dKff_dl += (K_ff_1 * dk_dD[:,:,None,None]).sum(axis=0) #/l*2
        return np.concatenate((Kff, dKff_dsigma, dKff_dl), axis=-1)
    else:
        tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None] #n1, n2, d, d
        _kff1 = (dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None]).sum(axis=(0,2)) # n1,n2,3
        kff = (_kff1[:,:,:,None] * dx2dr[:,:,None,:]).sum(axis=1)  # n2, 3, 9
        if wrap:
            kff = kff.sum(axis=0)
        return kff

def K_ef(x1, x2, dx2dr, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8, device='gpu'):
    """
    Compute the Kef between one structure and many atomic configurations
    """
    m = len(x1)
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
    kef1 = (-dD_dx2[:,:,:,None]*dx2dr[None,:,:,:]).sum(axis=2) #[m, n, 9]
    Kef = (kef1*dk_dD[:,:,None]).sum(axis=0) #[n, 9]
    if grad:
        l = np.sqrt(l2)
        l3 = l2*l
        d2k_dDdsigma = 2*dk_dD/np.sqrt(sigma2)
        d2k_dDdl = -((D-1)/l3 + 2/l)*dk_dD

        dKef_dsigma = (kef1*d2k_dDdsigma[:,:,None]).sum(axis=0) 
        dKef_dl     = (kef1*d2k_dDdl[:,:,None]).sum(axis=0)
        #print(dKef_dl/m)#; import sys; sys.exit()
        return np.concatenate((Kef/m, dKef_dsigma/m, dKef_dl/m), axis=-1)
    else:
        return Kef/m


