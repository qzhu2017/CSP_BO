import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def distance(x1, x2):
    """
    Args:
        X1: N1*M
        X2: N2*M

    Returns:
        distance: scalor
    """
    return x1@x2.T/(1e-4+np.outer(np.linalg.norm(x1, axis=1), np.linalg.norm(x2, axis=1)))
    #return x1@x2.T/np.outer(x1@x1.T, x2@x2.T)
    #return cdist(x1, x2, metric='cosine')

class Dot():
    def __init__(self, para=[1., 1.], bounds=[[1e-1, 2e+2], [1e-1, 1e+2]]):
        self.name = 'Dot_mb'
        self.bounds = bounds
        self.update(para)

    def __str__(self):
        return "{:6.3f}**2 *Dot(sigma_0={:6.3f})".format(self.delta, self.sigma0)
 
    def parameters(self):
        return [self.delta, self.sigma0]
 
    def update(self, para):
        self.delta, self.sigma0 = para[0], para[1]

    def covariance(self, X1, X2=None):
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                C[i, j] = distance(x1, x2).mean()
        C += self.sigma0**2
        C *= self.delta**2
        return C

    def auto_covariance(self, X1, grad=False):
        m1 = len(X1)
        mat = np.zeros([m1, m1])
        if grad:
            C_grad = np.zeros([m1, m1, 2])
        for i, x1 in enumerate(X1):
            for j in range(i, m1):
                tmp = distance(X1[i], X1[j]).mean()
                mat[i, j] = tmp
                mat[j, i] = tmp
        mat += self.sigma0**2
        C = mat*self.delta**2
        #print(C[:5,:5])
        #import sys
        #sys.exit()
        if grad:
            C_grad[:,:,0] = 2*self.sigma0*mat
            C_grad[:,:,1] = 2*self.sigma0*self.delta**2*np.ones([m1, m1])

            return C, C_grad 
        else:
            return C

class RBF():
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 2e+1], [1e-1, 1e+1]]):
        self.name = 'RBF_mb'
        self.bounds = bounds
        self.update(para)

    def __str__(self):
        return "{:.3f}**2 *RBF(length={:.3f})".format(self.delta, self.sigma)
 
    def parameters(self):
        return [self.delta, self.sigma]
 
    def update(self, para):
        self.delta, self.sigma = para[0], para[1]

    def covariance(self, X1, X2=None):
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                #D = (1-distance(x1, x2))**2
                D = 1-distance(x1, x2)**2
                C[i, j] = np.mean(np.exp(-0.5 * D / self.sigma ** 2))
        return C*self.delta**2

    def auto_covariance(self, X1, grad=False):
        m1 = len(X1)
        mat = np.zeros([m1, m1])
        C = np.zeros([m1, m1])
        if grad:
            C_grad = np.zeros([m1, m1, 2])
        for i, x1 in enumerate(X1):
            for j in range(i, m1):
                D = 1-distance(X1[i], X1[j])**2
                tmp = self.delta**2*np.exp(-0.5*D/self.sigma**2)
                C[i, j] = np.mean(tmp)
                mat[i, j] = np.mean(tmp*D)
                mat[j, i] = mat[i, j]

                C[j, i] = C[i, j]

        if grad:
            C_grad[:,:,0] = 2*C/self.delta
            C_grad[:,:,1] = mat/self.sigma**3

            return C, C_grad
        else:
            return C

class RBF_2b():
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 2e+1], [1e-1, 1e+1]]):
        self.name = 'RBF_2b'
        self.bounds = bounds
        self.update(para)

    def __str__(self):
        return "{:.3f}**2 *RBF_2b(length={:.3f})".format(self.delta, self.sigma)
 
    def parameters(self):
        return [self.delta, self.sigma]
 
    def update(self, para):
        self.delta, self.sigma = para[0], para[1]

    def covariance(self, X1, X2=None):
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])
        for i, data1 in enumerate(X1):
            (x1, n1) = data1
            for j, data2 in enumerate(X2):
                (x2, n2) = data2
                f = np.outer(x1[1], x2[1])
                d2 = (x1[0][:, None] - x2[0][None, :])**2
                C[i, j] = np.sum(f*np.exp(-0.5*d2/self.sigma ** 2))
                C[i, j] = C[i, j]/(n1*n2)
        return C*self.delta**2

    def auto_covariance(self, X1, grad=False):
        m1 = len(X1)
        mat = np.zeros([m1, m1])
        C = np.zeros([m1, m1])
        if grad:
            C_grad = np.zeros([m1, m1, 2])
        for i, data1 in enumerate(X1): #struc1
            (x1, n1) = data1
            for j in range(i, m1):  #struc2
                (x2, n2) = X1[j]
                f = np.outer(x1[1], x2[1])

                d2 = (x1[0][:, None] - x2[0][None, :])**2
                tmp = self.delta**2*f*np.exp(-0.5*d2/self.sigma**2)

                mat[i, j] = np.sum(tmp*d2)/(n1*n2)
                mat[j, i] = mat[i, j]
                C[i, j] = np.sum(tmp)/(n1*n2)
                C[j, i] = C[i, j]

        if grad:
            C_grad[:,:,0] = 2*C/self.delta
            C_grad[:,:,1] = mat/self.sigma**3

            return C, C_grad
        else:
            return C


class Combo():
    def __init__(self, para=[1., 1., 1., 1.], bounds=[(1e-1, 2e+2), (1e-1, 2e+1), (1e-1, 2e+2), (1e-1, 2e+1)]):
        self.name = 'Combo'
        self.bounds = bounds
        self.update(para)

    def __str__(self):
        strs = "{:.3f}**2 *RBF_2b(length={:.3f}) + ".format(self.delta1, self.sigma1)
        strs += "{:.3f}**2 *RBF_mb(length={:.3f})".format(self.delta2, self.sigma2)
        return strs

    def parameters(self):
        return [self.delta1, self.sigma1, self.delta2, self.sigma2]
 
    def update(self, para):
        self.delta1, self.sigma1, self.delta2, self.sigma2 = para[0], para[1], para[2], para[3]

    def covariance(self, X1, X2=None):
        m1, m2 = len(X1), len(X2)
        C = np.zeros([m1, m2])

        for i, data1 in enumerate(X1):
            _x1, (x1, n1) = data1
            for j, data2 in enumerate(X2):
                _x2, (x2, n2) = data2

                f = np.outer(x1[1], x2[1])
                tmp = x1[0][:, None] - x2[0][None, :]
                C[i, j] += self.delta1**2*np.sum(f*np.exp(-0.5*tmp*tmp/self.sigma1 ** 2))

                D = 1-distance(_x1, _x2)**2
                C[i, j] += self.delta2**2*np.sum(np.exp(-0.5*D/self.sigma2**2))
 
                C[i, j] = C[i, j]/(n1*n2)
        return C

    def auto_covariance(self, X1, grad=False):
        m1 = len(X1)
        mat1 = np.zeros([m1, m1])
        mat2 = np.zeros([m1, m1])
        C1 = np.zeros([m1, m1])
        C2 = np.zeros([m1, m1])
        if grad:
            C_grad = np.zeros([m1, m1, 4])
        for i, data1 in enumerate(X1): #struc1
            _x1, (x1, n1) = data1
            for j in range(i, m1):  #struc2
                _x2, (x2, n2) = X1[j]

                f = np.outer(x1[1], x2[1])
                d1 = (x1[0][:, None] - x2[0][None, :])**2
                tmp1 = self.delta1**2*f*np.exp(-0.5*d1/self.sigma1**2)

                mat1[i, j] = np.sum(tmp1*d1)/(n1*n2)
                mat1[j, i] = mat1[i, j]

                C1[i, j] = np.sum(tmp1)/(n1*n2)
                C1[j, i] = C1[i, j]

                d2 = 1-distance(_x1, _x2)**2
                tmp2 = self.delta2**2*np.exp(-0.5*d2/self.sigma2**2)

                mat2[i, j] = np.mean(tmp2*d2)
                mat2[j, i] = mat2[i, j]
                C2[i, j] = np.mean(tmp2)
                C2[j, i] = C2[i, j]
        C = C1 + C2
        if grad:
            C_grad[:,:,0] = 2*C1/self.delta1
            C_grad[:,:,1] = mat1/self.sigma1**3
            C_grad[:,:,2] = 2*C2/self.delta2
            C_grad[:,:,3] = mat2/self.sigma2**3

            return C, C_grad
        else:
            return C



class GaussianProcess():
    """ Gaussian Process Regressor. """
    def __init__(self, kernel=None, noise=1e-3):
        self.noise = noise
        self.x = None
        self.kernel = kernel

    def fit(self, TrainData, show=True):
        # Set up optimizer to train the GPR
        if show:
            print("Strart Training: ", str(self.kernel))
        self.unpack_data(TrainData)

        def obj_func(params, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(
                    params, eval_gradient=True, clone_kernel=False)
                if show:
                    strs = "Loss: {:12.3f} ".format(-lml)
                    for para in params:
                        strs += "{:6.3f} ".format(para)
                    #from scipy.optimize import approx_fprime
                    #print("from: ", grad)
                    #print("scipy", approx_fprime(params, self.log_marginal_likelihood, 1e-6))
                    #import sys
                    #sys.exit()
                    print(strs)
                return (-lml, -grad)
            else:
                return -self.log_marginal_likelihood(params, clone_kernel=False)

        params, loss = self.optimize(obj_func, self.kernel.parameters(), self.kernel.bounds)
        self.kernel.update(params)
        K = self.kernel.auto_covariance(self.x)
        K[np.diag_indices_from(K)] += self.noise
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            print(K)
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel,) + exc.args
            raise

        self.alpha_ = cho_solve((self.L_, True), self.y)  # Line 3
        return self

    def predict(self, _X, return_std=False, return_cov=False):
        #print("Predict: ", len(_X))
        X = []
        for i, x in enumerate(_X[0]):
            if x is None or self.kernel.name in ["RBF_2b"]:
                if i < len(_X[1]):
                    X.append(_X[1][i])
            elif _X[1][i] is None or self.kernel.name in ["RBF_mb", "Dot_mb"]:
                X.append(x)
            else:
                X.append((x, _X[1][i]))
        K_trans = self.kernel.covariance(X, self.x)
        pred = K_trans.dot(self.alpha_)
        y_mean = pred[:, 0]

        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T) 
            y_cov = self.kernel.auto_covariance(X) - K_trans.dot(v) 
            return y_mean, y_cov.detach().numpy()
        elif return_std:
            if self._K_inv is None:
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                self._K_inv = L_inv.dot(L_inv.T)
            y_var = self.kernel_.diag(X)
            y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)
            y_var_negative = y_var < 0
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            if np.any(y_var_negative):
                 warnings.warn("Predicted variances smaller than 0. "
                                "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean
    
    def unpack_data(self, data):
        # decompose X
        (_X, _Y) = data
        X = []
        Y = np.zeros([len(_Y), 1])
        for i, x in enumerate(_X[0]):
            if x is None or self.kernel.name in ["RBF_2b"]:
                X.append(_X[1][i])
            elif _X[1][i] is None or self.kernel.name in ["RBF_mb", "Dot_mb"]:
                X.append(x)
            else:
                X.append((x, _X[1][i]))
            Y[i,0] += _Y[i]

        self.x = X
        self.y = Y

    def log_marginal_likelihood(self, params, eval_gradient=False, clone_kernel=False):
        
        if clone_kernel:
            kernel = self.kernel.update(params)
        else:
            kernel = self.kernel
            kernel.update(params)
        if eval_gradient:
            K, K_gradient = kernel.auto_covariance(self.x, True)
        else:
            K = kernel.auto_covariance(self.x)
        K[np.diag_indices_from(K)] += self.noise

        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(params)) if eval_gradient else -np.inf

        y_train = self.y
        alpha = cho_solve((L, True), y_train)

        # log marginal likelihood
        ll_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        ll_dims -= np.log(np.diag(L)).sum()
        ll_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        MLL = ll_dims.sum(-1)  #sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            llg_dims = 0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
            llg = llg_dims.sum(-1)
            #print("Loss: {:12.4f} {:s}".format(-MLL, str(kernel)))
            return MLL, llg
        else:
            return MLL


    def optimize(self, fun, theta0, bounds):
        opt_res = minimize(fun, theta0, method="L-BFGS-B", bounds=bounds, 
            jac=True, options={'maxiter': 10, 'ftol': 1e-3})
        #print(opt_res)
        return opt_res.x, opt_res.fun



def covariance_energy_force(X1, X2, dX1, dX2, SEQ1, SEQ2):
    m1, m2 = len(X1), len(X2)
    n1, n2 = 0, 0
    
    N1 = []
        for i, x1 in enumerate(X1):
            if dX1[i]:
                N = 0
                for ele in self.elements:
                    l = self.atoms[ele][i]
                    N += l
                    n1 += l * 3
                N1.append(N)
            else:
                N1.append(None)

    N2 = []
    for i, x2, in enumerate(X2):
        if dX2[i]:
            N = 0
            for ele in self.elements:
                l = self.atoms[ele][i]
                N += l
                n2 += l * 3
            N2.append(N)
        else:
            N2.append(None)

    out = torch.zeros((m1+n1, m2+n2), dtype=torch.float64)
    dout = torch.zeros((m1+n1, m2+n2, 2), dtype=torch.float64)

    ki, ni = m1, 0
    for ele in self.elements:
        for i, (x1, dx1, seq1, n1) in enumerate(zip(X1, dX1, SEQ1, N1)):
            kj = m2
            if dx1:
                ni = len(x1[ele])*3
            
            for j, (x2, dx2, seq2, n2) in enumerate(zip(X2, dX2, SEQ2, N2)):
                if dx2:
                    nj = len(x2[ele])*3

                # Covariance between E_i and E_j
                D = np.sum(x1*x1, axis=1, keepdims=True) + np.sum(x2*x2, axis=1) - 2*x1@x2.T
                K = self.sigmaF ** 2 * np.exp(-(0.5 / self.sigmaL ** 2) * D)
                out[i,j] = np.sum(K)
                
                KeePrime = np.zeros([2,])
                KeePrime[0] = (2 / self.sigmaF) * Kee
                KeePrime[1] = np.sum(K * D) / self.sigmaL ** 3
                dout[i,j,:] += KeePrime
                
                # Covariance between F_i and E_j
                if dx1:
                    shp1 = x1.shape
                    dxdr1 = np.zeros([shp1[0], n1, shp1[1], 3])
                    for _m in range(n1):
                        rows = np.where(seq1[:,1]==_m)[0]
                        dxdr1[seq1[rows, 0], _m, :, :] += dx1[rows, :, :]
                    grad_x1 = K[:,:,None] * -1 * (x1[:,None,:] - x2)
                    Kfe = np.einsum("ijk,ilkm->lm", grad_x1, dxdr1).ravel() / self.sigmaL ** 2

                    KfePrime = np.zeros([len(Kfe), 2])
                    KfePrime[:,0] = (2 / self.sigmaF) * Kfe
                    kfep = (-K*D/self.sigmaL**5 + (2/self.sigmaL**3)*K)[:,:,None] * (x1[:, None, :] - x2)
                    KfePrime[:,1] = np.einsum("ijk,ilkm->lm", kfep, dxdr1).ravel()

                    out[ki:ki+ni, j], dout[ki:ki+ni, j, :] = fc * Kfe, fc * KfePrime

                # Covariance between E_i and F_j
                if dx2:
                    shp2 = x2.shape
                    dxdr2 = np.zeros([shp2[0], n2, shp2[1], 3])
                    for _m in range(n2):
                        rows = np.where(seq2[:,1]==_m)[0]
                        dxdr2[seq2[rows, 0], _m, :, :] += dx2[rows, :, :]
                    grad_x2 = K[:,:,None] * (x1[:,None,:]-x2)
                    Kef = np.einsum("ijk,jlkm->lm", grad_x2, dxdr2).ravel() / self.sigmaL ** 2

                    KefPrime = np.zeros([len(Kef), 2])
                    KefPrime[:,0] = (2/self.sigmaF) * Kef
                    kefp = (K*D/self.sigmaL**5 - (2/self.sigmaL**3)*K)[:, :, None] * (x1[:, None, :] - x2)
                    KefPrime[:,1] = np.einsum("ijk,ilkm->lm", kefp, dxdr2).ravel()
                    
                    out[i, kj:kj+nj], dout[i, kj:kj+nj, :] = fc * Kef, fc * KefPrime

                # Covariance betweeen F_i and F_j
                if dx1 and dx2:
                    M = (x1[:, None, :] - x2) / self.sigmaL ** 2
                    Q = np.eye(shp1[1]) / self.sigmaL ** 2 - M[:,:,:,None] * M[:,:,None,:]
                    grad_x1x2 = K[:,:,None,None] * Q
                    grad_x1x2_dx1 = np.einsum("ijkl, ihkm->jlhm", dxdr1, grad_x1x2)
                    Kff = np.einsum("jlhm, hnmp -> jlnp", grad_x1x2_dx1, dxdr2).reshape(ni, nj)

                    KffPrime = np.zeros([ni, nj, 2])
                    KffPrime[:,:0] = (2 / self.sigmaF) * Kff
                    kffp = (K * D / self.sigmaL ** 3)[:,:,None,None] * Q
                    kffp += K[:,:,None,None] * ((-2 / self.sigmaL ** 3) * np.eye(shp1[1]) + ((4 / self.sigmaL **3) * M[:,:,:None] * M[:,:,None,:]))
                    KffPrime_dxdr1 = torch.einsum("ijkl,ihkm->jlhm", dxdr1, kffp)
                    KffPrime[:,:,1] = torch.einsum("jlhm, hnmp -> jlnp",KffPrime1_dxdr1, dxdr2).reshape(ni, nj)

                    out[ki:ki+ni, kj:kj+nj], dout[ki:ki+ni, kj:kj+nj, :] = fc**2*Kff, fc**2*KffPrime
                                       
                                    
                if dx2:
                    kj += nj

            if dx1:
                ki += ni
        
    return out, dout
