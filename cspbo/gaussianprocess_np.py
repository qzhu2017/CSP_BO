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
                mat[i, j] = D.mean()
                mat[j, i] = mat[i, j]
                C[i, j] = np.mean(np.exp(-0.5 * D / self.sigma ** 2))
                C[j, i] = C[i, j]
        C *= self.delta**2
        #print(C[:5,:5])
        #import sys
        #sys.exit()

        if grad:
            # dk/d_delta
            C_grad[:,:,0] = 2*C/self.sigma
            # dk/ds
            C_grad[:,:,1] = C*mat/self.sigma**3

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
                    print("Loss: {:12.3f} {:6.3f} {:6.3f}".format(-lml, *params))
                return -lml, -grad
            else:
                return -self.log_marginal_likelihood(theta, clone_kernel=False)

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
                X.append(_X[1][i])
            elif _X[1][i] is None or self.kernel.name in ["RBF_mb", "Dot_mb"]:
                X.append(x)
            else:
                X.append(x, _X[1][i])

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
                X.append(x, _X[1][i])
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
        opt_res = minimize(fun, theta0, method="L-BFGS-B", \
            jac=True, bounds=bounds, 
            options={'maxiter': 10, 'ftol': 1e-3})
        return opt_res.x, opt_res.fun




