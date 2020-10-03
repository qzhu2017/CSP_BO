import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

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
        self.set_train_pts(TrainData)

        def obj_func(params, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(
                    params, eval_gradient=True, clone_kernel=False)
                if show:
                    strs = "Loss: {:12.3f} ".format(-lml)
                    for para in params:
                        strs += "{:6.3f} ".format(para)
                    from scipy.optimize import approx_fprime
                    print("from: ", grad)
                    print("scipy", approx_fprime(params, self.log_marginal_likelihood, 1e-7))
                    #import sys
                    #sys.exit()
                    print(strs)
                return (-lml, -grad)
            else:
                return -self.log_marginal_likelihood(params, clone_kernel=False)

        params, loss = self.optimize(obj_func, self.kernel.parameters(), self.kernel.bounds)
        self.kernel.update(params)
        K = self.kernel.k_total(self.train_x)
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
        self.alpha_ = cho_solve((self.L_, True), self.y_train)  # Line 3
        return self

    def predict(self, X, return_std=False, return_cov=False):
        K_trans = self.kernel.k_total(X, self.train_x)
        pred = K_trans.dot(self.alpha_)
        y_mean = pred[:, 0]

        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T) 
            y_cov = self.kernel.k_total(X) - K_trans.dot(v) 
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
    
    def set_train_pts(self, data):
        self.train_x = {'energy': [], 'force': []}
        self.train_y = {'energy': [], 'force': []}
        for key in data.keys():
            if key == 'energy':
                for eng_data in data[key]:
                    self.add_train_pts_energy(eng_data)
            elif key == 'force':
                for force_data in data[key]:
                    self.add_train_pts_force(force_data)

        # convert self.train_y to 1D numpy array
        Npt_E = len(self.train_y["energy"])
        Npt_F = 3*len(self.train_y["force"])
        y_train = np.zeros([Npt_E+Npt_F, 1])
        count = 0
        for i in range(len(y_train)):
            if Npt_E > 0 and i < Npt_E:
                y_train[i,0] = self.train_y["energy"][i]
            else:
                if (i-Npt_E)%3 == 0:
                    y_train[i:i+3,0] = self.train_y["force"][count] 
                    count += 1
        self.y_train=y_train

    def add_train_pts_energy(self, energy_data):
        """
        energy_data is a list of tuples (X, E)
        X: the descriptors for a given structure: N1*d
        E: total energy: scalor
        N1 is the number of atoms in the given structure
        """
        (X, E) = energy_data
        self.train_x['energy'].append(X)
        self.train_y['energy'].append(E)
    
    def add_train_pts_force(self, force_data):
        """
        force_data is a list of tuples (X, dXdR, F)
        X: the descriptors for a given structure: N2*d
        dXdR: the descriptors: N2*d*3
        F: atomic force: 1*3
        N2 is the number of the centered atoms' neighbors within the cutoff
        """
        (X, dXdR, F) = force_data
        self.train_x['force'].append((X, dXdR))
        self.train_y['force'].append(F)

    def log_marginal_likelihood(self, params, eval_gradient=False, clone_kernel=False):
        
        if clone_kernel:
            kernel = self.kernel.update(params)
        else:
            kernel = self.kernel
            kernel.update(params)
        if eval_gradient:
            K, K_gradient = kernel.k_total_with_grad(self.train_x)
            #print(K[:3,:3])
            #print(K[3:,3:])
            #print(K[:3,3:])
            #print(K[3:,:3])
            #import sys
            #sys.exit()
        else:
            K = kernel.k_total(self.train_x)
        K[np.diag_indices_from(K)] += self.noise

        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(params)) if eval_gradient else -np.inf

        y_train = self.y_train

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




