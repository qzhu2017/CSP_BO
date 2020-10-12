import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
import warnings

class GaussianProcess():
    """ Gaussian Process Regressor. """
    def __init__(self, kernel=None, noise_e=1e-3, noise_f=5e-2):
        self.noise_e = noise_e
        self.noise_f = noise_f
        self.x = None
        self.kernel = kernel

    def __str__(self):
        s = "------Gaussian Process Regression------\n"
        s += "Kernel: {:s}".format(str(self.kernel))
        if hasattr(self, "train_x"):
            s += " {:d} energy ".format(len(self.train_x["energy"]))
            s += " {:d} forces\n".format(len(self.train_x["force"]))
        return s

    def __repr__(self):
        return str(self)


    def fit(self, TrainData=None, show=True):
        # Set up optimizer to train the GPR
        if TrainData is not None:
            self.set_train_pts(TrainData)
        else:
            self.update_y_train()

        if show:
            print(self)

        def obj_func(params, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(
                    params, eval_gradient=True, clone_kernel=False)
                if show:
                    strs = "Loss: {:12.3f} ".format(-lml)
                    for para in params:
                        strs += "{:6.3f} ".format(para)
                    #from scipy.optimize import approx_fprime
                    #print("from ", grad)
                    #print("scipy", approx_fprime(params, self.log_marginal_likelihood, 1e-5))
                    #import sys
                    #sys.exit()
                    print(strs)
                return (-lml, -grad)
            else:
                return -self.log_marginal_likelihood(params, clone_kernel=False)

        params, loss = self.optimize(obj_func, self.kernel.parameters(), self.kernel.bounds)
        self.kernel.update(params)
        K = self.kernel.k_total(self.train_x)

        # add noise matrix
        #K[np.diag_indices_from(K)] += self.noise
        noise = np.eye(len(K))
        NE = len(self.train_x['energy'])
        noise[:NE,:NE] *= self.noise_e**2
        noise[NE:,NE:] *= self.noise_f**2
        K += noise

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

    def predict(self, X, kff_quick=False, total_E=False, return_std=False, return_cov=False):
        K_trans = self.kernel.k_total(X, self.train_x, kff_quick=kff_quick)
        pred = K_trans.dot(self.alpha_)
        y_mean = pred[:, 0]

        Npts = 0
        if 'energy' in X:
            Npts += len(X["energy"])
        if 'force' in X:
            Npts += 3*len(X["force"])

        factors = np.ones(Npts)

        if total_E:
            N_atoms = np.array([len(x) for x in X["energy"]]) 
            factors[:len(N_atoms)] = N_atoms
        y_mean *= factors
        
        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T) 
            y_cov = self.kernel.k_total(X) - K_trans.dot(v) 
            return y_mean, y_cov
        elif return_std:
            if self._K_inv is None:
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                self._K_inv = L_inv.dot(L_inv.T)
            y_var = self.kernel.diag(X)
            y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)
            #ans = np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)
            #if ans.shape[0] < 3:
            #    print("y_mean", y_mean)
            #    print("sigma", self.kernel.diag(X))
            #    print("K_trans", K_trans.shape, np.sum(K_trans))
            #    print("KCK", ans, "y_var", y_var)
            #    print(K_trans)
            y_var_negative = y_var < 0
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                                "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)*factors
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
        self.update_y_train()
    
    def update_y_train(self):
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

    def validate_data(self, test_data=None, total_E=False, return_std=False):
        if test_data is None:
            test_X_E = {"energy": self.train_x['energy']}
            test_X_F = {"force": self.train_x['force']}
            E = self.y_train[:len(test_X_E['energy'])].flatten()
            F = self.y_train[len(test_X_E['energy']):].flatten()
        else:
            test_X_E = {"energy": [data[0] for data in test_data['energy']]}
            test_X_F = {"force": [(data[0], data[1]) for data in test_data['force']]}
            E = np.array([data[1] for data in test_data['energy']])
            F = np.array([data[2] for data in test_data['force']]).flatten()

        if total_E:
            for i in range(len(E)):
                E[i] *= len(test_X_E['energy'][i])
                
        E_Pred, E_std, F_Pred, F_std = None, None, None, None
        if return_std:
            if len(test_X_E['energy']) > 0:
                E_Pred, E_std = self.predict(test_X_E, total_E=total_E, return_std=True)  
            if len(test_X_F['force']) > 0:
                F_Pred, F_std = self.predict(test_X_F, return_std=True)
            return E, E_Pred, E_std, F, F_Pred, F_std
        else:
            if len(test_X_E['energy']) > 0:
                E_Pred = self.predict(test_X_E, total_E=total_E)  
            if len(test_X_F['force']) > 0:
                F_Pred = self.predict(test_X_F)
            return E, E_Pred, F, F_Pred


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
        #self.update_y_train()

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
        #self.update_y_train()

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
        # add noise matrix
        #K[np.diag_indices_from(K)] += self.noise
        noise = np.eye(len(K))
        NE = len(self.train_x['energy'])
        noise[:NE,:NE] *= self.noise_e**2
        noise[NE:,NE:] *= self.noise_f**2
        K += noise



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




