import numpy as np
import cupy as cp
from pyxtal.database.element import Element
from .utilities import new_pt, convert_train_data
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
import warnings
import json
from ase.db import connect
import os
from copy import deepcopy

class GaussianProcess():
    """ 
    Gaussian Process Regressor. 

    This class is to fit the interatomic potential from the reference data of energy/forces.

    Arg:
        kernel (callable): object to compute the covariance matrix
        descriptor (callable): object to compute the structure to descriptor
        base_potential (callable): object to compute the base potential before GPR (default is None)
        f_coef (float): the coefficient of force noise relative to energy
        noise_e (list): define the energy noise
    """

    def __init__(self, kernel=None, descriptor=None, base_potential=None, f_coef=10, noise_e=[5e-3, 2e-3, 1e-1]):
        
        self.noise_e = noise_e[0]
        self.f_coef = f_coef
        self.noise_f = self.f_coef*self.noise_e
        self.noise_bounds = noise_e[1:]

        self.descriptor = descriptor
        self.kernel = kernel
        self.base_potential = base_potential

        self.x = None
        self.train_x = None
        self.train_y = None
        self.train_db = None
        self.alpha_ = None

    def __str__(self):
        s = "------Gaussian Process Regression------\n"
        s += "Kernel: {:s}".format(str(self.kernel))
        if hasattr(self, "train_x"):
            s += " {:d} energy ({:.3f})".format(len(self.train_x["energy"]), self.noise_e)
            s += " {:d} forces ({:.3f})\n".format(len(self.train_x["force"][0][-1]), self.noise_f)
        return s

    def __repr__(self):
        return str(self)

    def fit(self, TrainData=None, show=True, opt=True):
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
        hyper_params = self.kernel.parameters() + [self.noise_e]
        hyper_bounds = self.kernel.bounds + [self.noise_bounds]
        if opt:
            params, loss = self.optimize(obj_func, hyper_params, hyper_bounds, )
            self.kernel.update(params[:-1])
            self.noise_e = params[-1]
            self.noise_f = self.f_coef*params[-1]
        K = self.kernel.k_total(self.train_x, same=True)

        # add noise matrix
        #K[np.diag_indices_from(K)] += self.noise
        noise = np.eye(len(K))
        NE = len(self.train_x['energy'])
        noise[:NE,:NE] *= self.noise_e**2
        noise[NE:,NE:] *= (self.f_coef*self.noise_e)**2
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

    def predict(self, X, stress=False, total_E=False, return_std=False, return_cov=False):
        if stress:
            K_trans, K_trans1 = self.kernel.k_total_with_stress(X, self.train_x, same=False)
            pred1 = K_trans1.dot(self.alpha_)
        else:
            K_trans = self.kernel.k_total(X, self.train_x)
        
        pred = K_trans.dot(self.alpha_)
        y_mean = pred[:, 0]

        Npts = 0
        if 'energy' in X:
            Npts += len(X["energy"])
        if 'force' in X:
            Npts += 3*len(X["force"][0][-1])

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
    
    def set_train_pts(self, data, mode="w"):
        """
        Set the training pts for the GPR mode
        two modes ("write" and "append") are allowed

        Args:
            data: a dictionary of energy/force/db data
            mode: "w" or "a+"
        """
        if mode == "w" or self.train_x is None: #reset
            self.train_x = {'energy': [], 'force': []}
            self.train_y = {'energy': [], 'force': []}
            self.train_db = []

        N_E = len(self.train_x['energy'])
        N_F = len(self.train_x['force'])
        for d in data["db"]:
            (atoms, energy, force, energy_in, force_in) = d
            if energy_in:
                e_id = deepcopy(N_E + 1)
                N_E += 1
            else:
                e_id = None

            if len(force_in) > 0:
                f_ids = [N_F+i for i in range(len(force_in))]
                N_F += len(force_in)
            else:
                f_ids = []
            self.train_db.append((atoms, energy, force, energy_in, force_in, e_id, f_ids))

        for key in data.keys():
            if key == 'energy':
                for eng_data in data[key]:
                    self.add_train_pts_energy(eng_data)
            elif key == 'force':
                #for force_data in data[key]:
                #    self.add_train_pts_force(force_data)
                self.add_train_pts_force(data[key])
        self.update_y_train()

    
    def remove_train_pts(self, e_ids, f_ids):
        """
        delete the training pts for the GPR model

        Args:
            e_ids: ids to delete in K_EE
            f_ids: ids to delete in K_FF
        """
        data = {"energy":[], "force": [], "db": []}
        N_E, N_F = len(self.train_x['energy']), len(self.train_x['force'])
        for id in range(N_E):
            if id not in e_ids:
                (X, ele) = self.train_x['energy'][id]
                E = self.train_y['energy'][id]
                data['energy'].append((X, E, ele))

        for id in range(N_F):
            if id not in f_ids:
                (X, dxdr, _, ele) = self.train_x['force'][id]
                F = self.train_y['force'][id]
                data["force"].append((X, dxdr, _, F, ele))

        for d in self.train_db:
            (atoms, energy, force, energy_in, force_in, e_id, _f_ids) = d
            keep = False
            if e_id in e_ids:
                energy_in = False
            _force_in = []       
            for i, f_id in enumerate(_f_ids):
                if f_id not in f_ids:
                    _force_in.append(force_in[i])
            if energy_in or len(_force_in)>0:
                data['db'].append((atoms, energy, force, energy_in, _force_in))
        
        self.set_train_pts(data) # reset the train data
        self.fit()

    def update_y_train(self):
        """ 
        convert self.train_y to 1D numpy array
        """
        Npt_E = len(self.train_y["energy"])
        Npt_F = 3*len(self.train_y["force"])
        y_train = np.zeros([Npt_E+Npt_F, 1])
        count = 0
        for i in range(len(y_train)):
            if Npt_E > 0 and i < Npt_E:
                y_train[i,0] = self.train_y["energy"][i]
            else:
                if (i-Npt_E)%3 == 0:
                    #print(i, count, y_train.shape, self.train_y["force"][count])
                    y_train[i:i+3,0] = self.train_y["force"][count] 
                    count += 1
        self.y_train=y_train

    def validate_data(self, test_data=None, total_E=False, return_std=False):
        """
        validate the given dataset
        """
        if test_data is None:
            test_X_E = {"energy": self.train_x['energy']}
            test_X_F = {"force": self.train_x['force']}
            E = self.y_train[:len(test_X_E['energy'])].flatten()
            F = self.y_train[len(test_X_E['energy']):].flatten()
        else:
            test_X_E = {"energy": [(data[0], data[2]) for data in test_data['energy']]}
            test_X_F = {"force": [(data[0], data[1], data[2], data[4]) for data in test_data['force']]}
            E = np.array([data[1] for data in test_data['energy']])
            F = np.array([data[3] for data in test_data['force']]).flatten()

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

    def predict_structure(self, struc, stress=True, return_std=False):
        """
        make prediction for a given structure
        """
        d = self.descriptor.calculate(struc) 
        ele = [Element(ele).z for ele in d['elements']]
        ele = np.array(ele)
        data = {"energy": [(d['x'], ele)]}
        data["force"] = []
        _n, l = d['x'].shape
        for i in range(len(struc)):
            ids = np.argwhere(d['seq'][:,1]==i).flatten()
            _i = d['seq'][ids, 0] 
            _x, _dxdr, ele0 = d['x'][_i,:], d['dxdr'][ids], ele[_i]

            if stress:
                _rdxdr = d['rdxdr'][ids]
                _rdxdr = _rdxdr.reshape([len(ids), l, 9])[:, :, [0, 4, 8, 1, 2, 5]]
                data["force"].append((_x, _dxdr, _rdxdr, ele0))
            else:
                data["force"].append((_x, _dxdr, None, ele0))
        if stress:
            K_trans, K_trans1 = self.kernel.k_total_with_stress(data, self.train_x)
        else:
            K_trans = self.kernel.k_total(data, self.train_x)

        pred = K_trans.dot(self.alpha_)
        y_mean = pred[:, 0]
        y_mean[0] *= len(struc) #total energy
        E = y_mean[0]
        F = y_mean[1:].reshape([len(struc), 3])
        if stress:
            S = K_trans1.dot(self.alpha_)[:,0].reshape([len(struc), 6])
        else:
            S = None

        # substract the energy/force offsets due to the base_potential
        if self.base_potential is not None:
            energy_off, force_off, stress_off = self.compute_base_potential(struc)
            E += energy_off
            F += force_off
            if stress:
                S += stress_off

        if return_std:
            if self._K_inv is None:
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                self._K_inv = L_inv.dot(L_inv.T)
            y_var = self.kernel.diag(data)
            y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)
            y_var_negative = y_var < 0
            y_var[y_var_negative] = 0.0
            y_var = np.sqrt(y_var)
            E_std = y_var[0]
            F_std = y_var[1:].reshape([len(struc), 3])
            return E, F, S, E_std, F_std
        else:
            return E, F, S

    def add_train_pts_energy(self, energy_data):
        """
        energy_data is a list of tuples (X, E)
        X: the descriptors for a given structure: N1*d
        E: total energy: scalor
        N1 is the number of atoms in the given structure
        """
        (X, E, ele) = energy_data
        self.train_x['energy'].append((X, ele))
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
        icol = 0
        for fd in force_data:
            (x, dxdr, _, f, ele) = fd
            icol += x.shape[0]
        jcol = x.shape[1]
        
        ELE = []
        indices = []
        X = np.zeros([icol, jcol])
        dXdR = np.zeros([icol, jcol, 3])

        count = 0
        for i, fd in enumerate(force_data):
            (x, dxdr, _, f, ele) = fd
            shp = x.shape[0]
            indices.append(shp)
            X[count:count+shp, :jcol] = x
            dXdR[count:count+shp, :jcol, :3] = dxdr
            ELE.extend(ele)
            self.train_y['force'].append(f)
            count += shp
        ELE = np.ravel(ELE)

        if self.kernel.ncpu == 'gpu':
            self.train_x['force'].append((X, cp.array(dXdR), None, ELE, indices))
        else:
            self.train_x['force'].append((X. dXdR, None, ELE, indices))

        #if self.kernel.ncpu == 'gpu':
        #    self.train_x['force'].append((X, dXdR, None, ele, cp.array(dXdR), None))
        #else:
        #    self.train_x['force'].append((X, dXdR, None, ele))
        #self.update_y_train()

    def log_marginal_likelihood(self, params, eval_gradient=False, clone_kernel=False):
        
        if clone_kernel:
            kernel = self.kernel.update(params[:-1])
        else:
            kernel = self.kernel
            kernel.update(params[:-1])
        if eval_gradient:
            K, K_gradient = kernel.k_total_with_grad(self.train_x)
        else:
            K = kernel.k_total(self.train_x)
        # add noise matrix
        #K[np.diag_indices_from(K)] += self.noise
        noise = np.eye(len(K))
        NE = len(self.train_x['energy'])
        noise[:NE,:NE] *= params[-1]**2
        noise[NE:,NE:] *= (self.f_coef*params[-1])**2
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
            base = np.zeros([len(K), len(K), 1]) #for energy and force noise
            base[:NE,:NE, 0] += 2*params[-1]*np.eye(NE)
            base[NE:,NE:, 0] += 2*self.f_coef**2*params[-1]*np.eye(len(K)-NE)
            K_gradient = np.concatenate((K_gradient, base), axis=2)
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

    def save(self, filename, db_filename):
        """
        Save the model
        Args:
            filename: the file to save txt information
            db_filename: the file to save structural information
        """
        dict0 = self.save_dict(db_filename)
        with open(filename, "w") as fp:
            json.dump(dict0, fp)
        self.export_ase_db(db_filename, permission="w")

        print("save the GP model to", filename, "and database to", db_filename)

    def load(self, filename, N_max=None, opt=False, device=1):
        """
        Save the model
        Args:
            filename: the file to save txt information
            db_filename: the file to save structural information
        """
        #print(filename)
        with open(filename, "r") as fp:
            dict0 = json.load(fp)
        self.load_from_dict(dict0, N_max=N_max, device=device)
        self.fit(opt=opt)
        print("load the GP model from ", filename)

    def save_dict(self, db_filename):
        """
        save the model as a dictionary in json
        """
        noise = {"energy": self.noise_e, "f_coef": self.f_coef, "bounds": self.noise_bounds}
        dict0 = {"noise": noise,
                "kernel": self.kernel.save_dict(),
                "descriptor": self.descriptor.save_dict(),
                "db_filename": db_filename,
                }
        if self.base_potential is not None:
            dict0["base_potential"] = self.base_potential.save_dict(),
        return dict0


    def load_from_dict(self, dict0, N_max=None, device=1):
        
        #keys = ['kernel', 'descriptor', 'Noise']

        if dict0["kernel"]["name"] == "RBF_mb":
            from .RBF_mb import RBF_mb
            self.kernel = RBF_mb()
        elif dict0["kernel"]["name"] == "Dot_mb":
            from .Dot_mb import Dot_mb
            self.kernel = Dot_mb()
        else:
            msg = "unknow kernel {:s}".format(dict0["kernel"]["name"])
            raise NotImplementedError(msg)
        self.kernel.load_from_dict(dict0["kernel"])

        if dict0["descriptor"]["_type"] == "SO3":
            from pyxtal_ff.descriptors.SO3 import SO3
            self.descriptor = SO3()
            self.descriptor.load_from_dict(dict0["descriptor"])
        else:
            msg = "unknow descriptors {:s}".format(dict0["descriptor"]["name"])
            raise NotImplementedError(msg)

        if "base_potential" in dict0.keys():
            if dict0["base_potential"]["name"] == "LJ":
                from .calculator import LJ
                self.base_potential = LJ()
                self.base_potential.load_from_dict(dict0["base_potential"])
            else:
                msg = "unknow base potential {:s}".format(dict0["base_potential"]["name"])
                raise NotImplementedError(msg)
        self.kernel.ncpu = device

        self.noise_e = dict0["noise"]["energy"]
        self.f_coef = dict0["noise"]["f_coef"]
        self.noise_bounds = dict0["noise"]["bounds"]
        self.noise_f = self.f_coef*self.noise_e
        # save structural file
        self.extract_db(dict0["db_filename"], N_max)

    def export_ase_db(self, db_filename, permission="w"):
        """
        export the structural information in ase db format
            - atoms:
            - energy:
            _ forces:
            - energy_in:
            - forces_in:
        """
        if permission=="w" and os.path.exists(db_filename):
            os.remove(db_filename)

        with connect(db_filename) as db:
            for _data in self.train_db:
                (struc, energy, force, energy_in, force_in, _, _) = _data
                data = {"energy": energy,
                        "force": force,
                        "energy_in": energy_in,
                        "force_in": force_in,
                       }
                db.write(struc, data=data)

    def extract_db(self, db_filename, N_max=None):
        """
        convert the structures to the descriptors from a given ase db
        """
               
        pts_to_add = {"energy": [], "force": [], "db": []}

        with connect(db_filename) as db: 
            count = 0
            for row in db.select():
                count += 1
                atoms = db.get_atoms(id=row.id)
                energy = row.data.energy
                force = row.data.force

                # substract the energy/force offsets due to the base_potential
                if self.base_potential is not None:
                    energy_off, force_off, _ = self.compute_base_potential(atoms)
                    energy -= energy_off
                    force -= force_off

                energy_in = row.data.energy_in
                force_in = row.data.force_in

                d = self.descriptor.calculate(atoms)
                ele = [Element(ele).z for ele in d['elements']]
                ele = np.array(ele)

                if energy_in:
                    pts_to_add["energy"].append((d['x'], energy/len(atoms), ele))
                for id in force_in:
                    ids = np.argwhere(d['seq'][:,1]==id).flatten() 
                    _i = d['seq'][ids, 0]
                    pts_to_add["force"].append((d['x'][_i,:], d['dxdr'][ids], None, force[id], ele[_i]))
                pts_to_add["db"].append((atoms, energy, force, energy_in, force_in))

                if count % 50 == 0:
                    print("Processed {:d} structures".format(count))
                if N_max is not None and count == N_max:
                    break

        self.set_train_pts(pts_to_add, "w")

    def add_structure(self, data, N_max=6, tol_e_var=1.2, tol_f_var=1.2):
        """
        add the training points from a given structure base on the followings:
            1, compute the (E, F, E_var, F_var) based on the current model
            2, if E_var is greater than tol_e_var, add energy data
            3, if F_var is greater than tol_f_var, add force data
        """
        tol_e_var *= self.noise_e
        tol_f_var *= self.noise_f
               
        pts_to_add = {"energy": [], "force": [], "db": []}
        (atoms, energy, force) = data

        # substract the energy/force offsets due to the base_potential
        if self.base_potential is not None:
            energy_off, force_off, _ = self.compute_base_potential(atoms)
        else:
            energy_off, force_off = 0, np.zeros((len(atoms), 3))

        energy -= energy_off
        force -= force_off
        my_data = convert_train_data([(atoms, energy, force)], self.descriptor)
        if self.alpha_ is not None:
            E, E1, E_std, F, F1, F_std = self.validate_data(my_data, return_std=True)
            E_std = E_std[0]
            F_std = F_std.reshape((len(atoms), 3))
        else:
            E = E1 = [energy]
            F = F1 = force.flatten()
            E_std = 2*tol_e_var
            F_std = 2*tol_f_var*np.ones((len(atoms), 3))
           
        if E_std > tol_e_var:
            pts_to_add["energy"] = my_data["energy"]
            N_energy = 1
            energy_in = True
        else:
            N_energy = 0
            energy_in = False

        force_in = []

        xs_added = []
        for f_id in range(len(atoms)):
            include = False
            if np.max(F_std[f_id]) > tol_f_var: 
                X = my_data["energy"][0][0][f_id]
                _ele = my_data["energy"][0][2][f_id]
                if len(xs_added) == 0:
                    include = True
                else:
                    if new_pt((X, _ele), xs_added):
                        include = True
            if include:
                force_in.append(f_id)
                xs_added.append((X, _ele))
                pts_to_add["force"].append(my_data["force"][f_id])
            if len(force_in) == N_max:
                break

        N_forces = len(force_in)
        N_pts = N_energy + N_forces
        if N_pts > 0:
            pts_to_add["db"].append((atoms, energy, force, energy_in, force_in))
        errors = (E[0]+energy_off, E1[0]+energy_off, E_std, F+force_off.flatten(), F1+force_off.flatten(), F_std) 
        return pts_to_add, N_pts, errors

    def compute_base_potential(self, atoms):
        """
        compute the energy/forces/stress from the base_potential
        """

        return self.base_potential.calculate(atoms) 


    def sparsify(self, l_tol=1e-10):
        """
        sparify the covariance matrix by removing unimportant configurations from the training database
        """
        K = self.kernel.k_total(self.train_x)
        N_e = len(self.train_x["energy"])
        N_f = len(self.train_x["force"])
        pts_e = CUR(K[:N_e,:N_e], l_tol)
        pts = CUR(K[N_e:,N_e:], l_tol)
        pts_f = []
        if N_f > 0:
            for i in range(N_f):
                if len(pts[pts==i*3])==1 and len(pts[pts==(i*3+1)])==1 and len(pts[pts==(i*3+2)])==1:
                    pts_f.append(i)
        print("{:d} energy and {:d} force points will be removed".format(len(pts_e), len(pts_f)))
        self.remove_train_pts(pts_e, pts_f)


def CUR(K, l_tol=1e-10):
    """
    This is a code to perform CUR decomposition for the covariance matrix:
    Appendix D in Jinnouchi, et al, Phys. Rev. B, 014105 (2019)

    Args:
        K: N*N covariance matrix
        N_e: number of 
    """
    L, U = np.linalg.eigh(K)
    N_low = len(L[L<l_tol])
    omega = np.zeros(len(L))
    for j in range(len(L)):
        for eta in range(len(L)):
            if L[eta] < l_tol:
                omega[j] += U[j,eta]*U[j,eta]
    ids = np.argsort(-1*omega)
    return ids[:N_low]

