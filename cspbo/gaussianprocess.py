import math
import numpy as np
import torch
from .lbfgsb import LBFGSScipy 
#torch.set_default_tensor_type(torch.DoubleTensor)

class GaussianProcess():
    """ Gaussian Process Regressor. """
    def __init__(self, noise=1e-4, para=[100, 10], device='cpu'):
        self.noise = noise
        self.para = para
        self.x = None
        self.models = {'model': RBFKernel(para=self.para, device=device)}
        #print("Initial parameters:", str(self.models['model']))

    def fit(self, TrainData, para=[10, 10], bounds=[[5e-1, 1e+4], [1e-2, 1e+1]]):
        # Set up optimizer to train the GPR
        if para is not None:
            self.models['model'].update_parameters(para)
        print("Strart Training: ", str(self.models['model']))
        params = self.models['model'].parameters()
        self.optimizer = LBFGSScipy(params, bounds=bounds)
        self.unpack_data(TrainData)
        def closure():
            loss = self.log_marginal_likelihood(self.x, self.y)
            print("Loss: {:10.6f} {:s}".format(loss, str(self.models['model'])))
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        self.optimizer.step(closure)
        loss = self.log_marginal_likelihood(self.x, self.y)
        print("Loss: {:10.6f} {:s}".format(loss, str(self.models['model'])))


    def predict(self, _X, return_std=False, return_cov=False):
        #print("Predict: ", len(_X))
        X = []
        for x in _X:
            X.append(torch.from_numpy(x))
        Ks = self.get_covariance_matrix(X, self.x)
        pred = Ks @ self.alpha
        y_mean = np.asarray(pred[:, 0].detach().numpy()) 

        if return_cov:
            v = torch.cholesky_solve(Ks.T, self.L)
            y_cov = self.get_covariance_matrix_self(X) - torch.mm(Ks, v)
            return y_mean, y_cov.detach().numpy()
        elif return_std:
            raise NotImplementedError("no std yet")
            #return y_mean, np.sqrt(y_var)
        else:
            return y_mean
    
    def unpack_data(self, data):
        # decompose X
        (_X, _Y) = data
        X = []
        Y = torch.zeros([len(_Y), 1])
        for i, x in enumerate(_X):
            X.append(torch.from_numpy(x))
            Y[i,0] += _Y[i]

        self.x = X
        self.y = Y

    def log_marginal_likelihood(self, X, Y):
        K = self.get_covariance_matrix_self(X)
        try:
            self.L = torch.cholesky(K + self.noise * torch.eye(len(K)), upper=False)
        except RuntimeError:
            print(K)
            raise ValueError("kernel parameters too small:", str(self.models['model']))
        self.alpha = torch.cholesky_solve(Y, self.L)

        # log marginal likelihood
        MLL = -0.5 * torch.einsum("ik,ik", Y, self.alpha)
        MLL -= torch.log(torch.diag(self.L)).sum()
        MLL -= self.L.shape[0] / 2 * math.log(2 * math.pi)

        #mu = K.T @ self.alpha
        #error = torch.abs(mu - Y)
        #print("E_mae:", error.mean())
        return -MLL


    def get_covariance_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        m1, m2 = len(X1), len(X2)
        models = self.models['model']

        out = torch.zeros((m1, m2), dtype=torch.float64)
        # This is not necessary true for multi=species needs to fix this in the near future.
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                # Covariance between E_i and E_j
                out[i, j] = torch.sum(models.covariance(x1, x2))

        return out

    def get_covariance_matrix_self(self, X1):
        m1 = len(X1)
        models = self.models['model']
        out = torch.zeros((m1, m1), dtype=torch.float64)
        # This is not necessary true for multi=species needs to fix this in the near future.
        for i, x1 in enumerate(X1):
            for j in range(i, m1):
                # Covariance between E_i and E_j
                #print(i, j)
                out[i, j] = torch.sum(models.covariance(X1[i], X1[j]))
                if j > i:
                    out[j, i] = out[i, j]
                #print(x1)
                #import sys
                #sys.exit()

        return out



class RBFKernel():
    def __init__(self, para=None, device='cpu'):
        """ d is no of descriptors """
        super().__init__()
        self.device = device
        if para is None:
            para = [1., 1.]
        self.update_parameters(para)

    def covariance(self, x1, x2):
        # x1: m x d, x2: n x d, E: m x n
        #_x1 = x1.clone()
        #_x2 = x2.clone()
        #_x1 /= self.sigmaL
        #_x2 /= self.sigmaL
        #D = torch.sum(_x1*_x1, axis=1, keepdims=True) + torch.sum(_x2*_x2, axis=1) - 2*_x1@_x2.T
        #E = torch.exp(-0.5 * D)
        D = torch.sum(x1*x1, axis=1, keepdims=True) + torch.sum(x2*x2, axis=1) - 2*x1@x2.T
        E = torch.exp(-0.5 * D / self.sigmaL ** 2)
 
        
        return self.sigmaF ** 2 * E

    def __str__(self):
        return "RBF(length={:.3f}, coef={:.3f})".format(self.sigmaL.item(), self.sigmaF.item())
 
    def parameters(self):
        return [self.sigmaL, self.sigmaF]
 
    def update_parameters(self, para):
        # I wonder if I can combine sigmaF and sigmaL in one Tensor.
        self.sigmaF = torch.DoubleTensor([para[0]])
        self.sigmaF.requires_grad_()

        # Init the shapes of the RBF kernel as normal distribution
        self.sigmaL= torch.DoubleTensor([para[1]])
        self.sigmaL.requires_grad_()


