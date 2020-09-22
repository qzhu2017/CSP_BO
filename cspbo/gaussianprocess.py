import math
import numpy as np
import torch
from .lbfgsb import LBFGSScipy 
torch.set_default_tensor_type(torch.DoubleTensor)

class RBF():
    def __init__(self, para=[5, 5.], bounds=[[1e-1, 1e+2], [5e-1, 5e+2]]):
        """ d is no of descriptors """
        super().__init__()
        self.name = 'RBF_mb'
        self.bounds = bounds
        self.update_parameters(para)

    def covariance(self, x1, x2):
        # need to normalize
        D = torch.sum(x1*x1, axis=1, keepdims=True) + torch.sum(x2*x2, axis=1) - 2*x1@x2.T
        d1 = torch.sum(x1*x1, dim=1, keepdims=True)
        d2 = torch.sum(x2*x2, dim=1, keepdims=True)
        D = D/(torch.matmul(d1, d2.T)+1e-2)
        E = torch.exp(-0.5 * D / self.sigmaL ** 2)
        
        
        return torch.sum(self.sigmaF ** 2 * E)/(len(x1)*len(x2))

    def __str__(self):
        return "{:.3f}**2 *RBF(length={:.3f})".format(self.sigmaF.item(), self.sigmaL.item())
 
    def parameters(self):
        return [self.sigmaF, self.sigmaL]
 
    def update_parameters(self, para):
        self.sigmaF = torch.DoubleTensor([para[0]])
        self.sigmaF.requires_grad_()
        self.sigmaL= torch.DoubleTensor([para[1]])
        self.sigmaL.requires_grad_()

class Dot():
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 2e+1], [1e-2, 1e+1]]):
        """ d is no of descriptors """
        super().__init__()
        self.name = 'Dot_mb'
        self.bounds = bounds
        self.update_parameters(para)

    def covariance(self, x1, x2):
        D1 = torch.sum(x1@x2.T)
        D2 = torch.sum(x1@x1.T) + 1e-3
        D3 = torch.sum(x2@x2.T) + 1e-3
        return self.delta**2*(self.sigma0**2 + D1/torch.sqrt(D2*D3))

    def __str__(self):
        return "{:.3f}**2 *Dot(sigma_0={:.3f})".format(self.delta.item(), self.sigma0.item())
 
    def parameters(self):
        return [self.delta, self.sigma0]
 
    def update_parameters(self, para):
        self.delta = torch.DoubleTensor([para[0]])
        self.delta.requires_grad_()
        self.sigma0 = torch.DoubleTensor([para[1]])
        self.sigma0.requires_grad_()

class Combo():
    def __init__(self, coef=0.01, para=[1.0, 1.0, 1.0], bounds=[[1e-2, 1e+3], [3e-1, 1e+2], [1e-2, 1e+2]]):
        super().__init__()
        self.name = 'Combo'
        self.bounds = bounds
        self.coef = coef
        self.update_parameters(para)

    def covariance(self, X1, X2):
        (x1, d1), (x2, d2) = X1, X2
        D1 = torch.sum(x1@x2.T)
        D2 = torch.sum(x1@x1.T) + 1e-3
        D3 = torch.sum(x2@x2.T) + 1e-3
        K_dot = self.coef**2 *self.delta**2*(self.sigma0**2 + D1/torch.sqrt(D2*D3))
        D = torch.sum((d1-d2)**2) 
        E = torch.exp(-0.5 * D / self.sigmaL ** 2)
        K_rbf = self.delta ** 2 * E
        return K_dot/(len(x1)*len(x2)) + K_rbf

    def __str__(self):
        strs =  "{:.3f}**2 *RBF_2b(length={:.3f})".format(self.delta.item(), self.sigmaL.item())
        strs +=  "+ {:.3f}**2 *Dot(sigma_0={:.3f})".format(self.coef*self.delta.item(), self.sigma0.item())
        return strs

    def parameters(self):
        return [self.delta, self.sigmaL, self.sigma0]

    def update_parameters(self, para):
        self.delta = torch.DoubleTensor([para[0]])
        self.delta.requires_grad_()
        self.sigmaL = torch.DoubleTensor([para[1]])
        self.sigmaL.requires_grad_()
        self.sigma0 = torch.DoubleTensor([para[2]])
        self.sigma0.requires_grad_()

class RBF_2b():
    def __init__(self, coef=0.01, para=[1.0, 1.0], bounds=[[1e-2, 1e+3], [3e-1, 1e+2]]):
        super().__init__()
        self.name = 'RBF_2b'
        self.bounds = bounds
        self.coef = coef
        self.update_parameters(para)

    def covariance(self, d1, d2):
        D = torch.sum((d1-d2)**2) 
        E = torch.exp(-0.5 * D / self.sigmaL ** 2)
        K_rbf = self.delta ** 2 * E
        return K_rbf

    def __str__(self):
        strs =  "{:.3f}**2 *RBF_2b(length={:.3f})".format(self.delta.item(), self.sigmaL.item())
        return strs

    def parameters(self):
        return [self.delta, self.sigmaL]

    def update_parameters(self, para):
        self.delta = torch.DoubleTensor([para[0]])
        self.delta.requires_grad_()
        self.sigmaL = torch.DoubleTensor([para[1]])
        self.sigmaL.requires_grad_()


class GaussianProcess():
    """ Gaussian Process Regressor. """
    def __init__(self, noise=1e-4, kernel=None):
        self.noise = noise
        self.x = None
        self.models = {'model': kernel}

    def fit(self, TrainData):
        # Set up optimizer to train the GPR
        print("Strart Training: ", str(self.models['model']))
        params = self.models['model'].parameters()
        self.optimizer = LBFGSScipy(params, bounds=self.models['model'].bounds)
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
        for i, x in enumerate(_X[0]):
            if x is None or self.models['model'].name in ["RBF_2b"]:
                X.append(torch.from_numpy(_X[1][i]))
            elif _X[1][i] is None or self.models['model'].name in ["RBF_mb", "Dot_mb"]:
                X.append(torch.from_numpy(x))
            else:
                X.append((torch.from_numpy(x), torch.from_numpy(_X[1][i])))

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
        for i, x in enumerate(_X[0]):
            if x is None or self.models['model'].name in ["RBF_2b"]:
                X.append(torch.from_numpy(_X[1][i]))
            elif _X[1][i] is None or self.models['model'].name in ["RBF_mb", "Dot_mb"]:
                X.append(torch.from_numpy(x))
            else:
                X.append((torch.from_numpy(x), torch.from_numpy(_X[1][i])))
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


    def get_covariance_matrix(self, X1, X2):
        m1, m2 = len(X1), len(X2)
        models = self.models['model']

        out = torch.zeros((m1, m2), dtype=torch.float64)
        # This is not necessary true for multi=species needs to fix this in the near future.
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                # Covariance between E_i and E_j
                out[i, j] = models.covariance(x1, x2)

        return out

    def get_covariance_matrix_self(self, X1):
        m1 = len(X1)
        models = self.models['model']
        out = torch.zeros((m1, m1), dtype=torch.float64)
        for i, x1 in enumerate(X1):
            for j in range(i, m1):
                out[i, j] = torch.sum(models.covariance(X1[i], X1[j]))
                if j > i:
                    out[j, i] = out[i, j]
        return out




