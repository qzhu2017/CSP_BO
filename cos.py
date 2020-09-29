import numpy as np
import torch

a = torch.rand([3, 5]).requires_grad_()
b = torch.rand([2, 5]).requires_grad_()

def distance_torch(x1, x2):
    return x1@x2.T/(torch.ger(torch.norm(x1, dim=1), torch.norm(x2, dim=1)))

total = (1-(distance_torch(a, b)**2)).sum()
print(torch.autograd.grad(total, a))

def distance_prime(x1, x2):
    """ Derivative w.r.t. x1. Quotient rule is used."""
    term1 = torch.einsum("ij,k->kij", x2, torch.norm(x1, dim=1))
    term2 = (x1@x2.T)[:,:,None] * (x1 / torch.norm(x1, dim=1)[:,None])[:,None,:]
    term3 = (torch.norm(x1, dim=1) ** 2)[:, None, None] * (torch.norm(x2, dim=1))[None, :, None]
    return (term1 - term2) / term3

dist = -2*distance_torch(a, b)
prime = distance_prime(a, b)

print(torch.einsum("ij, ijk->ik", dist, prime))