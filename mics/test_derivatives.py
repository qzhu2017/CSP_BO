import numpy as np

def dfun1(x1, x2):
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    x1_norm3 = x1_norm**3        
    x2_norm3 = x2_norm**3       
    tmp0 = np.ones(x2.shape)
    x1x1 = np.einsum("ijl,ik->ijkl", x1[:,None,:]*tmp0[None,:,:], x1)
    tmp1 = x1x1/x2_norm[None,:,None,None]

    x1x2 = np.einsum("ik,jl->ijkl", x1, x2/x2_norm3[:,None])
    tmp2 = np.einsum("ijkl,ij->ijkl", x1x2, x1@x2.T)
    
    return (tmp1-tmp2)/(x1_norm3[:,None,None,None])

def fun1(x1, x2):
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    x1_norm3 = x1_norm**3        

    d = (x1@x2.T)[:,:,None]*x1[:,None,:]/x2_norm[None,:,None]
    return d/(x1_norm3[:,None,None])

def fun2(x1, x2):
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    return x2[None,:,:]/x1_norm[:,None,None]/x2_norm[None,:,None] #m,n,d2

def fun3(x1, x2):
    x2_norm = np.linalg.norm(x2, axis=1)
    return x2/x2_norm[None,:] #m,n,d2

def dfun3(x1, x2):
    x2_norm = np.linalg.norm(x2, axis=1)
    x2_norm2 = x2_norm**2
    x2_norm3 = x2_norm**3        
    (n, d) = x2.shape
    x2x2 = x2[:,:,None] * x2[:,None,:] #n*d1*d2
    #tmp2 = 1-x2x2/x2_norm2[:,None,None] # n*d1*d2
    tmp2 = np.eye(d)[None,:, :]-x2x2/x2_norm2[:,None,None] # n*d1*d2
    out = tmp2/x2_norm[:,None,None]
    return out
    #x2x2 = np.outer(x2, x2) #n*d1*d2
    #return 1/(x2_norm[:,None,None]) - x2x2/(x2_norm3[:,None,None])

def fun4(x1, x2):
    x2_norm = np.linalg.norm(x2, axis=1)
    return 1/x2_norm

def dfun4(x1, x2):
    x2_norm = np.linalg.norm(x2, axis=1)
    x2_norm3 = x2_norm**3
    return -x2/x2_norm3[:,None]

def dfun2(x1, x2):
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    x2_norm3 = x2_norm**3
    x2_norm2 = x2_norm**2
    tmp0 = np.ones(x1.shape)
    
    #x2x2 = np.einsum("ij,ik->ijk", x2, x2)
    x2x2 = x2[:,:,None] * x2[:,None,:] #n*d1*d2
    tmp2 = 1-x2x2/x2_norm2[:,None,None] # n*d1*d2
    out = tmp2[None,:,:,:]/x1_norm[:,None,None,None]/x2_norm[None,:,None,None]
    #out = tmp2[None,:,:,:]/x1_norm[:,None,None,None]/x2_norm[None,:,None,None]
    #tmp2 = 1-x2*x2/x2_norm2[:,None] # n*d2
    #out = tmp2[None,:,:]/x1_norm[:,None,None]/x2_norm[None,:,None]
    return out

m, n, k, sigma2, l2 = 1, 1, 3, 1.1, 0.9
x1 = np.random.random([m, k])
x2 = np.random.random([n, k])
eps = 1e-6
for i in range(x2.shape[0]):
    for j in range(x2.shape[1]):
        tmp = x2.copy()
        tmp[i, j] += eps
        d1 = fun1(x1, x2)
        d2 = fun1(x1, tmp)
        print((d2-d1)/eps)
print('dfun1')
print(dfun1(x1, x2))

print('4444444444444444444')
print(x2)
for i in range(x2.shape[0]):
    for j in range(x2.shape[1]):
        tmp = x2.copy()
        tmp[i, j] += eps
        d1 = fun3(x1, x2)
        d2 = fun3(x1, tmp)
        print(((d2-d1)/eps).flatten())

print('dfun3')
print(dfun3(x1, x2))
