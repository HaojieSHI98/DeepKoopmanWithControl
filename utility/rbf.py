import numpy as np
from copy import copy

def rbf(X,C,type,eps=1,k=1):
    ## X: nXN 
    ## C:rbf centers nx1 or nxK
    ## eps: kernel width for Gaussian type rbfs
    ## polyharmonic coefficient for polyharmonic rbfs

    Cbig = copy(C)
    n,N = X.shape
    K = C.shape[-1]
    Y = np.zeros((K,N))
    for i in range(K):
        C = Cbig[:,i].reshape(n,1).repeat(N,axis=-1)
        r_squared = np.sum((X-C)**2,axis=0)
        if type == "thinplate":
            y = r_squared*np.log(np.sqrt(r_squared))
            y[np.isnan(y)] = 0
            y[np.isinf(y)] = 0
        elif type == "gauss":
            y = np.exp(-eps**2*r_squared)
        elif type == "invquad":
            y = 1.0/(1+eps**2*r_squared)
        elif type == "invmultquad":
            y = 1.0/np.sqrt((1.0+eps**2*r_squared))
        elif type == "polyharmonic":
            y = r_squared**(k/2.0)*np.log(np.sqrt(r_squared))
            y[np.isnan(y)] = 0
            y[np.isinf(y)] = 0    
        else:
            raise NotImplementedError
        Y[i,:] = y 
    return Y   

        


