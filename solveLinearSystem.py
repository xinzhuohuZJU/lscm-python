"""
Created on 2021-07-29 14：45
Last Update Time: 2021-07-29
@author: Xinzhuo Hu

"""

import numpy as np
import pymesh
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

# Ku = f with 2 constrained vertices
def solveLSCMsystem(K, fixedVars, fixedVarValues):

    # number of vertices
    nv = int(K.shape[0]/2)
    n = K.shape[0]
    # full matrix Kf
    Kind_lower = np.tril_indices(n,-1)
    Kf = np.copy(K)
    Kf[Kind_lower] = Kf.T[Kind_lower]
    # lower triangle matrix Kl
    Kl = np.zeros((n,n),dtype=float)
    Kl[Kind_lower] = K.T[Kind_lower]

    # build right hand f
    # original f
    fori = np.zeros([2*nv, ],dtype=float)
    c = np.zeros([2*nv, ],dtype=float)
    c[fixedVars,] = fixedVarValues

    # true f
    # K is a upper triangle symmetric sparse matrix
    f = np.dot(Kf,c)
    ft = np.copy(f)
    ft = fori - f
    ft[fixedVars] = fixedVarValues
    # test
    ft_ind = np.where(ft!=0)

    # Modify LSCM Matrix
    num_fixedvars = fixedVars.shape[0]
    for i in range(num_fixedvars):
        idx = fixedVars[i]
        K[idx,:] = 0
        K[:,idx] = 0
        K[idx,idx] = 1
        # try Kf
        Kf[idx,:] = 0
        Kf[:,idx] = 0
        Kf[idx,idx] = 1
        # try Kl
        Kl[idx,:] = 0
        Kl[:,idx] = 0
        Kl[idx,idx] = 1

    # convert to sparse matrix
    # sK = sparse.csr_matrix(K)
    # x = spsolve_triangular(sK, ft, lower=False)
    # x_ind = np.where(x!=0)
    # print(np.allclose(sK.dot(x), ft))

    # PyMesh solvers
    sKf = sparse.csc_matrix(Kf)
    solver = pymesh.SparseSolver.create("LDLT")
    solver.compute(sKf)
    uv_vec = solver.solve(ft)
    print('Constrained Values Test: ',uv_vec[fixedVars, 0])
    print(np.allclose(sKf.dot(uv_vec), ft))

    # assemble uv
    u = uv_vec[0:nv]
    v = uv_vec[nv:2*nv]
    uv = np.hstack((u,v))

    return uv
    
