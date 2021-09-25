"""
Created on 2021-07-20 22：06
Last Update Time: 2021-07-20
@author: Xinzhuo Hu

"""
import os,sys
import numpy as np
from scipy import sparse

def TrianglesEmbed(faces, vertex):

    if faces.shape[1] != 3:
        raise RuntimeError('Only Support Triangular Mesh Element For Now.')

    trinum = faces.shape[0]
    gradBarycentricList = [None for i in range(trinum)]
    volumeList = [None for i in range(trinum)]

    # compute gradBarycentricList and volumeList for each triangle
    for tri in range(trinum):

        p0 = vertex[faces[tri,0], :]
        p1 = vertex[faces[tri,1], :]
        p2 = vertex[faces[tri,2], :]

        p0 = np.reshape(p0,(1,3))
        p1 = np.reshape(p1,(1,3))
        p2 = np.reshape(p2,(1,3))

        e0 = p2 - p1
        e1 = p0 - p2
        e2 = p1 - p0

        m_normal = np.cross(e1,e2)
        doubleA = np.linalg.norm(m_normal)
        m_normal /= doubleA
        m_volume = doubleA/2.0

        m_gradBarycentric = np.zeros((3,3),dtype=float)
        m_gradBarycentric[:,0] = np.cross(m_normal,e0)/doubleA
        m_gradBarycentric[:,1] = np.cross(m_normal,e1)/doubleA
        m_gradBarycentric[:,2] = np.cross(m_normal,e2)/doubleA

        gradBarycentricList[tri] = m_gradBarycentric
        volumeList[tri]=m_volume
  
    return gradBarycentricList, volumeList



def assembleLSCMMatrix(mesh):

    print('assemble LSCM Matrix Start')
    # read faces and meshes information of the mesh
    tris = mesh.cells[0].data
    vertex = mesh.points   
    nv = vertex.shape[0]
    trinum = tris.shape[0]
    # K is a UPPER_TRIANGLE Symmetry Matrix
    K = np.zeros((2*nv,2*nv),dtype=float)

    gradBarycentricList, volumeList = TrianglesEmbed(tris,vertex)
    for tri in range(trinum):

        gradLambda = gradBarycentricList[tri]
        volume = volumeList[tri]
        # to do Symmetric Laplacian blocks
        for ni_local in range(3):
            for nj_local in range(3):
                # ni_global = tri*3+ni_local
                # nj_global = tri*3+nj_local
                ni_global = tris[tri,ni_local]
                nj_global = tris[tri,nj_local]
                if ni_global > nj_global:
                    continue
                # Symmetric Laplacian blocks
                val = np.dot(gradLambda[:,ni_local],gradLambda[:,nj_local])*volume
                K[ni_global, nj_global] += val #(u,u) block
                K[nv+ni_global, nv+nj_global] += val #(v,v) block

                # Skew symmetric A block (u,v)
                if ni_local == nj_local:
                    continue
                s = 1.0 if (ni_local == (nj_local+1)%3) else -1.0
                K[ni_global, nv+nj_global] += 0.5*s
                K[nj_global, nv+ni_global] += -0.5*s

    # convert to sparse matrix
    sK = sparse.csc_matrix(K)
    return K, sK
