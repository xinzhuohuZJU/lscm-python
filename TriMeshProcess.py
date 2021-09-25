"""
Created on 2021-07-22 22：09
Last Update Time: 2021-07-22
@author: Xinzhuo Hu

"""

import numpy as np
from scipy import sparse

def BoundaryEdgeNodes(mesh):

    # read faces and meshes information of the mesh
    tris = mesh.cells[0].data
    vertex = mesh.points   
    nv = vertex.shape[0]
    trinum = tris.shape[0]

    # EdgeMatrix
    VAdj = np.zeros((nv,nv),dtype=int)
    # Traverse all triangles
    for tri in range(trinum):

        node_global_index_1 = tris[tri,0]
        node_global_index_2 = tris[tri,1]
        node_global_index_3 = tris[tri,2]

        # order
        VAdj[node_global_index_1,node_global_index_2] += 1
        VAdj[node_global_index_1,node_global_index_3] += 1
        VAdj[node_global_index_2,node_global_index_3] += 1

        # reverse order
        VAdj[node_global_index_2, node_global_index_1] += 1
        VAdj[node_global_index_3, node_global_index_1] += 1
        VAdj[node_global_index_3, node_global_index_2] += 1

    boundaryindicies = np.where(VAdj == 1)
    xindices = boundaryindicies[0]
    yindices = boundaryindicies[1]
    indnum = np.size(xindices)

    BoundaryEdgeDuplicate = np.zeros((indnum,2),dtype=int)
    for i in range(indnum):
        if xindices[i]<yindices[i]:
            BoundaryEdgeDuplicate[i,0] = xindices[i]
            BoundaryEdgeDuplicate[i,1] = yindices[i]
        else:
            BoundaryEdgeDuplicate[i,0] = yindices[i]
            BoundaryEdgeDuplicate[i,1] = xindices[i]
    
    BoundaryEdge = np.unique(BoundaryEdgeDuplicate, axis=0)
    BoundaryNodeInds = np.unique(np.reshape(BoundaryEdge,(indnum)))   

    return BoundaryEdge, BoundaryNodeInds