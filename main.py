"""
Created on 2021-07-20 22：06
Last Update Time: 2021-07-20
@author: Xinzhuo Hu

"""
import os,sys
import pymesh
import meshio
# from matplotlib import pyplot as plt
from Parametrization import *
from TriMeshProcess import *
from solveLinearSystem import solveLSCMsystem

if __name__ == '__main__':
    
    print('This the start !')
    # mesh_path = sys.argv[1]
    mesh_path = "meshes/saddle2.obj"
    # read mesh using meshio
    # We do assume the input mesh's triangles are consistently oriented counter-clockwise.
    mesh = meshio.read(mesh_path)

    # PyMesh test
    mesh2 = pymesh.load_mesh(mesh_path)
    assembler = pymesh.Assembler(mesh2)
    L = assembler.assemble("laplacian")
    
    # Assemble LSCM Matrix for a given mesh
    K, sK = assembleLSCMMatrix(mesh)
    # plt.spy(sK, markersize=1)
    # plt.show()
    
    BoundaryEdges, BoundaryNodeInds = BoundaryEdgeNodes(mesh)
    vertex = mesh.points 
    nv = vertex.shape[0]
    # pin down 2 vertices on the boundary
    idx0 = BoundaryNodeInds[0]
    v0 = vertex[idx0,:]
    furthestDist = 0
    furthestIdx = 0
    bn_ind_num = len(BoundaryNodeInds)

    for i in range(1,bn_ind_num):

        tempidx = BoundaryNodeInds[i]
        tempv = vertex[tempidx,:]
        dist = np.linalg.norm(tempv-v0)
        if dist > furthestDist:
            furthestDist = dist
            furthestIdx = tempidx    
    
    fixedVars = np.asarray([idx0, furthestIdx, idx0+nv, furthestIdx+nv],dtype=int)
    fixedVarValues = np.asarray([0, furthestDist, 0, 0], dtype=float)
    
    # Ku = f with 2 constrained vertices
    uvmap = solveLSCMsystem(K,fixedVars,fixedVarValues)

    # Write Test VTK
    uv_zcoord = np.zeros((nv,1),dtype=float)
    uv_3d = np.hstack((uvmap,uv_zcoord))

    modelfolder = 'conformalmodels'
    conmodel_name = 'saddle2.vtk'
    write_path = os.path.join(modelfolder,conmodel_name)
    points = uv_3d
    cells = {'triangle':mesh.cells[0].data}
    meshio.write_points_cells(write_path,points,cells)

    print('debug here')



