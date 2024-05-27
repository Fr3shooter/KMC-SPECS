from mayavi import mlab
import numpy as np
from numpy.linalg import norm

# Checking for boundaries (Oxygen connections purpose)
def check_PBC(coord,lattice_param, oxygen=False):
    updated_coords = coord.copy()
    for i in range(len(coord[:,0])):
        if updated_coords[i,0] == 0:
            updated_coords[i,0] += lattice_param[0]

        if updated_coords[i,1] == 0:
            updated_coords[i,1] += lattice_param[1]

        if updated_coords[i,2] == 0:
            updated_coords[i,2] += lattice_param[2]

    A = np.argwhere(coord == 0)[:,0]
    B = np.append(updated_coords,coord[A,:])

    if oxygen:
        flag = len(A)
        B = B.reshape(int(len(B)/3),3)
        return np.unique(B, axis=0), flag

    B = B.reshape(int(len(B)/3),3)

    return np.unique(B, axis=0)


# Adding N O unit cells along axis
def add_patternO(H2O_pos, axis, lattice_param, N=1):
    N-=1
    q = H2O_pos

    for i in range(1,N+1):
        H2O_pos = np.append(H2O_pos,q,axis=0)
        H2O_pos[i*len(q):,axis] += i*lattice_param[axis]

    return H2O_pos

# Connecting Oxygen (USE ONLY WHEN FULL rO IS BUILT)
def connectionsO(H20_pos, N_H20, lattice_param):
    connectionsO = np.array([[0,0]])
    added_atoms = check_PBC(H20_pos,lattice_param,True)[1]
    for i in range(0,len(H20_pos[:,0])):

        B = norm([H20_pos[i,0]-H20_pos[i:i+added_atoms+N_H20,0],
                H20_pos[i,1]-H20_pos[i:i+added_atoms+N_H20,1],
                H20_pos[i,2]-H20_pos[i:i+added_atoms+N_H20,2]], 
                axis=0)

        A = np.where(B<3)[0]+i

        C = np.array([A[1:],[i]*len(A[1:])]).transpose()

        connectionsO = np.append(connectionsO,C,axis=0)

    connectionsO = np.unique(connectionsO, axis=0)

    return connectionsO

# Drawing unit cell 
def unit_cell(lattice_param):
    mlab.plot3d([lattice_param, 2*lattice_param], [lattice_param, lattice_param], [lattice_param, lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [lattice_param, 2*lattice_param], [lattice_param, lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [lattice_param, lattice_param], [lattice_param, 2*lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param],  color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param],  color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [lattice_param, lattice_param], [lattice_param, 2*lattice_param],  color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param],   color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param], [lattice_param, lattice_param],   color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, 2*lattice_param], [lattice_param, lattice_param], [2*lattice_param, 2*lattice_param],   color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, 2*lattice_param],[2*lattice_param, 2*lattice_param], [lattice_param, lattice_param],    color=(1,1,1),line_width=10)
    
def render(R, pattern_size, rO, N_O, lattice_param, mapping, cages):
    # Nitrogen plots
    figure = mlab.figure()
    for i in range(0,len(R[:,0]),pattern_size):

        mlab.points3d(R[i:i+pattern_size,0],R[i:i+pattern_size,1],R[i:i+pattern_size,2], scale_factor=1, color=(0,0,1))

    print(np.max(rO[:,0]), np.max(rO[:,1]), np.max(rO[:,2]))
    print(np.max(R[:,0]), np.max(R[:,1]), np.max(R[:,2]))
    ptsO = mlab.points3d(rO[:,0],rO[:,1],rO[:,2], scale_factor=.50, color=(1,0,0))
    lines = mlab.pipeline.stripper(ptsO)
    mlab.pipeline.surface(lines, color=(1,0,0), line_width=5, opacity=.4)
    ptsO.mlab_source.dataset.lines = connectionsO(rO, N_O, lattice_param)

    figure.scene.disable_render = True
    res = {}

    for i, x in enumerate(R):
        res[f'{i}'] = (np.append(mapping[i,:],cages[i]))
        # print(res[f'{i}'])
        mlab.text3d(x[0]+0.1, x[1]+0.1, x[2]+0.5, f'{i,cages[i]}:{R[i,:]}', scale=(.25, .25, .25))
    figure.scene.disable_render = False

    # unit_cell(lattice_param)
    mlab.show()