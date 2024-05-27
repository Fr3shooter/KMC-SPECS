import numpy as np
from numpy.linalg import norm
import os, gc
from parameters import *

#_____________________________________________________________________________________
def gravity_center(r, m1=1, m2=1):
    """Computes the nitrogen gravity centers

    Args:
        r (numpy ndarray): array containing the nitrogen positions
        m1 (int, optional): mass to ponderate if CO. Defaults to 1.
        m2 (int, optional): mass to ponderate if CO. Defaults to 1.

    Returns:
        numpy ndarray: array containing the nitrogen gravity centers
    """

    # Initialize the result array
    centers = np.zeros((int(len(r)/2), 3))

    # Compute gravity centers along each axis
    for i in range(len(r[0])):

        # coordinates of particles with mass m1
        r_plus = r[1::2, i]  

        # coordinates of particles with mass m2
        r_minus = r[0::2, i] 

        # Compute gravity centers
        centers[:, i] = (r_plus * m1 + r_minus * m2) / (m1 + m2)

    return centers


#_____________________________________________________________________________________
def gravity_center(r, m1=1, m2=1):
    """Computes the nitrogen gravity centers

    Args:
        r (numpy ndarray): array containing the nitrogen positions
        m1 (int, optional): mass to ponderate if CO. Defaults to 1.
        m2 (int, optional): mass to ponderate if CO. Defaults to 1.

    Returns:
        numpy ndarray: array containing the nitrogen gravity centers
    """

    # Initialize the result array
    centers = np.zeros((int(len(r)/2), 3))

    # Compute gravity centers along each axis
    for i in range(len(r[0])):

        # coordinates of particles with mass m1
        r_plus = r[1::2, i]  

        # coordinates of particles with mass m2
        r_minus = r[0::2, i] 

        # Compute gravity centers
        centers[:, i] = (r_plus * m1 + r_minus * m2) / (m1 + m2)

    return centers


#_____________________________________________________________________________________
def replace_float_with_int(r):
    """Replacing cartesian coordinates by a map of integer indices

    Args:
        r (numpy ndarray): nitrogen gravity centers positions

    Returns:
        numpy ndarray: integer map for the nitrogen atoms
    """

    # Sort the unique coordinates
    print(r)
    r = np.array(r, dtype=int)
    sorted_list = sorted(set(r))
    print(r)
    print(50*'_')

    # Create a mapping of floating-point values to integers
    float_to_int_mapping = {value: index for index, value in enumerate(sorted_list)}

    # Replace float coordinates with corresponding integers
    integer_map = [float_to_int_mapping[item] for item in r]

    return integer_map


#_____________________________________________________________________________________
def add_pattern(centers, int_map, cages, axis, lattice_param, structure, N=1):
    """Function adding N unit cells along a given axis
    !!! WARNING: when used multiple times (eg. N=2 along x), the next use
    (eg. N=2 along y) will add x*y cells !!!

    Args:
        centers (numpy ndarray): array containing the coordinates of gravity centers
        int_map (numpy ndarray): array containing the new cage classification
        cages (list): list containing cages type for unit cell
        axis (int): axis along which to add the pattern, 0:x, 1:y, 2:z
        lattice_param (float): lattice parameter
        N (int, optional): amount of added pattern. Defaults to 1.

    Returns:
        numpy ndarray: updated array of gravity centers
        numpy ndarray: updated integer map
        list: updated cage types
    """

    # Decrementing for N to be user friendly
    N-=1

    q = centers
    p = int_map.reshape(-1, 3)

    new_centers = np.empty((len(centers) * (N + 1), 3), dtype=centers.dtype)
    new_int_map = np.empty((len(p) * (N + 1), 3), dtype=int_map.dtype)
    lattice_param = np.array(lattice_param)
    # Updating cages list
    new_cages = cages * (N + 1)
    size = np.max(int_map, axis=0)+1
    # Updating centers and integer map
    for i in range(N + 1):

        new_centers[i * len(q):(i + 1) * len(q), :] = q + lattice_param[axis] * i * np.eye(3)[axis]
        new_int_map[i * len(p):(i + 1) * len(p), :] = p + size[axis] * i * np.eye(3)[axis]

    return new_centers, new_int_map, new_cages


#_____________________________________________________________________________________
def is_interacting(centers, int_map, pattern_size, small, large, structure):
    """Checks the neighbors of a given cage

    Args:
        centers (numpy ndarray): array containing nitrogen gravity centers
        int_map (numpy ndarray): array containing nitrogen gravity centers integer mapping
        pattern_size (int): amount of cages in the unit cell
        small (int): amount of small cages in the unit cell
        large (int): amount of large cages in the unit cell

    Returns:
        numpy ndarray: neighbors_coords: contains neighbors relative to a site (real units)
        numpy ndarray: relative_map: contains neighbors relative to a site (integer coords)
        numpy ndarray: molecules in the centered unit cell
    """

    neigbors_coords = []
    new_int_map = []
    if structure =='sI':
        N = large * [15] + small * [13]
    elif structure == 'sII':
        N = large * [17] + small * [13]
    elif structure == 'sH':
        N = 2 * [21] + 6 * [13] + 4 * [13]

    # Loop through the centered cell
    centers = np.round(np.array(centers), 2)

    for i in range(13 * pattern_size, 14 * pattern_size):

        # Compute distances between particles
        B = norm([centers[i, 0] - centers[:, 0],
                  centers[i, 1] - centers[:, 1],
                  centers[i, 2] - centers[:, 2]],
                  axis=0)
        # Get indices of nearest neighbors
        # Uhm, I guess, I don't remember how it works but it works
        idx = np.argpartition(B, N[i - 13 * pattern_size])

        # Collect nearest neighbors and their mappings
        neigbors_coords.append(centers[idx[:N[i - 13 * pattern_size]], :])
        new_int_map.append(int_map[idx[:N[i - 13 * pattern_size]], :] - int_map[i, :])

    neigbors_coords = np.array(neigbors_coords, dtype='object')
    relative_map = np.array(new_int_map, dtype='object')

    return neigbors_coords, relative_map, centers[13 * pattern_size:14 * pattern_size, :]


#_____________________________________________________________________________________
def exclude_self_from_neighbors(int_map, relative_positions, structure):
    """Function excluding self cage from its own neighbors

    Args:
        int_map (numpy ndarray): array containing nitrogen gravity centers integer mapping
        relative_positions (numpy ndarray): array containing integer relative positions to cages
        structure (str): 'sI' or 'sII'

    Returns:
        numpy ndarray: updated relative positions
    """
    

    size = np.max(int_map, axis=0)+1

    # Loop through relative positions
    for i in range(len(relative_positions)):
        relative_positions[i] = relative_positions[i].tolist()
        # Loop through relative positions in reverse order (to avoid out bounds)
        flag = True
        for relative in reversed(range(len(relative_positions[i]))):
            # Remove self-mapping positions

            if np.array_equal((relative_positions[i][relative] + int_map[i]) % size, int_map[i]) and flag:
                relative_positions[i].remove(relative_positions[i][relative])
                flag = False

    gc.collect()
    return relative_positions


#_____________________________________________________________________________________
def build_system(centers, system_size, lattice_param, structure):
    """Function building the system, ie. cages, sites and locations

    Args:
        centers (numpy ndarray): contains the gravity centers of molecules.
        system_size (int): size of the system
        lattice_param (float): lattice parameter extracted from the POSCAR file
        structure (str): 'sI' or 'sII'

    Returns:
        numpy ndarray: centers: updated centers
        numpy ndarray: int_map: integer coordinates
        numpy ndarray: cages: contains the type of cage, 0 (1) small (large)
    """

    script_dir = os.path.dirname(__file__)

    if structure == "sI":
        cages = [0] * 6 + [1] * 2
        int_map = np.loadtxt(script_dir+'/input/stats/sites_sI.dat', dtype=int)
        
    elif structure == 'sII':
        cages = [0] * 8 + [1] * 16
        int_map = np.loadtxt(script_dir+'/input/stats/sites_sII.dat', dtype=int)

    elif structure == 'sH':
        cages = [2] * 2 + [1] * 6 + [0] * 4
        int_map = np.loadtxt(script_dir+'/input/stats/sites_sH.dat', dtype=int)

    centers, int_map, cages = add_pattern(centers, int_map, cages, 0, lattice_param, structure, system_size)
    centers, int_map, cages = add_pattern(centers, int_map, cages, 1, lattice_param, structure , system_size)
    centers, int_map, cages = add_pattern(centers, int_map, cages, 2, lattice_param, structure, system_size)

    del structure, script_dir
    gc.collect()

    return centers, int_map, cages


#_____________________________________________________________________________________
def build_cages_catalog(centers, int_map, cages, pattern_size):
    """Function building the object containing all informations on cages (engine room)

    Args:
        centers (numpy ndarray): array containing nitrogen gravity centers
        int_map (numpy ndarray): array containing nitrogen gravity centers integer mapping
        cages (list): contains the cages type [0->large, 1->small]
        pattern_size (int): size of the unit cell in the integer coordinate system.

    Returns:
        numpy ndarray: object containing information of cages
        (format: [ID, coords, type, status, deflect])
    """
#________________________________________________________________________________________________#########
# Add two columns for site IDs
    sites = np.zeros((len(centers), 9), dtype=int)
    size = 0

    # Loop through R
    for i in range(len(centers)):

        # Define the current site
        current_site = int_map[i, :]
        current_site = np.append(i, current_site)
        current_site = np.append(current_site, [cages[size], 2, 0, -1, -1])
        size += 1
        if size == pattern_size:
            size = 0

        sites[i, :] = current_site

    return sites


#_____________________________________________________________________________________
def get_neighbors(structure, pattern_size):
    """Gets neighbors of each cage in the system.

    Args:
        structure (str): Structure type (sI or sII).
        pattern_size (int): size of the unit cell in the integer coordinate system.

    Returns:
        numpy ndarray: Array containing the neighbors of each cage.
    """

    if structure == "sI":
        script_dir = os.path.dirname(__file__)
        neighbors = np.loadtxt(f'{script_dir}' + '/input/stats/relatives_sI.dat', dtype=int)
        neighbors_map = np.zeros(pattern_size, dtype='object')
        edges_large = 14
        edges_small = 12
        delta_cage = pattern_size - 6

    elif structure == "sII":
        script_dir = os.path.dirname(__file__)
        neighbors = np.loadtxt(f'{script_dir}' + '/input/stats/relatives_sII.dat', dtype=int)
        neighbors_map = np.zeros(pattern_size, dtype='object')
        edges_large = 16
        edges_small = 12
        delta_cage = pattern_size - 8
    
    elif structure == 'sH':
        script_dir = os.path.dirname(__file__)
        neighbors = np.loadtxt(f'{script_dir}' + '/input/stats/relatives_sH.dat', dtype=int)
        neighbors_map = np.zeros(pattern_size, dtype='object')
        edges_large = 20
        edges_small = 12
        delta_cage = pattern_size - 2

    k = 0
    for i in range(pattern_size - delta_cage):

        neighbors_map[i] = neighbors[k:k + edges_large, :]
        k += edges_large

    for i in range(pattern_size - delta_cage, pattern_size):
        neighbors_map[i] = neighbors[k:k + edges_small, :]
        k += edges_small

    return neighbors_map


#_____________________________________________________________________________________
def get_neighbors_cage_type(neighbors_map, catalog, pattern_size):
    """Function giving type of cage for each site neighbor

    Args:
        neighbors_map (numpy ndarray): Array containing every neighbor for every site
        catalog (numpy ndarray): Array containing every site information
        pattern_size (int): size of the unit cell in the integer coordinate system
        structure (str): 'sI' or 'sII'

    Returns:
        numpy ndarray: Array containing 0 (1) for large (small) cages for the unit cell
    """

    size = np.max(catalog[:pattern_size,1:4], axis=0)+1
    cages_type = np.zeros(pattern_size, dtype='object')

    for i in range(len(neighbors_map)):
        # Applying PBC

        temp_map = (neighbors_map[i]+catalog[i,1:4])%(size)
        temp_cages = np.zeros(len(temp_map), dtype=int)

        for j in range(len(temp_map)):

            idx = np.where(temp_map[j,0]==catalog[:,1])[0]
            idy = np.where(temp_map[j,1]==catalog[:,2])[0]
            idz = np.where(temp_map[j,2]==catalog[:,3])[0]

            intersect_xy = np.intersect1d(idx, idy)
            intersect_xyz = np.intersect1d(idz, intersect_xy)
            temp_cages[j] = intersect_xyz[0]

        cages_type[i] = catalog[temp_cages, 4]

        del intersect_xy, intersect_xyz, temp_cages
        gc.collect()

    return cages_type


#_____________________________________________________________________________________
def get_neighbors_face_type(neighbors_map, catalog, pattern_size, structure):
    """Function getting face type of transition

    Args:
        neighbors_map (numpy ndarray): Array containing every neighbor for every site
        catalog (numpy ndarray): Array containing every site information
        pattern_size (int): size of the unit cell in the new coordinate system
        structure (str): 'sI' or 'sII'

    Returns:
        numpy ndarray: transitions for the unit cell: 0: S5L, 1: L5S, 2: L5L, 3:L6L
    """
    unit_size = np.max(catalog[:pattern_size,1:4], axis=0)+1

    if structure == 'sI':
        faces_type = np.zeros(pattern_size, dtype='object')

        for i in range(pattern_size):

            temp_map = (neighbors_map[i]+catalog[i,1:4])%unit_size
            temp_face_type = np.zeros(len(temp_map))

            for j in range(len(temp_map)):

                # Checking if departure of large cage and arrival to small cage (L5S)
                if i<6 and (np.array_equal(temp_map[j,:],catalog[6,1:4]) or\
                    np.array_equal(temp_map[j,:],catalog[7,1:4])):
                    temp_face_type[j] = 1

                # Checking if departure of large cage arrival to large cage (L6L)
                elif (temp_map[j,0] == catalog[i,1] and temp_map[j,1] == catalog[i,2]) or \
            (temp_map[j,1] == catalog[i,2] and temp_map[j,2] == catalog[i,3]) or \
            (temp_map[j,0] == catalog[i,1] and temp_map[j,2] == catalog[i,3]):
                    temp_face_type[j] = 3

                # If not any condition but it is a small cage (S5L)
                elif i>=6:
                    temp_face_type[j] = 0

                # Else it can only be (L5L)
                else:
                    temp_face_type[j] = 2

            faces_type[i] = temp_face_type

            del temp_map, temp_face_type

    elif structure == 'sII':
        faces_type = np.zeros(pattern_size, dtype='object')
        for i in range(pattern_size):
            temp_map = (neighbors_map[i] + catalog[i,1:4])%unit_size
            temp_face_type = np.zeros(len(temp_map))

            for j in range(len(temp_map)):

                for k in range(pattern_size):
                    # L6L
                    if np.array_equal(temp_map[j,:], catalog[k,1:4]) and i<8 and k<8:
                        temp_face_type[j] = 3

                    # L5S
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i<8 and k>=8:
                        temp_face_type[j] = 1

                    # S5L
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i>=8 and k<8:
                        temp_face_type[j] = 0

                    # S5S
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i>=8 and k>=8:
                        temp_face_type[j] = 2

            faces_type[i] = temp_face_type

    elif structure == 'sH':
        faces_type = np.zeros(pattern_size, dtype='object')
        # i is for the departure site, k for the arrival
        for i in range(pattern_size):
            temp_map = (neighbors_map[i] + catalog[i,1:4])%unit_size
            temp_face_type = np.zeros(len(temp_map))

            for j in range(len(temp_map)):
                for k in range(pattern_size):
                    # L6M
                    if np.array_equal(temp_map[j,:], catalog[k,1:4]) and i < 2 and k >= 8:
                        temp_face_type[j] = 0

                    # L6L
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i < 2 and k < 2:
                        temp_face_type[j] = 1

                    # L5S
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i < 2 and 2 <= k < 8:
                        temp_face_type[j] = 2

                    # M4M
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i >= 8 and k >= 8:
                        temp_face_type[j] = 3

                    # M6L
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i >= 8 and k < 2:
                        temp_face_type[j] = 4

                    # M5S
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i >= 8 and 2 <= k < 8:
                        temp_face_type[j] = 5

                    # S5L
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and 2 <= i < 8 and k < 2:
                        temp_face_type[j] = 6

                    # S5M
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and 2 <= i < 8 and k >= 8:
                        temp_face_type[j] = 7

                    # S5S
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and 2 <= i < 8 and 2 <= k < 8:
                        temp_face_type[j] = 8

            faces_type[i] = temp_face_type

    gc.collect()

    return faces_type


#_____________________________________________________________________________________
def associate_ids_to_map(neighbors_map, catalog, system_size, pattern_size):
    """Function associating IDs of neighbors in catalog to their neighbor map

    Args:
        neighbors_map (numpy ndarray): Array containing every neighbor for every site
        catalog (numpy ndarray): Array containing every site information
        system_size (int): System size
        structure (str): 'sI' or 'sII'

    Returns:
        numpy ndarray: Updated neighbor map
    """

    # Defining a dictionary containing as key the integer coordinates of each sites,
    # and a values its ID.

    size = np.max(catalog[:pattern_size,1:4], axis=0)+1

    inventory = {}

    for i in range(len(catalog)):
        inventory[tuple(catalog[i,1:4])] = i

    gc.collect()

    # Tiling unit map to system size
    whole_map = np.tile(neighbors_map , int(len(catalog)/len(neighbors_map)))
    new_map = np.zeros(len(whole_map), dtype=object)

    # Setting ID wrt neighboring site
    for i in range(len(whole_map)):
        temp_ids = np.zeros(len(whole_map[i]), dtype=int)
        pbc = (whole_map[i][:,0:3]+catalog[i,1:4])%(size*system_size)

        for j in range(len(whole_map[i])):
            temp_ids[j] = inventory[tuple(pbc[j])]

        new_map[i] = temp_ids
        del temp_ids, pbc

    del whole_map
    gc.collect()

    return new_map


#_____________________________________________________________________________________
def build_complete_neighbors_map(structure, catalog, pattern_size, system_size):
    """Function building complete neighbors map object.
    Format: [ID, type, transition]

    Args:
        structure (str): 'sI' or 'sII'
        catalog (numpy ndarray): array containing every site information
        pattern_size (int): amount of cages in the unit cell
        system_size (int): size of the generated system

    Returns:
        numpy ndarray: complete neighbors map
    """

    neighbors_map = get_neighbors(structure,pattern_size)
    cages_type = get_neighbors_cage_type(neighbors_map,catalog, pattern_size)
    faces_type = get_neighbors_face_type(neighbors_map, catalog, pattern_size, structure)
    ids = associate_ids_to_map(neighbors_map, catalog, system_size, pattern_size)

    final_map = np.zeros(len(ids), dtype='object')

    for i in range(len(ids)):
        final_map[i] = np.column_stack((ids[i], cages_type[i%pattern_size]))
        final_map[i] = np.column_stack((final_map[i], faces_type[i%pattern_size])).astype(int)

    return final_map


#_____________________________________________________________________________________
def fill_system(catalog, density, mode, structure):
    """Function to randomly fill the structure with molecules

    Args:
        catalog (numpy ndarray): array containing every site information
        density (float): density of filling
        mode (str): 'SO' or 'DO'

    Returns:
        numpy ndarray: updated catalog

    """
#________________________________________________________________________________________________#########
# Add index for sites IDs depending on SO/DO
    k = 0

    if mode == 'DO':
        large = np.where(catalog[:,4]==0)[0]
        len_do = len(large)
        large = np.concatenate((large,large))
        small = np.where(catalog[:,4]==1)[0]
        cages = np.concatenate((large,small))
        np.random.shuffle(cages)
        cages = cages[0:int(density*(len(catalog)+len_do))]
        a = 0
        b = 0
        n = 0
        for i in range(len(cages)):
            if catalog[cages[i],7] != -1 and not catalog[cages[i],4] == 1:
                # print(True)
                catalog[cages[i],8] = k
                catalog[cages[i],5] = 0
                k += 1
                b += 1
            else:
                catalog[cages[i],7] = k
                catalog[cages[i],5] = 1
                k += 1
                if catalog[cages[i],4] == 1:
                    a +=1
                else:
                    b += 1
                    n += 1
        # print(a,b,k,n)

    elif mode == 'SO':
        cages = np.where(catalog[:,4]>=0)[0]
        np.random.shuffle(cages)
        cages = cages[0:int(density*len(catalog))]

        for i in range(len(cages)):
            catalog[cages[i], 5] = 1
            catalog[cages[i], 7] = k
            k += 1

    elif mode == 'Single':
        molecule = np.random.randint(0,len(catalog)+1)
        catalog[molecule, 5] = 1
        catalog[molecule, 7] = k

    return catalog


#_____________________________________________________________________________________
def init_system(structure, positions, system_size, size, occupation_density, mode):
    """Function initializing the structure

    Args:
        structure (str): 'sI' or 'sII'
        positions (numpy ndarray): array containing positions of N2 in the POSCAR file
        system_size (int): size of the system (system_size**3)
        size (list): contains the lattice parameter and pattern size
        occupation_density (float): density of filling

    Returns:
        numpy ndarray: catalog which contains all the information about each site
        numpy ndarray: neighbors_map which contains all the information of neighbors of each site
    """

    lattice_param, pattern_size = size
    centers = gravity_center(positions)

    centers, integer_map, cages = build_system(centers, system_size, lattice_param, structure)
    catalog = build_cages_catalog(centers, integer_map, cages, pattern_size)
    neighbors_map = build_complete_neighbors_map(structure, catalog, pattern_size, system_size)

    catalog = fill_system(catalog, occupation_density, mode, structure)

    return catalog, neighbors_map, centers