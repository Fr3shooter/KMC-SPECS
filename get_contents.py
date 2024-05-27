import numpy as np
import os
from parameters import input_file, structure

#____________________________________________________________________________________
def get_structure_stats(structure):
    """Function gathering the unit cell topology

    Args:
        structure (str): 'sI', 'sII' or 'sH'

    Returns:
        str, file: file to be readed
        list, cages: list containing the cage types
        int, large: amount of large cages in the unit cell
        int, small: amount of small cages in the unit cell
    """

    if structure == 'sI':
        file_name = "I"
        script_dir = os.path.dirname(__file__)
        file = script_dir+'/input/structure/'+file_name
        large = 6
        small = 2
        cages = large* [0] + small* [1]

        return file, cages, large, small

    elif structure == 'sII':
        file_name = "II"
        script_dir = os.path.dirname(__file__)
        file = script_dir+'/input/structure/'+file_name
        large = 8
        small = 16
        cages = large* [0] + small* [1]

        return file, cages, large, small

    elif structure == 'sH':
        file_name = 'H'
        script_dir = os.path.dirname(__file__)
        file = script_dir+'/input/structure/'+file_name
        large = 8
        small = 16
        cages = large* [0] + small* [1]

        return file, cages, large, small


#_____________________________________________________________________________________
def load_data(file):
    """Well... It load the data from a VASP POSCAR file ?

    Args:
        file (str): File to load

    Returns:
        float, lattice_param: lattice parameter
        numpy ndarray, r: array containing the nitrogen positions
        numpy ndarray, rO: array containing the oxygen positions
        integer, N_O: amount of oxygens
        integer, N_N: amount of nitrogens
        integer, pattern_size: amount of cages in the unit cell
    """

    with open(file, 'r') as f:
        lines = f.readlines()
        x, y, z = np.array([]), np.array([]), np.array([])
        scaling_factor = float(lines[1])
        lparam_x = float(list(filter(None, lines[2].split(' ')))[0])
        lparam_y = float(list(filter(None, lines[3].split(' ')))[1])
        lparam_z = float(list(filter(None, lines[4].split(' ')))[2])
        N_atoms = list(filter(None, lines[6].split(' ')))
        N_O, N_H, N_N = int(N_atoms[0]), int(N_atoms[1]), int(N_atoms[2])

        for i in range(8, len(lines)):
            r = list(filter(None, lines[i].split(' ')))
            x = np.append(x, float(r[0]))
            y = np.append(y, float(r[1]))
            z = np.append(z, float(r[2].strip()))

        r = np.array([x, y, z]).transpose()
        rO, rH, r = r[0:N_O, :], r[N_O + 1:N_O + 1 + N_H, :], r[N_O + N_H:, :]
        pattern_size = int(len(r)/2)

    return [lparam_x, lparam_y, lparam_z], r, rO, N_O, N_N, pattern_size


#_____________________________________________________________________________________
def load_transitions(T=225):
    """Function loading transitions from file

    Args:
        structure (str, optional): Considered structure. Defaults to "sI".
        T (float, optional): Temperature. Defaults to 225.

    Returns:
        numpy ndarray: transitions catalog
    """

    A = [3e12, 1.5e12]
    if structure == 'sI':
        pre_exp = 4*[0] + 12*[1]
    elif structure == 'sII':
        pre_exp = 4*[0] + 4*[1] + 4*[0] + 4*[1]
    script_dir = os.path.dirname(__file__)

    transitions = np.loadtxt(f'{script_dir}' + f'/input/stats/{input_file}', dtype=float)
    # pre_exp = np.array(transitions[:,1], dtype=int)
    # transitions = transitions[:,0]

    for i in range(len(pre_exp)):
        if transitions[i] == -1:
            transitions[i] = 0
        else:
            # transitions[i] = transitions[i]*1000
            transitions[i] = A[pre_exp[i]]*np.exp(-(1.6e-19*6.02e23*transitions[i])/(8.314*T))

    return transitions


#_____________________________________________________________________________________
def load_structure(structure, T, debug=False):
    """Global function loading structure properties

    Args:
        structure (str): 'sI', 'sII' or 'sH'
        T (float): Temperature

    Returns:
        numpy ndarray, positions: array containing sites positions (gravity centers in 
        given coordinates system)
        list, lattice_param: list containing the unit cell lattice parameters
        int, pattern_size: amount of sites in the unit cell
        int, cages: amount of cages in the unit cell
        numpy ndarray, transitions: array containing the velocity constants
    """

    file, _, _, _ = get_structure_stats(structure)
    lattice_param, positions, rO, N_O, _, pattern_size = load_data(file)
    transitions = load_transitions(T)
    if debug:
        return positions, [lattice_param, pattern_size], transitions, rO, N_O

    return positions, [lattice_param, pattern_size], transitions