import numpy as np
from display_tools import check_PBC, add_patternO, render
from system import *
from get_contents import get_structure_stats, load_data

structure = 'sI'
file, cages, large, small = get_structure_stats(structure)

lattice_param, positions, rO, N_O, N_N, pattern_size = load_data(file)

# Getting gravity centers of N2
centers = gravity_center(positions)

# Applying periodic boundary conditions around pattern
rO = check_PBC(rO, lattice_param)

# Adding N patterns along coord r for r0 (r0,coord,N)
rO = add_patternO(rO,0,lattice_param,3)
rO = add_patternO(rO,1,lattice_param,3)
rO = add_patternO(rO,2,lattice_param,3)

# Creating new coordinate system
XR = replace_float_with_int(centers[:,0])
YR = replace_float_with_int(centers[:,1])
ZR = replace_float_with_int(centers[:,2])

# Setting id (numerotation) to (x,y,z) sites
mapping = np.array([XR,YR,ZR], dtype=int).transpose()

# np.savetxt(f'sites_{structure}.dat', mapping, fmt='%i')

centers, mapping, cages = build_system(centers, 3,lattice_param, structure)


# Only use when full pattern is built
neighbor_coords, relative_positions, unit_atoms = is_interacting(centers, mapping, pattern_size, small, large, structure)
relative_positions = exclude_self_from_neighbors(mapping, relative_positions, structure)

catalog = build_cages_catalog(centers, mapping, cages, pattern_size)

for i in range(len(centers)):
    print(catalog[i,0], centers[i,:])

print(centers[:8,:])

# np.savetxt(f'relatives_{structure}.dat', np.vstack(relative_positions), fmt='%i')

render(centers, pattern_size, rO, N_O, lattice_param, mapping, cages)