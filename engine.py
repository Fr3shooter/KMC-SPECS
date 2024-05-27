import numpy as np
import random
from get_contents import load_structure
from system import init_system
from parameters import *
from numpy.linalg import norm
import scipy.sparse as sp

#_____________________________________________________________________________________
def init_event(catalog, neighbors_map, states, transition_type, candidate_length):
    """Function computing the amount of possible events corresponding to a given transition
    and inventoring the candidates

    Args:
        catalog (numpy ndarray): array containing informations for each site
        neighbors_map (numpy ndarray): array containing informations for each site's neighbors
        states (list): list containing the state of the departure and arrival sites (how they are filled)
        transition_type (int): integer corresponding to the transition (L6L......)

    Returns:
        list, candidates: candidate IDs for a given transition type
        int, amount_of_candidates: amount of candidates for a given transition type
    """

    departure, arrival = states
    # candidates = 10000*[-1]

    if mode == 'DO':
        amplifier = 3
    else:
        amplifier = 1
    
    if structure == 'sI':
        max_site = 6
    elif structure == 'sII':
        max_site = 16

    candidates = -1 * np.ones(int(max_site*amplifier*occupancy_density*candidate_length*system_size**3), dtype=int)
    amount_of_candidates = 0
    idx = 0

    # Checking for transitions for occupied sites
    for i in catalog[catalog[:,5]==departure][:,0]:
        # Checking where are given transitions
        transitions = np.where(neighbors_map[i][:,-1]==transition_type)[0]

        # Checking where are available neighboring sites
        available = np.where(catalog[neighbors_map[i][transitions,0]][:,5]==arrival)[0]

        # Adding corresponding amount of transitions
        amount_of_candidates += len(available)

        # Adding transition times the considered ID
        if len(available)>0:
            # print(len(available) + idx)
            candidates[idx:len(available)+idx] = len(available)*[i]
            idx += len(available)
    return candidates, amount_of_candidates


#_____________________________________________________________________________________
def init_events(catalog, neighbors_map, transitions, forbidden_transitions, candidate_length):
    """Function initializing all possibles transitions and their corresponding candidates

    Args:
        catalog (numpy ndarray): array containing informations for each site
        neighbors_map (numpy ndarray): array containing informations for each site's neighbors
        transitions (numpy ndarray): array containing the allowed transitions
        forbidden_transitions (dict): dictionary containing the forbidden transitions

    Returns:
        numpy ndarray, amounts: amount of candidates for each transition
        numpy ndarray, candidates: candidates IDs for each transition
    """

    candidates = np.zeros(len(transitions[transitions != 0]),dtype='object')
    amounts = np.zeros(len(transitions[transitions != 0]),dtype=int)
    if structure == 'sH':
        transition_type = list(range(9))
    else:
        transition_type = list(range(4))

    status = [[1,2], [1,1], [0,2], [0,1]]

    pos = 0
    for i in range(len(transition_type)):
        for j in range(len(status)):
            if not tuple([*status[j], transition_type[i]]) in forbidden_transitions:
                candidates[pos], amounts[pos] = init_event(catalog, neighbors_map, status[j], transition_type[i], candidate_length)
                pos+=1

    return amounts, candidates


#_____________________________________________________________________________________
def build_gauge(event_amount, transitions):
    """Function building BKL gauge based on possible events

    Args:
        event_amount (list): Amount of candidates for each transition
        transitions (list): List containing the different transitions

    Returns:
        numpy ndarray: Gauge (len=len(transitions)+1)
    """

    transitions = transitions[transitions != 0]

    gauge = np.zeros(len(transitions)+1)

    for i in range(len(transitions)):
        gauge[i+1] = gauge[i] + event_amount[i]*transitions[i]
    return gauge


#_____________________________________________________________________________________
def pick_in_gauge(gauge):
    """Function picking transition from probablity gauge

    Args:
        gauge (np.ndarray): probability gauge

    Returns:
        int: Index to be picked in transitions
    """

    rho1 = np.random.uniform(low=0.0, high=np.max(gauge), size=(1,))
    res = np.searchsorted(gauge, rho1, side="right") - 1

    return res[0]


#_____________________________________________________________________________________
def build_transition_inventory(transitions):
    """Function building the transition inventory, ie. all the possible and forbidden transitions

    Args:
        transitions (numpy ndarray): Array containing all the transitions

    Returns:
        list of dicts: dict contains index as keys and allowed transitions as values
        dict2 contains allowed transitions as keys, and index as values
        list of lists: list containing the forbidden transitions and their index
    """

    # ID to transition inventory
    dict = {}
    # transition to ID inventory
    dict2 = {}

    # forbidden transitions inventory
    forbidden_transitions = []

    # forbidden transitions IDs
    forbidden_transitions_idx = []

    k, m = 0, 0

    # SO/DO IDs (2: empty, 1: single occupancy, 0: full)
    occupancies = [[1,2], [1,1], [0,2], [0,1]]

    # For 'sH': (0: L6M, 1: L6L, 2: L5S, 3: M4M, 
    # 4: M6L, 5: M5S, 6: S5L, 7: S5M, 8: S5S)
    if structure == 'sH':
        transitions_type = list(range(9))

    # For 'sI/II': (0: S5L, 1: L5S, 2: L5L [resp. S5S sII], 3: L6L)
    else:
        transitions_type = list(range(4))

    for transition in transitions_type:
        for occupancy in occupancies:
            if transitions[k] != 0:
                dict[m] = occupancy + [transition]
                dict2[tuple(occupancy+[transition])] = m
                m+=1
            k+=1

    k = 0
    for transition in transitions_type:
        for occupancy in occupancies:
            if transitions[k] == 0:
                forbidden_transitions.append(tuple(occupancy+[transition]))
                forbidden_transitions_idx.append(m)
                m += 1
            k+=1

    return [dict, dict2], [forbidden_transitions, forbidden_transitions_idx]


#_____________________________________________________________________________________
def clear_departure_and_arrival_transitions(candidates, hopping_molecule, chosen_site, transitions, gauge):
    """Function clearing all the available transitions for the departure and arrival sites

    Args:
        candidates (numpy ndarray): array containing informations on sites
        hopping_molecule (int): ID of hopping molecule (departure cage)
        chosen_site (int): ID of arrival cage
        transitions (numpy ndarray): Array containing all the transitions
        gauge (numpy ndarray): Probability gauge

    Returns:
        list of lists, candidates: Candidates for each transition
        numpy ndarray, gauge: Updated probablity gauge
    """

    # Clearing departure and arrival sites from candidates
    for i in range(len(candidates)):
        idx = np.where(np.logical_or(candidates[i] == chosen_site, candidates[i] == hopping_molecule))[0]
        candidates[i][idx] = -1
        gauge[i+1:] -= len(idx)*transitions[i]
        # while hopping_molecule in candidates[i]:
        #     idx = candidates[i].index(hopping_molecule)
        #     candidates[i][idx] = -1
        #     gauge[i+1:] -= transitions[i]

        # while chosen_site in candidates[i]:
        #     idx = candidates[i].index(chosen_site)
        #     candidates[i][idx] = -1
        #     gauge[i+1:] -= transitions[i]

    return candidates, gauge


#_____________________________________________________________________________________
def refresh_departure_site(departure, catalog, neighbors_map, hopping_molecule, chosen_site, transition_inv, candidates, gauge, transitions):
    """Function refreshing transitions to the departure site and from the departure site

    Args:
        departure (int): State of the departure site
        catalog (numpy ndarray): array containing informations for each site
        neighbors_map (numpy ndarray): array containing informations for each site's neighbors
        hopping_molecule (int): ID of departure site
        chosen_site (int): ID of arrival site
        transition_inv (dict): transitions inventory
        candidates (list of lists): list containing candidates for each transition
        gauge (numpy ndarray): probability gauge
        transitions (numpy ndarray): array containing the allowed transitions

    Returns:
        list, candidates: updated candidates list for each transition
        numpy ndarray, gauge: updated probability gauge
    """

    gauge_index, transition_gauge_index = transition_inv[0]
    forbidden_transitions = transition_inv[1][0]
    transitions_of_neighbors = neighbors_map[hopping_molecule][:,-1]
    ids_of_neighbors = neighbors_map[hopping_molecule][:,0]
    occupied_neighbors = catalog[ids_of_neighbors][:,5]

    for i in range(len(occupied_neighbors)):
    # Refreshing s -> v transitions
        test_transition = (departure+1, occupied_neighbors[i], transitions_of_neighbors[i])
        if occupied_neighbors[i] > 0 and departure+1 < 2 and not test_transition in forbidden_transitions:
            transition = transition_gauge_index[test_transition]
            idx = np.where(candidates[transition] == -1)[0][0]
            # idx = candidates[transition].index(-1)
            candidates[transition][idx] = hopping_molecule
            gauge[transition+1:] += transitions[transition]

        if occupied_neighbors[i] < 2:
#________________________________________________________________________________________________#########
# Make a function for this to be compatible with sI/II and sH
            if transitions_of_neighbors[i] == 0:
                new_transition = 1
            elif transitions_of_neighbors[i] == 1:
                new_transition = 0
            else:
                new_transition = transitions_of_neighbors[i]
            # if transitions_of_neighbors[i] == 0:
            #     new_transition = 4
            # elif transitions_of_neighbors[i] == 4:
            #     new_transition = 0
            # elif transitions_of_neighbors[i] == 2:
            #     new_transition = 6
            # elif transitions_of_neighbors[i] == 6:
            #     new_transition = 2
            # elif transitions_of_neighbors[i] == 5:
            #     new_transition = 7
            # elif transitions_of_neighbors[i] == 7:
            #     new_transition = 5
            # else:
            #     new_transition = transitions_of_neighbors[i]

            # Removing old v -> s transitions
            test_transition = (occupied_neighbors[i], departure, new_transition)
            if departure==1 and not ids_of_neighbors[i] == chosen_site and not test_transition in forbidden_transitions:
                old_transition = transition_gauge_index[test_transition]
                # idx = candidates[old_transition].index(ids_of_neighbors[i])
                idx = np.where(candidates[old_transition] == ids_of_neighbors[i])[0][0]
                candidates[old_transition][idx] = -1
                gauge[old_transition+1:] -= transitions[old_transition]

            # Adding new v -> s transitions
            test_transition = (occupied_neighbors[i], departure+1, new_transition)
            if not test_transition in forbidden_transitions:
                transition = transition_gauge_index[test_transition]
                # idx = candidates[transition].index(-1)
                idx = np.where(candidates[transition] == -1)[0][0]
                candidates[transition][idx] = ids_of_neighbors[i]
                gauge[transition+1:] += transitions[transition]

    return candidates, gauge


#_____________________________________________________________________________________
def refresh_arrival_site(arrival, catalog, neighbors_map, hopping_molecule, chosen_site, transition_inv, candidates, gauge, transitions):
    """Function refreshing transitions to the arrival site and from the arrival site

    Args:
        arrival (int): State of the arrival site
        catalog (numpy ndarray): array containing informations for each site
        neighbors_map (numpy ndarray): array containing informations for each site's neighbors
        hopping_molecule (int): ID of departure site
        chosen_site (int): ID of arrival site
        transition_inv (dict): transitions inventory
        candidates (list of lists): list containing candidates for each transition
        gauge (numpy ndarray): probability gauge
        transitions (numpy ndarray): array containing the allowed transitions

    Returns:
        list, candidates: updated candidates list for each transition
        numpy ndarray, gauge: updated probability gauge
    """

    gauge_index, transition_gauge_index = transition_inv[0]
    transitions_of_new_neighbors = neighbors_map[chosen_site][:,-1]

    forbidden_transitions = transition_inv[1][0]
    ids_of_new_neighbors = neighbors_map[chosen_site][:,0]
    occupied_new_neighbors = catalog[ids_of_new_neighbors][:,5]

    for j in range(len(occupied_new_neighbors)):
        test_transition = (arrival-1, occupied_new_neighbors[j] , transitions_of_new_neighbors[j])
        if occupied_new_neighbors[j] > 0 and ids_of_new_neighbors[j] != hopping_molecule and not test_transition in forbidden_transitions:
            transition = transition_gauge_index[test_transition]
            gauge[transition+1:] += transitions[transition]
            # idx = candidates[transition].index(-1)
            idx = np.where(candidates[transition] == -1)[0][0]
            candidates[transition][idx] = chosen_site

        if occupied_new_neighbors[j] < 2 and ids_of_new_neighbors[j] != hopping_molecule:

#________________________________________________________________________________________________#########
# Make a function for this to be compatible with sI/II and sH

            if transitions_of_new_neighbors[j] == 1:
                new_transition = 0
            elif transitions_of_new_neighbors[j] == 0:
                new_transition = 1
            # if transitions_of_new_neighbors[j] == 0:
            #     new_transition = 4
            # elif transitions_of_new_neighbors[j] == 4:
            #     new_transition = 0
            # elif transitions_of_new_neighbors[j] == 2:
            #     new_transition = 6
            # elif transitions_of_new_neighbors[j] == 6:
            #     new_transition = 2
            # elif transitions_of_new_neighbors[j] == 5:
            #     new_transition = 7
            # elif transitions_of_new_neighbors[j] == 7:
            #     new_transition = 5
            else:
                new_transition = transitions_of_new_neighbors[j]

            test_transition = (occupied_new_neighbors[j], arrival, new_transition)
            if not test_transition in forbidden_transitions and not ids_of_new_neighbors[j] == chosen_site:
                old_transition = transition_gauge_index[test_transition]
                gauge[old_transition+1:] -= transitions[old_transition]
                # print(ids_of_new_neighbors[j], test_transition)
                # idx = candidates[old_transition].index(ids_of_new_neighbors[j])
                idx = np.where(candidates[old_transition] == ids_of_new_neighbors[j])[0][0]
                candidates[old_transition][idx] = -1
        
            test_transition = (occupied_new_neighbors[j], arrival-1, new_transition)
            if arrival - 1 !=0 and not test_transition in forbidden_transitions:
                transition = transition_gauge_index[test_transition]
                gauge[transition+1:] += transitions[transition]
                # idx = candidates[transition].index(-1)
                idx = np.where(candidates[transition] == -1)[0][0]
                candidates[transition][idx] = ids_of_new_neighbors[j]

    return candidates, gauge


#_____________________________________________________________________________________
def move(neighbors_map : np.ndarray, catalog : np.ndarray, gauge : np.ndarray, candidates : list, transitions : list, transition_inv, trajectories, events_amount):
    """Main function allowing to perform a move from a chosen site i to a site j

    Args:
        neighbors_map (np.ndarray): array containing informations on site's neighbors
        catalog (np.ndarray): array containing informations on sites
        gauge (np.ndarray): probability gauge
        candidates (list): list containing candidates for each transition
        transitions (list): list containing all velocity constants
        transition_inv (object): contains the transitions ids

    Returns:
        np.ndarray, neighbors_map: updated neighbors_map
        np.ndarray, catalog: updated catalog
        np.ndarray, gauge: updated probability gauge
        list, candidates: updated candidates
        list of lists, occupancies: amount of occupied cages for each type
    """

    gauge_index, transition_gauge_index = transition_inv[0]

    # Picking a transition
    selected = pick_in_gauge(gauge)

    events_amount[selected] += 1
    # Picking a particle among the ones satisfying the transition
    # hop_index = random.choice(range(len(candidates[selected])))
    # s_candidates = [candidate for candidate in candidates[selected] if candidate > -1]
    s_candidates = candidates[selected][candidates[selected] > -1]
    hop_index = int(random.random()*len(s_candidates))
    hopping_molecule = s_candidates[hop_index]
    departure, arrival, transition = gauge_index[selected]
    # Gathering neighbors of selected particle
    potential_sites = neighbors_map[hopping_molecule][neighbors_map[hopping_molecule][:,-1]==transition]

    # Filtering occupied neighbors
    available_sites_mask = catalog[potential_sites[:,0],5] == arrival
    available_sites = potential_sites[available_sites_mask, 0]
    # Choosing an empty neighbor randomly
    chosen_index = int(random.random()*len(available_sites))
    chosen_site = available_sites[chosen_index]

#________________________________________________________________________________________________#########
# The idea would be to add a if condition to empty the departure site and fill the arrival one
# Check if -1, if not, set to -1 and displace the molecule to the arrival site ?
# HERE
    catalog[hopping_molecule, 5] += 1
    catalog[chosen_site, 5] -= 1

    if departure == 0:
        if arrival == 1:
            trajectories[catalog[hopping_molecule, 8]] = catalog[chosen_site, 0]
            catalog[chosen_site, 8] = catalog[hopping_molecule, 8]
            catalog[hopping_molecule, 8] = -1
        else:
            trajectories[catalog[hopping_molecule, 8]] = catalog[chosen_site, 0]
            catalog[chosen_site, 7] = catalog[hopping_molecule, 8]
            catalog[hopping_molecule, 8] = -1

    elif departure == 1:
        if arrival == 1:
            trajectories[catalog[hopping_molecule, 7]] = catalog[chosen_site, 0]
            catalog[chosen_site, 8] = catalog[hopping_molecule, 7]
            catalog[hopping_molecule, 7] = -1
        else:
            trajectories[catalog[hopping_molecule, 7]] = catalog[chosen_site, 0]
            catalog[chosen_site, 7] = catalog[hopping_molecule, 7]
            catalog[hopping_molecule, 7] = -1
    
    candidates, gauge = clear_departure_and_arrival_transitions(candidates, 
                                                                hopping_molecule, 
                                                                chosen_site, 
                                                                transitions, 
                                                                gauge)
    
    
    candidates, gauge = refresh_departure_site(departure,
                                               catalog, 
                                               neighbors_map,
                                               hopping_molecule, 
                                               chosen_site, 
                                               transition_inv,
                                               candidates, 
                                               gauge, 
                                               transitions)

    candidates, gauge = refresh_arrival_site(arrival,
                                               catalog, 
                                               neighbors_map,
                                               hopping_molecule, 
                                               chosen_site, 
                                               transition_inv,
                                               candidates, 
                                               gauge, 
                                               transitions)

    small_cages = catalog[catalog[:,4]==1]
    large_cages = catalog[catalog[:,4]==0]
    ss_occupancy = int(len(small_cages[small_cages[:,5]==1]))
    ls_occupancy = int(len(large_cages[large_cages[:,5]==1]))
    ld_occupancy = int(len(large_cages[large_cages[:,5]==0]))
    t = -np.log(np.random.rand())/(gauge[-1])

    return neighbors_map, catalog, gauge, candidates, [ss_occupancy, ls_occupancy, ld_occupancy], trajectories, events_amount, t


#_____________________________________________________________________________________
def init_BKL(catalog, neighbors_map, transitions, transition_inv, candidate_length):
    """Function initializing the BKL algorithm

    Args:
        catalog (np.ndarray): array containing informations on sites
        neighbors_map (np.ndarray): array containing informations on site's neighbors
        transitions (list): list containing all velocity constants
        transition_inv (object): contains the transitions ids

    Returns:
        list of lists, candidates: initial candidates for each transition
        np.ndarray, gauge: initial probability gauge
    """

    events_amount, candidates = init_events(catalog, neighbors_map, transitions, transition_inv[1][0], candidate_length)
    gauge = build_gauge(events_amount, transitions)

    return candidates, gauge


def init_probes(catalog, centers):
    probe_stats = np.zeros((iterations, probe_amount), dtype=int)
    probes = np.zeros((probe_size, probe_amount), dtype=int)
    for i in range(probe_amount):
        dist = norm(centers[np.random.randint(0,len(catalog)),:] - centers, axis=1)
        probes[:,i] = np.argsort(dist)[1:probe_size+1]
        probe_stats[0,i] = np.sum(catalog[probes[:,i],5]==1) + 2*np.sum(catalog[probes[:,i],5]==0)
    return probe_stats, probes

def refresh_probes(catalog, probes, probe_stats, simulation_time):
    for i in range(probe_amount):
        probe_stats[simulation_time,i] = np.sum(catalog[probes[:,i],5]==1) + 2*np.sum(catalog[probes[:,i],5]==0)
    return probe_stats

def recompute_gauge(gauge, candidates, transitions):
    A = np.zeros(len(gauge))
    for i in range(1,len(candidates)+1):
        A[i] = len(candidates[i-1][candidates[i-1]>-1])*transitions[i-1] + A[i-1]
    return A

def init_occupancies(catalog):
    small_cages = catalog[catalog[:,4]==1]
    large_cages = catalog[catalog[:,4]==0]
    ss_occupancy = len(small_cages[small_cages[:,5]==1])
    ls_occupancy = len(large_cages[large_cages[:,5]==1])
    ld_occupancy = len(large_cages[large_cages[:,5]==0])
    occupancies = np.zeros((3,iterations), dtype=int)
    occupancies[:,0] = [ss_occupancy, ls_occupancy, ld_occupancy]
    return occupancies

#_____________________________________________________________________________________
def run_simulation(iterations, neighbors_map, catalog, gauge, candidates, transitions, transition_inv, centers, trajectories, MP=False):
    """Simulation loop

    Args:
        iterations (int): Max amount of iterations
        neighbors_map (np.ndarray): array containing informations on site's neighbors
        catalog (np.ndarray): array containing informations on sites
        gauge (np.ndarray): probability gauge
        candidates (list): list containing candidates for each transition
        transitions (list): list containing all velocity constants
        transition_inv (object): contains the transitions ids

    Returns:
        list of lists: Cages occupancy along simulation
    """
    t_tot = 0
    t = 0
    N = len(catalog[catalog[:,7] != -1]) + len(catalog[catalog[:,8] != -1])
    print(N)
    times = np.zeros(iterations)
    trajectories0 = trajectories.copy()
    overall_trajectories = sp.lil_matrix((iterations,N), dtype=int)
    overall_trajectories[0,:] = trajectories0
    occupancies = init_occupancies(catalog)
    probe_stats, probes = init_probes(catalog, centers)
    for _ in range(1,iterations):

        neighbors_map, catalog, gauge, candidates, occupancy, trajectories, events_amount, t = move(neighbors_map, catalog, gauge, candidates, transitions, transition_inv, trajectories, events_amount)
        occupancies[:,_] = occupancy

        gauge = recompute_gauge(gauge, candidates, transitions)
        probe_stats = refresh_probes(catalog, probes, probe_stats, _)
        overall_trajectories[_,:] = trajectories - trajectories0
        trajectories0 = trajectories.copy()
        t_tot += t
        times[_] = t_tot
        total_events = events_amount

    xdiff = xdiff.tocoo()

    return [occupancies, probe_stats, times, xdiff, total_events]


#_____________________________________________________________________________________
def get_params():
    """Function gathering which parameter to scale the mp simulation on

    Returns:
        list, densities
        list, temperatures
    """

    if parameter == 'Density':
        densities = np.array(range(5,100,10))/100
        temperatures = [T]*len(densities)
    if parameter == 'Temperature':
        temperatures = np.array(range(75,325,25))
        densities = [occupancy_density]*len(temperatures)

    return densities, temperatures


#_____________________________________________________________________________________
def mp_run(occupancy_density, T):
    """Function allowing to run the simulation either in single processing or multi processing mode

    Args:
        occupancy_density (float): occupancy density
        T (float): temperature

    Returns:
        list: occupancies
    """

    positions, size, transitions = load_structure(structure, T)
    transition_inv = build_transition_inventory(transitions)
    catalog, neighbors_map, centers, candidate_length = init_system(structure, positions, system_size, size, occupancy_density, mode)
    if mode == 'DO':
        sp_xdiff = np.zeros(int(len(catalog[catalog[:,7] != -1])+len(catalog[catalog[:,8] != -1])))
    else:
        sp_xdiff = np.zeros(len(catalog[catalog[:,7] != -1]))
    # tmp_diff = 0
    np.savetxt('centers.dat', centers)
    # centers = centers*1e-10

    for i in range(len(catalog[catalog[:,7] != -1])):
        sp_xdiff[catalog[catalog[:,7] != -1][i,7]] = catalog[catalog[:,7] != -1][i,0]
    if mode == 'DO':
        for j in range(len(catalog[catalog[:,8] != -1])):
            sp_xdiff[catalog[catalog[:,8] != -1][j,8]] = catalog[catalog[:,8] != -1][j,0]

    candidates, gauge = init_BKL(catalog, neighbors_map, transitions, transition_inv, candidate_length)
    func_args = [neighbors_map, catalog, gauge, candidates, transitions, transition_inv, centers, sp_xdiff]

    data = run_simulation(iterations, *func_args, MP)

    return data