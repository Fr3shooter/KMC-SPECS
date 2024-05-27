import numpy as np
import gc

#_____________________________________________________________________________________
def init_events(catalog, neighbors_map):
    """Function inventoring possible events and candidate sites to these events

    Args:
        catalog (numpy ndarray): Array containing every site information
        neighbors_map (numpy ndarray): Array containing every neighbor for every site

    Returns:
        list: List containing the amount of candidates for each transition (s5l, l5s, l5l, l6l)
        list of lists: List containing IDs for each candidate for a given transition
    """
    candidates = np.zeros(4, dtype='object')
    amount_of_candidates = np.zeros(4, dtype=int)
    s5l_candidates = []
    l5s_candidates = []
    l5l_candidates = []
    l6l_candidates = []
    s5l_amount = 0
    l5s_amount = 0
    l5l_amount = 0
    l6l_amount = 0

    # Checking for s5l transitions for occupied sites
    for i in catalog[catalog[:,5]==0][:,0]:
        # Checking where are s5l transitions
        s5l_transitions = np.where(neighbors_map[i][:,-1]==0)[0]

        # Checking where are empty neighbor sites
        s5l_available = np.where(catalog[neighbors_map[i][s5l_transitions,0]][:,-2]==1)[0]

        # Adding corresponding amount of transitions
        s5l_amount += len(s5l_available)

        # Adding transition times the considered ID
        if len(s5l_available)>0:
            s5l_candidates += len(s5l_available)*[i]

        # Same as before for the other possible transitions
        l5s_transitions = np.where(neighbors_map[i][:,-1]==1)[0]
        l5s_available = np.where(catalog[neighbors_map[i][l5s_transitions,0]][:,-2]==1)[0]
        l5s_amount += len(l5s_available)

        if len(l5s_available)>0:
            l5s_candidates += len(l5s_available)*[i]

        l5l_transitions = np.where(neighbors_map[i][:,-1]==2)[0]
        l5l_available = np.where(catalog[neighbors_map[i][l5l_transitions,0]][:,-2]==1)[0]
        l5l_amount += len(l5l_available)

        if len(l5l_available)>0:
            l5l_candidates += len(l5l_available)*[i]

        l6l_transitions = np.where(neighbors_map[i][:,-1]==3)[0]
        l6l_available = np.where(catalog[neighbors_map[i][l6l_transitions,0]][:,-2]==1)[0]
        l6l_amount += len(l6l_available)

        if len(l6l_available)>0:
            l6l_candidates += len(l6l_available)*[i]

    gc.collect()

    return [s5l_amount, l5s_amount, l5l_amount, l6l_amount],\
        [s5l_candidates, l5s_candidates, l5l_candidates, l6l_candidates]

def init_event(catalog, neighbors_map, states, transition_type):
    departure, arrival = states
    candidates = []
    amount_of_candidates = 0
    # Checking for s5l transitions for occupied sites
    for i in catalog[catalog[:,5]==departure][:,0]:
        # Checking where are s5l transitions
        transitions = np.where(neighbors_map[i][:,-1]==transition_type)[0]

        # Checking where are empty neighbor sites
        available = np.where(catalog[neighbors_map[i][transitions,0]][:,-2]==arrival)[0]

        # Adding corresponding amount of transitions
        amount_of_candidates += len(available)

        # Adding transition times the considered ID
        if len(available)>0:
            candidates += len(available)*[i]

    return candidates, available

def init_events(catalog, neighbors_map):
    events = np.zeros(16,dtype=object)
    amounts = np.zeros(16,dtype=int)
    status = [[1,2], [1,1], [0,2], [0,1]]
    transition_type = [0, 1, 2, 3]
    pos = 0
    for i in range(len(transition_type)):
        for j in range(len(status)):
            events[pos], amounts[pos] = init_event(catalog, neighbors_map, status[j], transition_type[i])
            pos+=1

    return events, amounts