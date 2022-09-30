import numpy as np
import networkx as nx

from utils import dist_point_segments
from iterative_geometry_solver import iterative_geometry_solver

def kernel(dist_arr):
    p = np.exp(-dist_arr**2/np.min(dist_arr)**2)
    return p / np.sum(p)

def acceptance_probability(old_cost, new_cost, temperature):
    if temperature < 1e-8:
        if new_cost < old_cost:
            return 1
        else:
            return 0
    arg = -(new_cost - old_cost) / temperature
    if arg >= 0:
        return 1.
    else:
        return np.exp(arg)


def monte_carlo_step(topo, sample_edge_list, cost, coords_arr, bot_problem_dict, temperature):
    """
    input: tree topology, sample_edge_list, cost, and node positions, bot_problem_dict
    output: tree topology, cost, and node positions, acceptance as boolean

    for the topology use the usual labelling conventions.
    """
    topo_old = topo.copy()  # used to be deepcopy.
    coords_arr_old = np.copy(coords_arr)

    # extract problem:
    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]
    dim = len(coords_sources[0])

    # assert dim == 2, "so far this works only in 2D."

    # randomly sample an edge which is cut:
    edge = sample_edge_list[np.random.choice(np.arange(len(sample_edge_list)))]
    sample_edge_list.remove(edge)

    # remove edge and get two components:
    topo.remove_edge(edge[0], edge[1])
    comp1 = list(nx.connected_components(topo))[0]
    comp2 = list(nx.connected_components(topo))[1]

    # decide who is connector (the component with smaller end) and who must hence be a BP.
    if len(comp1) <= len(comp2):
        if edge[0] in comp1:
            connector = edge[0]
            bp = edge[1]
        else:
            connector = edge[1]
            bp = edge[0]
    else:
        if edge[0] in comp2:
            connector = edge[0]
            bp = edge[1]
        else:
            connector = edge[1]
            bp = edge[0]

    # Cutting the above edge splits the tree into 2 components:
    bp_free = None
    if nx.degree(topo, bp) == 2:
        # remove bp and connect neighbors:
        n1, n2 = nx.neighbors(topo, bp)
        topo.remove_node(bp)
        topo.add_edge(n1, n2)
        bp_free = bp
        if bp in comp1:
            comp1.remove(bp)
        else:
            comp2.remove(bp)

    # --> the chosen edge must be an edge from the other component:
    if connector in comp1:
        comp_of_interest = comp2
    else:
        comp_of_interest = comp1

    # setup edges_arr:
    edges_of_interest = []
    edges_arr = np.array([])
    for node in comp_of_interest:
        for neighbour in nx.neighbors(topo, node):
            if node < neighbour:
                edge = (node, neighbour)
                if bp_free is not None:
                    if n1 in edge and n2 in edge:
                        #pass
                        continue  # do not want to end up with the same topology after the modification
                edges_of_interest.append((node, neighbour))
                edges_arr = np.append(edges_arr, coords_arr[node])
                edges_arr = np.append(edges_arr, coords_arr[neighbour])

    edges_arr = edges_arr.reshape((-1, 2*dim))
    if len(edges_of_interest) == 0:
        # we would try again the same revert to old state:
        print("no edges to connect to.")
        return topo_old, sample_edge_list, cost, coords_arr_old, False

    # randomly sample a close edge in the other component:
    connector_coords = coords_arr[connector]
    dist_child_edges = dist_child_edges = dist_point_segments(connector_coords, edges_arr[:, :dim], edges_arr[:, dim:])
    dist_propabilities = kernel(dist_child_edges)

    # else, continue by choosing an edge:
    ind_chosen = np.random.choice(np.arange(len(dist_propabilities)), p=dist_propabilities)
    chosen_edge = edges_of_interest[ind_chosen]

    # connect closest edge to connector with bp_new:
    if bp_free is not None:
        bp_new = bp_free
    else:
        bp_new = np.min(topo.nodes()) - 1

    topo.remove_edge(chosen_edge[0], chosen_edge[1])
    topo.add_edges_from([(chosen_edge[0], bp_new), (bp_new, chosen_edge[1]), (bp_new, connector)])

    new_cost, coords_iter = iterative_geometry_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                                      relative_improvement_threshold=1e-6, min_iterations=-1,
                                                      max_iterations=1000,
                                                      plot=False, title="", fov=None, save=False, save_name="img")

    # analyse how often the same topo is tried, which is unnecessarily costly:
    if bp_free is not None and n1 in chosen_edge and n2 in chosen_edge:
        print("same topo tried., cost=", cost, "new_cost=", new_cost)

    p_acceptance = acceptance_probability(cost, new_cost, temperature)

    if np.random.random() < p_acceptance:
        # new state accepted:
        return topo, list(topo.edges()), new_cost, coords_iter, True
    else:
        # revert to old state:
        return topo_old, sample_edge_list, cost, coords_arr_old, False
