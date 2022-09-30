import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from general_preprocessing import preprocess_from_topo_to_flows
from utils import visualise_BOT_solution


# Given a BOT problem locate the BPs at the COM. COM heuristic.
def COM_locate_BPs(topo, bot_problem_dict, plot=False):
    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    # first step general preprocessing:
    topo, coords_arr, list_bp_idx = preprocess_from_topo_to_flows(topo, supply_arr, demand_arr, coords_sources,
                                                                  coords_sinks, al)

    # topo[node][neighbour]["weight"]
    # set up system of equations Ax=b.
    dim = len(coords_arr[0])
    A = np.zeros((len(list_bp_idx), len(list_bp_idx)))
    b = np.zeros((len(list_bp_idx), dim))
    list_bp_arr = np.array(list_bp_idx)
    for i, bp in enumerate(list_bp_idx):
        total_flow = 0
        for neighbour in nx.neighbors(topo, bp):
            flow = abs(topo[bp][neighbour]["weight"]) ** al
            total_flow += flow
            if neighbour in list_bp_idx:
                # get the respective index:
                neighbour_ind = np.where(list_bp_arr == neighbour)[0][0]
                A[i, neighbour_ind] = - flow  # will contribute like weight * coordinate / sum of weights
            else:
                # i.e. if neighbour = terminal
                b[i, :] += flow * coords_arr[neighbour]

        # normalize by total mass:
        A[i, :] /= total_flow
        b[i, :] /= total_flow
        A[i, i] = 1

    # solve the system:
    coords_arr[len(supply_arr) + len(demand_arr):] = np.flip(np.linalg.solve(A, b), axis=0)

    # visualise COM solution:
    if plot:
        visualise_BOT_solution(topo, coords_arr, supply_arr, demand_arr, title="", fov=None, save=False,
                               save_name="img")

    return coords_arr


# input is coords_arr and topo as graph. Output dict: Assuming root = 0, for each BP give the label of the left child.
def left_child_halfplane_decider(topo, coords_arr):
    assert len(coords_arr[0]) == 2, "this method only works for 2d"
    # create children and parent dict with respect to root node 0:
    children_dict = dict(nx.bfs_successors(topo, 0))
    parent_dict = dict(nx.bfs_predecessors(topo, 0))

    left_label_dict = {}
    for node in topo.nodes():
        if node >= 0:
            continue

        # now node is a bp:
        parent = parent_dict[node]
        child1, child2 = children_dict[node]

        # get all coordinates:
        a0 = coords_arr[parent]
        b = coords_arr[node]
        a1 = coords_arr[child1]
        a2 = coords_arr[child2]

        left_pointer = np.array([[0, -1], [1, 0]]) @ (b - a0)  # rotate b-a0 by 90 degrees counter clockwise

        if (a1 - b) @ left_pointer > 0:
            left_label_dict[node] = child1
        else:
            left_label_dict[node] = child2

    return left_label_dict