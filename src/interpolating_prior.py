import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt

sys.path.append('/')
sys.path.append('../numerical BP optimization/')

from MST_prior import get_MST
from utils import *
from scipy.optimize import linprog
import ot
import time

# return a topo where all terminals have degree 1 that is a beta-interpolation between the MST and OT edges.
def interpolated_prior(bot_problem_dict, beta):
    time0 = time.time()

    # solve OT:
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    # use OT package to solve optimal transport
    M = ot.dist(coords_sources, coords_sinks, 'euclidean')
    flow_mat = ot.emd(supply_arr, demand_arr, M)  #solve optimal transport

    OT_topo = nx.Graph()
    coords_ext = np.vstack((coords_sources, coords_sinks))
    for n in range(len(coords_ext)):
        OT_topo.add_node(n, pos=coords_ext[n])

    # setup OT edges and their cost:
    num_edges = len(supply_arr) + len(demand_arr) - 1
    OT_edges_arr = np.zeros((num_edges, 2))
    OT_edges_dist = np.zeros(num_edges)
    count = 0
    for i, flow in enumerate(flow_mat.reshape(-1)):
        if flow > 1e-9:
            ind_pair = np.array(np.unravel_index(i, M.shape))
            OT_edges_dist[count] = M[ind_pair[0], ind_pair[1]]
            ind_pair[1] += len(supply_arr)
            OT_edges_arr[count] = ind_pair
            count += 1
    OT_topo.add_edges_from(OT_edges_arr)

    time1 = time.time()
    #print("time for OT solution:", time1-time0)

    # find the MST and get edges and cost of edges there:
    coords = np.vstack((coords_sources, coords_sinks))

    # construct all index pairs
    edges_arr = np.zeros((int(len(coords_ext) * (len(coords_ext) - 1) / 2), 2), dtype=int)
    count = 0
    for i in range(len(coords_ext)):
        for j in range(i):
            edges_arr[count] = np.array([i, j])
            count += 1

    # fast way to calculate distances:
    weight_arr = np.sqrt(np.sum((coords_ext[edges_arr[:, 0]] - coords_ext[edges_arr[:, 1]]) ** 2, axis=1))

    MST_topo, MST_edges_arr, MST_edges_dist = get_MST(edges_arr, weight_arr, coords_ext)

    time2 = time.time()
    #print("time for MST solution:", time2-time1)

    # consider the joint set of edges with reduces edge cost for OT edges:
    edges_arr_joint = np.vstack((MST_edges_arr, OT_edges_arr))
    dist_arr_joint = np.append(MST_edges_dist, OT_edges_dist * beta)

    MST_joint, _, _ = get_MST(edges_arr_joint, dist_arr_joint, coords)

    # last step turn this into a suitable tree topology, where each terminal has degree 1.
    # for every node in MST joint with more than degree 1 add a branching point:
    label_min = 0
    for node in list(MST_joint.nodes()):
        if MST_joint.degree(node) == 1:
            continue
        else:
            neighbours = list(nx.neighbors(MST_joint, node))
            neighbours.append(node)
            label_min -= 1
            add_edges_arr = np.ones((len(neighbours), 2), dtype=int) * label_min
            add_edges_arr[:, 1] = neighbours
            MST_joint.remove_node(node)
            MST_joint.add_edges_from(add_edges_arr)

    time3 = time.time()
    #print("time for joint solution:", time3-time2)

    return MST_joint
