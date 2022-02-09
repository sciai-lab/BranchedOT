import numpy as np
import networkx as nx
import ot
from scipy.optimize import linprog

# first get the cost matix:
# at position (i,j) it holds the distance of the ith source to the jth sink
def cost_mat(coords_sources, coords_sinks):
    distance_matrix = np.zeros((len(coords_sources), len(coords_sinks)))

    # construct all index pairs
    index_pairs = np.zeros((len(coords_sources) * len(coords_sinks), 2), dtype=int)
    count = 0
    for i in range(len(coords_sources)):
        for j in range(len(coords_sinks)):
            index_pairs[count] = np.array([i ,j])
            count += 1

    # fast way to calculate distances:
    distance_matrix[index_pairs[: ,0], index_pairs[: ,1]] = np.sqrt(np.sum((coords_sources[index_pairs[: ,0]]
                                                                    - coords_sinks[index_pairs[: ,1]] )**2, axis=1))
    return distance_matrix


# next set up the A_eq, i.e. constraint matrix:
def constr_mat(coords_sources, coords_sinks):
    n = len(coords_sources)
    m = len(coords_sinks)
    A = np.zeros(( n +m , n *m))

    # upper part for a_i:
    for i in range(0 ,n):
        new_line = np.zeros( n *m)
        new_line[ i *m: i * m +m] = 1
        A[i ,:] = new_line * 1

    # lower part for b_j:
    for i in range(n, n+ m):
        j = i - n
        new_line = np.zeros(n * m)
        new_line[j::m] = 1
        A[i, :] = new_line * 1

    return A


"""
input:
- coords_sources and coords_sinks of BOT problem
- supply and demand_arr of BOT problem

output:
- OT topology turned into a tree topology (probably not binary) where all terminals have degree 1  
"""

def OT_prior_topology(coords_sources, coords_sinks, supply_arr, demand_arr):

    num_terminals = len(supply_arr) + len(demand_arr)

    # use OT package to solve optimal transport
    M = ot.dist(coords_sources, coords_sinks, 'euclidean')
    X = ot.emd(supply_arr, demand_arr, M)  # solve optimal transport # X[i,j] is amount of mass which source i sends to sink j

    non_zero_ind1, non_zero_ind2 = np.where(X > 1e-9)
    non_zero_indeces = np.vstack((non_zero_ind1, non_zero_ind2 + len(supply_arr))).T

    # create topology:
    topo = nx.Graph()
    terminal_bp_edges = np.zeros((num_terminals, 2), dtype=int)
    terminal_bp_edges[:, 0] = np.arange(num_terminals)
    terminal_bp_edges[:, 1] = - np.arange(num_terminals) - 1
    topo.add_edges_from(terminal_bp_edges)

    non_zero_indeces_bps = -non_zero_indeces - 1
    topo.add_edges_from(non_zero_indeces_bps)

    # now remove branching points of degree 2:
    for n in range(num_terminals):
        if topo.degree(-n - 1) == 2:
            # connect the two neighbours:
            n1, n2 = nx.neighbors(topo, -n - 1)
            topo.add_edge(n1, n2)
            topo.remove_node(-n - 1)

    # last step: use the labelling convention for the remaining bps:
    count = -1
    mapping_dict = {}
    for node in topo:
        if topo.degree(node) > 1:
            mapping_dict[node] = count
            count -= 1

    return nx.relabel.relabel_nodes(topo, mapping_dict, copy=True)