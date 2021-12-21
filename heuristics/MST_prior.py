import numpy as np
import networkx as nx


def get_MST(edges_arr, weight_arr, coords):
    MST = nx.Graph()
    accepted_edges_idx = []
    for n in range(len(coords)):
        MST.add_node(n, pos=coords[n])
    weight_arr_c = np.copy(weight_arr)

    while True:
        argmin = np.argmin(weight_arr_c)
        index_pair = edges_arr[argmin]
        weight_arr_c[argmin] = np.inf

        # add edge if no cycle
        if not nx.has_path(MST, index_pair[0], index_pair[1]):
            MST.add_edge(index_pair[0], index_pair[1])
            accepted_edges_idx.append(argmin)

        if len(list(nx.connected_components(MST))) == 1:
            break

    return MST, edges_arr[accepted_edges_idx], weight_arr[accepted_edges_idx]



"""
input:
- coords_sources and coords_sinks of BOT problem

output:
- MST turned into a tree topology (probably not binary) where all terminals have degree 1
"""

def MST_prior_topology(coords_sources, coords_sinks):
    # construct the MST from the positions of the sources and sinks:
    # for all pairs of terminals calculate distance and sort edges according to that:
    coords = np.vstack((coords_sources, coords_sinks))
    distance_matrix = np.ones((len(coords), len(coords))) * np.inf

    # construct all index pairs
    index_pairs = np.zeros((int(len(coords) * (len(coords)-1) / 2), 2), dtype=int)
    count = 0
    for i in range(len(coords)):
        for j in range(i):
            index_pairs[count] = np.array([i,j])
            count += 1

    # fast way to calculate distances:
    pair_distance = np.sqrt(np.sum((coords[index_pairs[:,0]] - coords[index_pairs[:,1]])**2, axis = 1))
    distance_matrix[index_pairs[:,0], index_pairs[:,1]] = pair_distance
    distance_matrix[index_pairs[:,1], index_pairs[:,0]] = pair_distance

    # init MST and later topo graph:
    MST = nx.Graph()
    topo = nx.Graph()
    for n in range(len(coords)):
        MST.add_node(n)
        topo.add_edge(n, -n-1)  #the negative indeces are the branching points

    while True:
        argmin_indeces = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        distance_matrix[argmin_indeces] = np.inf

        # add edge if no cycle
        if not nx.has_path(MST, argmin_indeces[0], argmin_indeces[1]):
            MST.add_edge(argmin_indeces[0], argmin_indeces[1])

        if len(list(nx.connected_components(MST))) == 1:
            break

    # turn the completed MST into a topology with branching points where flow is conserved,
    # such that all terminals have degree 1:
    for edge in MST.edges():
        topo.add_edge(-edge[0]-1, -edge[1]-1)  #add an edge between the corresponding branching points

    # now remove branching points of degree 2:
    for n in range(len(coords)):
        if topo.degree(-n-1) == 2:
            #connect the two neighbours:
            n1, n2 = nx.neighbors(topo, -n-1)
            topo.add_edge(n1, n2)
            topo.remove_node(-n-1)

    # last step: use the labelling convention for the remaining bps:
    count = -1
    mapping_dict = {}
    for node in topo:
        if topo.degree(node) > 1:
            mapping_dict[node] = count
            count -= 1

    return nx.relabel.relabel_nodes(topo, mapping_dict, copy=True)